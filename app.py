import os
import logging
import subprocess
import tempfile
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
from flask import Flask, request, jsonify, render_template, session, send_from_directory, make_response
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from pdf2image import convert_from_bytes
import io
import torch
import hashlib
from concurrent.futures import ThreadPoolExecutor
import gc
from langdetect import detect, DetectorFactory
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import psutil
from timeout_decorator import timeout
import re
from fuzzywuzzy import fuzz
from difflib import get_close_matches
from datetime import datetime
from init_db import initialize_database
from db_utils import execute_query

# Ensure consistent language detection
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(process)d] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set environment variable for Hugging Face cache
os.environ["HF_HOME"] = "/data/models"

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Configure SECRET_KEY
SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY:
    logger.error("SECRET_KEY environment variable is not set. Set it in Hugging Face Spaces Secrets.")
    raise ValueError("SECRET_KEY is required for session management.")
app.secret_key = SECRET_KEY
logger.debug(f"SECRET_KEY hash: {hashlib.sha256(SECRET_KEY.encode()).hexdigest()[:8]}...")

# Session configuration
app.config.update(
    SESSION_COOKIE_NAME='session',
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=True,  # Enable for HTTPS in Spaces
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_PATH='/',
    SESSION_COOKIE_DOMAIN=os.environ.get('SPACE_DOMAIN', None),
    PERMANENT_SESSION_LIFETIME=7200,
    APPLICATION_ROOT='/'
)

# Set SERVER_NAME for Spaces
app.config['SERVER_NAME'] = os.environ.get('SPACE_DOMAIN', None)
logger.debug(f"Flask SERVER_NAME set to: {app.config['SERVER_NAME']}")

# Fallback in-memory cache
translation_cache = {}
cache_timeout = 7200

# Model path
MODEL_PATH = "Helsinki-NLP/opus-mt-bn-en"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
CACHE_DIR = "/tmp/ocr_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
MAX_IMAGE_DIMENSION = 600
OCR_TIMEOUT = 30
REQUEST_DELAY = 2
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

model = None
tokenizer = None
cancel_crawl_flag = False

try:
    initialize_database()
    logger.debug("Database initialization completed")
except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}")
    exit(1)

def init_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"user-agent={USER_AGENT}")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.binary_location = os.getenv("CHROMIUM_PATH", "/usr/bin/chromium")
    service = Service(os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver"))
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(30)
    return driver

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    width, height = image.size
    logger.debug(f"Original image dimensions: {width}x{height}")
    target_dpi = 300
    scale = min(target_dpi / 72, MAX_IMAGE_DIMENSION / max(width, height))
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    logger.debug(f"Resized image to: {new_width}x{new_height}")
    image = image.convert("L")
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = ImageEnhance.Sharpness(image).enhance(2.0)
    image_np = np.array(image)
    threshold = 150
    image_np = (image_np > threshold) * 255
    image = Image.fromarray(image_np.astype(np.uint8))
    image = image.filter(ImageFilter.MedianFilter(size=3))
    return image

def split_image(image, max_dim=400):
    width, height = image.size
    segments = []
    x_splits = (width + max_dim - 1) // max_dim
    y_splits = (height + max_dim - 1) // max_dim
    for i in range(x_splits):
        for j in range(y_splits):
            left = i * max_dim
            upper = j * max_dim
            right = min(left + max_dim, width)
            lower = min(upper + max_dim, height)
            segment = image.crop((left, upper, right, lower))
            segments.append(segment)
    return segments

def get_file_hash(file):
    file.seek(0)
    data = file.read()
    file.seek(0)
    return hashlib.md5(data).hexdigest()

def extract_text(file):
    try:
        file_hash = get_file_hash(file)
        cache_path = os.path.join(CACHE_DIR, f"{file_hash}.txt")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                logger.debug(f"Cache hit for file hash: {file_hash}")
                return f.read().strip()
        start_time = time.time()
        logger.debug(f"Memory usage before OCR: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        if file.filename.rsplit('.', 1)[1].lower() == 'pdf':
            file_bytes = file.read()
            images = convert_from_bytes(file_bytes, dpi=300, fmt='png')
            extracted_texts = []
            for img in images:
                img = preprocess_image(img)
                segments = split_image(img) if max(img.size) > 400 else [img]
                for idx, segment in enumerate(segments):
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img_file:
                        segment.save(temp_img_file.name)
                        logger.debug(f"Saved temporary segment {idx}: {temp_img_file.name}")
                        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_txt_file:
                            tesseract_cmd = [
                                'tesseract', temp_img_file.name, temp_txt_file.name[:-4],
                                '-l', 'ben', '--psm', '4', '--oem', '1'
                            ]
                            try:
                                result = subprocess.run(
                                    tesseract_cmd,
                                    timeout=OCR_TIMEOUT,
                                    check=True,
                                    capture_output=True,
                                    text=True
                                )
                                logger.debug(f"Tesseract stdout (segment {idx}): {result.stdout}")
                            except subprocess.TimeoutExpired:
                                logger.error(f"OCR timed out for segment {idx}")
                                os.unlink(temp_img_file.name)
                                os.unlink(temp_txt_file.name)
                                return "OCR timed out. Try a simpler image or PDF."
                            except subprocess.CalledProcessError as e:
                                logger.error(f"Tesseract failed for segment {idx}: {e.stderr}")
                                os.unlink(temp_img_file.name)
                                os.unlink(temp_txt_file.name)
                                return f"Error extracting text: {e.stderr}"
                            with open(temp_txt_file.name, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                                extracted_texts.append(text)
                        os.unlink(temp_img_file.name)
                        os.unlink(temp_txt_file.name)
            text = " ".join(extracted_texts)
        else:
            img = Image.open(file)
            img = preprocess_image(img)
            segments = split_image(img) if max(img.size) > 400 else [img]
            extracted_texts = []
            for idx, segment in enumerate(segments):
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img_file:
                    segment.save(temp_img_file.name)
                    logger.debug(f"Saved temporary segment {idx}: {temp_img_file.name}")
                    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_txt_file:
                        tesseract_cmd = [
                            'tesseract', temp_img_file.name, temp_txt_file.name[:-4],
                            '-l', 'ben', '--psm', '4', '--oem', '1'
                        ]
                        try:
                            result = subprocess.run(
                                tesseract_cmd,
                                timeout=OCR_TIMEOUT,
                                check=True,
                                capture_output=True,
                                text=True
                            )
                            logger.debug(f"Tesseract stdout (segment {idx}): {result.stdout}")
                        except subprocess.TimeoutExpired:
                            logger.error(f"OCR timed out for segment {idx}")
                            os.unlink(temp_img_file.name)
                            os.unlink(temp_txt_file.name)
                            return "OCR timed out. Try a simpler image or PDF."
                        except subprocess.CalledProcessError as e:
                            logger.error(f"Tesseract failed for segment {idx}: {e.stderr}")
                            os.unlink(temp_img_file.name)
                            os.unlink(temp_txt_file.name)
                            return f"Error extracting text: {e.stderr}"
                        with open(temp_txt_file.name, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            extracted_texts.append(text)
                    os.unlink(temp_img_file.name)
                    os.unlink(temp_txt_file.name)
            text = " ".join(extracted_texts)
        if not text.strip() or len(text.strip()) < 10:
            return "No meaningful text extracted. Ensure the file contains clear Bangla text."
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.debug(f"OCR took {time.time() - start_time:.2f} seconds")
        logger.debug(f"Memory usage after OCR: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        gc.collect()
        return text.strip()
    except Exception as e:
        logger.error(f"Error in extract_text: {str(e)}")
        return f"Error extracting text: {str(e)}"

def crawl_single_url(url, headers, use_selenium=False):
    global cancel_crawl_flag
    if cancel_crawl_flag:
        logger.info(f"Crawl cancelled for {url}")
        return "", []
    try:
        time.sleep(REQUEST_DELAY)
        logger.debug(f"Memory usage before crawling {url}: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        if use_selenium:
            driver = init_driver()
            try:
                driver.get(url)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                html = driver.page_source
            finally:
                driver.quit()
        else:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3'], limit=100)
        texts = []
        for element in text_elements:
            text = element.get_text(strip=True)
            if text and len(text) > 10:
                try:
                    if detect(text) == 'bn':
                        texts.append(text)
                except:
                    continue
        bangla_text = " ".join(texts)
        logger.debug(f"Memory usage after crawling {url}: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        return bangla_text, []
    except Exception as e:
        logger.error(f"Error crawling {url}: {str(e)}")
        return "", []

def load_model():
    try:
        logger.debug(f"Loading model and tokenizer from {MODEL_PATH}...")
        start_time = time.time()
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, cache_dir='/data/models')
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir='/data/models')
        logger.debug(f"Model and tokenizer loading took {time.time() - start_time:.2f} seconds")
        if torch.cuda.is_available():
            model = model.cuda()
            logger.debug("Model moved to GPU")
        start_time = time.time()
        dummy_input = tokenizer("আমি", return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            dummy_input = {k: v.cuda() for k, v in dummy_input.items()}
        _ = model.generate(**dummy_input)
        logger.debug(f"Model warm-up took {time.time() - start_time:.2f} seconds")
        logger.debug("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def initialize_model():
    global model, tokenizer
    if model is None and tokenizer is None:
        logger.debug(f"Loading model in process {os.getpid()}...")
        model, tokenizer = load_model()
    else:
        logger.debug(f"Model already loaded in process {os.getpid()}.")
    return model, tokenizer

@timeout(300, timeout_exception=TimeoutError)
def translate_text(sentence, model, tokenizer, url=None):
    start_time = time.time()
    logger.debug(f"Memory usage before translation: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    sentence = sentence[:10000]
    max_length = 512
    inputs = []
    current_chunk = []
    current_length = 0
    sentences = re.split(r'(?<=[।!?])\s+', sentence.strip())
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        token_length = len(tokenizer.tokenize(sent))
        if current_length + token_length > max_length:
            inputs.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_length = token_length
        else:
            current_chunk.append(sent)
            current_length += token_length
    if current_chunk:
        inputs.append(" ".join(current_chunk))
    def translate_chunk(chunk):
        try:
            input_ids = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            if torch.cuda.is_available():
                input_ids = {k: v.cuda() for k, v in input_ids.items()}
            output_ids = model.generate(
                **input_ids,
                max_length=512,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True
            )
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error translating chunk: {str(e)}")
            return f"Error translating chunk: {str(e)}"
    with ThreadPoolExecutor(max_workers=2) as executor:
        translated_chunks = list(executor.map(translate_chunk, inputs))
    translated = " ".join(translated_chunks)
    translated_sentences = re.split(r'(?<=[.!?])\s+', translated.strip())
    try:
        translation_id = execute_query(
            query="INSERT INTO translations (url, extracted_text, translated_text, translated_sentences) VALUES (?, ?, ?, ?)",
            params=(url, sentence, translated, "|".join(translated_sentences))
        )
        logger.debug(f"Inserted translation with ID: {translation_id}")
    except Exception as e:
        logger.error(f"Failed to insert translation: {str(e)}")
        raise
    cache_key = hashlib.md5(f"{url}_{time.time()}".encode()).hexdigest()
    translation_cache[cache_key] = {
        'translation_id': translation_id,
        'timestamp': time.time()
    }
    logger.debug(f"Stored translation_id {translation_id} in cache with key: {cache_key}")
    expired_keys = [k for k, v in translation_cache.items() if time.time() - v['timestamp'] > cache_timeout]
    for k in expired_keys:
        del translation_cache[k]
        logger.debug(f"Removed expired cache key: {k}")
    logger.debug(f"Translation took {time.time() - start_time:.2f} seconds")
    logger.debug(f"Memory usage after translation: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    del inputs, translated_chunks
    gc.collect()
    return translated, translated_sentences, translation_id, cache_key

try:
    model, tokenizer = initialize_model()
except Exception as e:
    logger.error(f"Startup error: {e}")
    exit(1)

@app.before_request
def log_request():
    logger.debug(f"Incoming request: {request.method} {request.path} Cookies: {request.cookies.get('session', 'None')}")

@app.after_request
def log_response(response):
    logger.debug(f"Response headers: {dict(response.headers)}")
    if 'Set-Cookie' in response.headers:
        logger.debug(f"Set-Cookie header: {response.headers['Set-Cookie']}")
    logger.debug(f"Session after response: {dict(session)}")
    return response

@app.route("/", methods=["GET"])
def home():
    logger.debug(f"Current session: {dict(session)}")
    response = make_response(render_template("index.html"))
    response.headers['Cache-Control'] = 'no-store'
    return response

@app.route("/debug_session", methods=["GET"])
def debug_session():
    session['test_key'] = 'test_value'
    session.modified = True
    logger.debug(f"Set test session key: {dict(session)}")
    response = make_response(jsonify({"session": dict(session), "cookies": request.cookies.get('session', 'None')}))
    response.headers['Cache-Control'] = 'no-store'
    return response

def process_web_translate():
    start_time = time.time()
    logger.debug(f"Memory usage before web_translate: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    text = request.form.get("text")
    file = request.files.get("file")
    logger.debug(f"Received text: {text}, file: {file.filename if file else None}")
    if not text and not file:
        return render_template("index.html", error="Please provide text or upload a file.")
    if file and allowed_file(file.filename):
        logger.debug("Starting OCR extraction for uploaded file")
        extracted_text = extract_text(file)
        logger.debug(f"OCR result: {extracted_text}")
        if extracted_text.startswith("Error") or extracted_text.startswith("OCR"):
            return render_template("index.html", error=extracted_text, text=text)
        try:
            if detect(extracted_text) != 'bn':
                return render_template("index.html", error="Extracted text is not in Bangla.", text=text)
        except:
            return render_template("index.html", error="Could not detect language of extracted text.", text=text)
        text_to_translate = extracted_text
    else:
        text_to_translate = text
        if text_to_translate:
            try:
                if detect(text_to_translate) != 'bn':
                    return render_template("index.html", error="Input text is not in Bangla.", text=text)
            except:
                return render_template("index.html", error="Could not detect language of input text.", text=text)
    if not text_to_translate:
        return render_template("index.html", error="No valid text to translate.", text=text)
    logger.debug("Starting translation")
    try:
        translated, translated_sentences, translation_id, cache_key = translate_text(text_to_translate, model, tokenizer)
        session['translation_id'] = translation_id
        session['cache_key'] = cache_key
        session['translated_text'] = translated
        session.permanent = True
        session.modified = True
        logger.debug(f"Set session translation_id: {translation_id}, cache_key: {cache_key}, translated_text: {translated[:50]}...")
    except TimeoutError:
        logger.error("Translation timed out after 300 seconds")
        return render_template("index.html", error="Translation timed out. Try a shorter text or check your subscription for higher limits.", text=text)
    logger.debug(f"Translation result: {translated[:50]}...")
    if translated.startswith("Error"):
        return render_template("index.html", error=translated, text=text, extracted_text=text_to_translate)
    logger.debug(f"Total web_translate took {time.time() - start_time:.2f} seconds")
    logger.debug(f"Memory usage after web_translate: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    logger.debug(f"Current session: {dict(session)}")
    response = make_response(render_template(
        "index.html",
        extracted_text=text_to_translate,
        translated_text=translated,
        text=text,
        cache_key=cache_key
    ))
    response.headers['Cache-Control'] = 'no-store'
    return response

@app.route("/web_translate", methods=["POST"])
def web_translate():
    try:
        start_time = time.time()
        result = process_web_translate()
        if time.time() - start_time > 180:
            raise TimeoutError("Request timed out after 180 seconds")
        return result
    except TimeoutError as e:
        logger.error(f"Request timed out: {str(e)}")
        return render_template("index.html", error="Request timed out. Try a simpler input or check your subscription for higher limits.", text=None)
    except Exception as e:
        logger.error(f"Error in web_translate: {str(e)}")
        return render_template("index.html", error=f"Error processing request: {str(e)}", text=None)

def process_crawl_and_translate():
    global cancel_crawl_flag
    cancel_crawl_flag = False
    start_time = time.time()
    url = request.form.get("url")
    if not url:
        return render_template("index.html", error="Please enter a website URL.")
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        return render_template("index.html", error="Invalid URL format.", url=url)
    logger.debug(f"Starting crawl for URL: {url}")
    headers = {"User-Agent": USER_AGENT}
    extracted_text, _ = crawl_single_url(url, headers, use_selenium=False)
    if not extracted_text:
        logger.debug(f"No Bangla text found with requests for {url}, retrying with Selenium")
        extracted_text, _ = crawl_single_url(url, headers, use_selenium=True)
    if not extracted_text:
        return render_template("index.html", error="No Bangla text found on the page.", url=url)
    try:
        if detect(extracted_text) != 'bn':
            return render_template("index.html", error="Crawled text is not in Bangla.", url=url)
    except:
        return render_template("index.html", error="Could not detect language of crawled text.", url=url)
    logger.debug("Starting translation")
    try:
        translated, translated_sentences, translation_id, cache_key = translate_text(extracted_text, model, tokenizer, url=url)
        session['translation_id'] = translation_id
        session['cache_key'] = cache_key
        session['translated_text'] = translated
        session.permanent = True
        session.modified = True
        logger.debug(f"Set session translation_id: {translation_id}, cache_key: {cache_key}, translated_text: {translated[:50]}...")
    except TimeoutError:
        logger.error("Translation timed out after 300 seconds")
        return render_template("index.html", error="Translation timed out. Try a different URL or check your subscription for higher limits.", url=url)
    if translated.startswith("Error"):
        return render_template("index.html", error=translated, url=url, extracted_text=extracted_text)
    logger.debug(f"Total crawl and translate took {time.time() - start_time:.2f} seconds")
    logger.debug(f"Current session: {dict(session)}")
    response = make_response(render_template(
        "index.html",
        extracted_text=extracted_text,
        translated_text=translated,
        url=url,
        cache_key=cache_key
    ))
    response.headers['Cache-Control'] = 'no-store'
    return response

@app.route("/crawl_and_translate", methods=["POST"])
def crawl_and_translate():
    try:
        start_time = time.time()
        result = process_crawl_and_translate()
        logger.debug(f"Crawl and translate took {time.time() - start_time:.2f} seconds")
        logger.debug(f"Session translation_id: {session.get('translation_id')}")
        if time.time() - start_time > 900:
            raise TimeoutError("Request timed out after 900 seconds")
        return result
    except TimeoutError as e:
        logger.error(f"Crawl and translate request timed out: {str(e)}")
        return render_template("index.html", error="Request timed out. Try a different URL or check your subscription for higher limits.", url=None)
    except Exception as e:
        logger.error(f"Error in crawl_and_translate: {str(e)}")
        return render_template("index.html", error=f"Error processing request: {str(e)}", url=None)

@app.route("/search", methods=["POST"])
def search():
    keyword = request.form.get("keyword")
    page = int(request.form.get("page", 1))
    context_size = int(request.form.get("context_size", 2))
    context_size = max(1, min(5, context_size))
    cache_key = request.form.get("cache_key")
    logger.debug(f"Search request: keyword={keyword}, page={page}, context_size={context_size}, cache_key={cache_key}")
    logger.debug(f"Form data: {dict(request.form)}")
    logger.debug(f"Current session: {dict(session)}")
    if not keyword:
        return render_template("index.html", error="Please enter a search keyword.", translated_text=session.get('translated_text', ''))
    try:
        translation_id = session.get('translation_id')
        session_cache_key = session.get('cache_key')
        translated_text = session.get('translated_text', '')
        logger.debug(f"Session translation_id for search: {translation_id}, session_cache_key: {session_cache_key}, translated_text: {translated_text[:50]}...")
        effective_cache_key = cache_key or session_cache_key
        if not translation_id and effective_cache_key in translation_cache:
            cached = translation_cache.get(effective_cache_key)
            if time.time() - cached['timestamp'] < cache_timeout:
                translation_id = cached['translation_id']
                logger.debug(f"Restored translation_id {translation_id} from cache with key: {effective_cache_key}")
            else:
                del translation_cache[effective_cache_key]
                logger.debug(f"Cache key {effective_cache_key} expired")
        if not translation_id:
            logger.error("No translation_id in session or cache")
            return render_template("index.html", error="No translated text available. Please crawl and translate a page first.", translated_text=translated_text)
        result = execute_query(
            query="SELECT translated_sentences, translated_text FROM translations WHERE id = ?",
            params=(translation_id,),
            fetch=True
        )
        logger.debug(f"Query result: {result}")
        if not result:
            logger.error(f"No translation found for ID: {translation_id}")
            return render_template("index.html", error="Translation not found in database.", translated_text=translated_text)
        translated_sentences = result[0][0].split("|") if result[0][0] else []
        translated_text = result[0][1] or translated_text  # Fallback to database
        logger.debug(f"Retrieved {len(translated_sentences)} sentences, translated_text: {translated_text[:50]}...")
        if not translated_sentences:
            logger.warning("No translated sentences available")
            return render_template("index.html", error="No translated sentences available.", translated_text=translated_text)
        matches = []
        keyword_lower = keyword.lower().strip()
        keywords = keyword_lower.split()
        all_words = set()
        for sentence in translated_sentences:
            all_words.update(sentence.lower().split())
        suggestions = get_close_matches(keyword_lower, all_words, n=3, cutoff=0.8)
        FUZZY_THRESHOLD = 90
        for idx, sentence in enumerate(translated_sentences):
            sentence_lower = sentence.lower()
            exact_match = any(kw in sentence_lower for kw in keywords)
            fuzzy_score = fuzz.partial_ratio(keyword_lower, sentence_lower)
            if exact_match or fuzzy_score >= FUZZY_THRESHOLD:
                start_idx = max(0, idx - context_size)
                end_idx = min(len(translated_sentences), idx + context_size + 1)
                context = " ".join(translated_sentences[start_idx:end_idx])
                matches.append({"id": idx, "context": context, "score": fuzzy_score if not exact_match else 100})
        matches.sort(key=lambda x: x['score'], reverse=True)
        RESULTS_PER_PAGE = 5
        total_matches = len(matches)
        logger.debug(f"Found {total_matches} matches: {matches}")
        total_pages = (total_matches + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * RESULTS_PER_PAGE
        end_idx = start_idx + RESULTS_PER_PAGE
        paginated_matches = matches[start_idx:end_idx]
        logger.debug(f"Paginated matches (page {page}): {paginated_matches}")
        template_vars = {
            "search_results": paginated_matches,
            "keyword": keyword,
            "context_size": context_size,
            "current_page": page,
            "total_pages": total_pages,
            "translated_text": translated_text,
            "cache_key": cache_key,
            "suggestions": suggestions if suggestions else None
        }
        logger.debug(f"Rendering template with variables: {template_vars}")
        response = make_response(render_template(
            "index.html",
            **template_vars
        ))
        response.headers['Cache-Control'] = 'no-store'
        logger.debug(f"Template rendering completed for /search")
        return response
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return render_template("index.html", error=f"Error processing search: {str(e)}", translated_text=session.get('translated_text', ''))

@app.route("/cancel_crawl", methods=["POST"])
def cancel_crawl():
    global cancel_crawl_flag
    cancel_crawl_flag = True
    logger.info("Crawl cancelled by user")
    return jsonify({"status": "cancelled"})

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        sentence = data["text"]
        try:
            if detect(sentence) != 'bn':
                return jsonify({"error": "Input text is not in Bangla."}), 400
        except:
            return jsonify({"error": "Could not detect language of input text."}), 400
        try:
            translated, translated_sentences, translation_id, cache_key = translate_text(sentence, model, tokenizer)
            session['translation_id'] = translation_id
            session['cache_key'] = cache_key
            session['translated_text'] = translated
            session.permanent = True
            session.modified = True
            logger.debug(f"Set session translation_id: {translation_id}, cache_key: {cache_key}, translated_text: {translated[:50]}...")
        except TimeoutError:
            logger.error("Translation timed out after 300 seconds")
            return jsonify({"error": "Translation timed out. Try a shorter text or check your subscription for higher limits."}), 500
        logger.debug(f"Current session: {dict(session)}")
        if translated.startswith("Error"):
            return jsonify({"error": translated}), 500
        return jsonify({"translated_text": translated, "cache_key": cache_key})
    except Exception as e:
        logger.error(f"Error in translate: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    logger.debug(f"Serving static file: {path}")
    return send_from_directory('static', path)

@app.route("/debug_db", methods=["GET"])
def debug_db():
    try:
        result = execute_query("SELECT id, url, timestamp FROM translations", fetch=True)
        logger.debug(f"Database debug: {len(result)} records retrieved")
        return jsonify({"records": result, "count": len(result)})
    except Exception as e:
        logger.error(f"Debug DB error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/debug_search", methods=["GET"])
def debug_search():
    try:
        translation_id = session.get('translation_id', 1)
        translated_text = session.get('translated_text', '')
        keyword = request.args.get("keyword", "Shimon Peres")
        context_size = int(request.args.get("context_size", 2))
        result = execute_query(
            query="SELECT translated_sentences, translated_text FROM translations WHERE id = ?",
            params=(translation_id,),
            fetch=True
        )
        if not result:
            return jsonify({"error": "No translation found for ID", "session": dict(session)}), 404
        translated_sentences = result[0][0].split("|") if result[0][0] else []
        translated_text = result[0][1] or translated_text
        matches = []
        all_words = set()
        for sentence in translated_sentences:
            all_words.update(sentence.lower().split())
        keyword_lower = keyword.lower().strip()
        keywords = keyword_lower.split()
        suggestions = get_close_matches(keyword_lower, all_words, n=3, cutoff=0.8)
        FUZZY_THRESHOLD = 90
        for idx, sentence in enumerate(translated_sentences):
            sentence_lower = sentence.lower()
            exact_match = any(kw in sentence_lower for kw in keywords)
            fuzzy_score = fuzz.partial_ratio(keyword_lower, sentence_lower)
            if exact_match or fuzzy_score >= FUZZY_THRESHOLD:
                start_idx = max(0, idx - context_size)
                end_idx = min(len(translated_sentences), idx + context_size + 1)
                context = " ".join(translated_sentences[start_idx:end_idx])
                matches.append({"id": idx, "context": context, "score": fuzzy_score if not exact_match else 100})
        matches.sort(key=lambda x: x['score'], reverse=True)
        logger.debug(f"Debug search matches: {matches}")
        return jsonify({
            "keyword": keyword,
            "context_size": context_size,
            "matches": matches,
            "sentences": translated_sentences,
            "translated_text": translated_text,
            "suggestions": suggestions,
            "session": dict(session)
        })
    except Exception as e:
        logger.error(f"Debug search error: {str(e)}")
        return jsonify({"error": str(e), "session": dict(session)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)