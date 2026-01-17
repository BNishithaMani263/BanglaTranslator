document.addEventListener('DOMContentLoaded', function () {
    const translateForm = document.getElementById('translateForm');
    const crawlForm = document.getElementById('crawlForm');
    const searchForm = document.getElementById('searchForm');
    const translateBtn = document.getElementById('translateBtn');
    const crawlBtn = document.getElementById('crawlBtn');
    const loading1 = document.getElementById('loading1');
    const loading2 = document.getElementById('loading2');
    const loadingSearch = document.getElementById('loadingSearch');

    // Log page load
    console.log('Page loaded. Initializing forms and animations.');

    // Debug translated text section
    const translatedSection = document.querySelector('[data-section="translated_text"]');
    if (translatedSection) {
        const translatedText = translatedSection.querySelector('.text-muted');
        console.log('Translated text section found:', {
            present: !!translatedText,
            content: translatedText?.getAttribute('data-content') || 'None',
            fullLength: translatedText?.textContent?.length || 0
        });
    } else {
        console.warn('Translated text section not found.');
    }

    // Debug results section
    const resultsSection = document.querySelector('[data-section="results"]');
    console.log('Results section rendered:', !!resultsSection);

    // Debug search results section
    const searchResultsSection = document.querySelector('[data-section="search_results"]');
    console.log('Search results section found:', {
        present: !!searchResultsSection,
        resultsCount: searchResultsSection?.getAttribute('data-results-count') || 0
    });

    // Handle Translate Form Submission
    if (translateForm) {
        translateForm.addEventListener('submit', function (e) {
            const text = document.getElementById('text').value.trim();
            const file = document.getElementById('file').value;
            console.log('Translate form submitted:', { text, file });
            if (!text && !file) {
                e.preventDefault();
                alert('Please provide Bangla text or upload a file.');
                return;
            }
            if (loading1) {
                console.log('Showing translate spinner');
                loading1.classList.remove('d-none');
                loading1.style.opacity = '0';
                loading1.animate([
                    { opacity: 0, transform: 'translateY(20px)' },
                    { opacity: 1, transform: 'translateY(0)' }
                ], { duration: 500, easing: 'ease-in-out', fill: 'forwards' });
                translateBtn.disabled = true;
                translateBtn.style.transform = 'scale(0.95)';
            }
        });
    } else {
        console.warn('Translate form not found.');
    }

    // Handle Crawl Form Submission
    if (crawlForm) {
        crawlForm.addEventListener('submit', function (e) {
            const url = document.getElementById('url').value.trim();
            console.log('Crawl form submitted:', { url });
            if (!url || !url.match(/^https?:\/\/.+/)) {
                e.preventDefault();
                alert('Please enter a valid URL starting with http:// or https://.');
                return;
            }
            if (loading2) {
                console.log('Showing crawl spinner');
                loading2.classList.remove('d-none');
                loading2.style.opacity = '0';
                loading2.animate([
                    { opacity: 0, transform: 'translateY(20px)' },
                    { opacity: 1, transform: 'translateY(0)' }
                ], { duration: 500, easing: 'ease-in-out', fill: 'forwards' });
                crawlBtn.disabled = true;
                crawlBtn.style.transform = 'scale(0.95)';
            }
        });
    } else {
        console.warn('Crawl form not found.');
    }

    // Handle Search Form Submission
    if (searchForm) {
        searchForm.addEventListener('submit', function (e) {
            const keyword = document.getElementById('keyword').value.trim();
            const cacheKey = searchForm.querySelector('input[name="cache_key"]').value;
            console.log('Search form submitted:', { keyword, cacheKey });
            if (!keyword) {
                e.preventDefault();
                alert('Please enter a search keyword.');
            } else {
                if (loadingSearch) {
                    console.log('Showing search spinner');
                    loadingSearch.classList.remove('d-none');
                    loadingSearch.style.opacity = '0';
                    loadingSearch.animate([
                        { opacity: 0, transform: 'translateY(20px)' },
                        { opacity: 1, transform: 'translateY(0)' }
                    ], { duration: 500, easing: 'ease-in-out', fill: 'forwards' });
                }
            }
        });
    } else {
        console.warn('Search form not found.');
    }

    // Highlight search keywords and debug results
    const highlightElements = document.querySelectorAll('.search-result');
    console.log('Found search result elements:', highlightElements.length);
    highlightElements.forEach(function (element, index) {
        const keywordInput = element.querySelector('.highlight-text')?.getAttribute('data-keyword');
        if (keywordInput) {
            const keywords = keywordInput.toLowerCase().split(/\s+/);
            let text = element.querySelector('.highlight-text').innerHTML;
            keywords.forEach(keyword => {
                const regex = new RegExp(`(${keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
                text = text.replace(regex, '<mark>$1</mark>');
            });
            element.querySelector('.highlight-text').innerHTML = text;
            console.log('Highlighted keywords:', keywords, 'in element:', element.id);
            element.style.opacity = '0';
            element.animate([
                { opacity: 0, transform: 'translateX(-20px)' },
                { opacity: 1, transform: 'translateX(0)' }
            ], { duration: 500, easing: 'ease-out', fill: 'forwards', delay: index * 100 });
        } else {
            console.warn('No keyword found for element:', element.id);
        }
    });

    // Pagination function
    window.goToPage = function (page) {
        if (searchForm) {
            const pageInput = searchForm.querySelector('input[name="page"]');
            const cacheKeyInput = searchForm.querySelector('input[name="cache_key"]');
            console.log('Navigating to page:', page, 'Cache key:', cacheKeyInput.value);
            pageInput.value = page;
            searchForm.submit();
        }
    };

    // Cancel crawl function
    window.cancelCrawl = function () {
        console.log('Cancel crawl requested');
        fetch('/cancel_crawl', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'cancelled') {
                    alert('Crawling cancelled.');
                    if (loading2) {
                        loading2.classList.add('d-none');
                        crawlBtn.disabled = false;
                        crawlBtn.style.transform = 'scale(1)';
                    }
                }
            })
            .catch(error => console.error('Error cancelling crawl:', error));
    };
});