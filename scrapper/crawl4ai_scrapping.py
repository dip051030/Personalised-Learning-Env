import logging
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    LLMExtractionStrategy,
    LLMConfig, CacheMode
)
import json
from keys.apis import set_env
from schemas import WebCrawlerConfig

logging.basicConfig(level=logging.INFO)

async def crawl_and_extract_json(urls: list) -> list:

    browser_cfg = BrowserConfig(
        browser_type="firefox",
        headless=False,
        verbose=True,
        light_mode=False
    )



    api_token = set_env('GROQ_DEEPSEEK_API_KEY')
    if not api_token:
        raise ValueError("Environment variable 'GROQ_DEEPSEEK_API_KEY' is not set or invalid.")

    llm_cfg = LLMConfig(
        provider='ollama/llama3',
        temperature=0,
    )

    extraction_strategy = LLMExtractionStrategy(
        llm_config=llm_cfg,
        instruction='''Extract the main educational content from the webpage, ignoring all HTML tags, scripts, and styles.
        Return the content in a structured JSON format according to the schema.''',
        input_format='markdown',
        schema=WebCrawlerConfig.model_json_schema()
    )

    crawl_cfg = CrawlerRunConfig(
        excluded_tags=[
            'footer',
            'nav',
            'aside',
            'script',
            'style',
            'link',
            'form',
            'noscript',
            'iframe',
            'svg',
            'canvas',
            'input',
            'button',
            'select',
            'option',
            'label',
            'object',
            'embed',
            'video',
            'audio'
        ],
        excluded_selector=
            '.ads, .advertisement, .sponsored, .promo, .sidebar, .related-links, .comments, .comment, '
            '.social-links, .share-buttons, .social-media, .footer, .footer-links, .footer-info, '
            '.footer-text, .footer-logo, .footer-social, .footer-contact, .footer-legal, .footer-privacy, '
            '.footer-terms, .footer-copyright, .footer-disclaimer, .footer-sitemap, '
            '.footer-subscribe, .footer-newsletter, .footer-contact-form, .footer-address, '
            '.footer-menu, .cookie-notice, .cookie-banner, .cookies, .popup, .modal, '
            '.popup-overlay, .popup-content, .popup-close, .popup-header, .popup-body, '
            '.popup-footer, .popup-buttons, .popup-link, .login, .signup, .login-form, '
            '.register, .auth, .nav, .navbar, .navigation, .menu, .topbar, .toolbar, '
            '.header, .masthead, .banner, .cta, .newsletter, .subscribe, .sticky, '
            '.chatbot, .livechat, .intercom-launcher, .notifications, .alert, .announcement, '
            '.breadcrumb, .pagination, .loader, .loading, .spinner, .hero, .widget, .widget-area, '
            '.search-box, .search-form, .search-bar, .scroll-to-top, .back-to-top, .branding, '
            '.related-posts, .related-articles, .more-articles, .external-links, .print-button',
        only_text = True,
        remove_forms=True,
        magic=True,
        exclude_external_links=True,
        exclude_social_media_links=True,
        verbose=True,
        extraction_strategy=extraction_strategy,
    )

    results = []
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for url in urls:
            try:
                logging.info(f"Crawling -> {url}")

                result = await crawler.arun(
                    url=url,
                    config=crawl_cfg
                )

                if result.success:
                    logging.info(f"Successfully crawled {url}")

                    results.append({
                        'url': url,
                        'extracted_json': json.loads(result.extracted_content),
                        'status': 'success'
                    })

                    extraction_strategy.show_usage()

            except Exception as e:
                logging.error(f"Error crawling {url}: {e}")
                results.append({
                    'url': url,
                    'error': str(e),
                    'status': 'failed'
                })

    return results