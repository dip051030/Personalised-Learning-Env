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
        instruction="""
        You are an intelligent extractor designed to process educational web pages and return clean, structured JSON content. Follow the schema and rules **exactly**. Do **not** generate, hallucinate, rewrite, or summarize. Only extract visible, factual educational content.

        ---

        JSON Output Schema:
        {
          "url": "<original URL>",
          "title": "<main educational article title or H1>",
          "headings": ["<H2–H4 section and sub-section headings>"],
          "main_findings": ["<concise, factual educational points or definitions stated clearly on the page>"],
          "content": "<single combined text of all educational findings, preserving original phrasing and order>"
        }

        ---

        Extraction Rules:

        Include:
        - Main content title (usually from H1)
        - All H2–H4 headings from the main article body only
        - Main educational points (definitions, explanations, laws, formulas, factual sentences)
        - Bullet lists **only** if they are full sentences and educational
        - For `main_findings`, only extract visible, standalone informative sentences — not filler or layout text
        - The final `content` field must be the joined version of all main findings in correct order

        Exclude:
        - HTML tags, CSS, scripts, styles
        - Navigation bars, sidebars, footers, menus, headers
        - Ads, cookie banners, related articles, timestamps, author bios, comment sections
        - External links, social media elements, promotional blurbs
        - Code blocks, embedded widgets, forms, or UI elements
        - Any placeholder text like “Here is the content from...”
        - Partial sentences, UI text, quotes, or vague phrases

        ---

        Output Guidelines:
        - If a field like "headings" or "main_findings" is not found, return it as an empty list []
        - If "title" or "content" is missing, return as null
        - Always return the original "url" as-is
        - Output must be valid JSON (not markdown, not natural language)
        - The JSON must be fully parseable for downstream ML or curriculum pipelines

        ---

        Your Role:
        You are a non-generative extractor — you do not invent or alter data. Only return factual, clean content that visibly exists on the page and fits the schema. This content will power curriculum systems, knowledge graphs, and educational AI agents. Be strict, accurate, and consistent.
        """,
    input_format='markdown',
        schema=WebCrawlerConfig.model_json_schema()
    )

    crawl_cfg = CrawlerRunConfig(
        magic=True,
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

                    extracted_list = json.loads(result.extracted_content)

                    extracted = extracted_list[0] if isinstance(extracted_list, list) else extracted_list

                    results.append({
                        "url": extracted.get("url", url),
                        "title": extracted.get("title"),
                        "headings": extracted.get("headings", []),
                        "main_findings": extracted.get("main_findings", []),
                        "content": extracted.get("content"),
                        "word_count": extracted.get("word_count",
                                                    len(" ".join(extracted.get("main_findings", [])).split())),
                        "status": "success"
                    })
                    extraction_strategy.show_usage()

            except Exception as e:
                logging.error(f"Error crawling {url}: {e}")
                results.append({
                    "url": url,
                    "title": None,
                    "headings": [],
                    "main_findings": [],
                    "content": None,
                    "word_count": 0,
                    "status": "failed"
                })

    return results