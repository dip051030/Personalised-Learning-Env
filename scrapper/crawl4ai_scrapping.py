import logging
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    LLMExtractionStrategy,
    LLMConfig, CacheMode
)
from datetime import datetime
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
        - The H1 title from the main article body (not the website name)
        - All H2–H4 headings in visible order, from the main body only
        - Educational facts, definitions, explanations, laws, formulas — as individual sentences under `main_findings`
        - Bullet lists **only if** items are full sentences and convey educational meaning
        - For `content`, concatenate all `main_findings` in order, preserving exact wording

        Exclude:
        - Any HTML, CSS, JavaScript, or styling elements
        - Page structure content (headers, footers, navbars, sidebars, menus)
        - Ads, cookie notices, related articles, timestamps, author bios, comments
        - Social media buttons, links, external references, or promotional content
        - Code blocks, UI labels, subscription forms, quotes, and placeholder text (e.g., "Click here to...")
        - Incomplete fragments, layout-only text, or decorative headings

        ---

        Output Guidelines:
        - If a field like `headings` or `main_findings` is not present, return it as an empty list: []
        - If `title` or `content` is missing, return it as: null
        - Always include the exact original `url` as received
        - Output must be **valid, strict JSON** — no markdown, no explanation, no freeform text
        - JSON must be fully parseable for machine learning and curriculum ingestion pipelines

        ---

        Your Role:
        You are a non-generative extractor. You must not invent, paraphrase, summarize, or infer any content. Only return **visible**, **factual**, and **educational** material that appears on the webpage. Maintain high precision and consistency — this data will power curriculum tools, knowledge graphs, and AI tutors.
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
                        "source": extracted.get("source", ""),  # e.g. "byjus.com"
                        "subject": extracted.get("subject", ""),  # e.g. "Physics"
                        "grade": extracted.get("grade", None),  # e.g. 11 (int) or None
                        "unit": extracted.get("unit", ""),  # e.g. "Electricity and Magnetism"
                        "topic_title": extracted.get("topic_title", None),  # optional, e.g. "Coulomb’s law"
                        "title": extracted.get("title"),
                        "headings": extracted.get("headings", []),
                        "main_findings": extracted.get("main_findings", []),
                        "content": extracted.get("content"),
                        "keywords": extracted.get("keywords", []),  # new field for keywords
                        "word_count": extracted.get(
                            "word_count",
                            len(" ".join(extracted.get("main_findings", [])).split())
                        ),
                        "status": "success",
                        "scraped_at": datetime.utcnow().isoformat() + "Z"
                    })
                    extraction_strategy.show_usage()

            except Exception as e:
                logging.error(f"Error crawling {url}: {e}")
                results.append({
                    "url": url,
                    "source": "",
                    "subject": "",
                    "grade": None,
                    "unit": "",
                    "topic_title": None,
                    "title": None,
                    "headings": [],
                    "main_findings": [],
                    "content": None,
                    "keywords": [],
                    "word_count": 0,
                    "status": "failed",
                    "scraped_at": datetime.utcnow().isoformat() + "Z"
                })

    return results