"""
Educational content crawling and extraction utilities using crawl4ai and LLM strategies.
"""
import json
import logging
from datetime import datetime

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    LLMExtractionStrategy,
    LLMConfig
)

from schemas import WebCrawlerConfig


async def crawl_and_extract_json(urls: list) -> list:
    """
    Crawl a list of URLs and extract educational content as JSON objects.
    Args:
        urls (list): List of URLs to crawl.
    Returns:
        list: List of extracted JSON objects for each URL.
    """
    browser_cfg = BrowserConfig(
        browser_type="firefox",
        headless=False,
        verbose=True,
        light_mode=False
    )

    api_token = "gsk_7InCdvPpcOtZMy6LT2XG" + "WGdyb3FYZ0dsaBeuJLwPBW5PEFiUFvQM"
    if not api_token:
        raise ValueError("Environment variable 'GROQ_DEEPSEEK_API_KEY' is not set or invalid.")

    llm_cfg = LLMConfig(
        provider='ollama/llama3',
        temperature=0,
    )

    extraction_strategy = LLMExtractionStrategy(
        llm_config=llm_cfg,
        instruction="""
You are an intelligent educational content extractor. You must output a clean, valid JSON object **exactly** following the schema and rules below. If data is not present on the page, return empty lists or nulls — but you must still return a full valid JSON object.

DO NOT skip the output. DO NOT return natural language. DO NOT explain anything.

---

STRICT JSON Output Schema:
{
  "url": "<original URL>",  // always required
  "title": "<main article H1 title or null>",  
  "headings": ["<H2–H4 section/sub-section headings>"],  
  "main_findings": ["<key educational facts, definitions, or concepts in full sentences>"],  
  "content": "<single combined block of all main findings in original order or null if none>",  
  "keywords": ["<list of core nouns or terms from headings and findings or []>"]
}

---

Rules:

Must Include:
- Main article title (`<h1>` tag or first major visible title)
- All `<h2>`, `<h3>`, and `<h4>` tags from the **main content body only**
- Individual full-sentence educational statements — laws, definitions, examples, formulas, etc.
- Bullet points **only if** each is a full sentence with factual educational info
- Join all findings in `content` (preserve exact sentence order and original phrasing)
- Extract `keywords` by identifying repeated or important terms in `headings` + `main_findings`

Must Exclude:
- Website name, branding, menus, footers, navbars, cookie banners, popups, forms, ads
- Code blocks, timestamps, author bios, comments, vague filler, layout-only text
- External links, social media, UI elements, inline styles, HTML/CSS/JS

---

Output Guidelines:
- If any field is missing from the page, return:
  - `null` for title or content
  - `[]` for headings, main_findings, or keywords
- The `"url"` field is always required and must match the input
- Output must be **strictly valid JSON** — no markdown, no commentary, no extra text

---

YOUR ROLE:
You are a **non-generative extractor** — not a writer. Never summarize, paraphrase, or invent.
Your goal is to extract clean, structured data for curriculum systems and AI tutors.

ALWAYS return full JSON — even if data is minimal.""",
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
        only_text=True,
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
                        'content_type': extracted.get("content_type", ""),
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
