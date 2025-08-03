import asyncio
import logging
import json
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    LLMExtractionStrategy,
    LLMConfig, CacheMode
)
from keys.apis import set_env


def safe_model_dump(obj):
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, '__dict__'):
        return {
            k: safe_model_dump(v)
            for k, v in obj.__dict__.items()
            if not isinstance(getattr(type(obj), k, None), property)
        }
    elif isinstance(obj, dict):
        return {k: safe_model_dump(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [safe_model_dump(i) for i in obj]
    else:
        return obj


logging.basicConfig(level=logging.INFO)

async def crawl_and_extract_json(urls: list) -> list:
    browser_cfg = BrowserConfig(headless=False, verbose=True)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.ENABLED)

    api_token = set_env('GROQ_DEEPSEEK_API_KEY')
    if not api_token:
        raise ValueError("Environment variable 'GROQ_DEEPSEEK_API_KEY' is not set or invalid.")

    llm_cfg = LLMConfig(
        provider='groq/deepseek-r1-distill-llama-70b',
        api_token=api_token,
        temperature=0
    )

    extraction_strategy = LLMExtractionStrategy(
        llm_config=llm_cfg,
        instruction="""
From crawl content
Strictly ignore:
- any HTML tags
- embedded CSS
- navigation bars
- sidebars, footers, cookie banners, subscription boxes
- scripting or metadata

Only return the clean content in this JSON format:
{{
  "title": "string",
  "grade_level": "string or null",
  "main_concepts": ["..."],
  "summary": "..."
  "keywords": ["..."]
}0  
""", input_format='markdown')

    results = []
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for url in urls:
            try:
                logging.info(f"Crawling -> {url}")
                result = await crawler.arun(
                    url=url,
                    config=run_config,
                    extraction_strategy=extraction_strategy
                )

                raw = safe_model_dump(result)
                results.append({
                    'url': url,
                    'extracted_json': {
                        "title": raw.get("title"),
                        "grade_level": raw.get("grade_level"),
                        "subject": raw.get("subject"),
                        "main_concepts": raw.get("main_concepts"),
                        "summary": raw.get("summary")
                    },
                    'status': 'success'
                })
                print(results[0])
            except Exception as e:
                logging.error(f"Error crawling {url}: {e}")
                results.append({
                    'url': url,
                    'error': str(e),
                    'status': 'failed'
                })

    return results