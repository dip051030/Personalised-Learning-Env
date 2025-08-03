import asyncio
import logging
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    LLMExtractionStrategy,
    LLMConfig, CacheMode
)
from keys.apis import set_env

logging.basicConfig(level=logging.INFO)

async def crawl_and_extract_json(urls: list) -> list:
    browser_cfg = BrowserConfig(headless=True, verbose=True)
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
        Extract the following fields from the webpage content:
        - title
        - grade_level
        - subject
        - main_concepts
        - summary
        Output strictly as JSON with these keys.
        """
    )

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

                results.append({
                    'url': url,
                    'extracted_json': result.json(),  # <- FIXED
                    'status': 'success'
                })
            except Exception as e:
                logging.error(f"Error crawling {url}: {e}")
                results.append({
                    'url': url,
                    'error': str(e),
                    'status': 'failed'
                })

    return results