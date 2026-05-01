
import os
import json
import logging
from typing import List, Dict, Optional
from firecrawl import FirecrawlApp
from urllib.parse import urlparse
from datetime import datetime
from mcp.server.fastmcp import FastMCP

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRAPE_DIR = "scraped_content"

mcp = FastMCP("llm_inference")

@mcp.tool()
def scrape_websites(
    websites: Dict[str, str],
    formats: List[str] = ['markdown', 'html'],
    api_key: Optional[str] = None
) -> List[str]:
    """
    Scrape multiple websites using Firecrawl and store their content.
    
    Args:
        websites: Dictionary of provider_name -> URL mappings
        formats: List of formats to scrape ['markdown', 'html'] (default: both)
        api_key: Firecrawl API key (if None, expects environment variable)
        
    Returns:
        List of provider names for successfully scraped websites
    """
    
    if api_key is None:
        api_key = os.getenv('FIRECRAWL_API_KEY')
        if not api_key:
            raise ValueError("API key must be provided or set as FIRECRAWL_API_KEY environment variable")
    
    app = FirecrawlApp(api_key=api_key)
    
    path = os.path.join(SCRAPE_DIR)
    os.makedirs(path, exist_ok=True)
    
    # save the scraped content to files and then create scraped_metadata.json as a summary file
    # check if the provider has already been scraped and decide if you want to overwrite
    # {
    #     "cloudrift_ai": {
    #         "provider_name": "cloudrift_ai",
    #         "url": "https://www.cloudrift.ai/inference",
    #         "domain": "www.cloudrift.ai",
    #         "scraped_at": "2025-10-23T00:44:59.902569",
    #         "formats": [
    #             "markdown",
    #             "html"
    #         ],
    #         "success": "true",
    #         "content_files": {
    #             "markdown": "cloudrift_ai_markdown.txt",
    #             "html": "cloudrift_ai_html.txt"
    #         },
    #         "title": "AI Inference",
    #         "description": "Scraped content goes here"
    #     }
    # }
    metadata_file = os.path.join(path, "scraped_metadata.json")

    # Load existing metadata
    try:
        with open(metadata_file, 'r') as file:
            scraped_metadata = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        scraped_metadata = {}                

    # Create placeholder for successful scrapes
    successful_scrape = []

    # Begin scraping each website
    for provider_name, url in websites.items():
        logger.info(f"Scraping {provider_name}: {url}")
        try:
            # Call Firecrawl
            scrape_result = app.scrape(url, formats=formats).model_dump()

            # Update metadata
            metadata = {
                "provider": provider_name,
                "url": url,
                "domain": urlparse(url).netloc,
                "scraped_at": datetime.now().isoformat(),
                "success": False,
            }

            # If we got any requested format back, treat it as a success
            content_found = any(scrape_result.get(format) for format in formats)

            # Return the scraped content in the specified formats and save to files                
            if content_found:
                content_files = {} # Initialize the dictionary
                
                # Save the scraped content to files based on formats
                for format in formats:
                    filename = f"{provider_name}_{format}.txt" # Fixed variable name
                    filepath = os.path.join(SCRAPE_DIR, filename)
                    with open(filepath, 'w', encoding='utf-8') as file:
                        file.write(scrape_result.get(format, ''))
                    
                    content_files[format] = filename # Record the filename

                # Add metadata
                metadata.update({
                    "title": scrape_result['metadata'].get('title', ''),
                    "description": scrape_result['metadata'].get('description', ''),
                    "content_files": content_files, # Add the mapping to metadata
                    "success": True
                })

                # Add Successful Scrape to the List
                successful_scrape.append(provider_name)

                logger.info(f"Successfully scraped {provider_name} and saved to {filepath}")
                
            else:

                logger.warning(f"No content scraped from {provider_name} at {url}")

        except Exception as e:
            logger.error(f"Error scraping {provider_name} at {url}: {e}")

        scraped_metadata[provider_name] = metadata

    # Save the updated metadata to file
    with open(metadata_file, 'w', encoding='utf-8') as file:
        json.dump(scraped_metadata, file, indent=2)

    # Log final results
    logger.info(f"Scraping completed. Successfully scraped: {successful_scrape}")

    return successful_scrape



@mcp.tool()
def extract_scraped_info(identifier: str) -> str:
    """
    Extract information about a scraped website.
    
    Args:
        identifier: The provider name, full URL, or domain to look for
        
    Returns:
        Formatted JSON string with the scraped information
    """
    
    logger.info(f"Extracting information for identifier: {identifier}")
    logger.info(f"Files in {SCRAPE_DIR}: {os.listdir(SCRAPE_DIR)}")

    metadata_file = os.path.join(SCRAPE_DIR, "scraped_metadata.json")
    logger.info(f"Checking metadata file: {metadata_file}")

    # contine your response here ...
    try:
        with open(metadata_file, 'r', encoding='utf-8') as file:
            scraped_metadata = json.load(file)

        # Search for the identifier in provider name, URL, or domain
        for provider_name, metadata in scraped_metadata.items():
            if (identifier == provider_name or 
                identifier == metadata.get('url') or 
                identifier == metadata.get('domain')):
                
                logger.info(f"Found matching metadata for identifier: {identifier}")
                result = metadata.copy()
                if 'content_files' in result.keys():
                    result['content'] = {}
                    for format_type, filename in result['content_files'].items():
                        content_path = os.path.join(SCRAPE_DIR, filename)
                        try:
                            with open(content_path, 'r', encoding='utf-8') as content_file:
                                result['content'][format_type] = content_file.read()
                        except Exception as e:
                            logger.error(f"Error reading content file {content_path}: {e}")
                            result['content'][format_type] = f"Error reading content: {e}"
                return json.dumps(result, indent=2)
        
        # If the loop finishes without returning, we didn't find it.
        logger.warning(f"No matching metadata found for identifier: {identifier}")
        return f"There's no saved information related to identifier '{identifier}'."
    except Exception as e:
        logger.error(f"Error reading metadata file {metadata_file}: {e}")
        return f"Error reading metadata: {e}"

if __name__ == "__main__":
    mcp.run(transport="stdio")