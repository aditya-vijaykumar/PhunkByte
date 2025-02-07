import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import uuid4

from agno.agent import Agent
from agno.tools.firecrawl import FirecrawlTools
from agno.tools.file import FileTools
from agno.models.google import Gemini
from agno.utils.log import logger

class ScraperAgent:
    def __init__(self, knowledge_dir: str = "knowledge"):
        """Initialize the Scraper Agent with necessary tools and models."""
        self.knowledge_dir = Path(knowledge_dir)
        self.structured_materials_dir = self.knowledge_dir / "structured_materials"
        
        # Initialize tools and models
        self.file_tools = FileTools()
        self.firecrawl_tools = FirecrawlTools(scrape=False, crawl=True)
        
        # Initialize the content processor agent with Gemini
        self.processor_agent = Agent(
            model=Gemini(id="gemini-2.0-flash-lite-preview-02-05"),
            tools=[self.file_tools],
            instructions=[
                "You are a content structuring expert that converts documentation into well-organized study material.",
                "Break down content into clear sections with learning objectives.",
                "Identify and highlight key concepts.",
                "Maintain source attribution and metadata.",
                "Format output in a way that's optimal for learning and quiz generation."
            ],
            show_tool_calls=True,
            markdown=True
        )

    def _ensure_directories(self, source_id: str) -> Dict[str, Path]:
        """Ensure all necessary directories exist and return their paths."""
        source_dir = self.structured_materials_dir / source_id
        raw_dir = source_dir / "raw"
        processed_dir = source_dir / "processed"

        for dir_path in [self.structured_materials_dir, source_dir, raw_dir, processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        return {
            "source": source_dir,
            "raw": raw_dir,
            "processed": processed_dir
        }

    def _get_url_identifier(self, url: str) -> str:
        """Convert URL to a valid filename."""
        return url.split('//')[-1].split('?')[0].replace('-', '_').replace('/', '_').replace('.', '_').lower()

    def _check_existing_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Check if content for the given URL already exists in knowledge directory."""
        # First check in knowledge directory for raw files
        url_id = self._get_url_identifier(url)
        raw_file = self.knowledge_dir / f"{url_id}.json"
        
        if raw_file.exists():
            logger.info(f"Using cached content for {url}")
            try:
                with open(raw_file, 'r') as f:
                    # We only cache the raw content, not the processed result
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Cache file corrupted: {url_id}.json")
                return None

        logger.info(f"No cache found for {url}")
        return None

    def _save_raw_content(self, url: str, content: Dict[str, Any]):
        """Save raw content to knowledge directory."""
        url_id = self._get_url_identifier(url)
        raw_file = self.knowledge_dir / f"{url_id}.json"
        
        # Ensure directory exists
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(raw_file, 'w') as f:
            json.dump(content, f, indent=2)
        logger.info(f"Saved raw content: {url_id}.json")

    def _update_manifest(self, url: str, source_id: str, metadata: Dict[str, Any]):
        """Update the main manifest file with new content information."""
        manifest_path = self.structured_materials_dir / "manifest.json"
        manifest = {}
        
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

        manifest[url] = {
            "source_id": source_id,
            "last_updated": datetime.now().isoformat(),
            "metadata": metadata
        }

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def _process_content(self, content: str) -> str:
        """Process raw content using Gemini to create structured study material in markdown format."""
        prompt = f"""
        Please structure the following documentation content into a well-organized study material format.
        Break it down into clear sections with the following structure:

        # Title
        Brief overview/introduction

        ## Learning Objectives
        - Objective 1
        - Objective 2
        ...

        ## Key Concepts
        - Concept 1: Brief explanation
        - Concept 2: Brief explanation
        ...

        ## Content Sections
        ### Section 1
        Content...

        ### Section 2
        Content...

        ## Summary
        Brief summary of main points

        ## Difficulty Level
        Indicate difficulty level (1-10) and explain why

        Content to process:
        {content}
        """
        
        response = self.processor_agent.run(prompt)
        return response.content

    def process_url(self, url: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Main method to process a URL and generate structured study material."""
        logger.info(f"Processing URL: {url}")

        # Initialize raw_content
        raw_content = None

        # Check existing content unless force refresh is requested
        if not force_refresh:
            raw_content = self._check_existing_content(url)
            if raw_content:
                logger.info("Found cached content, processing it...")
            else:
                logger.info("No cache found, will scrape fresh content")

        # If no cache or force refresh, scrape new content
        if raw_content is None:
            # Scrape content using Firecrawl
            logger.info("Scraping with Firecrawl...")
            raw_content = self.firecrawl_tools.crawl_website(url=url, limit=10)
            
            # Parse the raw content if it's a string
            if isinstance(raw_content, str):
                try:
                    raw_content = json.loads(raw_content)
                    logger.info("Successfully parsed Firecrawl response")
                except json.JSONDecodeError:
                    logger.error("Failed to parse Firecrawl response")
                    return {}

            # Save raw content
            self._save_raw_content(url, raw_content)

        # Generate source ID for new content
        source_id = str(uuid4())
        dirs = self._ensure_directories(source_id)
        
        # Process content sections
        logger.info("Processing content sections...")
        processed_content = []
        total_sections = len(raw_content.get("data", []))
        
        for idx, item in enumerate(raw_content.get("data", []), 1):
            logger.info(f"Processing section {idx}/{total_sections}")
            structured_content = self._process_content(item.get("markdown", ""))
            processed_content.append(structured_content)

        # Save processed content as markdown files
        logger.info("Saving processed content...")
        processed_dir = dirs["processed"]
        for idx, content in enumerate(processed_content, 1):
            content_file = processed_dir / f"section_{idx:03d}.md"
            with open(content_file, 'w') as f:
                f.write(content)

        # Create and save manifests
        logger.info("Updating manifests...")
        source_manifest = {
            "url": url,
            "source_id": source_id,
            "created_at": datetime.now().isoformat(),
            "sections_count": len(processed_content),
            "status": "completed"
        }

        # Save source manifest
        with open(dirs["source"] / "manifest.json", 'w') as f:
            json.dump(source_manifest, f, indent=2)

        # Update main manifest
        self._update_manifest(url, source_id, source_manifest)
        
        logger.info(f"Completed processing {url}")
        return source_manifest

# Example usage
if __name__ == "__main__":
    scraper = ScraperAgent()
    try:
        result = scraper.process_url("https://docs.uniswap.org/concepts/uniswap-protocol")
        logger.info(f"Successfully processed URL. Sections: {result['sections_count']}")
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")


