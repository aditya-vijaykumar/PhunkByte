import json
from typing import Dict, List, Optional, Iterator
from uuid import uuid4

from pydantic import BaseModel, Field
from agno.workflow import Workflow, RunResponse, RunEvent
from agno.storage.workflow.sqlite import SqliteWorkflowStorage
from agno.utils.log import logger

from scraper_agent import ScraperAgent
from question_generator_agent import QuestionGeneratorAgent
from validator_agent import ValidatorAgent
from reviser_agent import ReviserAgent

class ScrapedContent(BaseModel):
    content: str
    metadata: Dict[str, any]
    source_id: str

class GeneratedQuestions(BaseModel):
    questions: List[Dict[str, any]]
    source_id: str

class ValidationResults(BaseModel):
    results: Dict[str, any]
    source_id: str

class RevisedQuestions(BaseModel):
    revisions: Dict[str, any]
    source_id: str

class QuizMasterWorkflow(Workflow):
    """
    A comprehensive workflow that orchestrates the entire quiz generation process:
    1. Scrapes content from provided sources
    2. Generates questions from the content
    3. Validates the questions
    4. Revises questions that don't meet quality standards
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize all agents
        self.scraper = ScraperAgent()
        self.question_generator = QuestionGeneratorAgent()
        self.validator = ValidatorAgent()
        self.reviser = ReviserAgent()

    def run(self, source_url: str, use_cache: bool = True) -> Iterator[RunResponse]:
        """Main workflow execution method."""
        
        logger.info(f"Starting QuizMaster workflow for source: {source_url}")
        source_id = str(uuid4())  # Generate unique source ID

        # Step 1: Check cache for existing results
        if use_cache:
            cached_results = self._get_cached_results(source_url)
            if cached_results:
                yield RunResponse(
                    content=cached_results,
                    event=RunEvent.workflow_completed
                )
                return

        try:
            # Step 2: Scrape content
            yield RunResponse(content="🔍 Scraping content...", event=RunEvent.status_update)
            scraped_content = self._scrape_content(source_url, source_id)
            if not scraped_content:
                yield RunResponse(
                    content=f"Failed to scrape content from: {source_url}",
                    event=RunEvent.workflow_completed
                )
                return

            # Step 3: Generate questions
            yield RunResponse(content="📝 Generating questions...", event=RunEvent.status_update)
            generated_questions = self._generate_questions(scraped_content)
            if not generated_questions or not generated_questions.questions:
                yield RunResponse(
                    content="Failed to generate questions from content",
                    event=RunEvent.workflow_completed
                )
                return

            # Step 4: Validate questions
            yield RunResponse(content="✅ Validating questions...", event=RunEvent.status_update)
            validation_results = self._validate_questions(source_id)
            
            # Step 5: Revise flagged questions if any
            if validation_results and validation_results.results:
                yield RunResponse(content="🔄 Revising flagged questions...", event=RunEvent.status_update)
                revised_questions = self._revise_flagged_questions(source_id)
                
                # Add revised questions to final results
                final_results = {
                    "source_url": source_url,
                    "source_id": source_id,
                    "original_questions": generated_questions.questions,
                    "validation_results": validation_results.results,
                    "revised_questions": revised_questions.revisions if revised_questions else {}
                }
            else:
                final_results = {
                    "source_url": source_url,
                    "source_id": source_id,
                    "questions": generated_questions.questions,
                    "validation_results": "All questions passed validation"
                }

            # Cache the results
            self._cache_results(source_url, final_results)
            
            yield RunResponse(
                content=final_results,
                event=RunEvent.workflow_completed
            )

        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            yield RunResponse(
                content=f"Workflow failed: {str(e)}",
                event=RunEvent.workflow_completed
            )

    def _scrape_content(self, url: str, source_id: str) -> Optional[ScrapedContent]:
        """Scrape content from the provided URL."""
        try:
            content = self.scraper.scrape_url(url)
            return ScrapedContent(
                content=content,
                metadata={"url": url},
                source_id=source_id
            )
        except Exception as e:
            logger.error(f"Scraping failed: {str(e)}")
            return None

    def _generate_questions(self, content: ScrapedContent) -> Optional[GeneratedQuestions]:
        """Generate questions from scraped content."""
        try:
            questions = self.question_generator.generate_questions(
                content.content,
                source_id=content.source_id
            )
            return GeneratedQuestions(
                questions=questions,
                source_id=content.source_id
            )
        except Exception as e:
            logger.error(f"Question generation failed: {str(e)}")
            return None

    def _validate_questions(self, source_id: str) -> Optional[ValidationResults]:
        """Validate generated questions."""
        try:
            results = self.validator.validate_batch_question_with_source(source_id)
            return ValidationResults(
                results=results,
                source_id=source_id
            )
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return None

    def _revise_flagged_questions(self, source_id: str) -> Optional[RevisedQuestions]:
        """Revise questions that were flagged during validation."""
        try:
            revisions = self.reviser.revise_flagged_questions_for_source(source_id)
            return RevisedQuestions(
                revisions=revisions,
                source_id=source_id
            )
        except Exception as e:
            logger.error(f"Revision failed: {str(e)}")
            return None

    def _get_cached_results(self, source_url: str) -> Optional[Dict]:
        """Get cached results for a source URL."""
        return self.session_state.get("quiz_results", {}).get(source_url)

    def _cache_results(self, source_url: str, results: Dict):
        """Cache results for a source URL."""
        self.session_state.setdefault("quiz_results", {})
        self.session_state["quiz_results"][source_url] = results


# Example usage
if __name__ == "__main__":
    from rich.prompt import Prompt

    # Get URL from user
    url = Prompt.ask(
        "[bold]Enter a URL to generate quiz from[/bold]\n✨",
        default="https://docs.uniswap.org/concepts/overview",
    )

    # Initialize workflow
    workflow = QuizMasterWorkflow(
        session_id=f"quiz-master-{url.replace('://', '-').replace('/', '-')}",
        storage=SqliteWorkflowStorage(
            table_name="quiz_master_workflows",
            db_file="tmp/workflows.db",
        ),
    )

    # Execute workflow
    for response in workflow.run(source_url=url, use_cache=True):
        if response.event == RunEvent.status_update:
            print(response.content)
        elif response.event == RunEvent.workflow_completed:
            print("\n✨ Final Results:")
            print(json.dumps(response.content, indent=2)) 