import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from uuid import uuid4
import time  # Add time import

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.file import FileTools
from agno.utils.log import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

import psycopg2
from psycopg2.extras import Json

class ValidationScores(BaseModel):
    accuracy: int = Field(..., description="Score for correctness of answer and content accuracy (0-100)")
    clarity: int = Field(..., description="Score for question clarity and unambiguity (0-100)")
    difficulty: int = Field(..., description="Score for appropriate difficulty level (0-100)")
    distractor_quality: int = Field(..., description="Score for plausibility of distractors (0-100)")
    relevance: int = Field(..., description="Score for alignment with source material (0-100)")

class ValidationResult(BaseModel):
    scores: ValidationScores = Field(..., description="Quality scores for different aspects")
    flags: List[str] = Field(default_factory=list, description="List of issues that need attention")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Specific suggestions for improvement")

class ValidatorAgent:
    def __init__(
        self,
        knowledge_dir: str = "knowledge",
        db_config: Dict[str, str] = {
            "dbname": "phunkbyte",
            "user": "postgres",
            "password": "password",
            "host": "localhost",
            "port": "5432"
        }
    ):
        """Initialize the Validator Agent with necessary tools and models."""
        self.knowledge_dir = Path(knowledge_dir)
        self.db_config = db_config
        
        # Initialize tools
        self.file_tools = FileTools()
        
        # Initialize the validator agent with Gemini
        self.validator_agent = Agent(
            model=Gemini(id="gemini-2.0-flash-lite-preview-02-05"),
            tools=[self.file_tools],
            instructions=[
                "You are an expert at validating multiple choice questions for quality and effectiveness.",
                "Evaluate questions based on accuracy, clarity, difficulty appropriateness, and distractor quality.",
                "Provide specific, actionable feedback for improvements.",
                "Flag any issues that need attention or revision."
            ],
            response_model=ValidationResult,
            show_tool_calls=True
        )
        
        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize the PostgreSQL database and required tables."""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                # Create validation_results table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS validation_results (
                        id UUID PRIMARY KEY,
                        question_id UUID REFERENCES questions(id),
                        validator_version VARCHAR(50),
                        quality_scores JSONB,
                        flags TEXT[],
                        improvement_suggestions TEXT[],
                        validated_at TIMESTAMP
                    )
                """)
                conn.commit()

    def _get_question_with_distractors(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a question and its distractors from the database."""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    logger.info(f"Fetching question {question_id} from database")
                    # Get question
                    cur.execute("""
                        SELECT q.*, array_agg(d.distractor_text) as distractors,
                               array_agg(d.similarity_score) as distractor_scores
                        FROM questions q
                        LEFT JOIN distractors d ON q.id = d.question_id
                        WHERE q.id = %s
                        GROUP BY q.id
                    """, (question_id,))
                    
                    result = cur.fetchone()
                    if not result:
                        logger.error(f"Question {question_id} not found in database")
                        return None
                        
                    # Convert to dictionary
                    columns = [desc[0] for desc in cur.description]
                    question_data = dict(zip(columns, result))
                    logger.info(f"Successfully retrieved question data: {question_id}")
                    return question_data
        except Exception as e:
            logger.error(f"Database error while fetching question {question_id}: {str(e)}")
            return None

    def _get_source_material(self, source_id: str) -> str:
        """Retrieve source material content for validation."""
        try:
            source_dir = self.knowledge_dir / "structured_materials" / source_id / "processed"
            if not source_dir.exists():
                logger.warning(f"Source directory not found: {source_dir}")
                return ""
                
            content = []
            for md_file in sorted(source_dir.glob("*.md")):
                logger.info(f"Reading source material from: {md_file}")
                with open(md_file, 'r') as f:
                    content.append(f.read())
                    
            return "\n\n".join(content)
        except Exception as e:
            logger.error(f"Error reading source material: {str(e)}")
            return ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _validate_question(self, question: Dict[str, Any], source_material: str) -> ValidationResult:
        """Validate a question against multiple criteria."""
        try:
            logger.info("Starting question validation")
            prompt = f"""
            Please validate the following multiple choice question against the source material.
            
            Question: {question['question_text']}
            Correct Answer: {question['correct_answer']}
            Explanation: {question['explanation']}
            Difficulty Level: {question['difficulty_level']}
            Distractors: {', '.join(question['distractors'])}
            
            Source Material:
            {source_material[:2000]}  # Limit source material length
            
            Evaluate the question on:
            1. Accuracy: Is the correct answer definitively right?
            2. Clarity: Is the question clear and unambiguous?
            3. Difficulty: Does it match the assigned level?
            4. Distractor Quality: Are the distractors plausible but clearly incorrect?
            5. Relevance: Does it align well with the source material?
            
            Provide scores (0-100) for each aspect and list any issues or improvement suggestions.
            """
            
            logger.info("Sending validation request to agent")
            response = self.validator_agent.run(prompt)
            logger.info("Received validation response from agent")
            
            # Add delay to respect rate limits
            time.sleep(2)
            return response.content
        except Exception as e:
            logger.error(f"Error during question validation: {str(e)}")
            raise

    def _save_validation_result(self, question_id: str, result: ValidationResult):
        """Save validation results to the database."""
        try:
            logger.info(f"Saving validation result for question {question_id}")
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO validation_results (
                            id, question_id, validator_version,
                            quality_scores, flags,
                            improvement_suggestions, validated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        str(uuid4()), question_id, "1.0",
                        Json(result.scores.model_dump()),
                        result.flags,
                        result.improvement_suggestions,
                        datetime.now()
                    ))
                    conn.commit()
                    logger.info(f"Successfully saved validation result for question {question_id}")
                    
        except Exception as e:
            logger.error(f"Failed to save validation result: {str(e)}")
            raise

    def validate_question(self, question_id: str) -> Optional[ValidationResult]:
        """Main method to validate a question."""
        try:
            logger.info(f"Starting validation process for question {question_id}")
            
            # Get question data
            question = self._get_question_with_distractors(question_id)
            if not question:
                logger.error(f"Could not retrieve question {question_id}")
                return None
                
            # Get source material
            source_material = self._get_source_material(question['source_material_id'])
            if not source_material:
                logger.warning(f"No source material found for question {question_id}")
                
            # Validate question
            try:
                logger.info("Performing validation")
                result = self._validate_question(question, source_material)
                
                # Save validation result
                logger.info("Saving validation results")
                self._save_validation_result(question_id, result)
                
                # Log validation summary
                logger.info(f"Validation completed for question {question_id}")
                logger.info(f"Scores: {result.scores}")
                if result.flags:
                    logger.warning(f"Flags: {result.flags}")
                
                return result
                
            except Exception as e:
                logger.error(f"Validation failed: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to validate question: {str(e)}")
            return None

    def validate_batch(self, question_ids: List[str]) -> Dict[str, ValidationResult]:
        """Validate a batch of questions."""
        results = {}
        for question_id in question_ids:
            result = self.validate_question(question_id)
            if result:
                results[question_id] = result
        return results
    
    def _get_question_ids_for_source(self, source_id: str) -> List[str]:
        """Get all question IDs for a given source ID."""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id FROM questions WHERE source_material_id = %s
                """, (source_id,))
                return [row[0] for row in cur.fetchall()]

    def validate_batch_question_with_source(self, source_id: str) -> Optional[ValidationResult]:
        """Validate a batch of questions with source material."""
        question_ids = self._get_question_ids_for_source(source_id)
        return self.validate_batch(question_ids)

# Example usage
if __name__ == "__main__":
    validator = ValidatorAgent()
    # Example question ID from your database
    result = validator.validate_question("d1dc0276-7c27-433f-8e91-8bda4f1e9383")
    if result:
        print(f"Validation scores: {result.scores}")
        if result.flags:
            print(f"Issues found: {result.flags}") 