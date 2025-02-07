import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from uuid import uuid4

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.file import FileTools
from agno.utils.log import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

import psycopg2
from psycopg2.extras import Json

class Question(BaseModel):
    question: str = Field(..., description="The question text that tests understanding rather than mere recall")
    correct_answer: str = Field(..., description="The definitive correct answer")
    explanation: str = Field(..., description="Detailed explanation of why this is the correct answer")
    concept_tags: List[str] = Field(..., description="Relevant concept tags for this question")
    difficulty_level: int = Field(..., description="Difficulty level from 1-10")
    difficulty_explanation: str = Field(..., description="Explanation of why this difficulty level was chosen")

class Distractor(BaseModel):
    text: str = Field(..., description="The distractor option text that is plausible but incorrect")
    similarity_score: float = Field(..., description="How similar this distractor is to the correct answer (0.0-1.0)")
    explanation: str = Field(..., description="Why this option is wrong but plausible")

class DistractorList(BaseModel):
    distractors: List[Distractor] = Field(..., description="List of plausible but incorrect options")

class QuestionGeneratorAgent:
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
        """Initialize the Question Generator Agent with necessary tools and models."""
        self.knowledge_dir = Path(knowledge_dir)
        self.db_config = db_config
        
        # Initialize tools
        self.file_tools = FileTools()
        
        # Initialize the question generator agent with Gemini
        self.question_agent = Agent(
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[self.file_tools],
            instructions=[
                "You are an expert at creating multiple choice questions from study material.",
                "Generate questions that test understanding rather than mere recall.",
                "Create questions of varying difficulty levels.",
                "Ensure questions are clear, unambiguous, and have one definitive correct answer."
            ],
            response_model=Question,
            show_tool_calls=True
        )
        
        # Initialize the distractor generator agent
        self.distractor_agent = Agent(
            model=Gemini(id="gemini-2.0-flash-exp"),
            instructions=[
                "You are an expert at creating plausible but incorrect options for multiple choice questions.",
                "Create distractors that are similar enough to be plausible but different enough to be clearly wrong.",
                "Ensure distractors maintain consistent length and style with the correct answer.",
                "Avoid obviously wrong or joke answers."
            ],
            response_model=DistractorList,
            show_tool_calls=True
        )
        
        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize the PostgreSQL database and required tables."""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                # Create questions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS questions (
                        id UUID PRIMARY KEY,
                        source_material_id UUID,
                        question_text TEXT,
                        correct_answer TEXT,
                        explanation TEXT,
                        difficulty_level INTEGER,
                        concept_tags TEXT[],
                        metadata JSONB DEFAULT NULL,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP
                    )
                """)
                
                # Create distractors table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS distractors (
                        id UUID PRIMARY KEY,
                        question_id UUID REFERENCES questions(id),
                        distractor_text TEXT,
                        similarity_score FLOAT DEFAULT NULL,
                        created_at TIMESTAMP
                    )
                """)
                conn.commit()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _generate_question(self, content: str, difficulty: int) -> Dict[str, Any]:
        """Generate a single question with specified difficulty."""
        prompt = f"""
        Generate a multiple choice question from the following content with difficulty level {difficulty}/10.
        
        Content:
        {content}
        
        The question should:
        1. Test understanding rather than mere recall
        2. Have a clear, unambiguous answer
        3. Be appropriate for the specified difficulty level
        4. Include relevant concept tags
        """
        
        response = self.question_agent.run(prompt)
        # Convert Pydantic model to dictionary
        result = response.content.model_dump()
        # Add delay to respect rate limits
        time.sleep(2)
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _generate_distractors(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate plausible distractors for a question."""
        prompt = f"""
        Generate 3 plausible but incorrect options for the following multiple choice question.
        
        Question: {question['question']}
        Correct Answer: {question['correct_answer']}
        
        Each distractor should:
        1. Be clearly wrong but plausible
        2. Have similar length and style to the correct answer
        3. Be tempting to someone who partially understands the concept
        """
        
        response = self.distractor_agent.run(prompt)
        # Convert Pydantic model to dictionary and get distractors list
        result = response.content.model_dump()["distractors"]
        # Add delay to respect rate limits
        time.sleep(2)
        return result

    def _save_question(self, question: Dict[str, Any], source_id: str) -> Optional[str]:
        """Save a question and its distractors to the database."""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    # Generate question ID
                    question_id = str(uuid4())
                    
                    # Insert question
                    cur.execute("""
                        INSERT INTO questions (
                            id, source_material_id, question_text, correct_answer,
                            explanation, difficulty_level, concept_tags, metadata,
                            created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        question_id, source_id, question['question'],
                        question['correct_answer'], question['explanation'],
                        question['difficulty_level'], question['concept_tags'],
                        Json({'difficulty_explanation': question['difficulty_explanation']}),
                        datetime.now(), datetime.now()
                    ))
                    
                    # Generate and insert distractors
                    distractors = self._generate_distractors(question)
                    logger.info(f"Generated {len(distractors)} distractors for question {question_id}")
                    for distractor in distractors:
                        cur.execute("""
                            INSERT INTO distractors (
                                id, question_id, distractor_text,
                                similarity_score, created_at
                            ) VALUES (%s, %s, %s, %s, %s)
                        """, (
                            str(uuid4()), question_id,
                            distractor['text'], distractor['similarity_score'],
                            datetime.now()
                        ))
                    
                    conn.commit()
                    return question_id
                    
        except Exception as e:
            logger.error(f"Failed to save question: {str(e)}")
            return None

    def process_material(self, source_id: str, difficulty_range: range = range(1, 11)) -> List[str]:
        """Process study material and generate questions for each difficulty level."""
        logger.info(f"Processing material from source {source_id}")
        
        # Get the source directory
        source_dir = self.knowledge_dir / "structured_materials" / source_id / "processed"
        if not source_dir.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return []
        
        question_ids = []
        
        # Process each markdown file in the directory
        for md_file in sorted(source_dir.glob("*.md")):
            logger.info(f"Processing file: {md_file.name}")
            
            # Read content
            with open(md_file, 'r') as f:
                content = f.read()
            
            # Generate questions for each difficulty level
            for difficulty in difficulty_range:
                logger.info(f"Generating question with difficulty {difficulty}/10")
                question = self._generate_question(content, difficulty)
                
                if question:
                    question_id = self._save_question(question, source_id)
                    if question_id:
                        question_ids.append(question_id)
                        logger.info(f"Saved question {question_id}")
        
        logger.info(f"Generated {len(question_ids)} questions")
        return question_ids

# Example usage
if __name__ == "__main__":
    generator = QuestionGeneratorAgent()
    # Use a source_id from your structured_materials directory
    question_ids = generator.process_material("e1a104f5-de79-4fc3-a8f6-47f5ee87fccd")
    print(f"Generated questions: {len(question_ids)}") 