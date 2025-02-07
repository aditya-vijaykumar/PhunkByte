import json
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

class RevisionType(BaseModel):
    type: str = Field(..., description="Type of revision (minor|major)")
    changes: List[str] = Field(..., description="List of changes made")
    feedback_addressed: List[str] = Field(..., description="Validator feedback points that were addressed")

class RevisedQuestion(BaseModel):
    question_text: str = Field(..., description="The revised question text")
    correct_answer: str = Field(..., description="The revised correct answer")
    explanation: str = Field(..., description="Updated explanation for the correct answer")
    difficulty_level: int = Field(..., description="Adjusted difficulty level if needed")
    concept_tags: List[str] = Field(..., description="Updated concept tags")
    distractors: List[Dict[str, Any]] = Field(..., description="List of revised distractor options")
    revision_info: RevisionType = Field(..., description="Information about the revision made")

class ReviserAgent:
    def __init__(
        self,
        knowledge_dir: str = "knowledge",
        db_config: Dict[str, str] = {
            "dbname": "phunkbyte",
            "user": "postgres",
            "password": "myPass",
            "host": "localhost",
            "port": "5435"
        }
    ):
        """Initialize the Reviser Agent with necessary tools and models."""
        self.knowledge_dir = Path(knowledge_dir)
        self.db_config = db_config
        
        # Initialize tools
        self.file_tools = FileTools()
        
        # Initialize the reviser agent with Gemini
        self.reviser_agent = Agent(
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[self.file_tools],
            instructions=[
                "You are an expert at improving multiple choice questions based on validation feedback.",
                "Make targeted improvements while maintaining the original learning objective.",
                "Ensure revisions address all validator concerns comprehensively.",
                "Generate completely new questions when major revisions are needed."
            ],
            response_model=RevisedQuestion,
            show_tool_calls=True
        )
        
        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize the PostgreSQL database and required tables."""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                # Create revision_history table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS revision_history (
                        id UUID PRIMARY KEY,
                        question_id UUID REFERENCES questions(id),
                        previous_version JSONB,
                        revision_type TEXT,
                        changes_made TEXT[],
                        feedback_addressed TEXT[],
                        revised_at TIMESTAMP
                    )
                """)
                conn.commit()

    def _get_question_with_validation(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a question and its validation results."""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            q.*,
                            array_agg(d.distractor_text) as distractors,
                            array_agg(d.similarity_score) as distractor_scores,
                            v.quality_scores,
                            v.flags,
                            v.improvement_suggestions
                        FROM questions q
                        LEFT JOIN distractors d ON q.id = d.question_id
                        LEFT JOIN validation_results v ON q.id = v.question_id
                        WHERE q.id = %s
                        GROUP BY q.id, v.quality_scores, v.flags, v.improvement_suggestions
                    """, (question_id,))
                    
                    result = cur.fetchone()
                    if not result:
                        return None
                    
                    columns = [desc[0] for desc in cur.description]
                    return dict(zip(columns, result))
        except Exception as e:
            logger.error(f"Failed to retrieve question with validation: {str(e)}")
            return None

    def _get_source_material(self, source_id: str) -> str:
        """Retrieve source material content for context."""
        source_dir = self.knowledge_dir / "structured_materials" / source_id / "processed"
        if not source_dir.exists():
            return ""
            
        content = []
        for md_file in sorted(source_dir.glob("*.md")):
            with open(md_file, 'r') as f:
                content.append(f.read())
                
        return "\n\n".join(content)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _revise_question(self, question: Dict[str, Any], source_material: str) -> RevisedQuestion:
        """Revise a question based on validation feedback."""
        prompt = f"""
        Please revise the following multiple choice question based on validator feedback.
        
        Original Question: {question['question_text']}
        Current Answer: {question['correct_answer']}
        Current Explanation: {question['explanation']}
        Current Difficulty: {question['difficulty_level']}
        Current Distractors: {', '.join(question['distractors'])}
        
        Validation Scores: {question['quality_scores']}
        Issues Flagged: {question['flags']}
        Suggested Improvements: {question['improvement_suggestions']}
        
        Source Material:
        {source_material[:2000]}  # Limit source material length
        
        Please provide a revised version that:
        1. Addresses all validator concerns
        2. Maintains or improves question quality
        3. Keeps the same learning objective
        4. Includes improved distractors if needed
        
        If the issues are substantial, create a completely new question testing the same concept.
        """
        
        response = self.reviser_agent.run(prompt)
        # Add delay to respect rate limits
        time.sleep(2)
        return response.content

    def _update_question(self, question_id: str, revision: RevisedQuestion):
        """Update the question with revised content."""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    # Store current version in revision history
                    cur.execute("""
                        INSERT INTO revision_history (
                            id, question_id, previous_version,
                            revision_type, changes_made,
                            feedback_addressed, revised_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        str(uuid4()), question_id,
                        Json(self._get_question_with_validation(question_id)),
                        revision.revision_info.type,
                        revision.revision_info.changes,
                        revision.revision_info.feedback_addressed,
                        datetime.now()
                    ))
                    
                    # Update question
                    cur.execute("""
                        UPDATE questions
                        SET question_text = %s,
                            correct_answer = %s,
                            explanation = %s,
                            difficulty_level = %s,
                            concept_tags = %s,
                            updated_at = %s
                        WHERE id = %s
                    """, (
                        revision.question_text,
                        revision.correct_answer,
                        revision.explanation,
                        revision.difficulty_level,
                        revision.concept_tags,
                        datetime.now(),
                        question_id
                    ))
                    
                    # Delete old distractors
                    cur.execute("DELETE FROM distractors WHERE question_id = %s", (question_id,))
                    
                    # Insert new distractors
                    for distractor in revision.distractors:
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
                    
        except Exception as e:
            logger.error(f"Failed to update question: {str(e)}")
            raise

    def revise_question(self, question_id: str) -> Optional[RevisedQuestion]:
        """Main method to revise a question."""
        logger.info(f"Revising question {question_id}")
        
        # Get question data with validation results
        question = self._get_question_with_validation(question_id)
        if not question:
            logger.error(f"Question {question_id} not found")
            return None
            
        # Get source material
        source_material = self._get_source_material(question['source_material_id'])
        if not source_material:
            logger.warning(f"Source material not found for question {question_id}")
            
        # Revise question
        try:
            revision = self._revise_question(question, source_material)
            
            # Update question in database
            self._update_question(question_id, revision)
            
            # Log revision summary
            logger.info(f"Revision completed for question {question_id}")
            logger.info(f"Revision type: {revision.revision_info.type}")
            logger.info(f"Changes made: {revision.revision_info.changes}")
            
            return revision
            
        except Exception as e:
            logger.error(f"Failed to revise question: {str(e)}")
            return None

    def revise_batch(self, question_ids: List[str]) -> Dict[str, RevisedQuestion]:
        """Revise a batch of questions."""
        results = {}
        for question_id in question_ids:
            result = self.revise_question(question_id)
            if result:
                results[question_id] = result
        return results

    def get_flagged_questions_for_source(self, source_id: str) -> List[Dict[str, Any]]:
        """
        Fetch all questions for a given source_id that have validation flags.
        Returns a list of dictionaries containing question details and their validation flags.
        """
        try:
            logger.info(f"Fetching flagged questions for source {source_id}")
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            q.id as question_id,
                            q.question_text,
                            q.correct_answer,
                            q.explanation,
                            q.difficulty_level,
                            q.concept_tags,
                            v.flags,
                            v.improvement_suggestions,
                            v.quality_scores
                        FROM questions q
                        INNER JOIN validation_results v ON q.id = v.question_id
                        WHERE q.source_material_id = %s
                        AND v.flags IS NOT NULL
                        AND array_length(v.flags, 1) > 0
                    """, (source_id,))
                    
                    columns = [desc[0] for desc in cur.description]
                    results = []
                    
                    for row in cur.fetchall():
                        question_data = dict(zip(columns, row))
                        logger.info(f"Found flagged question {question_data['question_id']} with {len(question_data['flags'])} flags")
                        results.append(question_data)
                    
                    logger.info(f"Found {len(results)} questions with validation flags for source {source_id}")
                    return results
                    
        except Exception as e:
            logger.error(f"Failed to fetch flagged questions: {str(e)}")
            return []

    def revise_flagged_questions_for_source(self, source_id: str) -> Dict[str, RevisedQuestion]:
        """
        Fetch and revise all questions with validation flags for a given source_id.
        Returns a dictionary mapping question IDs to their revised versions.
        """
        try:
            logger.info(f"Starting revision process for flagged questions from source {source_id}")
            
            # Get all flagged questions
            flagged_questions = self.get_flagged_questions_for_source(source_id)
            if not flagged_questions:
                logger.info(f"No flagged questions found for source {source_id}")
                return {}
            
            # Revise each flagged question
            revisions = {}
            for question in flagged_questions:
                logger.info(f"Revising question {question['question_id']} with flags: {question['flags']}")
                revision = self.revise_question(question['question_id'])
                if revision:
                    revisions[question['question_id']] = revision
            
            logger.info(f"Completed revision of {len(revisions)} questions for source {source_id}")
            return revisions
            
        except Exception as e:
            logger.error(f"Failed to process revisions for source {source_id}: {str(e)}")
            return {}

# Example usage
if __name__ == "__main__":
    reviser = ReviserAgent()
    # Example source ID from your database
    source_id = "e1a104f5-de79-4fc3-a8f6-47f5ee87fccd"
    
    # Get all flagged questions for the source
    flagged_questions = reviser.get_flagged_questions_for_source(source_id)
    print(f"Found {len(flagged_questions)} questions with flags")
    
    # Revise all flagged questions
    revisions = reviser.revise_flagged_questions_for_source(source_id)
    print(f"Revised {len(revisions)} questions") 