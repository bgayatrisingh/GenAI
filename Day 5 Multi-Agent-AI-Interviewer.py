import sys
import logging
import uuid
import sqlite3
import json
from pathlib import Path
from typing import Dict, List
from fastapi import FastAPI, HTTPException, status
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
class DatabaseManager:
    def __init__(self):
        self.db_path = Path("ai_mentor.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._initialize_tables()

    def _initialize_tables(self):
        cursor = self.conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                goal TEXT,
                target_company TEXT,
                current_level TEXT,
                preferred_topics TEXT
            );
            
            CREATE TABLE IF NOT EXISTS questions (
                question_id TEXT PRIMARY KEY,
                user_id TEXT,
                question TEXT,
                options TEXT,
                correct_answer TEXT,
                explanation TEXT,
                topic TEXT,
                difficulty TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS responses (
                response_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                question_id TEXT,
                answer TEXT,
                is_correct BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

# Base agent class
class BaseAgent:
    def __init__(self, db: DatabaseManager, llm: ChatOpenAI):
        self.db = db
        self.llm = llm

# Specialized agents
class AssessmentAgent(BaseAgent):
    def update_profile(self, user_id: str, profile_data: Dict):
        cursor = self.db.conn.cursor()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO users 
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                profile_data.get('goal'),
                profile_data.get('target_company'),
                profile_data.get('current_level'),
                json.dumps(profile_data.get('preferred_topics', []))
            ))
            self.db.conn.commit()
            return {"status": "Profile updated"}
        except sqlite3.Error as e:
            self.db.conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

class QuestionGeneratorAgent(BaseAgent):
    def generate_questions(self, user_id: str, num_questions: int = 3):
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        profile = cursor.fetchone()
        
        prompt = ChatPromptTemplate.from_template("""
        Generate {num_questions} interview questions for {target_company} targeting a {current_level} candidate.
        Focus on: {topics}
        Each question must have:
        - Clear question text
        - 4 options (A-D)
        - Correct answer (A-D)
        - Explanation
        - Relevant topic
        - Difficulty (Easy/Medium/Hard)
        
        Format as JSON list with keys: question, options, correct_answer, explanation, topic, difficulty
        """)
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            questions = chain.invoke({
                "num_questions": num_questions,
                "target_company": profile["target_company"],
                "current_level": profile["current_level"],
                "topics": json.loads(profile["preferred_topics"])
            })
            
            questions_with_ids = []
            stored_ids = []
            
            for i, q in enumerate(questions, 1):
                qid = f"Q_{i}"  # Simple sequential IDs
                stored_ids.append(qid)
                
                question_data = {
                    "id": qid,
                    "question": q["question"],
                    "options": q["options"],
                    "correct_answer": q["correct_answer"],
                    "explanation": q["explanation"],
                    "topic": q["topic"],
                    "difficulty": q["difficulty"]
                }
                questions_with_ids.append(question_data)
                
                # Store in database
                cursor.execute("""
                    INSERT INTO questions 
                    (question_id, user_id, question, options, correct_answer, explanation, topic, difficulty)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    qid,
                    user_id,
                    q["question"],
                    json.dumps(q["options"]),
                    q["correct_answer"],
                    q["explanation"],
                    q["topic"],
                    q["difficulty"]
                ))
            
            self.db.conn.commit()
            return {
                "message": "Questions generated successfully",
                "questions": questions_with_ids
            }
            
        except Exception as e:
            self.db.conn.rollback()
            raise HTTPException(status_code=500, detail=str(e))

class EvaluationAgent(BaseAgent):
    def evaluate_response(self, user_id: str, question_id: str, answer: str):
        cursor = self.db.conn.cursor()
        try:
            cursor.execute("""
                SELECT correct_answer FROM questions 
                WHERE question_id = ? AND user_id = ?
            """, (question_id, user_id))
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Question not found")
            
            is_correct = answer.upper() == result["correct_answer"]
            
            cursor.execute("""
                INSERT INTO responses (user_id, question_id, answer, is_correct)
                VALUES (?, ?, ?, ?)
            """, (user_id, question_id, answer.upper(), is_correct))
            
            self.db.conn.commit()
            return {
                "is_correct": is_correct,
                "correct_answer": result["correct_answer"]
            }
            
        except sqlite3.Error as e:
            self.db.conn.rollback()
            raise HTTPException(status_code=500, detail=str(e))

class ProgressAgent(BaseAgent):
    def get_progress(self, user_id: str):
        cursor = self.db.conn.cursor()
        try:
            # Overall stats
            cursor.execute("""
                SELECT 
                    COUNT(*) AS total_questions,
                    SUM(is_correct) AS correct_answers,
                    AVG(is_correct) AS accuracy
                FROM responses
                WHERE user_id = ?
            """, (user_id,))
            stats = cursor.fetchone()
            
            # Topic performance
            cursor.execute("""
                SELECT 
                    q.topic,
                    COUNT(*) AS total,
                    SUM(r.is_correct) AS correct
                FROM responses r
                JOIN questions q ON r.question_id = q.question_id
                WHERE r.user_id = ?
                GROUP BY q.topic
            """, (user_id,))
            topics = [dict(row) for row in cursor.fetchall()]
            
            return {
                "stats": dict(stats),
                "topics": topics
            }
        except sqlite3.Error as e:
            raise HTTPException(status_code=500, detail=str(e))

class Orchestrator:
    def __init__(self):
        self.db = DatabaseManager()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        self.agents = {
            "assessment": AssessmentAgent(self.db, self.llm),
            "question_generator": QuestionGeneratorAgent(self.db, self.llm),
            "evaluator": EvaluationAgent(self.db, self.llm),
            "progress": ProgressAgent(self.db, self.llm)
        }
        
        self.intent_parser = JsonOutputParser(pydantic_object=self.IntentSchema)
        self.intent_prompt = ChatPromptTemplate.from_template("""
        Classify the user's intent from their message:
        {message}
        
        Options:
        - update_profile: Setting/changing goals/preferences
        - get_questions: Requesting practice questions
        - submit_answer: Answer submission (looks like "A", "Answer: B", etc)
        - check_progress: Asking about stats/performance
        - other: Anything else
        
        Respond ONLY with JSON: {{"intent": "<intent>", "details": {{...}}}}
        """)

    class IntentSchema(BaseModel):
        intent: str
        details: Dict = {}

    def route_request(self, user_id: str, message: str):
        # Detect intent
        chain = self.intent_prompt | self.llm | self.intent_parser
        intent_data = chain.invoke({"message": message})
        
        # Route to appropriate agent
        match intent_data["intent"]:
            case "update_profile":
                return self._handle_profile_update(user_id, message)
            case "get_questions":
                return self.agents["question_generator"].generate_questions(user_id)
            case "submit_answer":
                return self._handle_answer_submission(user_id, message)
            case "check_progress":
                return self.agents["progress"].get_progress(user_id)
            case _:
                return {"response": "I can help with interview prep. Ask me to generate questions, check your progress, or update your profile."}

    def _handle_profile_update(self, user_id: str, message: str):
        extract_prompt = ChatPromptTemplate.from_template("""
        Extract user profile details from:
        {message}
        
        Look for:
        - Goal/role (e.g., "software engineer")
        - Target company
        - Experience level
        - Topics/skills of interest
        
        Return JSON with keys: goal, target_company, current_level, preferred_topics
        """)
        
        chain = extract_prompt | self.llm | JsonOutputParser()
        profile_data = chain.invoke({"message": message})
        return self.agents["assessment"].update_profile(user_id, profile_data)

    def _handle_answer_submission(self, user_id: str, message: str):
        """Handle answer submission with question ID"""
        try:
            # Extract question ID and answer from message
            # Expected format: "Q1: A" or "Question 1: A" or "1: A"
            message = message.strip().upper()
            
            # Different possible formats for answer submission
            if ":" in message:
                # Format: "Q1: A" or "1: A"
                q_id, answer = message.split(":")
                q_id = q_id.strip().replace("Q", "").replace("QUESTION", "")
                answer = answer.strip()
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid answer format. Please use format 'Q1: A' or '1: A'"
                )

            # Verify question exists for this user
            cursor = self.db.conn.cursor()
            cursor.execute("""
                SELECT question_id, question, options, correct_answer 
                FROM questions 
                WHERE user_id = ? AND question_id = ?
            """, (user_id, f"Q_{q_id}"))
            
            question = cursor.fetchone()
            if not question:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Question Q_{q_id} not found for this user"
                )

            # Evaluate answer
            return self.agents["evaluator"].evaluate_response(
                user_id=user_id,
                question_id=f"Q_{q_id}",
                answer=answer
            )

        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid answer format. Please use format 'Q1: A' or '1: A'"
            )

# FastAPI Setup
app = FastAPI(title="AI Mentor")
orchestrator = Orchestrator()

class UserRequest(BaseModel):
    user_id: str
    message: str

@app.post("/ai-mentor", status_code=200)
async def ai_mentor_endpoint(request: UserRequest):
    try:
        return orchestrator.route_request(request.user_id, request.message)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)