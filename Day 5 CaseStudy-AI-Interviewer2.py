import sys
import logging
import uuid
from pathlib import Path
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check environment
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    import os
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
        
    import sqlite3
    from fastapi import FastAPI, HTTPException, status
    from langchain_community.chat_models import ChatOpenAI
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from pydantic import BaseModel as PydanticBaseModel
    import json
    
    logger.info("All required packages imported successfully")
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    sys.exit(1)

# Initialize FastAPI app
app = FastAPI(title="AI Interviewer System")

# Database setup with error handling
try:
    db_path = Path("interview_system.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    cursor = conn.cursor()
    logger.info(f"Connected to database at {db_path.absolute()}")
    
    # Create necessary tables
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS user_objectives (
            user_id TEXT PRIMARY KEY,
            goal TEXT NOT NULL,
            target_company TEXT NOT NULL,
            current_level TEXT NOT NULL,
            preferred_topics TEXT NOT NULL,
            improvement_areas TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS questions (
            question_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            question_text TEXT NOT NULL,
            options TEXT NOT NULL,
            correct_answer TEXT NOT NULL,
            explanation TEXT NOT NULL,
            topic TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES user_objectives(user_id)
        );

        CREATE TABLE IF NOT EXISTS user_responses (
            response_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            user_answer TEXT NOT NULL,
            is_correct BOOLEAN NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user_objectives(user_id),
            FOREIGN KEY (question_id) REFERENCES questions(question_id)
        );
    """)

    conn.commit()
except Exception as e:
    logger.error(f"Database initialization error: {str(e)}")
    sys.exit(1)

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.7)

# Pydantic models for request validation
class QuestionSchema(PydanticBaseModel):
    question_text: str
    options: List[str] = Field(..., min_items=4, max_items=4)
    correct_answer: str = Field(..., pattern=r"^[A-D]$")
    explanation: str
    topic: str
    difficulty: str = Field(..., pattern=r"^(Easy|Medium|Hard)$")

class UserObjective(PydanticBaseModel):
    user_id: str
    goal: str
    target_company: str
    current_level: str
    preferred_topics: List[str]

class UserResponse(PydanticBaseModel):
    user_id: str
    question_id: str
    answer: str = Field(..., pattern=r"^[A-Da-d]$")

# LangChain setup for structured output
output_parser = JsonOutputParser(pydantic_object=QuestionSchema)
PROMPT_TEMPLATE = """
You are a professional interview question generator for {target_company}.
Generate {num_questions} {difficulty} level questions about {topics} for a {current_level} candidate applying for {goal}.
Each question must have:
- A clear question stem
- 4 plausible options (A-D)
- Correct answer (A-D)
- Concise explanation
- Relevant topic
- Difficulty level

Format each question as JSON matching this schema:
{format_instructions}
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
chain = prompt | llm | output_parser

@app.post("/set_objective", status_code=status.HTTP_201_CREATED)
async def set_user_objective(objective: UserObjective):
    """Stores user's interview objectives"""
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO user_objectives 
            VALUES (:user_id, :goal, :target_company, :current_level, :preferred_topics, '')
        """, {
            "user_id": objective.user_id,
            "goal": objective.goal,
            "target_company": objective.target_company,
            "current_level": objective.current_level,
            "preferred_topics": json.dumps(objective.preferred_topics)
        })
        conn.commit()
        return {"message": "User objectives updated successfully"}
    except sqlite3.Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/generate-questions/{user_id}", status_code=status.HTTP_201_CREATED)
async def generate_questions(user_id: str, num_questions: int = 5, difficulty: str = "Medium"):
    """Generate interview questions based on user profile"""
    try:
        # Get user objectives
        cursor.execute("SELECT * FROM user_objectives WHERE user_id = ?", (user_id,))
        user_data = cursor.fetchone()
        if not user_data:
            print("User not found")
            raise HTTPException(status_code=404, detail="User not found")

        # Generate questions using structured chain
        questions = []
        for _ in range(num_questions):
            response = chain.invoke({
                "goal": user_data["goal"],
                "target_company": user_data["target_company"],
                "current_level": user_data["current_level"],
                "topics": json.loads(user_data["preferred_topics"]),
                "difficulty": difficulty,
                "num_questions": 1,
                "format_instructions": output_parser.get_format_instructions()
            })
            questions.append(response)
        
        print(questions)

        # Store questions and return formatted response
        stored_ids = []
        for q in questions:
            q_id = f"Q_{uuid.uuid4().hex[:8]}"
            cursor.execute("""
                INSERT INTO questions 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                q_id,
                user_id,
                q["question_text"],
                json.dumps(q["options"]),
                q["correct_answer"].upper(),
                q["explanation"],
                q["topic"],
                q["difficulty"]
            ))
            stored_ids.append(q_id)
        
        conn.commit()
        return {"generated_questions": stored_ids}
    
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit-answer", status_code=status.HTTP_201_CREATED)
async def submit_answer(response: UserResponse):
    """Evaluate and store user's answer"""
    try:
        # Get question details
        cursor.execute("""
            SELECT correct_answer, explanation 
            FROM questions 
            WHERE question_id = ?
        """, (response.question_id,))
        question = cursor.fetchone()
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")

        # Validate and store response
        is_correct = response.answer.upper() == question["correct_answer"]
        cursor.execute("""
            INSERT INTO user_responses 
            (user_id, question_id, user_answer, is_correct)
            VALUES (?, ?, ?, ?)
        """, (
            response.user_id,
            response.question_id,
            response.answer.upper(),
            is_correct
        ))
        conn.commit()
        
        return {
            "is_correct": is_correct,
            "correct_answer": question["correct_answer"],
            "explanation": question["explanation"]
        }
        
    except sqlite3.Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/progress/{user_id}", status_code=status.HTTP_200_OK)
async def get_progress(user_id: str):
    """Get user's learning progress"""
    try:
        # Basic stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_attempted,
                SUM(is_correct) as correct_answers,
                AVG(is_correct) as accuracy
            FROM user_responses
            WHERE user_id = ?
        """, (user_id,))
        stats = cursor.fetchone()

        # Weak areas analysis (CORRECTED QUERY)
        cursor.execute("""
            SELECT q.topic, q.difficulty, 
                   COUNT(*) as total,
                   SUM(ur.is_correct) as correct
            FROM user_responses ur
            JOIN questions q ON ur.question_id = q.question_id
            WHERE ur.user_id = ?
            GROUP BY q.topic, q.difficulty
        """, (user_id,))
        performance = cursor.fetchall()

        return {
            "user_id": user_id,
            "stats": dict(stats),
            "performance": [dict(row) for row in performance]
        }
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)