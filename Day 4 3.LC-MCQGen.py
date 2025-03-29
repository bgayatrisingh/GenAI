import os
import json
import random
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

class QuizGenerator:
    def __init__(self, model_name: str = "gpt-4-1106-preview", temperature: float = 0.7):
        # Define the response schemas
        response_schemas = [
            ResponseSchema(name="question", description="The multiple choice question"),
            ResponseSchema(name="options", description="Dictionary of options (a-d)"),
            ResponseSchema(name="correct_answer", description="Key of correct answer (a-d)")
        ]
        
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = self.output_parser.get_format_instructions()
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            model_kwargs={'response_format': {'type': 'json_object'}}
        )
        
        # Define the system and human message templates
        self.system_template = """You are an expert quiz generator. Generate high-quality multiple choice questions that are:
        - Clear and unambiguous
        - Educational and thought-provoking
        - Include 1 correct answer and 3 plausible distractors
        - Options must be labeled a, b, c, d
        
        {format_instructions}
        """
        
        self.human_template = """Generate a multiple choice question about {subtopic} in the context of {topic}."""
        
        # Create the chat prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("human", self.human_template)
        ])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_question(self, topic: str, subtopic: str) -> Optional[Dict]:
        """Generate a single quiz question"""
        try:
            # Format the messages
            messages = self.prompt.format_messages(
                format_instructions=self.output_parser.get_format_instructions(),
                topic=topic,
                subtopic=subtopic
            )
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Parse the response
            if isinstance(response.content, str):
                result = json.loads(response.content)
            else:
                result = response.content
            
            # Validate the response structure
            self._validate_question(result)
            
            # Shuffle and return the question
            return self._shuffle_options(result)
            
        except Exception as e:
            print(f"Error generating question for {topic}/{subtopic}: {str(e)}")
            return None

    def _validate_question(self, question_data: Dict) -> None:
        """Validate the question structure"""
        required_keys = {"question", "options", "correct_answer"}
        if not all(key in question_data for key in required_keys):
            raise ValueError("Invalid question structure from AI")

        options = question_data["options"]
        if len(options) != 4 or not all(k in options for k in ['a', 'b', 'c', 'd']):
            raise ValueError("Invalid options format")

        if question_data["correct_answer"] not in options:
            raise ValueError("Correct answer key not found in options")

    def _shuffle_options(self, question_data: Dict) -> Dict:
        """Shuffle options while maintaining correct answer mapping"""
        options = question_data["options"]
        correct = options[question_data["correct_answer"]]
        option_items = list(options.items())
        random.shuffle(option_items)
        
        new_options = {k: v for k, v in option_items}
        new_correct = next(k for k, v in new_options.items() if v == correct)
        
        return {
            "question": question_data["question"],
            "options": new_options,
            "correct_answer": new_correct
        }

    def generate_quiz(self, input_data: List[Dict], max_questions: int = 5) -> Dict:
        """Generate a full quiz set from input data"""
        if not isinstance(input_data, list):
            raise ValueError("Input must be a list of topics")
            
        quiz_set = []
        for topic_data in input_data:
            try:
                topic = topic_data["topic"]
                subtopics = topic_data["subtopics"][:max_questions]
            except KeyError:
                raise ValueError("Invalid topic structure")
                
            questions = []
            for subtopic in subtopics:
                question = self.generate_question(topic, subtopic)
                if question:
                    questions.append(question)
                    
            if questions:  # Only add topics with successful questions
                quiz_set.append({
                    "topic": topic,
                    "questions": questions[:max_questions]
                })
            
        return {"quiz_set": quiz_set}

def main():
    """Main function to demonstrate usage"""
    generator = QuizGenerator()
    
    input_json = """
    [
        {
            "topic": "Machine Learning",
            "subtopics": [
                "Supervised Learning",
                "Neural Networks",
                "Feature Engineering",
                "Overfitting",
                "Hyperparameter Tuning"
            ]
        },
        {
            "topic": "Space Exploration",
            "subtopics": [
                "Orbital Mechanics",
                "Space Telescopes",
                "Exoplanets",
                "Rocket Propulsion",
                "Space Agencies"
            ]
        }
    ]
    """
    
    try:
        topics = json.loads(input_json)
        quiz = generator.generate_quiz(topics)
        print(json.dumps(quiz, indent=2))
    except Exception as e:
        print(f"Error generating quiz: {str(e)}")

if __name__ == "__main__":
    main()
