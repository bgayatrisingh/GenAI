import os
import json
import random
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QuizGenerator:
    def __init__(self):
        self.system_prompt = """You are a expert quiz generator. Generate high-quality multiple choice questions with:
        - 1 correct answer
        - 3 plausible distractors
        - Options shuffled in random order
        - Return valid JSON format with: question, options (a-d), and correct_answer"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_question(self, topic, subtopic):
        try:
            prompt = f"Generate a MCQ question about {subtopic} in the context of {topic}"
            
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            return self._validate_and_format(result)
            
        except json.JSONDecodeError:
            raise ValueError("Failed to parse AI response")
        except Exception as e:
            print(f"AI API Error: {str(e)}")
            raise

    def _validate_and_format(self, question_data):
        required_keys = {"question", "options", "correct_answer"}
        if not all(key in question_data for key in required_keys):
            raise ValueError("Invalid question structure from AI")

        options = question_data["options"]
        if len(options) != 4 or not all(k in options for k in ['a', 'b', 'c', 'd']):
            raise ValueError("Invalid options format")

        # Shuffle options while maintaining correct answer
        correct = options[question_data["correct_answer"]]
        option_items = list(options.items())
        random.shuffle(option_items)
        
        new_options = {k: v for k, v in option_items}
        new_correct = [k for k, v in new_options.items() if v == correct][0]

        return {
            "question": question_data["question"],
            "options": new_options,
            "correct_answer": new_correct
        }

    def generate_quiz(self, input_data):
        if not isinstance(input_data, list):
            raise ValueError("Input must be a list of topics")
            
        quiz_set = []
        for topic_data in input_data:
            try:
                topic = topic_data["topic"]
                subtopics = topic_data["subtopics"][:5]  # Use first 5 subtopics
            except KeyError:
                raise ValueError("Invalid topic structure")
                
            questions = []
            for subtopic in subtopics:
                try:
                    questions.append(self.generate_question(topic, subtopic))
                except Exception as e:
                    print(f"Skipping question for {subtopic}: {str(e)}")
                    continue
                    
            quiz_set.append({
                "topic": topic,
                "questions": questions[:5]  # Ensure max 5 questions
            })
            
        return {"quiz_set": quiz_set}

# Example Usage
if __name__ == "__main__":
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
