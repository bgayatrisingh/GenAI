import os
import json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QuizEvaluationAgent:
    def __init__(self):
        self.system_prompt = """You are a expert quiz evaluator. Analyze incorrect answers and provide:
        - Brief explanation of why the answer is wrong
        - Constructive feedback for improvement
        - 1-2 sentence maximum for each
        - Return JSON format: { "explanation": "...", "feedback": "..." }"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _generate_feedback(self, question_data, user_answer):
        try:
            prompt = f"""
            Question: {question_data['question']}
            Correct Answer: {question_data['correct_answer']} ({question_data['options'][question_data['correct_answer']]})
            User's Answer: {user_answer} ({question_data['options'].get(user_answer, 'invalid')})

            Explain why the user's answer is incorrect and provide feedback:"""

            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )

            return json.loads(response.choices[0].message.content)
        
        except json.JSONDecodeError:
            raise ValueError("Failed to parse AI feedback response")
        except Exception as e:
            print(f"Feedback generation failed: {str(e)}")
            raise

    def evaluate_submission(self, submission, original_quiz):
        results = {
            "score": 0,
            "total": 0,
            "details": [],
            "overall_feedback": ""
        }

        # Match submission to original quiz structure
        for topic in original_quiz["quiz_set"]:
            for question in topic["questions"]:
                qid = f"{topic['topic']}-{question['question'][:20]}"
                user_answer = submission.get(qid, {}).get("answer", "").lower()
                
                is_correct = user_answer == question["correct_answer"]
                result = {
                    "question_id": qid,
                    "question": question["question"],
                    "user_answer": user_answer,
                    "correct_answer": question["correct_answer"],
                    "is_correct": is_correct,
                    "explanation": "",
                    "feedback": ""
                }

                if not is_correct:
                    try:
                        feedback = self._generate_feedback(question, user_answer)
                        result.update(feedback)
                    except Exception as e:
                        print(f"Failed to generate feedback for {qid}: {str(e)}")
                        result.update({
                            "explanation": "Error generating explanation",
                            "feedback": "Contact instructor for feedback"
                        })

                results["details"].append(result)
                results["total"] += 1
                if is_correct:
                    results["score"] += 1

        # Generate overall feedback
        results["overall_feedback"] = self._generate_overall_feedback(results)
        return results

    def _generate_overall_feedback(self, results):
        try:
            prompt = f"""Student scored {results['score']}/{results['total']}. 
            Provide 2-3 sentence constructive feedback focusing on:
            - Key strengths
            - Main areas for improvement
            - Suggested study strategies"""

            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful learning coach"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=200
            )

            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Overall feedback failed: {str(e)}")
            return "Overall feedback unavailable at this time"

# Example Usage
if __name__ == "__main__":
    evaluator = QuizEvaluationAgent()

    # Original Quiz Data (from previous generator)
    original_quiz = {
        "quiz_set": [
            {
                "topic": "Machine Learning",
                "questions": [
                    {
                        "question": "Which algorithm is best for labeled training data?",
                        "options": {
                            "a": "Supervised Learning",
                            "b": "Unsupervised Learning",
                            "c": "Reinforcement Learning",
                            "d": "Self-Supervised Learning"
                        },
                        "correct_answer": "a"
                    }
                ]
            }
        ]
    }

    # Sample Submission
    submission = {
        "machine-learning-which algorithm": {
            "answer": "b"
        }
    }

    evaluation = evaluator.evaluate_submission(submission, original_quiz)
    print(json.dumps(evaluation, indent=2))
