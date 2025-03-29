import os
from dotenv import load_dotenv
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Dict

# Load environment variables
load_dotenv()

# Verify OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")

def create_few_shot_chain(examples: List[Dict[str, str]] = None):
    """Create a few-shot learning chain with examples"""
    
    # Default examples if none provided
    if examples is None:
        examples = [
            {
                "input": "What is the capital of France?",
                "output": "The capital of France is Paris."
            },
            {
                "input": "Who wrote Romeo and Juliet?",
                "output": "William Shakespeare wrote Romeo and Juliet."
            },
            {
                "input": "What is the boiling point of water?",
                "output": "The boiling point of water is 100 degrees Celsius."
            },
            {
                "input": "Who painted the Mona Lisa?",
                "output": "Leonardo da Vinci painted the Mona Lisa."
            }
        ]

    # Define the example template
    example_template = """
Input: {input}
Output: {output}
"""

    # Create the example prompt template
    example_prompt = PromptTemplate.from_template(example_template)

    # Create the few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Answer the following questions in a similar style to these examples. "
               "Provide accurate and concise answers:",
        suffix="Input: {input}\nOutput:",
        input_variables=["input"]
    )

    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1  # Lower temperature for more consistent answers
    )

    return few_shot_prompt, llm

def get_answer(question: str, few_shot_prompt: FewShotPromptTemplate, llm: ChatOpenAI) -> str:
    """Get answer for a question using the few-shot chain"""
    try:
        # Format the prompt with the question
        formatted_prompt = few_shot_prompt.format(input=question)
        
        # Get the response from the LLM
        response = llm.invoke(formatted_prompt)
        
        return response.content.strip()
    
    except Exception as e:
        return f"Error getting answer: {str(e)}"

def main():
    try:
        # Create the few-shot chain
        few_shot_prompt, llm = create_few_shot_chain()
        
        # Example questions to demonstrate
        questions = [
            "What is the speed of light?",
            "Who invented the telephone?",
            "What is the capital of Japan?"
        ]
        
        # Get and print answers for each question
        print("\n=== Few-Shot Learning Q&A ===\n")
        
        for question in questions:
            print(f"Question: {question}")
            answer = get_answer(question, few_shot_prompt, llm)
            print(f"Answer: {answer}\n")
            
        # Interactive mode
        print("Enter your questions (type 'quit' to exit):")
        while True:
            question = input("\nYour question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if question:
                answer = get_answer(question, few_shot_prompt, llm)
                print(f"Answer: {answer}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()