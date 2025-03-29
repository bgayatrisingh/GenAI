import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import Dict

# Load environment variables from .env file
load_dotenv()

# Verify OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")

def create_story_analysis_chain():
    """Create and return the story generation and analysis chains"""
    # Create the story generation chain
    story_prompt = PromptTemplate.from_template(
        "Write a very short story about {topic}. Keep it under 200 words."
    )
    
    story_chain = (
        story_prompt | 
        ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        ) | 
        StrOutputParser()
    )

    # Create the analysis chain
    analysis_prompt = PromptTemplate.from_template(
        """Analyze the following story focusing on these aspects:
        - Main theme
        - Key elements
        - Writing style
        
        Story: {story}
        
        Provide a concise analysis."""
    )
    
    analysis_chain = (
        analysis_prompt | 
        ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3
        ) | 
        StrOutputParser()
    )

    # Create the combined chain
    def combined_chain(inputs: Dict[str, str]) -> Dict[str, str]:
        try:
            # Generate the story
            story = story_chain.invoke({"topic": inputs["topic"]})
            
            # Analyze the story
            analysis = analysis_chain.invoke({"story": story})
            
            return {
                "story": story,
                "analysis": analysis
            }
        except Exception as e:
            return {
                "error": f"An error occurred: {str(e)}"
            }

    return combined_chain

def main():
    try:
        # Create the chain
        chain = create_story_analysis_chain()
        
        # Get user input for the topic
        topic = input("Enter a topic for the story: ").strip()
        if not topic:
            topic = "time travel"  # default topic
        
        # Run the chain
        result = chain({"topic": topic})
        
        # Check for errors
        if "error" in result:
            print(result["error"])
            return
        
        # Print results
        print("\n=== Generated Story ===")
        print(result["story"])
        print("\n=== Analysis ===")
        print(result["analysis"])
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()