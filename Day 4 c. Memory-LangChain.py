import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Verify OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")

class ConversationalAI:
    def __init__(self):
        # Initialize the language model
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="history"
        )
        
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and friendly assistant. You maintain context "
                      "of the conversation and provide relevant, accurate responses."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create the chain
        self.chain = self.prompt | self.llm

    def chat(self, input_text: str) -> str:
        """Process a single chat interaction"""
        try:
            # Load conversation history
            history = self.memory.load_memory_variables({})["history"]
            
            # Get response from the chain
            response = self.chain.invoke({
                "history": history,
                "input": input_text
            })
            
            # Save the interaction to memory
            self.memory.save_context(
                {"input": input_text},
                {"output": response.content}
            )
            
            return response.content
            
        except Exception as e:
            return f"Error processing message: {str(e)}"
    
    def view_memory(self) -> Dict[str, Any]:
        """View the current conversation history"""
        return self.memory.load_memory_variables({})
    
    def clear_memory(self) -> None:
        """Clear the conversation history"""
        self.memory.clear()

def main():
    try:
        # Create the conversational AI instance
        ai = ConversationalAI()
        
        print("=== Conversational AI Demo ===")
        print("Type 'quit' to exit, 'clear' to clear memory, 'memory' to view conversation history")
        print("Starting conversation...\n")
        
        # Demonstration of capabilities
        demo_interactions = [
            "Hello, my name is Alice.",
            "What's my name?",
            "Tell me about yourself.",
            "What was the first thing I told you?"
        ]
        
        print("=== Demo Interactions ===")
        for interaction in demo_interactions:
            print(f"\nHuman: {interaction}")
            response = ai.chat(interaction)
            print(f"AI: {response}")
            
        print("\n=== Interactive Mode ===")
        print("Now you can chat freely with the AI!")
        
        # Interactive chat loop
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            elif user_input.lower() == 'clear':
                ai.clear_memory()
                print("Memory cleared!")
                continue
                
            elif user_input.lower() == 'memory':
                memory = ai.view_memory()
                print("\n=== Conversation History ===")
                print(memory)
                continue
                
            elif user_input:
                response = ai.chat(user_input)
                print(f"AI: {response}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()