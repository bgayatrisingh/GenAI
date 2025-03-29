import os
from dotenv import load_dotenv
import autogen

# Load environment variables
load_dotenv()

# Configure OpenAI API key
config_list = [
    {
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
        
    }
]

# Create assistant and user agents
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list},
    system_message="You are a helpful AI assistant."
)

coder = autogen.AssistantAgent(
    name="coder",
    llm_config={"config_list": config_list},
    system_message="You are a Python programming expert. Write code to solve problems."
)

reviewer = autogen.AssistantAgent(
    name="reviewer",
    llm_config={"config_list": config_list},
    system_message="You are a code reviewer. Review code for bugs and suggest improvements."
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False
    }
)

def main():
    # Initialize a group chat
    groupchat = autogen.GroupChat(
        agents=[user_proxy, assistant, coder, reviewer],
        messages=[],
        max_round=12
    )
    
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config={"config_list": config_list}
    )

    # Start the conversation with a coding task
    user_proxy.initiate_chat(
        manager,
        message="""
        Please help me with the following task:
        1. Write a Python function to calculate the Fibonacci sequence
        2. Optimize it for better performance
        3. Add proper documentation
        """
    )

if __name__ == "__main__":
    main()