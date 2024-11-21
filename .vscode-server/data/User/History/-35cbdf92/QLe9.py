import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Load environment variables from .env file
load_dotenv()

def create_groq_chat():
    """Initialize the Groq chat client"""
    # Use the actual API key directly
    api_key = "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"
    
    try:
        chat = ChatGroq(
            groq_api_key=api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.7
        )
        return ConversationChain(
            llm=chat,
            memory=ConversationBufferMemory(),
            verbose=False
        )
    except Exception as e:
        print(f"Error initializing chat: {str(e)}")
        return None

def main():
    """Main chat loop"""
    print("\nSimple Groq Chatbot")
    print("Type 'quit' to exit\n")
    
    # Initialize chat
    conversation = create_groq_chat()
    if not conversation:
        return
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for quit command
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            # Get and print response
            if user_input:
                print("\nAssistant:", end=" ")
                response = conversation.predict(input=user_input)
                print(response)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            continue

if __name__ == "__main__":
    main()