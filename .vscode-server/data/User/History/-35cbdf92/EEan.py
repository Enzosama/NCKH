import os
from typing import Optional, List, Dict
from dataclasses import dataclass
import logging
from datetime import datetime
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt, IntPrompt
from rich.logging import RichHandler

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessage
)
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()

@dataclass
class ChatConfig:
    """Configuration settings for the chat application"""
    model: str
    system_prompt: str
    memory_length: int
    temperature: float = 0.7
    max_tokens: int = 1024

class GroqChatApp:
    """Terminal-based chat application class that handles the Groq integration"""
    
    AVAILABLE_MODELS = [
        'llama3-8b-8192',
        'mixtral-8x7b-32768',
        'gemma-7b-it'
    ]
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful, friendly AI assistant. Respond thoughtfully 
    and clearly to user questions while maintaining a conversational tone."""

    def __init__(self):
        """Initialize the chat application"""
        self.chat_history: List[Dict[str, str]] = []
        self.conversation_memory = None
        self.load_config()
        
    def load_config(self):
        """Load configuration through terminal input"""
        console.print("\n[bold blue]Chat Configuration[/bold blue]")
        
        # Display available models
        console.print("\nAvailable models:")
        for idx, model in enumerate(self.AVAILABLE_MODELS, 1):
            console.print(f"{idx}. {model}")
        
        # Get model choice
        model_idx = IntPrompt.ask(
            "Choose a model (enter number)",
            default=2,
            choices=[str(i) for i in range(1, len(self.AVAILABLE_MODELS) + 1)]
        )
        
        # Get other configurations
        system_prompt = Prompt.ask(
            "Enter system prompt",
            default=self.DEFAULT_SYSTEM_PROMPT
        )
        
        memory_length = IntPrompt.ask(
            "Enter conversation memory length",
            default=5
        )
        
        temperature = float(Prompt.ask(
            "Enter temperature (0.0 to 1.0)",
            default="0.7"
        ))
        
        self.config = ChatConfig(
            model=self.AVAILABLE_MODELS[model_idx - 1],
            system_prompt=system_prompt,
            memory_length=memory_length,
            temperature=temperature
        )

    def setup_groq_client(self) -> Optional[ChatGroq]:
        """Initialize the Groq client with error handling"""
        try:
            groq_api_key = os.environ.get('GROQ_API_KEY')
            if not groq_api_key:
                console.print("[bold red]Error: Please set the GROQ_API_KEY environment variable[/bold red]")
                return None
                
            return ChatGroq(
                groq_api_key=groq_api_key,
                model_name=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        except Exception as e:
            logger.error(f"Error setting up Groq client: {str(e)}")
            console.print(f"[bold red]Failed to initialize Groq client: {str(e)}[/bold red]")
            return None

    def create_conversation_chain(self, groq_chat: ChatGroq) -> LLMChain:
        """Create the conversation chain with the current configuration"""
        if not self.conversation_memory:
            self.conversation_memory = ConversationBufferWindowMemory(
                k=self.config.memory_length,
                memory_key="chat_history",
                return_messages=True
            )

        # Load existing chat history into memory
        for message in self.chat_history:
            self.conversation_memory.save_context(
                {'input': message['human']},
                {'output': message['AI']}
            )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.config.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        return LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=True,
            memory=self.conversation_memory
        )

    def display_chat_history(self):
        """Display the chat history with proper formatting"""
        if self.chat_history:
            console.print("\n[bold blue]Chat History:[/bold blue]")
            for message in self.chat_history:
                console.print("\n[bold green]You:[/bold green]")
                console.print(message['human'])
                console.print("\n[bold purple]Assistant:[/bold purple]")
                console.print(Markdown(message['AI']))
                console.print("\n" + "-" * 50)

    def run(self):
        """Main execution loop for the chat application"""
        console.print("\n[bold blue]Welcome to Groq Chat! Type 'quit' to exit.[/bold blue]\n")
        
        groq_chat = self.setup_groq_client()
        if not groq_chat:
            return

        conversation = self.create_conversation_chain(groq_chat)
        
        while True:
            try:
                # Display chat history
                self.display_chat_history()
                
                # Get user input
                user_question = Prompt.ask("\n[bold green]You")
                
                # Check for quit command
                if user_question.lower() in ['quit', 'exit', 'q']:
                    console.print("\n[bold blue]Goodbye![/bold blue]\n")
                    break
                
                # Get response
                console.print("\n[bold purple]Assistant:[/bold purple]")
                with console.status("[bold yellow]Thinking...[/bold yellow]"):
                    response = conversation.predict(human_input=user_question)
                
                # Save the new message to chat history
                message = {'human': user_question, 'AI': response}
                self.chat_history.append(message)
                
            except KeyboardInterrupt:
                console.print("\n[bold blue]Goodbye![/bold blue]\n")
                break
            except Exception as e:
                logger.error(f"Error during conversation: {str(e)}")
                console.print(f"\n[bold red]An error occurred: {str(e)}[/bold red]")
                if Prompt.ask("\nContinue? (y/n)", default="y").lower() != 'y':
                    break

def main():
    """Application entry point"""
    try:
        app = GroqChatApp()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        console.print("[bold red]An unexpected error occurred. Please try again.[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()