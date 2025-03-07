import google.generativeai as genai
from typing import List, Dict, Any, Optional
import base64
import mimetypes
import os

from .video_processor import VideoProcessor

class ChatManager:
    """
    Class to manage interactions with the AI model.
    Handles prompt creation, model configuration, and response generation.
    """
    
    def __init__(self, model_name: str, video_processor: VideoProcessor):
        """
        Initialize the ChatManager with a model name and processed content.
        
        Args:
            model_name: Name of the AI model to use
            video_processor: VideoProcessor instance with processed content
        """
        self.model_name = model_name
        self.video_processor = video_processor
        self.chat_session = None
        self.model = None
        
        # Initialize model and chat session
        self._initialize_model()
        self._initialize_chat_session()
    
    def _initialize_model(self) -> None:
        """Initialize the AI model."""
        try:
            # Configure model with appropriate parameters
            generation_config = {
                "temperature": 0.4,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]
            
            # Initialize the model
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        except Exception as e:
            raise Exception(f"Failed to initialize model: {e}")
    
    def _initialize_chat_session(self) -> None:
        """Initialize the chat session with the content context."""
        try:
            # Prepare content data for processing
            content_data = self.video_processor.prepare_for_gemini(max_frames=50)
            
            # Create initial system prompt with content data
            system_prompt = """
            You are an intelligent AI assistant created by Connective Labs. 
            You have been trained on specialized knowledge and can explain concepts, ideas, and processes clearly.
            
            As a knowledgeable assistant:
            1. Answer questions directly based on your specialized knowledge
            2. Explain technical concepts in a straightforward manner
            3. Provide detailed explanations when asked about topics in your knowledge base
            4. Be honest about your limitations if asked about something outside your knowledge
            
            Guidelines for answering:
            - Provide direct, concise answers as if the information is part of your general knowledge
            - Present information confidently and clearly
            - Never mention "content," "visuals," "frames," or any source material for your knowledge
            - Use a friendly, helpful tone that makes complex topics easy to understand
            
            Your goal is to be helpful and informative while making the user feel that they're interacting
            with a knowledgeable assistant rather than a content analysis tool.
            """
            
            # Start the chat session with the frames as initial context
            content_parts = [{"text": system_prompt}]
            
            # Add frames as parts
            for frame_data in content_data:
                # Add the frame as an inline image without any text about frames/timestamps
                mime_type = "image/jpeg"
                data_uri = f"data:{mime_type};base64,{frame_data['image']}"
                content_parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": frame_data['image']
                    }
                })
            
            # Initialize chat session
            self.chat_session = self.model.start_chat(history=[])
            
            # Add initial message with frames to establish context
            initial_prompt = "Please review this specialized content and prepare to answer questions based on the information provided."
            content_parts.append({"text": initial_prompt})
            
            # Send the initial message to establish context
            self.chat_session.send_message(content_parts)
        except Exception as e:
            raise Exception(f"Failed to initialize chat session: {e}")
    
    def generate_response(self, query: str) -> str:
        """
        Generate a response to a user query based on the knowledge context.
        
        Args:
            query: User query about the knowledge
            
        Returns:
            Generated response from the model
        """
        try:
            # Send the user query to the chat session
            response = self.chat_session.send_message(query)
            
            # Return the generated text
            return response.text
        except Exception as e:
            error_message = f"Error generating response: {e}"
            print(error_message)
            return f"I encountered an issue while processing your question. {error_message}"
    
    def restart_chat_session(self) -> None:
        """Restart the chat session with the same content context."""
        self._initialize_chat_session() 