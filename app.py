import os
import logging
import tempfile
import re
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from utils.video_processor import VideoProcessor
from utils.chat_manager import ChatManager
from utils.youtube_downloader import YouTubeDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set default model name
MODEL_NAME = "models/gemini-1.5-flash"

# Set up the Google API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    try:
        # Try to load from .env
        load_dotenv()
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    except Exception as e:
        logger.error(f"Error loading environment variables: {str(e)}", exc_info=True)

# Check if API key exists and configure Gemini
if GOOGLE_API_KEY:
    logger.info("Google API key found in environment variables")
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Gemini successfully configured")
    except Exception as e:
        logger.error(f"Error configuring Gemini: {str(e)}", exc_info=True)
else:
    logger.error("Google API key not found in environment variables")
    # We'll handle this in the app itself

# Set page config
st.set_page_config(
    page_title="Connective Video Chatbot Prototype",
    page_icon="🎬",
    layout="wide",
)
logger.info("Streamlit page configured")

# Initialize session state for chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    logger.debug("Initialized chat_history in session state")

if "video_processor" not in st.session_state:
    st.session_state.video_processor = None
    logger.debug("Initialized video_processor in session state")

if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = None
    logger.debug("Initialized chat_manager in session state")

if "video_path" not in st.session_state:
    st.session_state.video_path = None
    logger.debug("Initialized video_path in session state")

if "video_title" not in st.session_state:
    st.session_state.video_title = None
    logger.debug("Initialized video_title in session state")

if "direct_chat_mode" not in st.session_state:
    st.session_state.direct_chat_mode = False
    logger.debug("Initialized direct_chat_mode in session state")

if "direct_chat_session" not in st.session_state:
    st.session_state.direct_chat_session = None
    logger.debug("Initialized direct_chat_session in session state")

def validate_youtube_url(url):
    """Validate if the URL is a YouTube URL."""
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    match = re.match(youtube_regex, url)
    return match is not None

def initialize_direct_chat():
    """Initialize direct chat mode with Connective AI.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    try:
        logger.info("Initializing direct chat mode")
        
        # Set the direct chat mode flag
        st.session_state.direct_chat_mode = True
        
        # Create a fresh chat history
        st.session_state.chat_history = []
        
        # Initialize the generative model for direct chat
        from google.generativeai import GenerativeModel
        
        genai.configure(api_key=GOOGLE_API_KEY)
        
        try:
            # Test model initialization
            model = GenerativeModel(MODEL_NAME)
            logger.info(f"Direct chat model initialized successfully: {MODEL_NAME}")
            
            # Store the model in session state
            st.session_state.direct_chat_model = model
            
            return True
        except Exception as e:
            logger.error(f"Error initializing direct chat model: {str(e)}", exc_info=True)
            return False
            
    except Exception as e:
        logger.error(f"Error initializing direct chat: {str(e)}", exc_info=True)
        return False

def generate_direct_response(user_query):
    """Generate a response for direct chat mode.
    
    Args:
        user_query: The user's query text
        
    Returns:
        str: The generated response
    """
    try:
        logger.info("Generating direct chat response")
        
        if "direct_chat_model" not in st.session_state:
            logger.error("Direct chat model not initialized")
            return "Error: Direct chat not properly initialized. Please try restarting direct chat mode."
        
        model = st.session_state.direct_chat_model
        
        # Default system prompt for Connective AI
        system_prompt = """You are a knowledgeable assistant created by Connective Labs. 
        Your goal is to provide helpful, accurate, and educational responses. 
        Be conversational and friendly while focusing on delivering valuable information.
        Never mention that you are an AI model or refer to any underlying technologies.
        Always maintain the personality of a Connective Labs assistant."""
        
        # Generate the response
        response = model.generate_content(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            generation_config={"temperature": 0.2}
        )
        
        logger.info("Direct chat response generated successfully")
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating direct chat response: {str(e)}", exc_info=True)
        return f"I'm sorry, I encountered an error while processing your request: {str(e)}"

def process_mp4_upload(uploaded_file):
    """Process an uploaded MP4 file.
    
    Args:
        uploaded_file: The uploaded MP4 file from Streamlit's file_uploader
        
    Returns:
        tuple: (success, message) where success is a boolean and message is a string
    """
    try:
        # Reset direct chat mode
        st.session_state.direct_chat_mode = False
        
        logger.info(f"Processing uploaded MP4 file: {uploaded_file.name}")
        
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        # Add a file size limit for cloud deployment (e.g., 50MB)
        if file_size_mb > 50:
            logger.warning(f"File too large: {file_size_mb:.2f} MB")
            return False, f"File too large ({file_size_mb:.2f} MB). Please upload a file smaller than 50MB."
        
        # Create a temporary file to save the uploaded file
        logger.info("Creating temporary file for uploaded MP4")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        
        logger.info(f"Temporary file created at: {temp_file_path}")
        
        st.session_state.display_message = st.empty()
        st.session_state.display_message.info("Creating content processor for uploaded MP4...")
        
        # Create video processor with optimized settings for cloud deployment
        logger.info("Creating video processor with optimized settings")
        video_processor = VideoProcessor(
            temp_file_path,
            sample_rate=1,  # Sample every frame
            max_frames=20,  # Limit to max 20 frames
            max_resolution=360  # Limit resolution to 360p
        )
        
        # Store the video processor in session state
        st.session_state.video_processor = video_processor
        st.session_state.video_path = temp_file_path
        st.session_state.video_title = uploaded_file.name
        
        logger.info("Video processor created successfully")
        st.session_state.display_message.info("Creating chat manager...")
        
        # Create chat manager with processed video
        logger.info("Creating chat manager")
        chat_manager = ChatManager(MODEL_NAME, video_processor)
        st.session_state.chat_manager = chat_manager
        
        # Clear previous chat history
        st.session_state.chat_history = []
        
        logger.info("Chat manager created successfully")
        
        # Clear the display message
        if "display_message" in st.session_state:
            st.session_state.display_message.empty()
            del st.session_state.display_message
        
        return True, uploaded_file.name
    
    except Exception as e:
        logger.error(f"Error processing MP4 upload: {str(e)}", exc_info=True)
        
        # Clear the display message if it exists
        if "display_message" in st.session_state:
            st.session_state.display_message.empty()
            del st.session_state.display_message
            
        return False, str(e)

def process_youtube_url(youtube_url):
    """Download and process YouTube video, initialize chat manager."""
    try:
        # Display progress message
        progress_msg = st.info("Processing YouTube content... This may take a moment, please be patient.")
        
        # Initialize YouTube downloader
        downloader = YouTubeDownloader()
        
        # Download YouTube video
        logger.info(f"Processing YouTube content from URL: {youtube_url}")
        logger.info("Starting YouTube content download")
        video_path, content_title = downloader.download_video(youtube_url)
        logger.info(f"YouTube content downloaded successfully: {video_path}")
        logger.info(f"Content title: {content_title}")
        
        # Store the video path and title in session state
        st.session_state.video_path = video_path
        st.session_state.video_title = content_title
        
        # Create video processor with optimized settings for cloud
        logger.info("Creating content processor")
        video_processor = VideoProcessor(
            video_path, 
            sample_rate=1, 
            max_frames=20, 
            max_resolution=360
        )
        st.session_state.video_processor = video_processor
        logger.info("Content processor created successfully")
        
        # Create chat manager with processed video
        logger.info(f"Creating chat manager with Connective AI")
        chat_manager = ChatManager(MODEL_NAME, video_processor)
        st.session_state.chat_manager = chat_manager
        logger.info("Chat manager created successfully")
        
        # Turn off direct chat mode if it was on
        st.session_state.direct_chat_mode = False
        
        # Clear previous chat history
        st.session_state.chat_history = []
        
        # Clear progress message
        progress_msg.empty()
        
        logger.info("Content processing completed successfully")
        return True, content_title
    except Exception as e:
        logger.error(f"Error processing YouTube content: {str(e)}")
        return False, str(e)

def main():
    st.title("Connective Video Chatbot Prototype")
    st.subheader("Powered by Connective Labs")
    logger.info("Main application started")
    
    # Initialize session state variables if they don't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "direct_chat_mode" not in st.session_state:
        st.session_state.direct_chat_mode = False
    
    # Check if API key is available
    if not GOOGLE_API_KEY:
        st.error("Google API key is not configured. Please set the GOOGLE_API_KEY environment variable or add it to your .env file.")
    
    # Sidebar with mode selection and controls
    with st.sidebar:
        st.title("Controls")
        
        # Add model selection
        MODEL_OPTIONS = {
            "Connective AI": "models/gemini-1.5-flash"
        }
        
        model_option = st.selectbox(
            "Select AI Model",
            list(MODEL_OPTIONS.keys()),
            index=0
        )
        
        global MODEL_NAME
        MODEL_NAME = MODEL_OPTIONS[model_option]
        
        # Create horizontal tabs for different input modes
        mode_tabs = st.tabs(["YouTube", "MP4 Upload", "Direct Chat"])
        
        # YouTube tab
        with mode_tabs[0]:
            st.markdown("### Enter YouTube URL")
            youtube_url = st.text_input("Enter YouTube URL", key="youtube_url_input")
            
            if youtube_url:
                logger.debug(f"YouTube URL entered: {youtube_url}")
                if not validate_youtube_url(youtube_url):
                    st.error("Invalid YouTube URL. Please enter a valid YouTube URL.")
                    logger.warning(f"Invalid YouTube URL: {youtube_url}")
                    youtube_url = None
                
            if youtube_url and st.button("Process YouTube Content"):
                with st.spinner("Processing YouTube content..."):
                    # Use the optimized YouTube processing function
                    success, message = process_youtube_url(youtube_url)
                    
                    if success:
                        st.success(f"YouTube content processed successfully: {message}")
                    else:
                        st.error(f"Error processing YouTube content: {message}")
        
        # MP4 Upload tab
        with mode_tabs[1]:
            st.markdown("### Upload MP4 File")
            uploaded_file = st.file_uploader("Choose an MP4 file", type=["mp4"])
            
            if uploaded_file is not None:
                logger.debug(f"File uploaded: {uploaded_file.name}")
                if st.button("Process MP4 Video"):
                    with st.spinner("Processing uploaded video..."):
                        success, message = process_mp4_upload(uploaded_file)
                        
                        if success:
                            st.success(f"MP4 video processed successfully: {message}")
                        else:
                            st.error(f"Error processing MP4 video: {message}")
        
        # Direct Chat tab
        with mode_tabs[2]:
            st.markdown("### Direct Chat")
            st.write("Chat directly with Connective AI without any content")
            
            if st.button("Start Direct Chat"):
                with st.spinner("Initializing direct chat..."):
                    success = initialize_direct_chat()
                    if success:
                        st.success("Direct chat initialized")
                    else:
                        st.error("Error initializing direct chat")
        
        # Display app information
        st.markdown("---")
        st.markdown("## About")
        
        st.markdown("### 🌐 YouTube")
        st.markdown("Enter any YouTube URL for AI analysis")
        
        st.markdown("### 🎬 MP4 Upload")
        st.markdown("Upload your own MP4 files for analysis")
        
        st.markdown("### 💬 Direct Chat")
        st.markdown("Chat directly with Connective AI")
        
        st.markdown("---")
        st.caption("© 2025 Connective Labs")
    
    # Main chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "video_title" in st.session_state and not st.session_state.get("direct_chat_mode", False):
        st.markdown(f"## Now chatting about: {st.session_state.video_title}")
    elif st.session_state.get("direct_chat_mode", False):
        st.markdown("## Direct Chat Mode")
    
    # Display chat messages
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            st.markdown(content)
    
    # Chat input
    if ("chat_manager" in st.session_state) or st.session_state.get("direct_chat_mode", False):
        user_query = st.chat_input("Ask a question...")
        
        if user_query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Get AI response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                try:
                    logger.info(f"Received user query: {user_query}")
                    logger.info("Generating response")
                    
                    if st.session_state.get("direct_chat_mode", False):
                        response = generate_direct_response(user_query)
                    else:
                        response = st.session_state.chat_manager.generate_response(user_query)
                    
                    message_placeholder.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    logger.info("Response generated successfully")
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    message_placeholder.error(error_message)
                    logger.error(error_message, exc_info=True)
    else:
        st.info("Please process a video or initialize direct chat to start.")
    
    logger.info("Application running")

if __name__ == "__main__":
    try:
        logger.info("Starting application")
        main()
        logger.info("Application running")
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}") 