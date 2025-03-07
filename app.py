import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import re
from utils.video_processor import VideoProcessor
from utils.chat_manager import ChatManager
from utils.youtube_downloader import YouTubeDownloader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set API key directly rather than loading from .env
os.environ["GOOGLE_API_KEY"] = "AIzaSyCubhOuGrvMEi8NIdqT_g8bt9SYwN6r5Y8"
logger.info("API key set directly in code")

# Configure API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY not found in environment variables")
    st.error("Please set your GOOGLE_API_KEY in the .env file")
    st.stop()
else:
    logger.info("GOOGLE_API_KEY found in environment variables")

# Configure AI model
try:
    genai.configure(api_key=api_key)
    logger.info("AI model configured successfully")
except Exception as e:
    logger.error(f"Error configuring AI model: {str(e)}")
    st.error(f"Error configuring AI model: {str(e)}")
    st.stop()

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

# Hardcode the model to always use gemini-1.5-flash
MODEL_NAME = "gemini-1.5-flash"

def validate_youtube_url(url):
    """Validate if the URL is a YouTube URL."""
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    match = re.match(youtube_regex, url)
    return match is not None

def initialize_direct_chat():
    """Initialize a direct chat session without video context."""
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
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Start chat
        chat_session = model.start_chat(history=[])
        st.session_state.direct_chat_session = chat_session
        st.session_state.direct_chat_mode = True
        logger.info(f"Direct chat session initialized with Connective AI")
        return True
    except Exception as e:
        logger.error(f"Error initializing direct chat: {str(e)}")
        st.error(f"Error initializing direct chat: {str(e)}")
        return False

def process_mp4_upload(uploaded_file):
    """Process uploaded MP4 file and initialize chat manager."""
    try:
        # Save uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        video_path = temp_file.name
        
        # Create a title for the video from the filename
        content_title = uploaded_file.name
        
        # Store the video path and title in session state
        st.session_state.video_path = video_path
        st.session_state.video_title = content_title
        
        # Create video processor
        logger.info("Creating content processor for uploaded MP4")
        video_processor = VideoProcessor(video_path)
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
        
        return True, content_title
    except Exception as e:
        logger.error(f"Error processing MP4 upload: {str(e)}", exc_info=True)
        return False, str(e)

def main():
    st.title("Connective Video Chatbot Prototype")
    st.subheader("Powered by Connective Labs")
    logger.info("Main application started")
    
    # Sidebar for inputs and settings
    with st.sidebar:
        st.header("Settings")
        
        # Display information about the AI being used
        st.markdown("### Using Connective AI")
        
        # Mode selection
        st.markdown("### Choose Mode")
        mode_tabs = st.tabs(["YouTube", "MP4 Upload", "Direct Chat"])
        
        # YouTube tab
        with mode_tabs[0]:
            st.markdown("### YouTube Content")
            youtube_url = st.text_input("Enter YouTube URL")
            
            if youtube_url:
                logger.debug(f"YouTube URL entered: {youtube_url}")
                if not validate_youtube_url(youtube_url):
                    st.error("Please enter a valid YouTube URL")
                    logger.warning(f"Invalid YouTube URL: {youtube_url}")
                    youtube_url = None
            
            if youtube_url and st.button("Process YouTube Content"):
                # Reset direct chat mode
                st.session_state.direct_chat_mode = False
                
                logger.info(f"Processing YouTube content from URL: {youtube_url}")
                with st.spinner("Downloading and processing YouTube content..."):
                    try:
                        # Download the YouTube video
                        logger.info("Starting YouTube content download")
                        downloader = YouTubeDownloader()
                        video_path, content_title = downloader.download_video(youtube_url)
                        logger.info(f"YouTube content downloaded successfully: {video_path}")
                        logger.info(f"Content title: {content_title}")
                        
                        # Store the video path and title in session state
                        st.session_state.video_path = video_path
                        st.session_state.video_title = content_title
                        
                        # Create video processor
                        logger.info("Creating content processor")
                        video_processor = VideoProcessor(video_path)
                        st.session_state.video_processor = video_processor
                        logger.info("Content processor created successfully")
                        
                        # Create chat manager with processed video
                        logger.info(f"Creating chat manager with Connective AI")
                        chat_manager = ChatManager(MODEL_NAME, video_processor)
                        st.session_state.chat_manager = chat_manager
                        logger.info("Chat manager created successfully")
                        
                        # Clear previous chat history
                        st.session_state.chat_history = []
                        
                        # Success message
                        st.success(f"Content '{content_title}' processed successfully!")
                        logger.info("Content processing completed successfully")
                        
                    except Exception as e:
                        st.error(f"Error processing YouTube content: {str(e)}")
                        logger.error(f"Error processing YouTube content: {str(e)}", exc_info=True)
        
        # MP4 Upload tab
        with mode_tabs[1]:
            st.markdown("### Upload Content")
            uploaded_file = st.file_uploader("Choose an MP4 file", type=["mp4"])
            
            if uploaded_file is not None:
                if st.button("Process Content"):
                    # Reset direct chat mode
                    st.session_state.direct_chat_mode = False
                    
                    with st.spinner("Processing content..."):
                        success, message = process_mp4_upload(uploaded_file)
                        if success:
                            st.success(f"Content '{message}' processed successfully!")
                        else:
                            st.error(f"Error processing content: {message}")
        
        # Direct Chat tab
        with mode_tabs[2]:
            st.markdown("### Direct Chat with Connective AI")
            st.markdown("Chat directly without any additional content")
            
            if st.button("Start Direct Chat"):
                # Clear previous chat history
                st.session_state.chat_history = []
                
                # Initialize direct chat
                with st.spinner("Initializing chat..."):
                    if initialize_direct_chat():
                        st.success(f"Direct chat with Connective AI initialized successfully!")
    
    # Main chat area
    if st.session_state.chat_manager or st.session_state.direct_chat_mode:
        # Display content type and title if available
        if st.session_state.direct_chat_mode:
            st.markdown(f"## Direct Chat with Connective AI")
        elif st.session_state.video_title:
            st.markdown(f"## Knowledge Assistant: **{st.session_state.video_title}**")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
        
        # User input
        chat_prompt = "Ask something..." if st.session_state.direct_chat_mode else "Ask a question..."
        user_query = st.chat_input(chat_prompt)
        
        if user_query:
            logger.info(f"Received user query: {user_query}")
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            
            # Get response from appropriate source
            with st.spinner("Thinking..."):
                try:
                    logger.info("Generating response")
                    if st.session_state.direct_chat_mode and st.session_state.direct_chat_session:
                        # Get response from direct chat session
                        response = st.session_state.direct_chat_session.send_message(user_query).text
                    else:
                        # Get response from chat manager (video context)
                        response = st.session_state.chat_manager.generate_response(user_query)
                    
                    logger.info("Response generated successfully")
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}", exc_info=True)
                    response = f"I encountered an error while processing your question: {str(e)}"
                
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
    else:
        st.info("Please choose one of the options from the sidebar to start chatting.")
        
        # Show demo area
        st.markdown("---")
        st.markdown("## Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📺 YouTube Content")
            st.markdown("Enter a YouTube URL to analyze and chat about content")
        
        with col2:
            st.markdown("### 🎬 MP4 Upload")
            st.markdown("Upload your own MP4 files for analysis")
        
        with col3:
            st.markdown("### 💬 Direct Chat")
            st.markdown("Chat directly with Connective AI without any external content")
        
        st.markdown("---")
        st.markdown("## How It Works")
        st.markdown("""
        This application is powered by Connective Labs' advanced AI technology.
        
        1. **Choose your input method** from the sidebar tabs
        2. **Process the content** by clicking the appropriate button
        3. **Start asking questions** about the content or any topic (in direct chat mode)
        """)

if __name__ == "__main__":
    try:
        logger.info("Starting application")
        main()
        logger.info("Application running")
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}") 