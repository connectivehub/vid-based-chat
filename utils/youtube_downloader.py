import os
import tempfile
import time
import re
from typing import Optional, Tuple
from pytubefix import YouTube
import logging
import requests
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of common User-Agent strings to rotate through for better YouTube access
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (iPad; CPU OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
]

class YouTubeDownloader:
    """
    Class to download content from YouTube URLs.
    """
    
    @staticmethod
    def extract_video_id(youtube_url: str) -> str:
        """
        Extract the content ID from a YouTube URL.
        
        Args:
            youtube_url: URL of the YouTube content
            
        Returns:
            The content ID extracted from the URL
        """
        # Regular expression to match YouTube content IDs
        youtube_regex = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
        match = re.search(youtube_regex, youtube_url)
        
        if match:
            return match.group(1)
        else:
            raise ValueError("Could not extract content ID from URL")
    
    @staticmethod
    def download_video(youtube_url: str, output_path: Optional[str] = None, max_retries: int = 3) -> Tuple[str, Optional[str]]:
        """
        Download content from a YouTube URL with retry logic.
        
        Args:
            youtube_url: URL of the YouTube content
            output_path: Path where to save the content (if None, creates a temporary file)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple containing:
                - Path to the downloaded content file
                - Title of the content
        """
        logger.info(f"Starting download for YouTube URL: {youtube_url}")
        
        # Create a temporary file if no output path is provided
        if output_path is None:
            temp_dir = tempfile.mkdtemp()
            output_path = temp_dir
            logger.debug(f"Using temporary directory: {temp_dir}")
        
        # Extract content ID for more reliable access
        try:
            content_id = YouTubeDownloader.extract_video_id(youtube_url)
            # Standardize the URL format
            standard_url = f"https://www.youtube.com/watch?v={content_id}"
            logger.info(f"Extracted content ID: {content_id}")
        except ValueError as e:
            logger.error(f"Error extracting content ID: {str(e)}")
            raise Exception(f"Invalid YouTube URL: {str(e)}")
        
        # Try to download with retries
        retry_count = 0
        last_exception = None
        
        while retry_count < max_retries:
            try:
                # Select a random User-Agent for this attempt
                user_agent = random.choice(USER_AGENTS)
                
                # Check if the content is available by checking thumbnail availability
                # Use a proper User-Agent in the headers
                headers = {
                    'User-Agent': user_agent,
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Referer': 'https://www.google.com/',
                    'DNT': '1'
                }
                
                logger.debug(f"Using User-Agent: {user_agent}")
                
                response = requests.head(
                    f"https://img.youtube.com/vi/{content_id}/0.jpg",
                    headers=headers
                )
                
                if response.status_code == 404:
                    logger.error(f"Content ID {content_id} does not exist or is not available")
                    raise Exception(f"YouTube content not found or not available: {content_id}")
                
                # Create YouTube object with custom settings to avoid 403 errors
                yt = None
                for attempt in range(3):
                    try:
                        # Initialize YouTube with custom headers and proxies
                        yt = YouTube(
                            standard_url,
                            use_oauth=False,
                            allow_oauth_cache=False,
                            use_innertube=True
                        )
                        
                        # Set custom headers and disable use of cipher
                        yt.headers = headers
                        
                        # Wait for metadata to be populated
                        timeout = 0
                        while not hasattr(yt, 'title') or not yt.title:
                            time.sleep(1)
                            timeout += 1
                            if timeout > 10:  # Wait max 10 seconds
                                raise TimeoutError("Timed out waiting for content metadata")
                        break
                    except Exception as e:
                        logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                        time.sleep(3 + attempt)  # Increasing wait time for subsequent retries
                
                # If we still don't have a YouTube object, raise an exception
                if yt is None or not hasattr(yt, 'title') or not yt.title:
                    raise Exception("Failed to create YouTube object after multiple attempts")
                
                # Get content title more safely
                content_title = getattr(yt, 'title', f"YouTube Content {content_id}")
                logger.info(f"YouTube content title: {content_title}")
                
                # Get the highest resolution stream with both video and audio
                logger.debug("Getting stream with highest resolution")
                stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                
                if not stream:
                    logger.warning("No progressive stream found, trying to get highest resolution stream")
                    # If no progressive stream is available, get the highest resolution stream
                    stream = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()
                
                if not stream:
                    logger.error("No suitable stream found")
                    raise ValueError("No suitable stream found for this YouTube content")
                
                logger.info(f"Selected stream: resolution={stream.resolution}")
                
                # Download the content
                logger.info(f"Downloading content to: {output_path}")
                
                # Use the on_progress_callback to track download progress
                content_path = stream.download(output_path=output_path)
                logger.info(f"Download completed: {content_path}")
                
                return content_path, content_title
                
            except Exception as e:
                last_exception = e
                retry_count += 1
                logger.warning(f"Download attempt {retry_count} failed: {str(e)}")
                time.sleep(3 * retry_count)  # Exponential backoff
        
        # If we've exhausted all retries, log the error and raise an exception
        logger.error(f"Error downloading YouTube content after {max_retries} attempts: {str(last_exception)}")
        raise Exception(f"Failed to download YouTube content after multiple attempts: {str(last_exception)}") 