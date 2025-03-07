import os
import logging
import tempfile
from typing import List, Dict, Any, Optional
import numpy as np
import io
import base64
from PIL import Image

# Try to import OpenCV with fallbacks
try:
    import cv2
except ImportError:
    try:
        import cv2.cv2 as cv2
    except ImportError:
        logging.error("OpenCV (cv2) could not be imported. Video processing will not work.")
        cv2 = None

# Try to import moviepy with fallback
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    logging.error("MoviePy could not be imported. Audio extraction will not work.")
    VideoFileClip = None

# Configure logging
logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Class to process video files for use with the AI API.
    Handles video frame extraction, audio processing, and preparing data for model input.
    Optimized for cloud deployment with resource constraints.
    """
    
    def __init__(self, video_path: str, sample_rate: int = 1, max_frames: int = 20, max_resolution: int = 480):
        """
        Initialize the VideoProcessor with a video path.
        
        Args:
            video_path: Path to the video file
            sample_rate: Number of frames to extract per second (default: 1)
            max_frames: Maximum number of frames to extract total (default: 20)
            max_resolution: Maximum height of frames in pixels (default: 480)
        """
        self.video_path = video_path
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.max_resolution = max_resolution
        self.frames = []
        self.video_info = {}
        self.audio_path = None
        
        logger.info(f"Initializing VideoProcessor for {video_path}")
        
        try:
            # Extract video information
            self._extract_video_info()
            
            # Extract frames with optimized settings
            self._extract_frames_optimized()
            
            # Skip audio extraction for cloud deployment
            # self._extract_audio()
            
            logger.info(f"VideoProcessor initialized successfully with {len(self.frames)} frames")
        except Exception as e:
            logger.error(f"Error initializing VideoProcessor: {str(e)}", exc_info=True)
            raise
    
    def _extract_video_info(self) -> None:
        """Extract basic information about the video."""
        logger.info("Extracting video information")
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        self.video_info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        }
        
        logger.info(f"Video info extracted: {self.video_info}")
        cap.release()
    
    def _extract_frames_optimized(self) -> None:
        """Extract frames with optimized settings for cloud deployment."""
        logger.info("Extracting frames with optimized settings")
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file for frame extraction: {self.video_path}")
        
        fps = self.video_info["fps"]
        total_frames = self.video_info["frame_count"]
        
        # Limit maximum number of frames to extract
        frame_skip = max(1, int(total_frames / self.max_frames))
        frame_skip = max(frame_skip, int(fps / self.sample_rate))
        
        logger.info(f"Frame skip rate: {frame_skip} (extracting approximately 1 frame every {frame_skip} frames)")
        
        # Calculate scale factor for resizing
        height = self.video_info["height"]
        scale = 1.0
        if height > self.max_resolution:
            scale = self.max_resolution / height
            logger.info(f"Scaling frames by factor of {scale}")
        
        # Extract frames at calculated intervals
        count = 0
        for i in range(0, int(total_frames), frame_skip):
            if count >= self.max_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Resize frame if needed
            if scale < 1.0:
                new_width = int(self.video_info["width"] * scale)
                new_height = int(self.video_info["height"] * scale)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Convert from BGR to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames.append(frame_rgb)
            count += 1
            
            # Log progress periodically
            if count % 5 == 0:
                logger.info(f"Extracted {count} frames so far")
        
        logger.info(f"Completed frame extraction: {len(self.frames)} frames extracted")
        cap.release()
    
    def _extract_audio(self) -> None:
        """
        Extract audio from the video if present.
        Note: This is disabled in cloud deployment to save resources.
        """
        pass
    
    def get_video_data(self) -> Dict[str, Any]:
        """
        Get information about the processed video.
        
        Returns:
            Dict containing video information
        """
        return {
            "info": self.video_info,
            "frame_count": len(self.frames),
            "has_audio": False
        }
    
    def get_frames(self, start_time: Optional[float] = None, 
                   end_time: Optional[float] = None) -> List[np.ndarray]:
        """
        Get frames from the video within a specific time range.
        
        Args:
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            
        Returns:
            List of frames as numpy arrays
        """
        if start_time is None and end_time is None:
            return self.frames
        
        # Convert time to frame indices
        fps_equivalent = self.sample_rate  # Since we're sampling at this rate
        start_idx = 0 if start_time is None else int(start_time * fps_equivalent)
        end_idx = len(self.frames) if end_time is None else int(end_time * fps_equivalent)
        
        # Ensure indices are within bounds
        start_idx = max(0, min(start_idx, len(self.frames) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(self.frames)))
        
        return self.frames[start_idx:end_idx]
    
    def prepare_for_gemini(self, max_frames: int = 20) -> List[Dict[str, Any]]:
        """
        Prepare video data for AI model input.
        
        Args:
            max_frames: Maximum number of frames to include (default: 20)
            
        Returns:
            List of frames in AI-compatible format
        """
        logger.info(f"Preparing frames for Gemini API, limiting to {max_frames} frames")
        
        # If we have more frames than max_frames, sample evenly across the video
        frames_to_use = self.frames
        if len(self.frames) > max_frames:
            indices = np.linspace(0, len(self.frames) - 1, max_frames, dtype=int)
            frames_to_use = [self.frames[i] for i in indices]
            logger.info(f"Sampled {len(frames_to_use)} frames from {len(self.frames)} total frames")
        
        # Convert frames to PIL images and then to base64
        gemini_frames = []
        for i, frame in enumerate(frames_to_use):
            try:
                # Convert and compress image
                pil_image = Image.fromarray(frame)
                buffer = io.BytesIO()
                
                # Use JPEG with compression to reduce size
                pil_image.save(buffer, format="JPEG", quality=85)
                image_data = buffer.getvalue()
                
                # Encode as base64
                base64_image = base64.b64encode(image_data).decode("utf-8")
                
                # Calculate timestamp
                timestamp = i / self.sample_rate
                
                gemini_frames.append({
                    "image": base64_image,
                    "timestamp": timestamp
                })
                
                # Log progress periodically
                if (i+1) % 5 == 0 or i == len(frames_to_use) - 1:
                    logger.info(f"Processed {i+1}/{len(frames_to_use)} frames for API")
                    
            except Exception as e:
                logger.error(f"Error processing frame {i}: {str(e)}")
                continue
        
        logger.info(f"Completed preparing {len(gemini_frames)} frames for API")
        return gemini_frames
    
    def cleanup(self) -> None:
        """Clean up any temporary files and release memory."""
        logger.info("Cleaning up VideoProcessor resources")
        if self.audio_path and os.path.exists(self.audio_path):
            os.unlink(self.audio_path)
            self.audio_path = None
            
        # Clear frames to free memory
        self.frames = [] 