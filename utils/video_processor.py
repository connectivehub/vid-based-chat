import os
import cv2
import base64
import tempfile
from typing import List, Dict, Any, Optional
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image
import io

class VideoProcessor:
    """
    Class to process video files for use with the Gemini API.
    Handles video frame extraction, audio processing, and preparing data for model input.
    """
    
    def __init__(self, video_path: str, sample_rate: int = 1):
        """
        Initialize the VideoProcessor with a video path.
        
        Args:
            video_path: Path to the video file
            sample_rate: Number of frames to extract per second (default: 1 - matches Gemini's sampling rate)
        """
        self.video_path = video_path
        self.sample_rate = sample_rate
        self.frames = []
        self.video_info = {}
        self.audio_path = None
        
        # Extract video information and frames
        self._extract_video_info()
        self._extract_frames()
        self._extract_audio()
    
    def _extract_video_info(self) -> None:
        """Extract basic information about the video."""
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        self.video_info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        }
        
        cap.release()
    
    def _extract_frames(self) -> None:
        """Extract frames from the video at the specified sample rate."""
        cap = cv2.VideoCapture(self.video_path)
        fps = self.video_info["fps"]
        total_frames = self.video_info["frame_count"]
        
        # Calculate frame indices to extract based on sample rate
        frame_indices = []
        for i in range(0, int(total_frames), int(fps / self.sample_rate)):
            frame_indices.append(i)
        
        # Extract frames at calculated indices
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert from BGR to RGB (OpenCV uses BGR by default)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame_rgb)
        
        cap.release()
    
    def _extract_audio(self) -> None:
        """Extract audio from the video if present."""
        try:
            # Create temporary file for audio
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_audio.close()
            
            # Extract audio using moviepy
            video_clip = VideoFileClip(self.video_path)
            if video_clip.audio is not None:
                video_clip.audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
                self.audio_path = temp_audio.name
            
            video_clip.close()
        except Exception as e:
            print(f"Error extracting audio: {e}")
            if self.audio_path and os.path.exists(self.audio_path):
                os.unlink(self.audio_path)
            self.audio_path = None
    
    def get_video_data(self) -> Dict[str, Any]:
        """
        Get information about the processed video.
        
        Returns:
            Dict containing video information
        """
        return {
            "info": self.video_info,
            "frame_count": len(self.frames),
            "has_audio": self.audio_path is not None
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
    
    def prepare_for_gemini(self, max_frames: int = 50) -> List[Dict[str, Any]]:
        """
        Prepare video data for Gemini model input.
        
        Args:
            max_frames: Maximum number of frames to include (to manage token usage)
            
        Returns:
            List of frames in Gemini-compatible format
        """
        # If we have more frames than max_frames, sample evenly across the video
        frames_to_use = self.frames
        if len(self.frames) > max_frames:
            indices = np.linspace(0, len(self.frames) - 1, max_frames, dtype=int)
            frames_to_use = [self.frames[i] for i in indices]
        
        # Convert frames to PIL images and then to base64
        gemini_frames = []
        for i, frame in enumerate(frames_to_use):
            pil_image = Image.fromarray(frame)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            image_data = buffer.getvalue()
            
            # Encode as base64
            base64_image = base64.b64encode(image_data).decode("utf-8")
            
            # Calculate timestamp
            timestamp = i / self.sample_rate
            
            gemini_frames.append({
                "image": base64_image,
                "timestamp": timestamp
            })
        
        return gemini_frames
    
    def cleanup(self) -> None:
        """Clean up any temporary files."""
        if self.audio_path and os.path.exists(self.audio_path):
            os.unlink(self.audio_path)
            self.audio_path = None 