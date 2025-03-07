# Connective Video Chatbot Prototype

A powerful AI assistant by Connective Labs that can analyze and chat about content from YouTube videos and MP4 files, or chat directly without any external content.

## Features

- **YouTube Content Analysis**: Enter any YouTube URL to process and chat about its content
- **MP4 File Upload**: Upload your own MP4 files for AI analysis
- **Direct Chat Mode**: Chat with the AI assistant without any external content

## Technology Stack

- **Backend**: Python, Google AI API
- **Frontend**: Streamlit
- **Content Processing**: OpenCV, MoviePy
- **YouTube Integration**: PyTubeFix

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/connectivehub/vid-based-chat.git
cd vid-based-chat
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your Google API key:

   - Create a `.env` file in the root directory
   - Add your Google API key: `GOOGLE_API_KEY=your_api_key_here`

4. Run the application:

```bash
streamlit run app.py
```

## Deployment

This application can be deployed to Streamlit Cloud:

1. Push code to GitHub (this repository)
2. Connect Streamlit Cloud to this repository
3. Add your Google API key as a secret in Streamlit Cloud settings
4. Deploy!

## Usage

1. Select a mode from the sidebar (YouTube, MP4 Upload, or Direct Chat)
2. Process content or start a direct chat
3. Ask questions to the AI assistant

## License

Copyright © 2025 Connective Labs. All rights reserved.
