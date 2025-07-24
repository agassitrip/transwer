# Transwer - AI Meeting Assistant

Transwer is a powerful real-time AI-powered meeting assistant that provides live transcription, translation, and intelligent response suggestions. Perfect for multilingual meetings, accessibility needs, and content creation.

## 🚀 Features

### Real-time Speech Transcription
- **Multiple STT Engines**: Support for Vosk (offline), Faster Whisper (offline), Google Cloud Speech-to-Text, and Google Chirp
- **Live Transcription**: See both partial (real-time) and final transcriptions
- **Audio Device Selection**: Choose from available input devices
- **High Accuracy**: Professional-grade speech recognition

### Intelligent Translation
- **AI-Powered Translation**: Uses OpenAI GPT-4o-mini or Google Translate
- **Multiple Languages**: Supports Portuguese (BR), English (US), Spanish, Japanese, and Chinese
- **Background Processing**: Non-blocking translation with queue system
- **Force Translation**: Translate entire transcription buffer on demand

### Smart Response Suggestions
- **AI-Generated Suggestions**: Context-aware response suggestions using OpenAI
- **Professional Responses**: Concise suggestions (max 20 words) suitable for business meetings
- **Individual Refresh**: Update suggestions individually as needed
- **Real-time Updates**: Suggestions update based on conversation context

### Modern User Interface
- **Glassmorphism Design**: Beautiful dark theme with glass effects
- **Resizable Panels**: Adjust layout to your preference with drag handles
- **Real-time Status**: Visual indicators for API connection and processing status
- **Quick Language Switch**: Fast language selection in the footer
- **Responsive Design**: Works on various screen sizes

## 🛠️ Technical Stack

### Backend
- Python 3.x with Flask framework
- Flask-SocketIO for real-time WebSocket communication
- Multi-threaded architecture for audio processing
- Queue-based translation system
- Thread-safe state management

### Frontend
- Vanilla JavaScript (ES6+)
- TailwindCSS for styling
- Socket.IO client for real-time updates
- Local storage for settings persistence

### AI Services
- OpenAI API (GPT-4o-mini)
- Google Cloud Speech-to-Text API
- Google Translate API
- Vosk (offline STT)
- Faster Whisper (offline STT)

## 📋 Requirements

- Python 3.7 or higher
- Audio input device (microphone)
- API keys for cloud services (optional for offline modes)
- 4GB RAM minimum (8GB recommended for Faster Whisper)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transwer.git
cd transwer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/google/credentials.json
```

5. Download Vosk model (for offline STT):
```bash
mkdir -p models/vosk
cd models/vosk
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
mv vosk-model-en-us-0.22 model
```

## 🚀 Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Configure settings:
   - Click the settings icon
   - Select your audio device
   - Choose STT and translation engines
   - Enter API keys if using cloud services
   - Select target language

4. Start listening:
   - Click the play button or press Ctrl+Space
   - Speak in English
   - See real-time transcription, translation, and suggestions

## ⚙️ Configuration

The application uses a `transwer_config.json` file for persistent settings:

```json
{
  "sttEngine": "vosk",
  "translationEngine": "openai",
  "audioDevice": 0,
  "translationLang": "pt-BR",
  "apiKey": "your_api_key",
  "googleCreds": "path/to/credentials.json",
  "fastwhisperModel": "base",
  "computeType": "int8",
  "googleRegion": "us-central1"
}
```

## 🎯 Use Cases

- **International Business Meetings**: Real-time translation for global teams
- **Language Learning**: Practice with instant translations
- **Accessibility**: Live captions for hearing-impaired users
- **Content Creation**: Transcribe and translate podcasts/videos
- **Customer Support**: Multilingual support with response suggestions

## 🔒 Privacy & Security

- All offline modes (Vosk, Faster Whisper) process audio locally
- Cloud services only receive audio when their engines are selected
- API keys are stored locally and never transmitted except to their respective services
- No audio recording or storage - all processing is real-time

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Vosk for offline speech recognition
- OpenAI for translation and suggestions
- Google Cloud for speech and translation services
- The open-source community for various libraries used