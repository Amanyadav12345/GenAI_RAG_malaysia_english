# 🇲🇾 Malaysian English Transcription POC

A proof-of-concept system for transcribing Malaysian English (Manglish) with real-time monitoring and performance tracking.

## 🎯 Features

- **Speech-to-Text Transcription** optimized for Malaysian English
- **Code-Switching Detection** for English-Malay mixed conversations
- **Real-time Monitoring Dashboard** with performance metrics
- **RAG-based Context Enhancement** using vector database
- **Performance Tracking** with WER, CER, and confidence scoring
- **Interactive Web Interface** built with Streamlit

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│   Transcriber   │───▶│  RAG Enhancement│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │◀───│   Monitoring    │◀───│  Vector Database│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (for Whisper model)
- Audio files in WAV, MP3, FLAC, or M4A format

### Option 1: Automated Setup

```bash
# Clone or download the project files
# Run the setup script
python setup_and_run.py
```

The setup script will guide you through:
1. Installing requirements
2. Downloading models
3. Creating sample files
4. Running the dashboard

### Option 2: Manual Setup

```bash
# Install requirements
pip install -r requirements.txt

# Run the dashboard
streamlit run malaysian_transcription_poc.py
```

## 📊 Dashboard Features

### Real-time Metrics
- **Accuracy**: Overall transcription accuracy
- **Latency**: Average processing time per audio file
- **Throughput**: Files processed per minute
- **Word Error Rate (WER)**: Percentage of incorrectly transcribed words

### Performance Monitoring
- Live charts showing accuracy trends
- Error rate tracking over time
- Confidence score distributions
- Processing latency analysis

### Transcription Results
- Real-time transcription of uploaded audio
- Code-switching detection and highlighting
- Confidence scoring for each transcription
- Dialect marker identification

## 🎤 Malaysian English Support

### Supported Features
- **Manglish Particles**: Detection of "lah", "lor", "meh", etc.
- **Code-Switching**: English-Malay mixed conversations
- **Local Terms**: Kopitiam, mamak, tapau, shiok, etc.
- **Cultural Context**: Enhanced understanding through RAG

### Example Transcriptions
```
Input Audio: "Eh, you want to tapau this roti canai or not lah?"
Output: "Eh, you want to tapau (takeaway) this roti canai or not lah (emphasis particle)?"
Code-Switching: ✅ Detected (lah)
```

## 🔧