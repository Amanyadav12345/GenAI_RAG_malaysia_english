# Initialize system
    if 'system' not in st.session_state:
        with st.spinner("Initializing enhanced transcription system..."):
            st.session_state.system = EnhancedTranscriptionSystem()
    
    system = st.session_state.system
    
    # Manual text input section
    st.header("📝 Text Input for Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        manual_text = st.text_area(
            "Enter Malaysian English text for analysis:",
            placeholder="Eh, you want to makan at kopitiam or not lah?",
            help="Type any Malaysian English text to see translation and AI analysis"
        )
    
    with col2:
        st.write("**Examples:**")
        examples = [
            "Wah traffic jam so teruk!",
            "Can belanja me teh tarik meh?",
            "Alamak forgot my wallet lah!",
            "This laksa very shiok!"
        ]
        for example in examples:
            if st.button(example, key=f"example_{hash(example)}"):
                st.session_state.example_text = example
    
    # Use example text if selected
    if 'example_text' in st.session_state:
        manual_text = st.session_state.example_text
        del st.session_state.example_text
    
    if manual_text and st.button("🔍 Analyze Text", type="primary"):
        with st.spinner("Analyzing text..."):
            try:
                # Create a mock result for text analysis
                mock_result = TranscriptionResult(
                    id=f"text_{int(time.time#!/usr/bin/env python3
"""
Lightweight Malaysian English Transcription POC
===============================================

A streamlined version that avoids PyTorch/Streamlit conflicts
while still demonstrating the core concepts.
"""

import os
import json
import time
import sqlite3
import re
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Audio processing
try:
    import speech_recognition as sr
    import pydub
    from pydub import AudioSegment
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    st.warning("⚠️ Audio processing not available. Install: pip install SpeechRecognition pydub")

# AI/NLP for analysis
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    AI_ANALYSIS_AVAILABLE = True
except ImportError:
    AI_ANALYSIS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    """Data class for transcription results"""
    id: str
    timestamp: datetime
    audio_file: str
    transcription: str
    english_translation: str
    confidence_score: float
    processing_time: float
    language_detected: str
    code_switching_detected: bool
    word_count: int
    dialect_markers: List[str]
    ai_analysis: Dict[str, any]

class RealAudioTranscriber:
    """
    Real audio transcriber using speech recognition
    """
    
    def __init__(self):
        if not AUDIO_PROCESSING_AVAILABLE:
            raise ImportError("Audio processing libraries not available")
        
        self.recognizer = sr.Recognizer()
        
        # Malaysian English translations and enhancements
        self.manglish_to_english = {
            "lah": "",  # Remove particles for English translation
            "lor": "",
            "meh": "?",  # Convert to question mark
            "wah": "Wow",
            "eh": "Hey",
            "alamak": "Oh no",
            "shiok": "delicious/good",
            "teruk": "terrible",
            "sedap": "delicious",
            "gila": "very/extremely",
            "tapau": "takeaway",
            "kopitiam": "coffee shop",
            "mamak": "Indian-Muslim restaurant",
            "makan": "eat",
            "belanja": "treat/pay for"
        }
        
        self.code_switching_patterns = list(self.manglish_to_english.keys())
    
    def convert_audio_format(self, audio_path: str) -> str:
        """Convert audio to WAV format for better recognition"""
        try:
            audio = AudioSegment.from_file(audio_path)
            # Convert to WAV with optimal settings for speech recognition
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            wav_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
            audio.export(wav_path, format="wav")
            return wav_path
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return audio_path
    
    def transcribe_with_google(self, audio_path: str) -> tuple:
        """Transcribe using Google Speech Recognition"""
        try:
            wav_path = self.convert_audio_format(audio_path)
            
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
            
            # Try multiple recognition services
            try:
                text = self.recognizer.recognize_google(audio_data)
                confidence = 0.85  # Google doesn't provide confidence
                return text, confidence
            except sr.UnknownValueError:
                return "Could not understand audio", 0.0
            except sr.RequestError as e:
                logger.error(f"Google API error: {e}")
                return f"API Error: {e}", 0.0
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return f"Transcription failed: {e}", 0.0
    
    def detect_code_switching(self, text: str):
        """Enhanced code-switching detection"""
        detected_markers = []
        text_lower = text.lower()
        
        for pattern in self.code_switching_patterns:
            if re.search(r'\b' + pattern + r'\b', text_lower):
                detected_markers.append(pattern)
        
        return len(detected_markers) > 0, detected_markers
    
    def translate_to_english(self, text: str) -> str:
        """Convert Malaysian English to standard English"""
        english_text = text
        
        # Replace Manglish terms
        for manglish, english in self.manglish_to_english.items():
            if english:  # Only replace if there's an English equivalent
                pattern = r'\b' + re.escape(manglish) + r'\b'
                english_text = re.sub(pattern, english, english_text, flags=re.IGNORECASE)
            else:  # Remove particles
                pattern = r'\s*\b' + re.escape(manglish) + r'\b\s*'
                english_text = re.sub(pattern, ' ', english_text, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        english_text = re.sub(r'\s+', ' ', english_text).strip()
        
        # Fix grammar issues common in Malaysian English
        grammar_fixes = {
            r'\bcan or not\b': 'is it possible',
            r'\bwant or not\b': 'do you want it',
            r'\bhave or not\b': 'is it available',
            r'\bgot or not\b': 'is there any',
            r'\blidat\b': 'like that',
            r'\blidis\b': 'like this'
        }
        
        for pattern, replacement in grammar_fixes.items():
            english_text = re.sub(pattern, replacement, english_text, flags=re.IGNORECASE)
        
        return english_text
    
    def transcribe_audio(self, audio_path: str) -> TranscriptionResult:
        """Main transcription method"""
        start_time = time.time()
        
        try:
            # Transcribe audio
            transcription, confidence = self.transcribe_with_google(audio_path)
            
            # Detect code-switching
            code_switching, dialect_markers = self.detect_code_switching(transcription)
            
            # Translate to English
            english_translation = self.translate_to_english(transcription)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            
            # Create result (AI analysis will be added later)
            result = TranscriptionResult(
                id=f"trans_{int(time.time())}_{np.random.randint(1000)}",
                timestamp=datetime.now(),
                audio_file=audio_path,
                transcription=transcription,
                english_translation=english_translation,
                confidence_score=confidence,
                processing_time=processing_time,
                language_detected="malaysian_english" if code_switching else "english",
                code_switching_detected=code_switching,
                word_count=len(transcription.split()),
                dialect_markers=dialect_markers,
                ai_analysis={}  # Will be populated by AI analyzer
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

class AIAnalyzer:
    """
    AI-powered analysis of transcribed text
    """
    
    def __init__(self):
        if AI_ANALYSIS_AVAILABLE:
            try:
                # Load lightweight models for analysis
                self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                                 model="cardiffnlp/twitter-roberta-base-sentiment-latest")
                self.emotion_analyzer = pipeline("text-classification", 
                                                model="j-hartmann/emotion-english-distilroberta-base")
                self.summarizer = pipeline("summarization", 
                                         model="facebook/bart-large-cnn")
                self.available = True
            except Exception as e:
                logger.warning(f"AI models not available: {e}")
                self.available = False
        else:
            self.available = False
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of the text"""
        if not self.available:
            return {"sentiment": "neutral", "confidence": 0.5}
        
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                "sentiment": result['label'].lower(),
                "confidence": result['score']
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "confidence": 0.5}
    
    def analyze_emotion(self, text: str) -> Dict:
        """Analyze emotions in the text"""
        if not self.available:
            return {"emotion": "neutral", "confidence": 0.5}
        
        try:
            result = self.emotion_analyzer(text)[0]
            return {
                "emotion": result['label'].lower(),
                "confidence": result['score']
            }
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return {"emotion": "neutral", "confidence": 0.5}
    
    def extract_key_info(self, text: str) -> Dict:
        """Extract key information from the text"""
        info = {
            "topics": [],
            "entities": [],
            "intent": "unknown",
            "urgency": "normal"
        }
        
        text_lower = text.lower()
        
        # Topic detection
        topics = {
            "food": ["makan", "eat", "food", "restaurant", "kopitiam", "mamak", "laksa", "roti", "teh", "coffee"],
            "transport": ["traffic", "jam", "car", "bus", "train", "taxi", "grab", "drive"],
            "weather": ["hot", "cold", "rain", "sunny", "weather", "aircon"],
            "shopping": ["buy", "shop", "market", "pasar", "mall", "price"],
            "work": ["office", "meeting", "work", "boss", "colleague", "project"],
            "social": ["friend", "family", "party", "gathering", "lepak"]
        }
        
        for topic, keywords in topics.items():
            if any(keyword in text_lower for keyword in keywords):
                info["topics"].append(topic)
        
        # Intent detection
        if any(word in text_lower for word in ["want", "need", "can", "help"]):
            info["intent"] = "request"
        elif any(word in text_lower for word in ["?", "or not", "meh"]):
            info["intent"] = "question"
        elif any(word in text_lower for word in ["thanks", "thank you", "good", "nice"]):
            info["intent"] = "appreciation"
        elif any(word in text_lower for word in ["alamak", "wah", "oh no"]):
            info["intent"] = "exclamation"
        
        # Urgency detection
        if any(word in text_lower for word in ["urgent", "quickly", "fast", "now", "emergency"]):
            info["urgency"] = "high"
        elif any(word in text_lower for word in ["when free", "no rush", "whenever"]):
            info["urgency"] = "low"
        
        return info
    
    def generate_summary(self, text: str) -> str:
        """Generate a summary of the conversation"""
        if not self.available or len(text.split()) < 10:
            return text
        
        try:
            summary = self.summarizer(text, max_length=50, min_length=10, do_sample=False)[0]
            return summary['summary_text']
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return text
    
    def analyze_text(self, text: str, english_text: str) -> Dict:
        """Complete AI analysis of the text"""
        analysis = {}
        
        # Use English translation for better AI analysis
        analysis_text = english_text if english_text else text
        
        # Sentiment analysis
        analysis["sentiment"] = self.analyze_sentiment(analysis_text)
        
        # Emotion analysis
        analysis["emotion"] = self.analyze_emotion(analysis_text)
        
        # Key information extraction
        analysis["key_info"] = self.extract_key_info(text)  # Use original text for Malaysian context
        
        # Summary
        analysis["summary"] = self.generate_summary(analysis_text)
        
        # Language analysis
        analysis["language_analysis"] = {
            "code_switching_level": "high" if len(analysis["key_info"]["topics"]) > 0 else "low",
            "formality": "informal" if any(word in text.lower() for word in ["lah", "lor", "wah"]) else "formal",
            "complexity": "simple" if len(text.split()) < 10 else "complex"
        }
        
        return analysis

class MockVectorDatabase:
    """
    Mock vector database for RAG demonstration
    """
    
    def __init__(self):
        self.malaysian_context = [
            "Malaysian English (Manglish) includes particles like 'lah', 'lor', 'meh'",
            "Kopitiam refers to traditional coffee shops in Malaysia",
            "Mamak stalls are Indian-Muslim restaurants popular in Malaysia",
            "Tapau means takeaway in Malaysian English",
            "Code-switching between English and Malay is very common",
            "Shiok means delicious or enjoyable in Malaysian slang",
            "Alamak is an expression of surprise or dismay"
        ]
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Mock search - returns relevant context"""
        # Simple keyword matching for demo
        relevant = []
        query_lower = query.lower()
        
        for context in self.malaysian_context:
            if any(word in context.lower() for word in query_lower.split()):
                relevant.append(context)
        
        return relevant[:k] if relevant else self.malaysian_context[:k]

class MonitoringSystem:
    """
    Simplified monitoring system using only SQLite
    """
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS transcriptions (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                audio_file TEXT,
                transcription TEXT,
                english_translation TEXT,
                confidence_score REAL,
                processing_time REAL,
                language_detected TEXT,
                code_switching_detected BOOLEAN,
                word_count INTEGER,
                dialect_markers TEXT,
                ai_analysis TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp TEXT,
                wer REAL,
                cer REAL,
                latency REAL,
                throughput REAL,
                accuracy REAL,
                confidence_avg REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_transcription(self, result: TranscriptionResult):
        """Log transcription result"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO transcriptions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.id,
            result.timestamp.isoformat(),
            result.audio_file,
            result.transcription,
            result.english_translation,
            result.confidence_score,
            result.processing_time,
            result.language_detected,
            result.code_switching_detected,
            result.word_count,
            json.dumps(result.dialect_markers),
            json.dumps(result.ai_analysis)
        ))
        conn.commit()
        conn.close()
    
    def get_recent_transcriptions(self, limit: int = 20) -> List[Dict]:
        """Get recent transcriptions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT * FROM transcriptions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            result['dialect_markers'] = json.loads(result['dialect_markers']) if result['dialect_markers'] else []
            result['ai_analysis'] = json.loads(result['ai_analysis']) if result['ai_analysis'] else {}
            results.append(result)
        
        conn.close()
        return results
    
    def calculate_current_metrics(self) -> Dict:
        """Calculate current performance metrics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent transcriptions for metrics
        cursor = conn.execute('''
            SELECT confidence_score, processing_time 
            FROM transcriptions 
            WHERE timestamp > datetime('now', '-1 hour')
            ORDER BY timestamp DESC
        ''')
        
        recent_data = cursor.fetchall()
        conn.close()
        
        if not recent_data:
            # Return default metrics if no data
            return {
                "accuracy": 0.92,
                "latency": 1.5,
                "throughput": 8.5,
                "wer": 0.08,
                "confidence_avg": 0.90
            }
        
        confidences = [row[0] for row in recent_data]
        latencies = [row[1] for row in recent_data]
        
        avg_confidence = np.mean(confidences)
        avg_latency = np.mean(latencies)
        throughput = len(recent_data) / max(1, len(recent_data) * 0.1)  # Simplified
        wer = max(0, 1 - avg_confidence) * 0.3  # Estimated
        accuracy = 1 - wer
        
        return {
            "accuracy": accuracy,
            "latency": avg_latency,
            "throughput": throughput,
            "wer": wer,
            "confidence_avg": avg_confidence
        }

class EnhancedTranscriptionSystem:
    """
    Enhanced system with real audio processing and AI analysis
    """
    
    def __init__(self):
        # Initialize transcriber
        if AUDIO_PROCESSING_AVAILABLE:
            try:
                self.transcriber = RealAudioTranscriber()
                self.audio_enabled = True
            except Exception as e:
                logger.warning(f"Real audio transcriber failed, using mock: {e}")
                self.transcriber = self._create_mock_transcriber()
                self.audio_enabled = False
        else:
            self.transcriber = self._create_mock_transcriber()
            self.audio_enabled = False
        
        # Initialize AI analyzer
        self.ai_analyzer = AIAnalyzer()
        
        # Initialize other components
        self.vector_db = MockVectorDatabase()
        self.monitoring = MonitoringSystem()
        
        logger.info(f"Enhanced Transcription System initialized (Audio: {self.audio_enabled}, AI: {self.ai_analyzer.available})")
    
    def _create_mock_transcriber(self):
        """Create mock transcriber when real one isn't available"""
        class MockTranscriber:
            def __init__(self):
                self.sample_transcriptions = [
                    "Eh, you want to makan at the kopitiam or not lah?",
                    "Wah, the traffic jam so teruk today lor!",
                    "Can you belanja me one teh tarik meh?",
                    "Alamak! I forgot to bring my wallet lah!",
                    "This laksa very shiok, you must try!"
                ]
                
                self.manglish_to_english = {
                    "lah": "", "lor": "", "meh": "?", "wah": "Wow",
                    "eh": "Hey", "alamak": "Oh no", "shiok": "delicious",
                    "teruk": "terrible", "makan": "eat", "kopitiam": "coffee shop"
                }
            
            def detect_code_switching(self, text):
                patterns = ["lah", "lor", "meh", "wah", "eh", "alamak", "shiok", "teruk"]
                detected = [p for p in patterns if p in text.lower()]
                return len(detected) > 0, detected
            
            def translate_to_english(self, text):
                english_text = text
                for manglish, english in self.manglish_to_english.items():
                    if english:
                        english_text = english_text.replace(manglish, english)
                    else:
                        english_text = english_text.replace(manglish, "")
                return english_text.strip()
            
            def transcribe_audio(self, audio_path):
                transcription = np.random.choice(self.sample_transcriptions)
                code_switching, dialect_markers = self.detect_code_switching(transcription)
                english_translation = self.translate_to_english(transcription)
                
                return TranscriptionResult(
                    id=f"trans_{int(time.time())}_{np.random.randint(1000)}",
                    timestamp=datetime.now(),
                    audio_file=audio_path,
                    transcription=transcription,
                    english_translation=english_translation,
                    confidence_score=np.random.uniform(0.85, 0.98),
                    processing_time=np.random.uniform(1, 3),
                    language_detected="malaysian_english",
                    code_switching_detected=code_switching,
                    word_count=len(transcription.split()),
                    dialect_markers=dialect_markers,
                    ai_analysis={}
                )
        
        return MockTranscriber()
    
    def process_audio(self, audio_path: str) -> TranscriptionResult:
        """Process audio with full pipeline"""
        try:
            # Step 1: Transcribe audio
            result = self.transcriber.transcribe_audio(audio_path)
            
            # Step 2: AI Analysis
            if self.ai_analyzer.available:
                ai_analysis = self.ai_analyzer.analyze_text(
                    result.transcription, 
                    result.english_translation
                )
                result.ai_analysis = ai_analysis
            else:
                # Basic analysis without AI models
                result.ai_analysis = {
                    "sentiment": {"sentiment": "neutral", "confidence": 0.5},
                    "emotion": {"emotion": "neutral", "confidence": 0.5},
                    "key_info": {
                        "topics": ["conversation"],
                        "intent": "communication",
                        "urgency": "normal"
                    },
                    "summary": result.english_translation,
                    "language_analysis": {
                        "code_switching_level": "medium" if result.code_switching_detected else "low",
                        "formality": "informal",
                        "complexity": "simple"
                    }
                }
            
            # Step 3: Get relevant context
            context = self.vector_db.search(result.transcription)
            if context:
                logger.info(f"Found context: {context[0][:50]}...")
            
            # Step 4: Log result
            self.monitoring.log_transcription(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise

def create_dashboard():
    """
    Streamlit dashboard
    """
    st.set_page_config(
        page_title="Malaysian English Transcription (Lightweight)",
        page_icon="🇲🇾",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #4ECDC4;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .success-alert {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🇲🇾 Enhanced Malaysian English Transcription System")
    st.markdown("**Real audio processing** with AI analysis and English translation")
    
    # Show system capabilities
    col1, col2, col3 = st.columns(3)
    with col1:
        audio_status = "✅ Enabled" if AUDIO_PROCESSING_AVAILABLE else "❌ Mock Mode"
        st.info(f"🎤 Audio Processing: {audio_status}")
    with col2:
        ai_status = "✅ Enabled" if AI_ANALYSIS_AVAILABLE else "❌ Basic Mode"
        st.info(f"🤖 AI Analysis: {ai_status}")
    with col3:
        st.info("🇲🇾 Malaysian English: ✅ Enabled")
    
    # Installation instructions if needed
    if not AUDIO_PROCESSING_AVAILABLE:
        with st.expander("📦 Enable Audio Processing"):
            st.code("pip install SpeechRecognition pydub")
            st.write("Install these packages for real audio transcription")
    
    if not AI_ANALYSIS_AVAILABLE:
        with st.expander("🤖 Enable AI Analysis"):
            st.code("pip install transformers torch")
            st.write("Install these packages for sentiment analysis and AI insights")
    
    # Initialize system
    if 'system' not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.system = LightweightTranscriptionSystem()
    
    system = st.session_state.system
    
    # File upload section
    st.header("📤 Upload Audio for Transcription")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an audio file", 
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload Malaysian English audio (demo will generate sample transcription)"
        )
    
    with col2:
        st.info("💡 **Demo Mode**\n\nThis version generates sample Malaysian English transcriptions for demonstration.")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        audio_path = f"temp_{uploaded_file.name}"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🎯 Transcribe Audio", type="primary"):
            with st.spinner("Processing audio..."):
                try:
                    result = system.process_audio(audio_path)
                    
                    # Display results
                    st.markdown("""
                    <div class="success-alert">
                        <h4>✅ Transcription Complete!</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.subheader("📝 Transcription Result")
                        st.text_area("Transcribed Text", result.transcription, height=80)
                        
                        if result.code_switching_detected:
                            markers_text = ", ".join([f"'{m}'" for m in result.dialect_markers])
                            st.success(f"🔄 Code-switching detected: {markers_text}")
                        
                        # Show context
                        context = system.vector_db.search(result.transcription, k=2)
                        if context:
                            st.info(f"📚 Context: {context[0]}")
                    
                    with col2:
                        st.subheader("📊 Analysis")
                        st.metric("Confidence", f"{result.confidence_score:.1%}")
                        st.metric("Processing Time", f"{result.processing_time:.2f}s")
                        st.metric("Word Count", result.word_count)
                        st.metric("Language", result.language_detected.replace('_', ' ').title())
                
                except Exception as e:
                    st.error(f"❌ Processing failed: {e}")
                
                finally:
                    # Clean up
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
    
    # Monitoring dashboard
    st.header("📈 Performance Monitoring")
    
    # Get current metrics
    current_metrics = system.monitoring.calculate_current_metrics()
    
    # Display current metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Accuracy", f"{current_metrics['accuracy']:.1%}")
    
    with col2:
        st.metric("⚡ Avg Latency", f"{current_metrics['latency']:.2f}s")
    
    with col3:
        st.metric("📊 Throughput", f"{current_metrics['throughput']:.1f}/min")
    
    with col4:
        st.metric("🔢 Word Error Rate", f"{current_metrics['wer']:.1%}")
    
    # Sample performance charts
    st.subheader("📈 Performance Trends")
    
    # Generate sample time series data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='H')
    sample_metrics = pd.DataFrame({
        'timestamp': dates,
        'accuracy': np.random.normal(0.92, 0.03, 30).clip(0.85, 0.98),
        'latency': np.random.normal(1.5, 0.4, 30).clip(0.8, 3.0),
        'confidence': np.random.normal(0.90, 0.05, 30).clip(0.8, 0.98)
    })
    
    # Create performance chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy Over Time', 'Latency Trends', 
                       'Confidence Distribution', 'Error Rates'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accuracy
    fig.add_trace(
        go.Scatter(x=sample_metrics['timestamp'], y=sample_metrics['accuracy'],
                  name='Accuracy', line=dict(color='green')),
        row=1, col=1
    )
    
    # Latency
    fig.add_trace(
        go.Scatter(x=sample_metrics['timestamp'], y=sample_metrics['latency'],
                  name='Latency', line=dict(color='blue')),
        row=1, col=2
    )
    
    # Confidence histogram
    fig.add_trace(
        go.Histogram(x=sample_metrics['confidence'], name='Confidence',
                    marker_color='purple', nbinsx=20),
        row=2, col=1
    )
    
    # Error rates
    error_data = 1 - sample_metrics['accuracy']
    fig.add_trace(
        go.Scatter(x=sample_metrics['timestamp'], y=error_data,
                  name='Error Rate', line=dict(color='red')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent transcriptions
    st.header("📋 Recent Transcriptions")
    
    recent_transcriptions = system.monitoring.get_recent_transcriptions(10)
    
    if recent_transcriptions:
        df = pd.DataFrame(recent_transcriptions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Display key columns
        display_cols = ['timestamp', 'transcription', 'confidence_score', 
                       'processing_time', 'code_switching_detected']
        
        if all(col in df.columns for col in display_cols):
            st.dataframe(
                df[display_cols].sort_values('timestamp', ascending=False),
                use_container_width=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time"),
                    "transcription": st.column_config.TextColumn("Transcription", width="large"),
                    "confidence_score": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                    "processing_time": st.column_config.NumberColumn("Time (s)", format="%.2f"),
                    "code_switching_detected": st.column_config.CheckboxColumn("Code-Switch")
                }
            )
    else:
        st.info("No transcriptions yet. Upload an audio file to get started!")
    
    # Malaysian English examples
    with st.expander("🇲🇾 Malaysian English Examples", expanded=False):
        examples = [
            ("Casual greeting", "Eh, you want to makan at kopitiam or not lah?"),
            ("Traffic complaint", "Wah, the traffic jam so teruk today lor!"),
            ("Food request", "Can you belanja me one teh tarik meh?"),
            ("Surprise expression", "Alamak! I forgot my wallet lah!"),
            ("Food recommendation", "This laksa very shiok, you must try!")
        ]
        
        for context, example in examples:
            st.write(f"**{context}:** {example}")

if __name__ == "__main__":
    create_dashboard()


    
    # Initialize system
    if 'system' not in st.session_state:
        with st.spinner("Initializing enhanced transcription system..."):
            st.session_state.system = EnhancedTranscriptionSystem()
    
    system = st.session_state.system
    
    # Manual text input section
    st.header("📝 Text Input for Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        manual_text = st.text_area(
            "Enter Malaysian English text for analysis:",
            placeholder="Eh, you want to makan at kopitiam or not lah?",
            help="Type any Malaysian English text to see translation and AI analysis"
        )
    
    with col2:
        st.write("**Examples:**")
        examples = [
            "Wah traffic jam so teruk!",
            "Can belanja me teh tarik meh?",
            "Alamak forgot my wallet lah!",
            "This laksa very shiok!"
        ]
        for example in examples:
            if st.button(example, key=f"example_{hash(example)}"):
                st.session_state.example_text = example
    
    # Use example text if selected
    if 'example_text' in st.session_state:
        manual_text = st.session_state.example_text
        del st.session_state.example_text
    
    if manual_text and st.button("🔍 Analyze Text", type="primary"):
        with st.spinner("Analyzing text..."):
            try:
                # Create a mock result for text analysis
                mock_result = TranscriptionResult(
                    id=f"text_{int(time.time#!/usr/bin/env python3
"""
Lightweight Malaysian English Transcription POC
===============================================

A streamlined version that avoids PyTorch/Streamlit conflicts
while still demonstrating the core concepts.
"""

import os
import json
import time
import sqlite3
import re
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Audio processing
try:
    import speech_recognition as sr
    import pydub
    from pydub import AudioSegment
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    st.warning("⚠️ Audio processing not available. Install: pip install SpeechRecognition pydub")

# AI/NLP for analysis
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    AI_ANALYSIS_AVAILABLE = True
except ImportError:
    AI_ANALYSIS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    """Data class for transcription results"""
    id: str
    timestamp: datetime
    audio_file: str
    transcription: str
    english_translation: str
    confidence_score: float
    processing_time: float
    language_detected: str
    code_switching_detected: bool
    word_count: int
    dialect_markers: List[str]
    ai_analysis: Dict[str, any]

class RealAudioTranscriber:
    """
    Real audio transcriber using speech recognition
    """
    
    def __init__(self):
        if not AUDIO_PROCESSING_AVAILABLE:
            raise ImportError("Audio processing libraries not available")
        
        self.recognizer = sr.Recognizer()
        
        # Malaysian English translations and enhancements
        self.manglish_to_english = {
            "lah": "",  # Remove particles for English translation
            "lor": "",
            "meh": "?",  # Convert to question mark
            "wah": "Wow",
            "eh": "Hey",
            "alamak": "Oh no",
            "shiok": "delicious/good",
            "teruk": "terrible",
            "sedap": "delicious",
            "gila": "very/extremely",
            "tapau": "takeaway",
            "kopitiam": "coffee shop",
            "mamak": "Indian-Muslim restaurant",
            "makan": "eat",
            "belanja": "treat/pay for"
        }
        
        self.code_switching_patterns = list(self.manglish_to_english.keys())
    
    def convert_audio_format(self, audio_path: str) -> str:
        """Convert audio to WAV format for better recognition"""
        try:
            audio = AudioSegment.from_file(audio_path)
            # Convert to WAV with optimal settings for speech recognition
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            wav_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
            audio.export(wav_path, format="wav")
            return wav_path
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return audio_path
    
    def transcribe_with_google(self, audio_path: str) -> tuple:
        """Transcribe using Google Speech Recognition"""
        try:
            wav_path = self.convert_audio_format(audio_path)
            
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
            
            # Try multiple recognition services
            try:
                text = self.recognizer.recognize_google(audio_data)
                confidence = 0.85  # Google doesn't provide confidence
                return text, confidence
            except sr.UnknownValueError:
                return "Could not understand audio", 0.0
            except sr.RequestError as e:
                logger.error(f"Google API error: {e}")
                return f"API Error: {e}", 0.0
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return f"Transcription failed: {e}", 0.0
    
    def detect_code_switching(self, text: str):
        """Enhanced code-switching detection"""
        detected_markers = []
        text_lower = text.lower()
        
        for pattern in self.code_switching_patterns:
            if re.search(r'\b' + pattern + r'\b', text_lower):
                detected_markers.append(pattern)
        
        return len(detected_markers) > 0, detected_markers
    
    def translate_to_english(self, text: str) -> str:
        """Convert Malaysian English to standard English"""
        english_text = text
        
        # Replace Manglish terms
        for manglish, english in self.manglish_to_english.items():
            if english:  # Only replace if there's an English equivalent
                pattern = r'\b' + re.escape(manglish) + r'\b'
                english_text = re.sub(pattern, english, english_text, flags=re.IGNORECASE)
            else:  # Remove particles
                pattern = r'\s*\b' + re.escape(manglish) + r'\b\s*'
                english_text = re.sub(pattern, ' ', english_text, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        english_text = re.sub(r'\s+', ' ', english_text).strip()
        
        # Fix grammar issues common in Malaysian English
        grammar_fixes = {
            r'\bcan or not\b': 'is it possible',
            r'\bwant or not\b': 'do you want it',
            r'\bhave or not\b': 'is it available',
            r'\bgot or not\b': 'is there any',
            r'\blidat\b': 'like that',
            r'\blidis\b': 'like this'
        }
        
        for pattern, replacement in grammar_fixes.items():
            english_text = re.sub(pattern, replacement, english_text, flags=re.IGNORECASE)
        
        return english_text
    
    def transcribe_audio(self, audio_path: str) -> TranscriptionResult:
        """Main transcription method"""
        start_time = time.time()
        
        try:
            # Transcribe audio
            transcription, confidence = self.transcribe_with_google(audio_path)
            
            # Detect code-switching
            code_switching, dialect_markers = self.detect_code_switching(transcription)
            
            # Translate to English
            english_translation = self.translate_to_english(transcription)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            
            # Create result (AI analysis will be added later)
            result = TranscriptionResult(
                id=f"trans_{int(time.time())}_{np.random.randint(1000)}",
                timestamp=datetime.now(),
                audio_file=audio_path,
                transcription=transcription,
                english_translation=english_translation,
                confidence_score=confidence,
                processing_time=processing_time,
                language_detected="malaysian_english" if code_switching else "english",
                code_switching_detected=code_switching,
                word_count=len(transcription.split()),
                dialect_markers=dialect_markers,
                ai_analysis={}  # Will be populated by AI analyzer
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

class AIAnalyzer:
    """
    AI-powered analysis of transcribed text
    """
    
    def __init__(self):
        if AI_ANALYSIS_AVAILABLE:
            try:
                # Load lightweight models for analysis
                self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                                 model="cardiffnlp/twitter-roberta-base-sentiment-latest")
                self.emotion_analyzer = pipeline("text-classification", 
                                                model="j-hartmann/emotion-english-distilroberta-base")
                self.summarizer = pipeline("summarization", 
                                         model="facebook/bart-large-cnn")
                self.available = True
            except Exception as e:
                logger.warning(f"AI models not available: {e}")
                self.available = False
        else:
            self.available = False
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of the text"""
        if not self.available:
            return {"sentiment": "neutral", "confidence": 0.5}
        
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                "sentiment": result['label'].lower(),
                "confidence": result['score']
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "confidence": 0.5}
    
    def analyze_emotion(self, text: str) -> Dict:
        """Analyze emotions in the text"""
        if not self.available:
            return {"emotion": "neutral", "confidence": 0.5}
        
        try:
            result = self.emotion_analyzer(text)[0]
            return {
                "emotion": result['label'].lower(),
                "confidence": result['score']
            }
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return {"emotion": "neutral", "confidence": 0.5}
    
    def extract_key_info(self, text: str) -> Dict:
        """Extract key information from the text"""
        info = {
            "topics": [],
            "entities": [],
            "intent": "unknown",
            "urgency": "normal"
        }
        
        text_lower = text.lower()
        
        # Topic detection
        topics = {
            "food": ["makan", "eat", "food", "restaurant", "kopitiam", "mamak", "laksa", "roti", "teh", "coffee"],
            "transport": ["traffic", "jam", "car", "bus", "train", "taxi", "grab", "drive"],
            "weather": ["hot", "cold", "rain", "sunny", "weather", "aircon"],
            "shopping": ["buy", "shop", "market", "pasar", "mall", "price"],
            "work": ["office", "meeting", "work", "boss", "colleague", "project"],
            "social": ["friend", "family", "party", "gathering", "lepak"]
        }
        
        for topic, keywords in topics.items():
            if any(keyword in text_lower for keyword in keywords):
                info["topics"].append(topic)
        
        # Intent detection
        if any(word in text_lower for word in ["want", "need", "can", "help"]):
            info["intent"] = "request"
        elif any(word in text_lower for word in ["?", "or not", "meh"]):
            info["intent"] = "question"
        elif any(word in text_lower for word in ["thanks", "thank you", "good", "nice"]):
            info["intent"] = "appreciation"
        elif any(word in text_lower for word in ["alamak", "wah", "oh no"]):
            info["intent"] = "exclamation"
        
        # Urgency detection
        if any(word in text_lower for word in ["urgent", "quickly", "fast", "now", "emergency"]):
            info["urgency"] = "high"
        elif any(word in text_lower for word in ["when free", "no rush", "whenever"]):
            info["urgency"] = "low"
        
        return info
    
    def generate_summary(self, text: str) -> str:
        """Generate a summary of the conversation"""
        if not self.available or len(text.split()) < 10:
            return text
        
        try:
            summary = self.summarizer(text, max_length=50, min_length=10, do_sample=False)[0]
            return summary['summary_text']
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return text
    
    def analyze_text(self, text: str, english_text: str) -> Dict:
        """Complete AI analysis of the text"""
        analysis = {}
        
        # Use English translation for better AI analysis
        analysis_text = english_text if english_text else text
        
        # Sentiment analysis
        analysis["sentiment"] = self.analyze_sentiment(analysis_text)
        
        # Emotion analysis
        analysis["emotion"] = self.analyze_emotion(analysis_text)
        
        # Key information extraction
        analysis["key_info"] = self.extract_key_info(text)  # Use original text for Malaysian context
        
        # Summary
        analysis["summary"] = self.generate_summary(analysis_text)
        
        # Language analysis
        analysis["language_analysis"] = {
            "code_switching_level": "high" if len(analysis["key_info"]["topics"]) > 0 else "low",
            "formality": "informal" if any(word in text.lower() for word in ["lah", "lor", "wah"]) else "formal",
            "complexity": "simple" if len(text.split()) < 10 else "complex"
        }
        
        return analysis

class MockVectorDatabase:
    """
    Mock vector database for RAG demonstration
    """
    
    def __init__(self):
        self.malaysian_context = [
            "Malaysian English (Manglish) includes particles like 'lah', 'lor', 'meh'",
            "Kopitiam refers to traditional coffee shops in Malaysia",
            "Mamak stalls are Indian-Muslim restaurants popular in Malaysia",
            "Tapau means takeaway in Malaysian English",
            "Code-switching between English and Malay is very common",
            "Shiok means delicious or enjoyable in Malaysian slang",
            "Alamak is an expression of surprise or dismay"
        ]
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Mock search - returns relevant context"""
        # Simple keyword matching for demo
        relevant = []
        query_lower = query.lower()
        
        for context in self.malaysian_context:
            if any(word in context.lower() for word in query_lower.split()):
                relevant.append(context)
        
        return relevant[:k] if relevant else self.malaysian_context[:k]

class MonitoringSystem:
    """
    Simplified monitoring system using only SQLite
    """
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS transcriptions (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                audio_file TEXT,
                transcription TEXT,
                english_translation TEXT,
                confidence_score REAL,
                processing_time REAL,
                language_detected TEXT,
                code_switching_detected BOOLEAN,
                word_count INTEGER,
                dialect_markers TEXT,
                ai_analysis TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp TEXT,
                wer REAL,
                cer REAL,
                latency REAL,
                throughput REAL,
                accuracy REAL,
                confidence_avg REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_transcription(self, result: TranscriptionResult):
        """Log transcription result"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO transcriptions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.id,
            result.timestamp.isoformat(),
            result.audio_file,
            result.transcription,
            result.english_translation,
            result.confidence_score,
            result.processing_time,
            result.language_detected,
            result.code_switching_detected,
            result.word_count,
            json.dumps(result.dialect_markers),
            json.dumps(result.ai_analysis)
        ))
        conn.commit()
        conn.close()
    
    def get_recent_transcriptions(self, limit: int = 20) -> List[Dict]:
        """Get recent transcriptions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT * FROM transcriptions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            result['dialect_markers'] = json.loads(result['dialect_markers']) if result['dialect_markers'] else []
            result['ai_analysis'] = json.loads(result['ai_analysis']) if result['ai_analysis'] else {}
            results.append(result)
        
        conn.close()
        return results
    
    def calculate_current_metrics(self) -> Dict:
        """Calculate current performance metrics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent transcriptions for metrics
        cursor = conn.execute('''
            SELECT confidence_score, processing_time 
            FROM transcriptions 
            WHERE timestamp > datetime('now', '-1 hour')
            ORDER BY timestamp DESC
        ''')
        
        recent_data = cursor.fetchall()
        conn.close()
        
        if not recent_data:
            # Return default metrics if no data
            return {
                "accuracy": 0.92,
                "latency": 1.5,
                "throughput": 8.5,
                "wer": 0.08,
                "confidence_avg": 0.90
            }
        
        confidences = [row[0] for row in recent_data]
        latencies = [row[1] for row in recent_data]
        
        avg_confidence = np.mean(confidences)
        avg_latency = np.mean(latencies)
        throughput = len(recent_data) / max(1, len(recent_data) * 0.1)  # Simplified
        wer = max(0, 1 - avg_confidence) * 0.3  # Estimated
        accuracy = 1 - wer
        
        return {
            "accuracy": accuracy,
            "latency": avg_latency,
            "throughput": throughput,
            "wer": wer,
            "confidence_avg": avg_confidence
        }

class EnhancedTranscriptionSystem:
    """
    Enhanced system with real audio processing and AI analysis
    """
    
    def __init__(self):
        # Initialize transcriber
        if AUDIO_PROCESSING_AVAILABLE:
            try:
                self.transcriber = RealAudioTranscriber()
                self.audio_enabled = True
            except Exception as e:
                logger.warning(f"Real audio transcriber failed, using mock: {e}")
                self.transcriber = self._create_mock_transcriber()
                self.audio_enabled = False
        else:
            self.transcriber = self._create_mock_transcriber()
            self.audio_enabled = False
        
        # Initialize AI analyzer
        self.ai_analyzer = AIAnalyzer()
        
        # Initialize other components
        self.vector_db = MockVectorDatabase()
        self.monitoring = MonitoringSystem()
        
        logger.info(f"Enhanced Transcription System initialized (Audio: {self.audio_enabled}, AI: {self.ai_analyzer.available})")
    
    def _create_mock_transcriber(self):
        """Create mock transcriber when real one isn't available"""
        class MockTranscriber:
            def __init__(self):
                self.sample_transcriptions = [
                    "Eh, you want to makan at the kopitiam or not lah?",
                    "Wah, the traffic jam so teruk today lor!",
                    "Can you belanja me one teh tarik meh?",
                    "Alamak! I forgot to bring my wallet lah!",
                    "This laksa very shiok, you must try!"
                ]
                
                self.manglish_to_english = {
                    "lah": "", "lor": "", "meh": "?", "wah": "Wow",
                    "eh": "Hey", "alamak": "Oh no", "shiok": "delicious",
                    "teruk": "terrible", "makan": "eat", "kopitiam": "coffee shop"
                }
            
            def detect_code_switching(self, text):
                patterns = ["lah", "lor", "meh", "wah", "eh", "alamak", "shiok", "teruk"]
                detected = [p for p in patterns if p in text.lower()]
                return len(detected) > 0, detected
            
            def translate_to_english(self, text):
                english_text = text
                for manglish, english in self.manglish_to_english.items():
                    if english:
                        english_text = english_text.replace(manglish, english)
                    else:
                        english_text = english_text.replace(manglish, "")
                return english_text.strip()
            
            def transcribe_audio(self, audio_path):
                transcription = np.random.choice(self.sample_transcriptions)
                code_switching, dialect_markers = self.detect_code_switching(transcription)
                english_translation = self.translate_to_english(transcription)
                
                return TranscriptionResult(
                    id=f"trans_{int(time.time())}_{np.random.randint(1000)}",
                    timestamp=datetime.now(),
                    audio_file=audio_path,
                    transcription=transcription,
                    english_translation=english_translation,
                    confidence_score=np.random.uniform(0.85, 0.98),
                    processing_time=np.random.uniform(1, 3),
                    language_detected="malaysian_english",
                    code_switching_detected=code_switching,
                    word_count=len(transcription.split()),
                    dialect_markers=dialect_markers,
                    ai_analysis={}
                )
        
        return MockTranscriber()
    
    def process_audio(self, audio_path: str) -> TranscriptionResult:
        """Process audio with full pipeline"""
        try:
            # Step 1: Transcribe audio
            result = self.transcriber.transcribe_audio(audio_path)
            
            # Step 2: AI Analysis
            if self.ai_analyzer.available:
                ai_analysis = self.ai_analyzer.analyze_text(
                    result.transcription, 
                    result.english_translation
                )
                result.ai_analysis = ai_analysis
            else:
                # Basic analysis without AI models
                result.ai_analysis = {
                    "sentiment": {"sentiment": "neutral", "confidence": 0.5},
                    "emotion": {"emotion": "neutral", "confidence": 0.5},
                    "key_info": {
                        "topics": ["conversation"],
                        "intent": "communication",
                        "urgency": "normal"
                    },
                    "summary": result.english_translation,
                    "language_analysis": {
                        "code_switching_level": "medium" if result.code_switching_detected else "low",
                        "formality": "informal",
                        "complexity": "simple"
                    }
                }
            
            # Step 3: Get relevant context
            context = self.vector_db.search(result.transcription)
            if context:
                logger.info(f"Found context: {context[0][:50]}...")
            
            # Step 4: Log result
            self.monitoring.log_transcription(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise

def create_dashboard():
    """
    Streamlit dashboard
    """
    st.set_page_config(
        page_title="Malaysian English Transcription (Lightweight)",
        page_icon="🇲🇾",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #4ECDC4;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .success-alert {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🇲🇾 Enhanced Malaysian English Transcription System")
    st.markdown("**Real audio processing** with AI analysis and English translation")
    
    # Show system capabilities
    col1, col2, col3 = st.columns(3)
    with col1:
        audio_status = "✅ Enabled" if AUDIO_PROCESSING_AVAILABLE else "❌ Mock Mode"
        st.info(f"🎤 Audio Processing: {audio_status}")
    with col2:
        ai_status = "✅ Enabled" if AI_ANALYSIS_AVAILABLE else "❌ Basic Mode"
        st.info(f"🤖 AI Analysis: {ai_status}")
    with col3:
        st.info("🇲🇾 Malaysian English: ✅ Enabled")
    
    # Installation instructions if needed
    if not AUDIO_PROCESSING_AVAILABLE:
        with st.expander("📦 Enable Audio Processing"):
            st.code("pip install SpeechRecognition pydub")
            st.write("Install these packages for real audio transcription")
    
    if not AI_ANALYSIS_AVAILABLE:
        with st.expander("🤖 Enable AI Analysis"):
            st.code("pip install transformers torch")
            st.write("Install these packages for sentiment analysis and AI insights")
    
    # Initialize system
    if 'system' not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.system = LightweightTranscriptionSystem()
    
    system = st.session_state.system
    
    # File upload section
    st.header("📤 Upload Audio for Transcription")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an audio file", 
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload Malaysian English audio (demo will generate sample transcription)"
        )
    
    with col2:
        st.info("💡 **Demo Mode**\n\nThis version generates sample Malaysian English transcriptions for demonstration.")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        audio_path = f"temp_{uploaded_file.name}"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🎯 Transcribe Audio", type="primary"):
            with st.spinner("Processing audio..."):
                try:
                    result = system.process_audio(audio_path)
                    
                    # Display results
                    st.markdown("""
                    <div class="success-alert">
                        <h4>✅ Transcription Complete!</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.subheader("📝 Transcription Result")
                        st.text_area("Transcribed Text", result.transcription, height=80)
                        
                        if result.code_switching_detected:
                            markers_text = ", ".join([f"'{m}'" for m in result.dialect_markers])
                            st.success(f"🔄 Code-switching detected: {markers_text}")
                        
                        # Show context
                        context = system.vector_db.search(result.transcription, k=2)
                        if context:
                            st.info(f"📚 Context: {context[0]}")
                    
                    with col2:
                        st.subheader("📊 Analysis")
                        st.metric("Confidence", f"{result.confidence_score:.1%}")
                        st.metric("Processing Time", f"{result.processing_time:.2f}s")
                        st.metric("Word Count", result.word_count)
                        st.metric("Language", result.language_detected.replace('_', ' ').title())
                
                except Exception as e:
                    st.error(f"❌ Processing failed: {e}")
                
                finally:
                    # Clean up
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
    
    # Monitoring dashboard
    st.header("📈 Performance Monitoring")
    
    # Get current metrics
    current_metrics = system.monitoring.calculate_current_metrics()
    
    # Display current metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Accuracy", f"{current_metrics['accuracy']:.1%}")
    
    with col2:
        st.metric("⚡ Avg Latency", f"{current_metrics['latency']:.2f}s")
    
    with col3:
        st.metric("📊 Throughput", f"{current_metrics['throughput']:.1f}/min")
    
    with col4:
        st.metric("🔢 Word Error Rate", f"{current_metrics['wer']:.1%}")
    
    # Sample performance charts
    st.subheader("📈 Performance Trends")
    
    # Generate sample time series data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='H')
    sample_metrics = pd.DataFrame({
        'timestamp': dates,
        'accuracy': np.random.normal(0.92, 0.03, 30).clip(0.85, 0.98),
        'latency': np.random.normal(1.5, 0.4, 30).clip(0.8, 3.0),
        'confidence': np.random.normal(0.90, 0.05, 30).clip(0.8, 0.98)
    })
    
    # Create performance chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy Over Time', 'Latency Trends', 
                       'Confidence Distribution', 'Error Rates'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accuracy
    fig.add_trace(
        go.Scatter(x=sample_metrics['timestamp'], y=sample_metrics['accuracy'],
                  name='Accuracy', line=dict(color='green')),
        row=1, col=1
    )
    
    # Latency
    fig.add_trace(
        go.Scatter(x=sample_metrics['timestamp'], y=sample_metrics['latency'],
                  name='Latency', line=dict(color='blue')),
        row=1, col=2
    )
    
    # Confidence histogram
    fig.add_trace(
        go.Histogram(x=sample_metrics['confidence'], name='Confidence',
                    marker_color='purple', nbinsx=20),
        row=2, col=1
    )
    
    # Error rates
    error_data = 1 - sample_metrics['accuracy']
    fig.add_trace(
        go.Scatter(x=sample_metrics['timestamp'], y=error_data,
                  name='Error Rate', line=dict(color='red')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent transcriptions
    st.header("📋 Recent Transcriptions")
    
    recent_transcriptions = system.monitoring.get_recent_transcriptions(10)
    
    if recent_transcriptions:
        df = pd.DataFrame(recent_transcriptions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Display key columns
        display_cols = ['timestamp', 'transcription', 'confidence_score', 
                       'processing_time', 'code_switching_detected']
        
        if all(col in df.columns for col in display_cols):
            st.dataframe(
                df[display_cols].sort_values('timestamp', ascending=False),
                use_container_width=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time"),
                    "transcription": st.column_config.TextColumn("Transcription", width="large"),
                    "confidence_score": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                    "processing_time": st.column_config.NumberColumn("Time (s)", format="%.2f"),
                    "code_switching_detected": st.column_config.CheckboxColumn("Code-Switch")
                }
            )
    else:
        st.info("No transcriptions yet. Upload an audio file to get started!")
    
    # Malaysian English examples
    with st.expander("🇲🇾 Malaysian English Examples", expanded=False):
        examples = [
            ("Casual greeting", "Eh, you want to makan at kopitiam or not lah?"),
            ("Traffic complaint", "Wah, the traffic jam so teruk today lor!"),
            ("Food request", "Can you belanja me one teh tarik meh?"),
            ("Surprise expression", "Alamak! I forgot my wallet lah!"),
            ("Food recommendation", "This laksa very shiok, you must try!")
        ]
        
        for context, example in examples:
            st.write(f"**{context}:** {example}")

if __name__ == "__main__":
    create_dashboard()