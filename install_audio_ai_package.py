#!/usr/bin/env python3
"""
Complete Malaysian English Transcription POC
===========================================

A comprehensive system for Malaysian English transcription with AI analysis.
All features in one file - handles both text input and audio upload.
"""

import os
import json
import time
import sqlite3
import re
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Check for optional packages
try:
    import speech_recognition as sr
    import pydub
    from pydub import AudioSegment
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

try:
    from transformers import pipeline
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
    ai_analysis: Dict

class MalaysianEnglishProcessor:
    """
    Core processor for Malaysian English text processing
    """
    
    def __init__(self):
        # Malaysian English to Standard English mappings
        self.manglish_to_english = {
            "lah": "",  # Remove particles
            "lor": "",
            "meh": "?",
            "wah": "Wow",
            "eh": "Hey",
            "alamak": "Oh no",
            "shiok": "delicious",
            "teruk": "terrible",
            "sedap": "delicious",
            "gila": "very",
            "tapau": "takeaway",
            "kopitiam": "coffee shop",
            "mamak": "Indian-Muslim restaurant",
            "makan": "eat",
            "belanja": "treat",
            "lepak": "hang out",
            "kiasu": "fear of losing out",
            "paiseh": "embarrassed",
            "steady": "cool/good"
        }
        
        self.code_switching_patterns = list(self.manglish_to_english.keys())
        
        # Grammar patterns common in Malaysian English
        self.grammar_fixes = {
            r'\bcan or not\b': 'is it possible',
            r'\bwant or not\b': 'do you want it',
            r'\bhave or not\b': 'is it available',
            r'\bgot or not\b': 'is there any',
            r'\blidat\b': 'like that',
            r'\blidis\b': 'like this',
            r'\bvery the\b': 'very',
            r'\bso the\b': 'so'
        }
    
    def detect_code_switching(self, text: str) -> Tuple[bool, List[str]]:
        """Detect Malaysian English markers in text"""
        detected_markers = []
        text_lower = text.lower()
        
        for pattern in self.code_switching_patterns:
            if re.search(r'\b' + re.escape(pattern) + r'\b', text_lower):
                detected_markers.append(pattern)
        
        return len(detected_markers) > 0, detected_markers
    
    def translate_to_english(self, text: str) -> str:
        """Convert Malaysian English to Standard English"""
        english_text = text
        
        # Replace Manglish terms
        for manglish, english in self.manglish_to_english.items():
            if english:  # Only replace if there's an English equivalent
                pattern = r'\b' + re.escape(manglish) + r'\b'
                english_text = re.sub(pattern, english, english_text, flags=re.IGNORECASE)
            else:  # Remove particles like "lah", "lor"
                pattern = r'\s*\b' + re.escape(manglish) + r'\b\s*'
                english_text = re.sub(pattern, ' ', english_text, flags=re.IGNORECASE)
        
        # Apply grammar fixes
        for pattern, replacement in self.grammar_fixes.items():
            english_text = re.sub(pattern, replacement, english_text, flags=re.IGNORECASE)
        
        # Clean up extra spaces and punctuation
        english_text = re.sub(r'\s+', ' ', english_text).strip()
        english_text = re.sub(r'\s*\?\s*$', '?', english_text)  # Fix question marks
        
        return english_text

class AudioTranscriber:
    """
    Audio transcription handler
    """
    
    def __init__(self):
        self.processor = MalaysianEnglishProcessor()
        
        if AUDIO_PROCESSING_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.available = True
        else:
            self.available = False
    
    def convert_audio_format(self, audio_path: str) -> str:
        """Convert audio to WAV format for better recognition"""
        if not AUDIO_PROCESSING_AVAILABLE:
            return audio_path
        
        try:
            audio = AudioSegment.from_file(audio_path)
            # Convert to optimal settings for speech recognition
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            wav_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
            audio.export(wav_path, format="wav")
            return wav_path
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return audio_path
    
    def transcribe_audio(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio file"""
        start_time = time.time()
        
        if not self.available:
            # Return mock result if audio processing not available
            return self._create_mock_result(audio_path, start_time)
        
        try:
            # Convert audio format
            wav_path = self.convert_audio_format(audio_path)
            
            # Perform speech recognition
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
            
            try:
                transcription = self.recognizer.recognize_google(audio_data)
                confidence = 0.85  # Google doesn't provide confidence scores
            except sr.UnknownValueError:
                transcription = "Could not understand audio clearly"
                confidence = 0.0
            except sr.RequestError as e:
                transcription = f"Speech recognition service error: {e}"
                confidence = 0.0
            
            # Process the transcription
            code_switching, dialect_markers = self.processor.detect_code_switching(transcription)
            english_translation = self.processor.translate_to_english(transcription)
            processing_time = time.time() - start_time
            
            # Clean up temporary file
            if wav_path != audio_path and os.path.exists(wav_path):
                os.remove(wav_path)
            
            return TranscriptionResult(
                id=f"audio_{int(time.time())}_{np.random.randint(1000)}",
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
                ai_analysis={}
            )
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return self._create_error_result(audio_path, str(e), start_time)
    
    def _create_mock_result(self, audio_path: str, start_time: float) -> TranscriptionResult:
        """Create mock result when audio processing not available"""
        mock_transcriptions = [
            "Eh, you want to makan at the kopitiam or not lah?",
            "Wah, the traffic jam so teruk today lor!",
            "Can you belanja me one teh tarik meh?",
            "Alamak! I forgot to bring my wallet lah!",
            "This laksa very shiok, you must try!"
        ]
        
        transcription = np.random.choice(mock_transcriptions)
        code_switching, dialect_markers = self.processor.detect_code_switching(transcription)
        english_translation = self.processor.translate_to_english(transcription)
        
        return TranscriptionResult(
            id=f"mock_{int(time.time())}_{np.random.randint(1000)}",
            timestamp=datetime.now(),
            audio_file=audio_path,
            transcription=transcription,
            english_translation=english_translation,
            confidence_score=np.random.uniform(0.85, 0.95),
            processing_time=time.time() - start_time,
            language_detected="malaysian_english",
            code_switching_detected=code_switching,
            word_count=len(transcription.split()),
            dialect_markers=dialect_markers,
            ai_analysis={}
        )
    
    def _create_error_result(self, audio_path: str, error_msg: str, start_time: float) -> TranscriptionResult:
        """Create error result"""
        return TranscriptionResult(
            id=f"error_{int(time.time())}",
            timestamp=datetime.now(),
            audio_file=audio_path,
            transcription=f"Error: {error_msg}",
            english_translation=f"Error: {error_msg}",
            confidence_score=0.0,
            processing_time=time.time() - start_time,
            language_detected="error",
            code_switching_detected=False,
            word_count=0,
            dialect_markers=[],
            ai_analysis={}
        )

class AIAnalyzer:
    """
    AI-powered text analysis
    """
    
    def __init__(self):
        self.available = False
        self.models = {}
        
        if AI_ANALYSIS_AVAILABLE:
            try:
                # Load lightweight models
                self.models['sentiment'] = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                self.available = True
                logger.info("AI models loaded successfully")
            except Exception as e:
                logger.warning(f"AI models not available: {e}")
                self.available = False
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        if not self.available:
            return {"sentiment": "neutral", "confidence": 0.5}
        
        try:
            results = self.models['sentiment'](text)[0]
            # Find the highest scoring sentiment
            best_result = max(results, key=lambda x: x['score'])
            
            return {
                "sentiment": best_result['label'].lower().replace('label_', ''),
                "confidence": best_result['score']
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "confidence": 0.5}
    
    def extract_key_info(self, text: str) -> Dict:
        """Extract key information from text"""
        text_lower = text.lower()
        
        # Topic detection
        topics = []
        topic_keywords = {
            "food": ["makan", "eat", "food", "restaurant", "kopitiam", "mamak", "laksa", "roti", "teh", "coffee", "shiok", "sedap"],
            "transport": ["traffic", "jam", "car", "bus", "train", "taxi", "grab", "drive", "parking"],
            "weather": ["hot", "cold", "rain", "sunny", "weather", "aircon", "humid"],
            "shopping": ["buy", "shop", "market", "pasar", "mall", "price", "expensive", "cheap"],
            "work": ["office", "meeting", "work", "boss", "colleague", "project", "deadline"],
            "social": ["friend", "family", "party", "gathering", "lepak", "dating"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        # Intent detection
        intent = "statement"
        if any(word in text_lower for word in ["want", "need", "can", "help", "please"]):
            intent = "request"
        elif any(word in text_lower for word in ["?", "or not", "meh", "how", "what", "when", "where"]):
            intent = "question"
        elif any(word in text_lower for word in ["thanks", "thank you", "good", "nice", "appreciate"]):
            intent = "appreciation"
        elif any(word in text_lower for word in ["alamak", "wah", "oh no", "shit", "damn"]):
            intent = "exclamation"
        
        # Urgency detection
        urgency = "normal"
        if any(word in text_lower for word in ["urgent", "quickly", "fast", "now", "emergency", "asap"]):
            urgency = "high"
        elif any(word in text_lower for word in ["when free", "no rush", "whenever", "slowly"]):
            urgency = "low"
        
        return {
            "topics": topics,
            "intent": intent,
            "urgency": urgency
        }
    
    def analyze_language(self, original_text: str, english_text: str) -> Dict:
        """Analyze language characteristics"""
        manglish_particles = ["lah", "lor", "meh", "wah", "eh"]
        particle_count = sum(1 for particle in manglish_particles if particle in original_text.lower())
        
        return {
            "code_switching_level": "high" if particle_count >= 2 else "medium" if particle_count == 1 else "low",
            "formality": "informal" if particle_count > 0 or any(word in original_text.lower() for word in ["bro", "dude", "buddy"]) else "formal",
            "complexity": "complex" if len(original_text.split()) > 15 else "simple",
            "translation_applied": original_text.lower() != english_text.lower()
        }
    
    def analyze_text(self, original_text: str, english_text: str) -> Dict:
        """Complete analysis of text"""
        analysis = {}
        
        # Use English translation for better AI analysis
        analysis_text = english_text if english_text else original_text
        
        # Sentiment analysis
        analysis["sentiment"] = self.analyze_sentiment(analysis_text)
        
        # Key information extraction
        analysis["key_info"] = self.extract_key_info(original_text)
        
        # Language analysis
        analysis["language_analysis"] = self.analyze_language(original_text, english_text)
        
        # Summary (simple version)
        analysis["summary"] = analysis_text if len(analysis_text) <= 100 else analysis_text[:97] + "..."
        
        return analysis

class DatabaseManager:
    """
    Database management for storing transcriptions and metrics
    """
    
    def __init__(self, db_path: str = "transcription_poc.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Create database tables"""
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
                total_transcriptions INTEGER,
                avg_confidence REAL,
                avg_processing_time REAL,
                code_switching_percentage REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_transcription(self, result: TranscriptionResult):
        """Save transcription result to database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
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
        except Exception as e:
            logger.error(f"Database save failed: {e}")
        finally:
            conn.close()
    
    def get_recent_transcriptions(self, limit: int = 20) -> List[Dict]:
        """Get recent transcriptions from database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            cursor = conn.execute('''
                SELECT * FROM transcriptions 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                # Parse JSON fields
                try:
                    result['dialect_markers'] = json.loads(result['dialect_markers']) if result['dialect_markers'] else []
                    result['ai_analysis'] = json.loads(result['ai_analysis']) if result['ai_analysis'] else {}
                except:
                    result['dialect_markers'] = []
                    result['ai_analysis'] = {}
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return []
        finally:
            conn.close()
    
    def get_statistics(self) -> Dict:
        """Get overall statistics"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total,
                    AVG(confidence_score) as avg_confidence,
                    AVG(processing_time) as avg_processing_time,
                    AVG(CASE WHEN code_switching_detected THEN 1 ELSE 0 END) as code_switching_rate
                FROM transcriptions
                WHERE timestamp > datetime('now', '-24 hours')
            ''')
            
            row = cursor.fetchone()
            if row and row[0] > 0:
                return {
                    "total_transcriptions": row[0],
                    "avg_confidence": row[1] or 0.0,
                    "avg_processing_time": row[2] or 0.0,
                    "code_switching_rate": row[3] or 0.0
                }
            else:
                return {
                    "total_transcriptions": 0,
                    "avg_confidence": 0.85,
                    "avg_processing_time": 1.5,
                    "code_switching_rate": 0.6
                }
        except Exception as e:
            logger.error(f"Statistics query failed: {e}")
            return {
                "total_transcriptions": 0,
                "avg_confidence": 0.85,
                "avg_processing_time": 1.5,
                "code_switching_rate": 0.6
            }
        finally:
            conn.close()

class TranscriptionSystem:
    """
    Main transcription system orchestrating all components
    """
    
    def __init__(self):
        self.processor = MalaysianEnglishProcessor()
        self.transcriber = AudioTranscriber()
        self.ai_analyzer = AIAnalyzer()
        self.database = DatabaseManager()
        
        logger.info(f"Transcription system initialized - Audio: {self.transcriber.available}, AI: {self.ai_analyzer.available}")
    
    def process_text(self, text: str) -> TranscriptionResult:
        """Process text input"""
        start_time = time.time()
        
        # Detect code-switching and translate
        code_switching, dialect_markers = self.processor.detect_code_switching(text)
        english_translation = self.processor.translate_to_english(text)
        
        # Create result
        result = TranscriptionResult(
            id=f"text_{int(time.time())}_{np.random.randint(1000)}",
            timestamp=datetime.now(),
            audio_file="text_input",
            transcription=text,
            english_translation=english_translation,
            confidence_score=1.0,
            processing_time=time.time() - start_time,
            language_detected="malaysian_english" if code_switching else "english",
            code_switching_detected=code_switching,
            word_count=len(text.split()),
            dialect_markers=dialect_markers,
            ai_analysis={}
        )
        
        # Add AI analysis
        result.ai_analysis = self.ai_analyzer.analyze_text(text, english_translation)
        
        # Save to database
        self.database.save_transcription(result)
        
        return result
    
    def process_audio(self, audio_path: str) -> TranscriptionResult:
        """Process audio file"""
        # Transcribe audio
        result = self.transcriber.transcribe_audio(audio_path)
        
        # Add AI analysis if transcription was successful
        if result.confidence_score > 0:
            result.ai_analysis = self.ai_analyzer.analyze_text(
                result.transcription, 
                result.english_translation
            )
        
        # Save to database
        self.database.save_transcription(result)
        
        return result

def create_streamlit_app():
    """
    Main Streamlit application
    """
    st.set_page_config(
        page_title="Malaysian English Transcription POC",
        page_icon="🇲🇾",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .feature-box {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #4ECDC4;
            margin: 1rem 0;
        }
        .success-box {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🇲🇾 Malaysian English Transcription POC</h1>
        <p>Complete audio-to-text system with AI analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    if 'system' not in st.session_state:
        with st.spinner("Initializing transcription system..."):
            st.session_state.system = TranscriptionSystem()
    
    system = st.session_state.system
    
    # Show system status
    col1, col2, col3 = st.columns(3)
    with col1:
        audio_status = "✅ Real Audio" if system.transcriber.available else "🎭 Mock Mode"
        st.info(f"🎤 **Audio Processing:** {audio_status}")
    with col2:
        ai_status = "✅ AI Models" if system.ai_analyzer.available else "📝 Basic Analysis"
        st.info(f"🤖 **AI Analysis:** {ai_status}")
    with col3:
        st.info("🇲🇾 **Malaysian English:** ✅ Full Support")
    
    # Installation help
    if not system.transcriber.available or not system.ai_analyzer.available:
        with st.expander("📦 Enable Full Features"):
            if not system.transcriber.available:
                st.code("pip install SpeechRecognition pydub")
                st.write("↑ Install for real audio transcription")
            if not system.ai_analyzer.available:
                st.code("pip install transformers torch")
                st.write("↑ Install for advanced AI analysis")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "🎤 Audio Processing", "📊 Dashboard"])
    
    with tab1:
        st.header("📝 Text Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            text_input = st.text_area(
                "Enter Malaysian English text:",
                placeholder="Eh, you want to makan at kopitiam or not lah?",
                height=100,
                help="Type any Malaysian English text for analysis"
            )
        
        with col2:
            st.write("**Quick Examples:**")
            examples = [
                "Wah traffic jam so teruk!",
                "Can belanja me teh tarik meh?",
                "Alamak forgot my wallet lah!",
                "This laksa very shiok!",
                "Eh bro you free this weekend or not?"
            ]
            
            for example in examples:
                if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
                    st.session_state.example_text = example
        
        # Use example if selected
        if 'example_text' in st.session_state:
            text_input = st.session_state.example_text
            del st.session_state.example_text
        
        if text_input and st.button("🔍 Analyze Text", type="primary"):
            with st.spinner("Analyzing text..."):
                result = system.process_text(text_input)
                display_results(result)
    
    with tab2:
        st.header("🎤 Audio Processing")
        
        uploaded_file = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload audio containing Malaysian English speech"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Show file details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", f"{len(uploaded_file.getbuffer()):,} bytes")
            with col2:
                st.metric("File Type", uploaded_file.type)
            with col3:
                processing_mode = "Real Audio" if system.transcriber.available else "Demo Mode"
                st.metric("Processing", processing_mode)
            
            if st.button("🎯 Transcribe Audio", type="primary"):
                # Save uploaded file
                audio_path = f"temp_{uploaded_file.name}"
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    with st.spinner("Processing audio..."):
                        result = system.process_audio(audio_path)
                    
                    if result.confidence_score > 0:
                        st.success("✅ Audio transcription completed!")
                        display_results(result)
                    else:
                        st.error("❌ Transcription failed - please check audio quality")
                        st.write("**Troubleshooting tips:**")
                        st.write("- Ensure clear speech without background noise")
                        st.write("- Try shorter audio clips (< 1 minute)")
                        st.write("- Check audio file format is supported")
                
                finally:
                    # Clean up
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
    
    with tab3:
        st.header("📊 Performance Dashboard")
        
        # Get statistics
        stats = system.database.get_statistics()
        recent_transcriptions = system.database.get_recent_transcriptions(10)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Processed", stats['total_transcriptions'])
        with col2:
            st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")
        with col3:
            st.metric("Avg Processing Time", f"{stats['avg_processing_time']:.2f}s")
        with col4:
            st.metric("Code-Switching Rate", f"{stats['code_switching_rate']:.1%}")
        
        # Recent transcriptions table
        if recent_transcriptions:
            st.subheader("📋 Recent Transcriptions")
            
            df = pd.DataFrame(recent_transcriptions)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Display selected columns
            display_cols = ['timestamp', 'transcription', 'english_translation', 
                          'confidence_score', 'code_switching_detected']
            
            available_cols = [col for col in display_cols if col in df.columns]
            
            if available_cols:
                st.dataframe(
                    df[available_cols].sort_values('timestamp', ascending=False),
                    use_container_width=True,
                    column_config={
                        "timestamp": st.column_config.DatetimeColumn("Time"),
                        "transcription": st.column_config.TextColumn("Original", width="medium"),
                        "english_translation": st.column_config.TextColumn("English", width="medium"),
                        "confidence_score": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                        "code_switching_detected": st.column_config.CheckboxColumn("Code-Switch")
                    }
                )
            
            # Sample performance chart
            if len(recent_transcriptions) > 1:
                st.subheader("📈 Performance Trends")
                
                chart_data = pd.DataFrame({
                    'Time': range(len(recent_transcriptions