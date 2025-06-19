#!/usr/bin/env python3
"""
Download Required Models for Malaysian Transcription POC
========================================================

This script pre-downloads all required models to avoid delays during first use.
"""

import os
import sys
from pathlib import Path

def download_whisper_model():
    """Download Whisper model for transcription"""
    print("🎤 Downloading Whisper model...")
    try:
        import whisper
        
        # Download base model (good balance of speed vs accuracy)
        model = whisper.load_model("base")
        print("✅ Whisper 'base' model downloaded successfully!")
        print(f"   Model size: ~150MB")
        
        # Optional: Download small model for faster processing
        try:
            small_model = whisper.load_model("small")
            print("✅ Whisper 'small' model also downloaded!")
            print(f"   Model size: ~250MB")
        except:
            print("⚠️  Small model download failed (optional)")
        
        return True
        
    except ImportError:
        print("❌ Whisper not installed. Install with: pip install openai-whisper")
        return False
    except Exception as e:
        print(f"❌ Error downloading Whisper model: {e}")
        return False

def download_sentence_transformer():
    """Download sentence transformer for RAG functionality"""
    print("\n🔍 Downloading Sentence Transformer model...")
    try:
        from sentence_transformers import SentenceTransformer
        
        # Download the model used in our system
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence Transformer model downloaded successfully!")
        print(f"   Model: all-MiniLM-L6-v2 (~90MB)")
        
        # Test encoding
        test_text = "Testing Malaysian English transcription"
        embedding = model.encode([test_text])
        print(f"✅ Model test successful - embedding shape: {embedding.shape}")
        
        return True
        
    except ImportError:
        print("❌ Sentence Transformers not installed. Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ Error downloading Sentence Transformer: {e}")
        return False

def download_additional_models():
    """Download additional models for better Malaysian English support"""
    print("\n🇲🇾 Downloading Malaysian-specific models...")
    
    # Try to download multilingual models
    try:
        from sentence_transformers import SentenceTransformer
        
        # Multilingual model that might work better for mixed languages
        multilingual_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ Multilingual model downloaded (better for code-switching)")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Multilingual model download failed (optional): {e}")
        return False

def check_disk_space():
    """Check available disk space"""
    try:
        import shutil
        
        # Check space in home directory (where models are cached)
        home_dir = Path.home()
        total, used, free = shutil.disk_usage(home_dir)
        
        free_gb = free / (1024**3)
        print(f"💾 Available disk space: {free_gb:.1f} GB")
        
        if free_gb < 1:
            print("⚠️  Warning: Less than 1GB free space. Models need ~500MB")
            return False
        else:
            print("✅ Sufficient disk space available")
            return True
            
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True

def show_model_locations():
    """Show where models are stored"""
    print("\n📍 Model Storage Locations:")
    
    # Whisper cache location
    whisper_cache = Path.home() / ".cache" / "whisper"
    print(f"🎤 Whisper models: {whisper_cache}")
    
    # HuggingFace cache location  
    hf_cache = Path.home() / ".cache" / "huggingface"
    print(f"🤗 HuggingFace models: {hf_cache}")
    
    # Check if they exist and show sizes
    for cache_dir in [whisper_cache, hf_cache]:
        if cache_dir.exists():
            try:
                size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                size_mb = size / (1024**2)
                print(f"   ✅ {cache_dir.name}: {size_mb:.1f} MB")
            except:
                print(f"   📁 {cache_dir.name}: exists")
        else:
            print(f"   📁 {cache_dir.name}: will be created")

def main():
    """Main download function"""
    print("🇲🇾 Malaysian English Transcription POC - Model Download")
    print("=" * 60)
    
    # Check requirements
    if not check_disk_space():
        choice = input("Continue anyway? (y/n): ").strip().lower()
        if choice not in ['y', 'yes']:
            print("❌ Download cancelled")
            return
    
    show_model_locations()
    
    print("\n🚀 Starting model downloads...")
    print("Note: This may take 5-10 minutes depending on your internet speed")
    
    # Download core models
    whisper_success = download_whisper_model()
    transformer_success = download_sentence_transformer()
    
    # Download optional models
    multilingual_success = download_additional_models()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DOWNLOAD SUMMARY:")
    print(f"🎤 Whisper Model: {'✅ Success' if whisper_success else '❌ Failed'}")
    print(f"🔍 Sentence Transformer: {'✅ Success' if transformer_success else '❌ Failed'}")
    print(f"🌍 Multilingual Model: {'✅ Success' if multilingual_success else '⚠️ Optional'}")
    
    if whisper_success and transformer_success:
        print("\n🎉 Core models downloaded successfully!")
        print("You can now run: streamlit run malaysian_transcription_poc.py")
    else:
        print("\n⚠️  Some core models failed to download.")
        print("The app may still work but will download models on first use.")
    
    print("\n💡 TIP: Models are cached and only downloaded once.")
    print("=" * 60)

if __name__ == "__main__":
    main()