#!/usr/bin/env python3
"""
Step-by-step setup script with error handling
============================================
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print(f"🐍 Python version: {sys.version}")
    if sys.version_info < (3.8):
        print("❌ Error: Python 3.8+ required")
        return False
    print("✅ Python version OK")
    return True

def install_package(package):
    """Install a single package with error handling"""
    print(f"📦 Installing {package}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            package, "--user", "--upgrade"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def install_core_packages():
    """Install packages one by one"""
    packages = [
        "streamlit>=1.25.0",
        "pandas>=1.5.0", 
        "numpy>=1.21.0",
        "plotly>=5.0.0",
        "soundfile>=0.12.1",
        "librosa>=0.9.0",
        "openai-whisper>=20231117",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.0",
        "tqdm>=4.60.0"
    ]
    
    failed_packages = []
    
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  Failed to install: {failed_packages}")
        print("Try installing these manually:")
        for pkg in failed_packages:
            print(f"  pip install {pkg}")
    else:
        print("\n🎉 All packages installed successfully!")
    
    return len(failed_packages) == 0

def create_minimal_version():
    """Create a minimal version of the app for testing"""
    minimal_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

st.title("🇲🇾 Malaysian English Transcription POC - Minimal Version")

st.write("This is a minimal version to test if the basic setup works.")

# Test file upload
uploaded_file = st.file_uploader("Test file upload", type=['wav', 'mp3'])

if uploaded_file:
    st.success("File upload works!")
    st.write(f"File name: {uploaded_file.name}")
    st.write(f"File size: {len(uploaded_file.getvalue())} bytes")

# Test basic functionality
if st.button("Test Basic Functions"):
    st.write("✅ Streamlit working")
    st.write("✅ Pandas working") 
    st.write("✅ NumPy working")
    st.metric("Test Metric", "100%")
    
    # Test chart
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Accuracy', 'Latency', 'Confidence']
    )
    st.line_chart(chart_data)

st.write("If you can see this page, the basic setup is working!")
'''
    
    with open("minimal_test.py", "w") as f:
        f.write(minimal_code)
    
    print("📝 Created minimal_test.py for testing")

def main():
    print("🛠️  Step-by-step Setup for Malaysian Transcription POC")
    print("=" * 60)
    
    # Step 1: Check Python
    if not check_python_version():
        return
    
    # Step 2: Create directories
    print("\n📁 Creating directories...")
    directories = ["data", "data/audio", "logs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    # Step 3: Install packages
    print("\n📦 Installing packages...")
    success = install_core_packages()
    
    # Step 4: Create test version
    print("\n📝 Creating test files...")
    create_minimal_version()
    
    # Step 5: Instructions
    print("\n" + "="*60)
    print("🎯 NEXT STEPS:")
    print("1. Test minimal version first:")
    print("   streamlit run minimal_test.py")
    print()
    print("2. If that works, try the full version:")
    print("   streamlit run malaysian_transcription_poc.py")
    print()
    print("3. If you get import errors, install missing packages:")
    print("   pip install [package_name]")
    print()
    print("4. Access the app at: http://localhost:8501")
    print("="*60)

if __name__ == "__main__":
    main()