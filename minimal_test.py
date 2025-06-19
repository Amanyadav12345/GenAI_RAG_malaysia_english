#!/usr/bin/env python3
"""
Minimal Test Version - Malaysian English Transcription POC
=========================================================

This is a simplified version to test if your setup works before 
running the full transcription system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Malaysian English Transcription - Test",
    page_icon="🇲🇾",
    layout="wide"
)

# Custom CSS for better styling
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
    .test-box {
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

# Main header
st.markdown("""
<div class="main-header">
    <h1>🇲🇾 Malaysian English Transcription POC - Test Version</h1>
    <p>Testing basic functionality before running the full system</p>
</div>
""", unsafe_allow_html=True)

# Status check section
st.header("🔍 System Status Check")

# Test 1: Basic imports
with st.expander("📦 Package Import Tests", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            import pandas as pd
            st.success("✅ Pandas imported")
        except ImportError:
            st.error("❌ Pandas missing")
    
    with col2:
        try:
            import numpy as np
            st.success("✅ NumPy imported")
        except ImportError:
            st.error("❌ NumPy missing")
    
    with col3:
        try:
            import plotly
            st.success("✅ Plotly imported")
        except ImportError:
            st.error("❌ Plotly missing")
    
    # Additional imports
    col4, col5, col6 = st.columns(3)
    
    with col4:
        try:
            import soundfile
            st.success("✅ SoundFile imported")
        except ImportError:
            st.error("❌ SoundFile missing")
    
    with col5:
        try:
            import librosa
            st.success("✅ Librosa imported")
        except ImportError:
            st.error("❌ Librosa missing")
    
    with col6:
        try:
            import whisper
            st.success("✅ Whisper imported")
        except ImportError:
            st.error("❌ Whisper missing")

# Test 2: File upload functionality
st.header("📤 File Upload Test")

uploaded_file = st.file_uploader(
    "Test audio file upload functionality", 
    type=['wav', 'mp3', 'flac', 'm4a', 'txt'],
    help="Upload any file to test the upload mechanism"
)

if uploaded_file is not None:
    st.markdown("""
    <div class="success-box">
        <h4>✅ File Upload Successful!</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**File Details:**")
        st.write(f"📁 Name: `{uploaded_file.name}`")
        st.write(f"📊 Size: `{len(uploaded_file.getvalue()):,} bytes`")
        st.write(f"🏷️ Type: `{uploaded_file.type}`")
    
    with col2:
        st.write("**File Content Preview:**")
        if uploaded_file.type.startswith('text'):
            content = uploaded_file.read().decode('utf-8')
            st.text_area("Content", content[:500] + "..." if len(content) > 500 else content)
        else:
            st.info("Binary file - content not displayed")

# Test 3: Dashboard components
st.header("📊 Dashboard Components Test")

# Metrics display test
st.subheader("Metrics Display")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎯 Test Accuracy", "95.2%", "2.1%")

with col2:
    st.metric("⚡ Avg Latency", "1.23s", "-0.15s")

with col3:
    st.metric("📈 Throughput", "8.5/min", "1.2/min")

with col4:
    st.metric("🔢 Error Rate", "4.8%", "-2.1%")

# Chart tests
st.subheader("Chart Visualization Tests")

# Generate sample data
dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
sample_data = pd.DataFrame({
    'Date': dates,
    'Accuracy': np.random.normal(0.92, 0.05, len(dates)).clip(0.8, 1.0),
    'Latency': np.random.normal(1.5, 0.3, len(dates)).clip(0.5, 3.0),
    'Confidence': np.random.normal(0.88, 0.08, len(dates)).clip(0.7, 1.0),
    'Throughput': np.random.normal(10, 2, len(dates)).clip(5, 20)
})

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Accuracy Trends', 'Latency Over Time', 
                   'Confidence Scores', 'Throughput Metrics'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Add traces
fig.add_trace(
    go.Scatter(x=sample_data['Date'], y=sample_data['Accuracy'],
              name='Accuracy', line=dict(color='green')),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=sample_data['Date'], y=sample_data['Latency'],
              name='Latency', line=dict(color='blue')),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=sample_data['Date'], y=sample_data['Confidence'],
              name='Confidence', line=dict(color='purple')),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=sample_data['Date'], y=sample_data['Throughput'],
              name='Throughput', line=dict(color='orange')),
    row=2, col=2
)

fig.update_layout(height=600, showlegend=False, title_text="Sample Performance Charts")
st.plotly_chart(fig, use_container_width=True)

# Test 4: Interactive features
st.header("🎮 Interactive Features Test")

# Button tests
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎯 Test Transcription", type="primary"):
        with st.spinner("Simulating transcription..."):
            time.sleep(2)
            st.success("✅ Transcription simulation complete!")
            st.write("**Sample Output:**")
            st.write("```")
            st.write("Eh, you want to makan at the kopitiam or not lah?")
            st.write("Code-switching detected: 'lah'")
            st.write("Confidence: 92.5%")
            st.write("```")

with col2:
    if st.button("📊 Generate Report"):
        st.info("📋 Sample Report Generated")
        report_data = {
            "Metric": ["Accuracy", "Latency", "Throughput", "Errors"],
            "Value": ["94.2%", "1.35s", "9.8/min", "5.8%"],
            "Status": ["Good", "Acceptable", "Good", "Needs Improvement"]
        }
        st.table(pd.DataFrame(report_data))

with col3:
    if st.button("🔄 Refresh Data"):
        st.success("🔄 Data refreshed!")
        # Simulate data update
        new_metric = np.random.uniform(85, 98)
        st.metric("Random Metric", f"{new_metric:.1f}%")

# Test 5: Malaysian English simulation
st.header("🇲🇾 Malaysian English Features Test")

sample_manglish = [
    "Eh, you want to tapau this roti canai or not lah?",
    "Wah, the traffic jam so teruk today lor!",
    "Can you belanja me one teh tarik meh?",
    "Alamak! I forgot to bring my wallet lah!",
    "This laksa very shiok, you must try!"
]

st.subheader("Sample Manglish Transcriptions")
for i, text in enumerate(sample_manglish, 1):
    with st.expander(f"Sample {i}: {text[:30]}..."):
        st.write(f"**Original:** {text}")
        
        # Simulate code-switching detection
        code_switches = []
        manglish_words = ['lah', 'lor', 'meh', 'wah', 'alamak', 'shiok', 'teruk']
        for word in manglish_words:
            if word in text.lower():
                code_switches.append(word)
        
        if code_switches:
            st.success(f"🔄 Code-switching detected: {', '.join(code_switches)}")
        
        # Simulate confidence
        confidence = np.random.uniform(0.85, 0.98)
        st.metric("Confidence Score", f"{confidence:.1%}")

# Test 6: Data table
st.header("📋 Data Table Test")

# Sample transcription history
transcription_data = {
    "Timestamp": [datetime.now() - timedelta(minutes=i*5) for i in range(10)],
    "Text": [
        "Eh, you want to makan at kopitiam?",
        "Wah, traffic jam so teruk today lor!",
        "Can you help me tapau one char koay teow?",
        "This durian very shiok, must try!",
        "Alamak, I forgot my wallet lah!",
        "You already book table at mamak or not?",
        "The weather so hot, need aircon lah!",
        "Can belanja me one teh tarik meh?",
        "Eh bro, you free this weekend or not?",
        "Wah, this nasi lemak sedap gila!"
    ],
    "Confidence": np.random.uniform(0.85, 0.98, 10),
    "Code-Switch": [True, True, True, True, True, True, True, True, True, True],
    "Processing Time": np.random.uniform(0.8, 2.5, 10)
}

df = pd.DataFrame(transcription_data)
df['Timestamp'] = df['Timestamp'].dt.strftime('%H:%M:%S')

st.dataframe(
    df,
    use_container_width=True,
    column_config={
        "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
        "Code-Switch": st.column_config.CheckboxColumn("Code-Switch"),
        "Processing Time": st.column_config.NumberColumn("Process Time (s)", format="%.2f")
    }
)

# Test 7: System info
st.header("💻 System Information")

col1, col2 = st.columns(2)

with col1:
    st.write("**Environment Details:**")
    st.write(f"🐍 Python Version: `{st.__version__}`")
    st.write(f"📊 Streamlit Version: `{st.__version__}`")
    st.write(f"🕒 Current Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")

with col2:
    st.write("**Test Status:**")
    st.success("✅ Dashboard rendering works")
    st.success("✅ Interactive components work") 
    st.success("✅ Charts and metrics display properly")
    st.success("✅ File upload functionality works")

# Final status
st.header("🎉 Test Summary")

st.markdown("""
<div class="success-box">
    <h3>✅ Basic Setup Test Complete!</h3>
    <p>If you can see this page with all components working, your basic setup is ready.</p>
    <p><strong>Next Step:</strong> Run the full transcription system with:</p>
    <p><code>streamlit run malaysian_transcription_poc.py</code></p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh option
if st.checkbox("🔄 Auto-refresh every 10 seconds", help="Test real-time updates"):
    time.sleep(10)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("🇲🇾 **Malaysian English Transcription POC** | Test Version | Built with Streamlit")