import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from datasets import Audio
import requests
import librosa
import numpy as np
from typing import Union, Optional
import warnings
warnings.filterwarnings("ignore")

class MalaysianWhisperTranscriber:
    def __init__(self, model_name: str = "mesolitica/malaysian-whisper-medium"):
        """
        Initialize Malaysian Whisper transcriber
        
        Args:
            model_name: Model name from Hugging Face
        """
        self.model_name = model_name
        self.sr = 16000
        self.audio_processor = Audio(sampling_rate=self.sr)
        
        print(f"Loading {model_name}...")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Model loaded successfully on {self.device}")
    
    def transcribe_from_url(self, audio_url: str, language: str = 'ms') -> dict:
        """
        Transcribe audio from URL (your original approach)
        
        Args:
            audio_url: URL to audio file
            language: Language code (ms for Malay)
            
        Returns:
            Dictionary with transcription results
        """
        try:
            print(f"Downloading audio from: {audio_url}")
            
            # Download audio
            response = requests.get(audio_url)
            response.raise_for_status()
            
            # Decode audio using datasets Audio
            audio_data = self.audio_processor.decode_example(
                self.audio_processor.encode_example(response.content)
            )
            y = audio_data['array']
            
            # Process and transcribe
            return self._transcribe_array(y, language)
            
        except Exception as e:
            return {'error': str(e), 'transcription': ''}
    
    def transcribe_from_file(self, file_path: str, language: str = 'ms') -> dict:
        """
        Transcribe audio from local file
        
        Args:
            file_path: Path to local audio file
            language: Language code
            
        Returns:
            Dictionary with transcription results
        """
        try:
            print(f"Loading audio file: {file_path}")
            
            # Load audio using librosa (handles more formats)
            y, sr = librosa.load(file_path, sr=self.sr, mono=True)
            
            return self._transcribe_array(y, language)
            
        except Exception as e:
            return {'error': str(e), 'transcription': ''}
    
    def _transcribe_array(self, audio_array: np.ndarray, language: str = 'ms') -> dict:
        """
        Internal method to transcribe audio array
        
        Args:
            audio_array: Numpy array of audio data
            language: Language code
            
        Returns:
            Dictionary with results
        """
        try:
            # Ensure audio is float32 and normalized
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Normalize if needed
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            duration = len(audio_array) / self.sr
            print(f"Audio duration: {duration:.2f} seconds")
            
            # Process audio for model
            inputs = self.processor([audio_array], return_tensors='pt', sampling_rate=self.sr)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                # Method 1: Your approach with generate()
                generated_ids = self.model.generate(
                inputs['input_features'],
                task="translate",  # <--- NEW
                return_timestamps=True,
                max_length=512,
                num_beams=5,
                do_sample=False,
                temperature=0.0
            )
                
                # Decode transcription
                transcription = self.processor.tokenizer.decode(
                    generated_ids[0], 
                    skip_special_tokens=True
                )
            
            return {
                'transcription': transcription.strip(),
                'duration': duration,
                'language': language,
                'model_used': self.model_name,
                'audio_length': len(audio_array),
                'sample_rate': self.sr
            }
            
        except Exception as e:
            return {'error': str(e), 'transcription': ''}
    
    def transcribe_with_pipeline(self, audio_input: Union[str, np.ndarray], 
                                language: str = 'ms') -> dict:
        """
        Alternative method using Transformers pipeline
        
        Args:
            audio_input: File path, URL, or audio array
            language: Language code
            
        Returns:
            Dictionary with results
        """
        try:
            print("Using pipeline approach...")
            
            # Create pipeline
            pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=0 if self.device.type == 'cuda' else -1
            )
            
            # Handle different input types
            if isinstance(audio_input, str):
                if audio_input.startswith('http'):
                    # URL
                    response = requests.get(audio_input)
                    audio_data = self.audio_processor.decode_example(
                        self.audio_processor.encode_example(response.content)
                    )
                    audio_array = audio_data['array']
                else:
                    # File path
                    audio_array, _ = librosa.load(audio_input, sr=self.sr, mono=True)
            else:
                audio_array = audio_input
            
            # Transcribe using pipeline
            result = pipe(
                audio_array,
                generate_kwargs={
                    "language": language,
                    "task": "translate",
                    "return_timestamps": True
                }
            )
            
            return {
                'transcription': result['text'],
                'chunks': result.get('chunks', []),
                'language': language,
                'method': 'pipeline'
            }
            
        except Exception as e:
            return {'error': str(e), 'transcription': ''}

# Example usage and testing
def main():
    # Initialize transcriber
    transcriber = MalaysianWhisperTranscriber()
    
    # # Test 1: Your original approach (URL)
    # print("=== Test 1: Transcribe from URL ===")
    # test_url = 'https://huggingface.co/datasets/huseinzol05/malaya-speech-stt-test-set/resolve/main/test.mp3'
    # result1 = transcriber.transcribe_from_url(test_url)
    
    # if 'error' not in result1:
    #     print(f"Duration: {result1['duration']:.2f}s")
    #     print(f"Transcription: {result1['transcription']}")
    # else:
    #     print(f"Error: {result1['error']}")
    
    # Test 2: Local file (if you have one)
    print("\n=== Test 2: Transcribe from local file ===")
    # Uncomment and modify path as needed
    result2 = transcriber.transcribe_from_file('C:/Users/harsh/Desktop/GenAI+RAG/GenAI_RAG_malaysia_english/ElevenLabs_Text_to_Speech_audio.wav',language="ms")
    print(result2['transcription'])
    
    # Test 3: Pipeline approach
    # print("\n=== Test 3: Pipeline approach ===")
    # result3 = transcriber.transcribe_with_pipeline(test_url)
    
    # if 'error' not in result3:
    #     print(f"Transcription: {result3['transcription']}")
    #     if 'chunks' in result3:
    #         print("Timestamps:")
    #         for chunk in result3['chunks']:
    #             print(f"  {chunk['timestamp']}: {chunk['text']}")
    # else:
    #     print(f"Error: {result3['error']}")

if __name__ == "__main__":
    main()

# Quick test function (similar to your original code)
def quick_test():
    """Quick test function matching your original approach"""
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    from datasets import Audio
    import requests
    
    sr = 16000
    audio = Audio(sampling_rate=sr)
    
    processor = AutoProcessor.from_pretrained("mesolitica/malaysian-whisper-medium")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("mesolitica/malaysian-whisper-medium")
    
    # Download test audio
    r = requests.get('https://huggingface.co/datasets/huseinzol05/malaya-speech-stt-test-set/resolve/main/test.mp3')
    y = audio.decode_example(audio.encode_example(r.content))['array']
    
    # Process and generate
    inputs = processor([y], return_tensors='pt')
    result = model.generate(inputs['input_features'], language='ms', return_timestamps=True)
    
    # Decode result
    transcription = processor.tokenizer.decode(result[0])
    print(f"Quick test result: {transcription}")
    
    return transcription

# Uncomment to run quick test
quick_test()