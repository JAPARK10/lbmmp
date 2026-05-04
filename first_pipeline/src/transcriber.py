import whisper
import requests
import json
import torch
from pathlib import Path

class ContentTranscriber:
    def __init__(self, whisper_model_name="base", vllm_base_url="http://localhost:8000/v1"):
        """
        Initializes the Transcriber and Summarizer.
        """
        print(f"[*] Loading Whisper model: {whisper_model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(whisper_model_name, device=self.device)
        self.vllm_url = f"{vllm_base_url}/chat/completions"
        self.word_data = [] # Stores global transcript with timestamps
        print(f"[+] Whisper loaded on {self.device}. vLLM API set to {self.vllm_url}")

    def transcribe_full_audio(self, audio_path):
        """
        Transcribes the entire audio file once with word-level timestamps.
        
        Args:
            audio_path (str): Path to the full audio file.
        """
        print(f"[*] Transcribing full audio for global timestamp alignment...")
        try:
            # word_timestamps=True is the key feature here
            result = self.model.transcribe(
                str(audio_path), 
                fp16=(self.device == "cuda"),
                word_timestamps=True
            )
            
            # Extract and flatten word data
            self.word_data = []
            for segment in result.get('segments', []):
                for word in segment.get('words', []):
                    self.word_data.append({
                        "text": word['word'].strip(),
                        "start": word['start'],
                        "end": word['end']
                    })
            
            print(f"[+] Global transcription complete. Extracted {len(self.word_data)} words.")
            return self.word_data
        except Exception as e:
            print(f"[-] Global transcription error: {e}")
            self.word_data = []
            return []

    def get_text_in_range(self, start_time, end_time):
        """
        Retrieves text from the global transcript that falls within the given time range.
        
        Args:
            start_time (float): Start of the window in seconds.
            end_time (float): End of the window in seconds.
        """
        if not self.word_data:
            return ""

        # Find words whose start time falls within our window
        matching_words = [
            w['text'] for w in self.word_data 
            if start_time <= w['start'] < end_time
        ]
        
        return " ".join(matching_words).strip()

    def summarize(self, text):
        """
        Summarizes the text into 1-2 words using the local vLLM API.
        """
        if not text or len(text.split()) < 1:
            return "None"

        # System prompt to enforce the 1-2 word constraint
        messages = [
            {"role": "system", "content": "You are a concise summarizer. Summarize the user input in exactly 1 or 2 words. Do not use punctuation."},
            {"role": "user", "content": f"Text: {text}"}
        ]
        
        payload = {
            "model": "default", # vLLM usually handles whatever model is loaded
            "messages": messages,
            "max_tokens": 10,
            "temperature": 0.0
        }

        try:
            response = requests.post(self.vllm_url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            summary = data['choices'][0]['message']['content'].strip()
            # Basic cleanup to ensure it's not too long
            summary = " ".join(summary.split()[:3]) 
            return summary
        except Exception as e:
            print(f"[-] vLLM Summarization error: {e}")
            # Fallback to simple keyword if API fails
            return text.split()[0] if text.split() else "None"

if __name__ == "__main__":
    # Quick test logic
    import sys
    if len(sys.argv) > 1:
        transcriber = ContentTranscriber()
        txt = transcriber.transcribe(sys.argv[1])
        print(f"Transcript: {txt}")
        summary = transcriber.summarize(txt)
        print(f"Summary: {summary}")
    else:
        print("Usage: python transcriber.py <path_to_audio_segment>")
