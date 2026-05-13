import whisper
import requests
import json
import torch
import re
import gc
from pathlib import Path
from transformers import pipeline

class ContentTranscriber:
    def __init__(self, whisper_model_name="base", summarizer_model="Qwen/Qwen2.5-1.5B-Instruct"):
        """
        Initializes the ContentTranscriber structure. Does not load models into RAM yet.
        """
        self.whisper_model_name = whisper_model_name
        self.summarizer_model_name = summarizer_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = None # Whisper model holder
        self.summarizer = None # Summarizer pipeline holder
        self.sentence_segments = []
        self.all_words = [] # Stores raw word timestamps globally
        
    def load_whisper(self):
        if self.model is None:
            print(f"[*] Loading Whisper model into memory: {self.whisper_model_name}")
            self.model = whisper.load_model(self.whisper_model_name, device=self.device)
            print(f"[+] Whisper load completed on {self.device}.")
            
    def unload_whisper(self):
        if self.model is not None:
            print(f"[*] Unloading Whisper from memory to free space...")
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[+] Memory cleared.")
            
    def load_summarizer(self):
        if self.summarizer is None:
            print(f"[*] Loading Summarizer into memory: {self.summarizer_model_name}")
            hf_device = 0 if self.device == "cuda" else -1
            self.summarizer = pipeline("text-generation", model=self.summarizer_model_name, device=hf_device)
            print(f"[+] Summarizer load completed.")
            
    def unload_summarizer(self):
        if self.summarizer is not None:
            print(f"[*] Unloading Summarizer from memory to free space...")
            del self.summarizer
            self.summarizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[+] Memory cleared.")

    def transcribe_full_audio(self, audio_path):
        """
        Transcribes the entire audio file once and extracts sentence-level segments.
        
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
            
            # Extract words
            words = []
            for segment in result.get('segments', []):
                for word in segment.get('words', []):
                    words.append({
                        "text": word['word'].strip(),
                        "start": word['start'],
                        "end": word['end']
                    })
            
            self.all_words = words # Store words for 1-second window alignment
            
            # Group into sentences using Regex . ! ?
            self.sentence_segments = []
            if not words:
                return []
                
            current_sentence_words = []
            for w in words:
                current_sentence_words.append(w)
                if re.search(r'[.!?]', w['text']):
                    sentence_text = " ".join([cw['text'] for cw in current_sentence_words])
                    self.sentence_segments.append({
                        "text": sentence_text,
                        "start": current_sentence_words[0]['start'],
                        "end": current_sentence_words[-1]['end']
                    })
                    current_sentence_words = []
            
            # Leftover words without punctuation
            if current_sentence_words:
                sentence_text = " ".join([cw['text'] for cw in current_sentence_words])
                self.sentence_segments.append({
                    "text": sentence_text,
                    "start": current_sentence_words[0]['start'],
                    "end": current_sentence_words[-1]['end']
                })
            
            print(f"[+] Global transcription complete. Extracted {len(self.sentence_segments)} Regex sentences.")
            return self.sentence_segments
        except Exception as e:
            print(f"[-] Global transcription error: {e}")
            self.sentence_segments = []
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


    def get_text_emotion_scores(self, text, emotions_list):
        """
        Prompts the LLM to rate all emotions 0-10 and parses numerical outputs robustly.
        """
        # Default safely to 0 for everything if no text.
        default_scores = {em: 0 for em in emotions_list}
        if not text:
            return default_scores
            
        # 1. The core analysis prompt
        prompt_content = (
            "Analyze this specific text and rate the likelihood of it being expressed in ALL 8 categories on a scale of 0 to 10.\n"
            "Scoring Guide:\n"
            "- 0 to 1: Almost NEVER carries this emotion.\n"
            "- 5: COULD carry this emotion depending on context.\n"
            "- 10: CERTAIN or extremely common to carry this emotion.\n\n"
            f"Text: \"{text}\"\n\n"
            "You MUST follow this format exactly:\n"
            "Brief Analysis: [Explain the emotional tone in one sentence]\n"
            "angry: [number]\n"
            "calm: [number]\n"
            "disgust: [number]\n"
            "fearful: [number]\n"
            "happy: [number]\n"
            "neutral: [number]\n"
            "sad: [number]\n"
            "surprised: [number]"
        )
        
        # 2. Wrap in Native ChatML Messaging to unlock maximum Qwen IQ
        messages = [
            {"role": "system", "content": "You are an expert emotional analysis assistant. You perform objective semantic reasoning and always output precise integers."},
            {"role": "user", "content": prompt_content}
        ]
        
        try:
            print(f"    [*] Reasoning about text emotion with Qwen-1.5B...")
            # Using 'messages' automatically applies the model's specialized chat template formatting
            result = self.summarizer(
                messages, 
                max_new_tokens=150, # Higher tokens to allow the Brief Analysis to generate first
                do_sample=False, 
                temperature=None, 
                top_p=None, 
                top_k=None,
                pad_token_id=self.summarizer.tokenizer.eos_token_id
            )
            print(f"    [+] LLM reasoning complete.")
            
            # In message mode, generated content is inside generated_text's last element
            # or direct string depending on HF version. Handle robustly.
            generated = result[0]['generated_text']
            if isinstance(generated, list):
                response_text = generated[-1]['content'].lower()
            else:
                # Fallback for non-chat output structure
                response_text = generated.lower()
            
            found_scores = {}
            lines = response_text.split("\n")
            
            # Log reasoning briefly for user verification in terminal
            for line in lines:
                if "analysis:" in line:
                    print(f"    [LLM Thought]: {line.strip()}")
                    break

            for emotion in emotions_list:
                score = 0
                for line in lines:
                    # We check explicitly for "emotion:" to avoid matching common words in Analysis
                    target = emotion + ":"
                    if target in line:
                        matches = re.findall(r'\d+', line)
                        if matches:
                            try:
                                val = int(matches[0])
                                score = max(0, min(10, val))
                                break
                            except:
                                pass
                found_scores[emotion] = score
            
            # Fill missing categories safely with zeros to prevent key errors later
            for emotion in emotions_list:
                if emotion not in found_scores:
                    found_scores[emotion] = 0
            return found_scores

        except Exception as e:
            print(f"[-] LLM vector fusion inference failed: {e}")
            return default_scores

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
