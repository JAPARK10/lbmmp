import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch.nn.functional as F
from pathlib import Path

class EmotionDetector:
    def __init__(self, model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        """
        Initializes the Emotion Detector with a pre-trained Wav2Vec2 model.
        
        Args:
            model_name (str): HuggingFace model identifier.
        """
        print(f"[*] Loading emotion detection model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Get labels from config
        self.id2label = self.model.config.id2label
        print(f"[+] Model loaded on {self.device}. Labels: {list(self.id2label.values())}")

    def detect_emotion(self, audio_path):
        """
        Predicts emotion probabilities for a single audio file.
        
        Args:
            audio_path (str or Path): Path to the 2s audio segment.
            
        Returns:
            dict: A dictionary mapping emotion labels to probability scores.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Load audio - Wav2Vec2 expects 16kHz mono
            speech, sr = librosa.load(path, sr=16000)
            
            # Ensure input is not empty
            if len(speech) == 0:
                return {label: 0.0 for label in self.id2label.values()}

            # Preprocess
            inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
                
            # Apply Softmax to get probabilities
            probs = F.softmax(logits, dim=-1).squeeze()
            
            # Convert to CPU and list
            if self.device == "cuda":
                probs = probs.cpu()
            
            scores = probs.numpy().tolist()
            
            # Create a vector of {label: score}
            emotion_vector = {self.id2label[i]: float(scores[i]) for i in range(len(scores))}
            
            return emotion_vector
            
        except Exception as e:
            print(f"[-] Error during emotion detection for {path.name}: {e}")
            # Return neutral vector on error or handle as needed
            return {}

if __name__ == "__main__":
    # Quick test logic
    import sys
    import json
    if len(sys.argv) > 1:
        detector = EmotionDetector()
        result = detector.detect_emotion(sys.argv[1])
        print(json.dumps(result, indent=4))
    else:
        print("Usage: python emotion.py <path_to_audio_segment>")
