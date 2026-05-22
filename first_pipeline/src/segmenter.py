import librosa
import numpy as np
from pathlib import Path

class AudioSegmenter:
    def __init__(self):
        """
        Initializes the segmenter.
        """
        self.audio_path = None
        self.audio_array = None
        self.sample_rate = 16000

    def load_audio(self, input_file):
        """
        Loads the entire audio file into memory as a 16kHz numpy array.
        
        Args:
            input_file (str or Path): Path to the audio file.
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        print(f"[*] Segmenter: Loading full audio into memory ({input_path.name})...")
        try:
            self.audio_array, _ = librosa.load(input_path, sr=self.sample_rate)
            self.audio_path = input_path
            print(f"[+] Audio loaded successfully. Duration: {len(self.audio_array) / self.sample_rate:.2f} seconds.")
        except Exception as e:
            print(f"[-] Error loading audio file with librosa: {e}")
            raise

    def get_slice(self, start_time, end_time):
        """
        Extracts a slice of the loaded audio.
        
        Args:
            start_time (float): Start time in seconds.
            end_time (float): End time in seconds.
            
        Returns:
            np.ndarray: The audio slice.
        """
        if self.audio_array is None:
            raise RuntimeError("Audio not loaded. Call load_audio first.")
            
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        
        # Ensure we don't go out of bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio_array), end_sample)
        
        if start_sample >= end_sample:
             # Return an empty array if invalid slice
             return np.array([], dtype=np.float32)
             
        return self.audio_array[start_sample:end_sample]

    def clear(self):
        """
        Clears the loaded audio from memory.
        """
        self.audio_array = None
        self.audio_path = None

if __name__ == "__main__":
    # Quick test logic
    import sys
    if len(sys.argv) > 1:
        segmenter = AudioSegmenter()
        segmenter.load_audio(sys.argv[1])
        audio_slice = segmenter.get_slice(0.0, 2.0)
        print(f"Slice shape: {audio_slice.shape}")
    else:
        print("Usage: python segmenter.py <path_to_audio_file>")
