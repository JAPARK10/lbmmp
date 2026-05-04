import os
import json
from pathlib import Path
from tqdm import tqdm
import argparse

# Local imports
from isolator import AudioIsolator
from segmenter import AudioSegmenter
from emotion import EmotionDetector
from transcriber import ContentTranscriber

class AudioPipeline:
    def __init__(self, whisper_model="base", vllm_url="http://localhost:8000/v1"):
        """
        Initializes the pipeline configuration. Models are loaded lazily to save GPU memory.
        """
        print("="*50)
        print("INITIALIZING AUDIO PROCESSING PIPELINE (LAZY MODE)")
        print("="*50)
        
        self.whisper_model_name = whisper_model
        self.vllm_url = vllm_url
        
        # Initialize small utility classes
        self.isolator = AudioIsolator()
        self.segmenter = AudioSegmenter()
        
        # Models will be loaded after isolation
        self.emotion_detector = None
        self.transcriber = None
        
        print("="*50)
        print("PIPELINE CONFIGURATION READY")
        print("="*50)

    def run(self, input_file):
        """
        Runs the full pipeline with memory-aware loading.
        """
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"[!] Error: File {input_file} does not exist.")
            return

        # STAGE 1: Vocal Isolation (Subprocess)
        print("[*] Stage 1: Running Vocal Isolation...")
        try:
            vocal_track_path = self.isolator.isolate_vocals(input_path)
        except Exception as e:
            print(f"[!] Stage 1 Failed: {e}")
            return

        # CLEAR CACHE after heavy isolation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # STAGE 2: Segmentation
        print("[*] Stage 2: Segmenting Audio...")
        try:
            segments = self.segmenter.segment_audio(vocal_track_path)
        except Exception as e:
            print(f"[!] Stage 2 Failed: {e}")
            return

        # LOAD MODELS NOW (After isolation is done and cache is clear)
        print("[*] Loading Analysis Models into GPU...")
        self.emotion_detector = EmotionDetector()
        self.transcriber = ContentTranscriber(whisper_model_name=self.whisper_model_name, vllm_base_url=self.vllm_url)

        # NEW STAGE 2.5: Global Transcription
        self.transcriber.transcribe_full_audio(vocal_track_path)

        # STAGES 3 & 4: Analysis
        print(f"[*] Starting Analysis for {len(segments)} segments...")
        pipeline_results = []
        
        for seg in tqdm(segments, desc="Processing segments"):
            audio_path = seg['path']
            start_time = seg['start_time']
            end_time = seg['end_time']
            
            # Emotion Analysis (still on 2s chunks)
            emotions = self.emotion_detector.detect_emotion(audio_path)
            
            # Transcription (LOOKUP instead of RE-TRANSCRIBE)
            transcript = self.transcriber.get_text_in_range(start_time, end_time)
            
            # Summarization (vLLM)
            summary = self.transcriber.summarize(transcript)
            
            # Aggregate
            result_item = {
                "index": seg['index'],
                "timestamp": {
                    "start": seg['start_time'],
                    "end": seg['end_time']
                },
                "emotions": emotions,
                "transcript": transcript,
                "content_summary": summary
            }
            pipeline_results.append(result_item)

        # SAVE RESULTS
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{input_path.stem}_analysis.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pipeline_results, f, indent=4, ensure_ascii=False)
            
        print("="*50)
        print(f"[SUCCESS] Pipeline finished processing {input_path.name}")
        print(f"[SUCCESS] Results saved to: {output_file}")
        print("="*50)
        
        return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Emotion and Content Pipeline")
    parser.add_argument("input", help="Path to input audio file (MP3/WAV)")
    parser.add_argument("--whisper", default="base", help="Whisper model size (base, small, medium, large)")
    parser.add_argument("--vllm", default="http://localhost:8000/v1", help="vLLM API base URL")
    
    args = parser.parse_args()
    
    pipeline = AudioPipeline(whisper_model=args.whisper, vllm_url=args.vllm)
    pipeline.run(args.input)
