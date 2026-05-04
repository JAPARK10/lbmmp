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
        Initializes all stages of the pipeline.
        """
        print("="*50)
        print("INITIALIZING AUDIO PROCESSING PIPELINE")
        print("="*50)
        
        self.isolator = AudioIsolator()
        self.segmenter = AudioSegmenter()
        self.emotion_detector = EmotionDetector()
        self.transcriber = ContentTranscriber(whisper_model_name=whisper_model, vllm_base_url=vllm_url)
        
        print("="*50)
        print("PIPELINE READY")
        print("="*50)

    def run(self, input_file):
        """
        Runs the full pipeline on a single audio file.
        
        1. Isolate Vocals (Demucs)
        2. Segment Audio (pydub)
        3. Detect Emotion (Wav2Vec2)
        4. Transcribe (Whisper) & Summarize (vLLM)
        """
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"[!] Error: File {input_file} does not exist.")
            return

        # STAGE 1: Vocal Isolation
        try:
            vocal_track_path = self.isolator.isolate_vocals(input_path)
        except Exception as e:
            print(f"[!] Stage 1 Failed: {e}")
            return

        # STAGE 2: Segmentation
        try:
            segments = self.segmenter.segment_audio(vocal_track_path)
        except Exception as e:
            print(f"[!] Stage 2 Failed: {e}")
            return

        # NEW STAGE 2.5: Global Transcription
        # We transcribe the full vocal track once to get accurate context and timestamps
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
