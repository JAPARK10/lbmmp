import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import gc

# Local imports
from isolator import AudioIsolator
from segmenter import AudioSegmenter
from emotion import EmotionDetector
from transcriber import ContentTranscriber

class AudioPipeline:
    def __init__(self, whisper_model="base", summarizer_model="google/flan-t5-small"):
        """
        Initializes the pipeline configuration. Models are loaded lazily to save GPU memory.
        """
        print("="*50)
        print("INITIALIZING AUDIO PROCESSING PIPELINE (LAZY MODE)")
        print("="*50)
        
        self.whisper_model_name = whisper_model
        self.summarizer_model = summarizer_model
        
        self.script_dir = Path(__file__).parent.resolve()
        self.root_dir = self.script_dir.parent.resolve()
        
        # Initialize small utility classes with absolute paths
        self.isolator = AudioIsolator(output_dir=str(self.root_dir / "data" / "processed" / "isolated"))
        self.segmenter = AudioSegmenter()
        
        # Models will be loaded after isolation
        self.emotion_detector = None
        self.transcriber = None
        
        print("="*50)
        print("PIPELINE CONFIGURATION READY")
        print("="*50)

    def _globally_normalize_0_to_2(self, items, key_name):
        """
        Scans the entirety of the dictionaries and applies a 
        linear stretch mapping the absolute minimum to 0.0 and max to 2.0.
        
        Args:
            items: List of dictionaries.
            key_name: The key inside the dictionary that holds the emotion dict (e.g. 'fused_emotion')
        """
        if not items:
            return items
            
        # 1. Collect all scalar values globally
        all_values = []
        for item in items:
            all_values.extend(item[key_name].values())
            
        if not all_values:
            return items
            
        glob_min = min(all_values)
        glob_max = max(all_values)
        
        denominator = glob_max - glob_min
        if denominator == 0:
            return items
            
        # 2. Apply math: 2 * (val - min) / (max - min)
        for item in items:
            item[key_name] = {
                k: 2.0 * (float(v) - glob_min) / denominator
                for k, v in item[key_name].items()
            }
        
        return items

    def run(self, input_file):
        """
        Runs the full pipeline with memory-aware loading, using sentence-by-sentence windowing.
        """
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"[!] Error: File {input_file} does not exist.")
            return
            
        supported_extensions = ['.mp3', '.wav', '.m4a']
        if input_path.suffix.lower() not in supported_extensions:
            print(f"[!] Error: Unsupported file format '{input_path.suffix}'. Please provide an {', '.join(supported_extensions)} file.")
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

        # PREPARE DYNAMIC LIFECYCLE HANDLERS
        self.transcriber = ContentTranscriber(whisper_model_name=self.whisper_model_name, summarizer_model=self.summarizer_model)

        # MEMORY PHASE 1: GLOBAL TRANSCRIPTION (WHISPER ONLY)
        print("\n" + "="*60)
        print("[*] MEMORY PHASE 1/3: Loading Whisper Transcription...")
        print("="*60)
        self.transcriber.load_whisper()
        
        sentence_segments = self.transcriber.transcribe_full_audio(vocal_track_path)
        
        self.transcriber.unload_whisper() # Kills Whisper immediately

        # MEMORY PHASE 2: ACOUSTIC EXTRACTION (EMOTION DETECTOR ONLY)
        print("\n" + "="*60)
        print("[*] MEMORY PHASE 2/3: Loading Acoustic Emotion Detector...")
        print("="*60)
        
        # Load audio into memory for slicing
        self.segmenter.load_audio(vocal_track_path)
        self.emotion_detector = EmotionDetector()

        print(f"[*] Starting Acoustic Analysis for {len(sentence_segments)} sentences...")
        
        sentences_data = []
        
        for i, sentence in enumerate(tqdm(sentence_segments, desc="Extracting acoustic features")):
            s_start = sentence['start']
            s_end = sentence['end']
            s_text = sentence['text']
            
            # Extract audio slice for this exact sentence
            audio_slice = self.segmenter.get_slice(s_start, s_end)
            
            # Get acoustic emotion
            acoustic_emotion = self.emotion_detector.detect_emotion_from_array(audio_slice)
            
            sentences_data.append({
                "sentence_index": i,
                "timestamp": {
                    "start": s_start,
                    "end": s_end
                },
                "transcript": s_text,
                "acoustic_emotion": acoustic_emotion,
                "text_emotion": {}, # Filled in phase 3
                "fused_emotion": {} # Filled in phase 3
            })
            
        # Critical Cleanup
        print("[*] Acoustic Analysis Complete. Cleaning up memory...")
        self.segmenter.clear()
        del self.emotion_detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[+] RAM fully cleaned and optimized.")

        # MEMORY PHASE 3: LLM INFERENCE (QWEN ONLY)
        print("\n" + "="*60)
        print("[*] MEMORY PHASE 3/3: Loading LLM Summarizer into Clean RAM...")
        print("="*60)
        self.transcriber.load_summarizer()

        print(f"[*] Analyzing {len(sentences_data)} sentences via LLM...")
        all_emotion_options = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
        
        for sentence_obj in tqdm(sentences_data, desc="LLM Processing & Vector Fusion"):
            s_text = sentence_obj['transcript']
            
            # Get text emotions
            llm_text_scores = self.transcriber.get_text_emotion_scores(s_text, all_emotion_options)
            sentence_obj['text_emotion'] = llm_text_scores
            
            # CONFIDENCE GATE: Check how many emotions have a score > 0
            non_zero_count = sum(1 for val in llm_text_scores.values() if float(val) > 0)
            
            if non_zero_count <= 3:
                # CONFIDENT: Convert ratings directly into multiplier scale (1.0x to 1.5x)
                text_multipliers = {
                    em: 1.0 + 0.5 * (float(val) / 10.0)
                    for em, val in llm_text_scores.items()
                }
            else:
                # UNCERTAIN (NOISY): Set all multipliers to 1.0 to avoid boosting artifacts
                text_multipliers = {em: 1.0 for em in all_emotion_options}
            
            # Direct 1:1 Fusion
            fused = {}
            for em in all_emotion_options:
                val = sentence_obj['acoustic_emotion'].get(em, 0.0)
                scalar = text_multipliers.get(em, 1.0)
                fused[em] = val * scalar
                
            # Normalize fused distribution so it sums to 1.0 (L1 Normalization)
            if fused:
                total = sum(fused.values())
                if total > 0:
                    fused = {k: v / total for k, v in fused.items()}
                
            sentence_obj['fused_emotion'] = fused

        # APPLY GLOBAL MIN-MAX NORMALIZATION STRETCH (0.0 to 2.0)
        print("[*] Finalizing vectors: Normalizing acoustic emotions along entire audio track (Range 0 - 2)...")
        sentences_data = self._globally_normalize_0_to_2(sentences_data, 'acoustic_emotion')
        # We do NOT globally normalize 'fused_emotion' because it is already a sentence-level softmax distribution.

        # Output final payload strictly as a list of sentences
        final_output = {
            "sentences": sentences_data
        }

        # SAVE RESULTS
        output_dir = self.root_dir / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{input_path.stem}_analysis.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
            
        print("="*50)
        print(f"[SUCCESS] Pipeline finished processing {input_path.name}")
        print(f"[SUCCESS] Results saved to: {output_file}")
        print("="*50)
        
        # Final memory cleanup
        self.transcriber.unload_summarizer()
        
        return output_file

if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    root_dir = script_dir.parent.resolve()
    default_input = root_dir / "data" / "raw"

    parser = argparse.ArgumentParser(description="Audio Emotion and Content Pipeline")
    parser.add_argument("--input_dir", default=str(default_input), help="Directory containing input audio files")
    parser.add_argument("--whisper", default="base", help="Whisper model size (base, small, medium, large)")
    parser.add_argument("--summarizer", default="Qwen/Qwen2.5-1.5B-Instruct", help="HuggingFace summarization model")
    
    args = parser.parse_args()
    
    pipeline = AudioPipeline(whisper_model=args.whisper, summarizer_model=args.summarizer)
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[!] Error: Directory '{input_dir.absolute()}' does not exist.")
        exit(1)
        
    supported_extensions = ['.mp3', '.wav', '.m4a']
    audio_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]
    
    if not audio_files:
        print(f"[*] No supported audio files found in {input_dir}")
        exit(0)
        
    print(f"[*] Found {len(audio_files)} audio files. Starting batch processing...")
    
    for audio_file in audio_files:
        expected_output = root_dir / "data" / "processed" / f"{audio_file.stem}_analysis.json"
        if expected_output.exists():
            print(f"\n>> Skipping: {audio_file.name} (Already processed)")
            continue
            
        print(f"\n>> Processing: {audio_file.name}")
        pipeline.run(audio_file)
        print("-" * 50)
