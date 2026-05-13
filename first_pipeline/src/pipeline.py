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
        
        # Initialize small utility classes
        self.isolator = AudioIsolator()
        self.segmenter = AudioSegmenter()
        
        # Models will be loaded after isolation
        self.emotion_detector = None
        self.transcriber = None
        
        print("="*50)
        print("PIPELINE CONFIGURATION READY")
        print("="*50)

    def _globally_normalize_0_to_2(self, chunks):
        """
        Scans the entirety of the chunk emotions lists and applies a 
        linear stretch mapping the absolute minimum to 0.0 and max to 2.0.
        """
        if not chunks:
            return chunks
            
        # 1. Collect all scalar values globally
        all_values = []
        for c in chunks:
            all_values.extend(c['emotions'].values())
            
        if not all_values:
            return chunks
            
        glob_min = min(all_values)
        glob_max = max(all_values)
        
        denominator = glob_max - glob_min
        if denominator == 0:
            # Degenerate case (all values identical)
            return chunks
            
        # 2. Apply math: 2 * (val - min) / (max - min)
        for c in chunks:
            c['emotions'] = {
                k: 2.0 * (float(v) - glob_min) / denominator
                for k, v in c['emotions'].items()
            }
        
        return chunks

    def run(self, input_file):
        """
        Runs the full pipeline with memory-aware loading.
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

        # STAGE 2: Segmentation
        print("[*] Stage 2: Segmenting Audio...")
        try:
            segments = self.segmenter.segment_audio(vocal_track_path)
        except Exception as e:
            print(f"[!] Stage 2 Failed: {e}")
            return

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
        self.emotion_detector = EmotionDetector()

        print(f"[*] Starting Analysis for {len(segments)} chunks...")
        
        raw_chunks = []
        all_words = getattr(self.transcriber, 'all_words', [])
        
        for seg in tqdm(segments, desc="Extracting acoustic features"):
            audio_path = seg['path']
            c_start = seg['start_time']
            c_end = seg['end_time']
            
            # Find words that start or end in this 1 second window (temporal overlap)
            matching_words = []
            for w in all_words:
                w_start = w.get('start', 0)
                w_end = w.get('end', 0)
                # Test if overlap occurs
                if max(c_start, w_start) < min(c_end, w_end):
                    matching_words.append(w.get('text', ''))
                    
            chunk_transcript = " ".join(matching_words).strip() if matching_words else "None"
            
            emotions = self.emotion_detector.detect_emotion(audio_path)
            raw_chunks.append({
                "index": seg['index'],
                "timestamp": {
                    "start": c_start,
                    "end": c_end
                },
                "emotions": emotions,
                "transcript": chunk_transcript
            })
            
        # Critical Cleanup
        print("[*] Acoustic Analysis Complete. Nuking Emotion Detector to liberate RAM...")
        del self.emotion_detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[+] RAM fully cleaned and optimized.")

        # MEMORY PHASE 3: LLM INFERENCE (QWEN ONLY)
        print("\n" + "="*60)
        print("[*] MEMORY PHASE 3/3: Loading Qwen-3B into Clean RAM Environment...")
        print("="*60)
        self.transcriber.load_summarizer()

        print(f"[*] Analyzing {len(sentence_segments)} sentences via LLM...")
        sentences = []
        
        import copy
        boosted_chunks = copy.deepcopy(raw_chunks)
        
        for i, sentence in enumerate(tqdm(sentence_segments, desc="LLM Processing")):
            s_start = sentence['start']
            s_end = sentence['end']
            s_text = sentence['text']
            
            # Find overlapping chunks indices
            overlapping_indices = []
            for idx, chunk in enumerate(raw_chunks):
                c_start = chunk['timestamp']['start']
                c_end = chunk['timestamp']['end']
                if max(s_start, c_start) < min(s_end, c_end):
                    overlapping_indices.append(idx)
            
            # Get all available emotions from the dataset
            all_emotion_options = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
            avg_emotions = {}
            
            if overlapping_indices:
                for idx in overlapping_indices:
                    for em, val in raw_chunks[idx]['emotions'].items():
                        avg_emotions[em] = avg_emotions.get(em, 0.0) + val
                
                total = sum(avg_emotions.values())
                if total > 0:
                    avg_emotions = {k: v / total for k, v in avg_emotions.items()}
            
            # Perform Dynamic Bayesian Vector Fusion
            llm_text_scores = self.transcriber.get_text_emotion_scores(s_text, all_emotion_options)
            
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
                print(f"    [!] LLM Confidence Low ({non_zero_count} active emotions). Disabling boost to protect signal.")
                text_multipliers = {em: 1.0 for em in all_emotion_options}
            
            sentences.append({
                "sentence_index": i,
                "timestamp": {"start": s_start, "end": s_end},
                "transcript": s_text,
                "average_emotions": avg_emotions,
                "llm_text_scores": llm_text_scores
            })
            
            # Boost overlapping chunks by applying the full 8D Vector multiplication
            for idx in overlapping_indices:
                chunk = boosted_chunks[idx]
                
                # Apply element-wise multiplication scalars to the entire distribution
                for em in all_emotion_options:
                    if em in chunk['emotions']:
                        scalar = text_multipliers.get(em, 1.0)
                        chunk['emotions'][em] *= scalar
                
                # Re-normalize after scaling distribution
                total_em = sum(chunk['emotions'].values())
                if total_em > 0:
                    chunk['emotions'] = {k: v / total_em for k, v in chunk['emotions'].items()}

        # APPLY GLOBAL MIN-MAX NORMALIZATION STRETCH (0.0 to 2.0)
        print("[*] Finalizing vectors: Normalizing along entire audio track (Range 0 - 2)...")
        raw_chunks = self._globally_normalize_0_to_2(raw_chunks)
        boosted_chunks = self._globally_normalize_0_to_2(boosted_chunks)

        # Output final payload exactly in order
        final_output = {
            "raw_chunks": raw_chunks,
            "sentences": sentences,
            "boosted_chunks": boosted_chunks
        }

        # SAVE RESULTS
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{input_path.stem}_analysis.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
            
        print("="*50)
        print(f"[SUCCESS] Pipeline finished processing {input_path.name}")
        print(f"[SUCCESS] Results saved to: {output_file}")
        print("="*50)
        
        # Final memory cleanup for next iteration of batch
        self.transcriber.unload_summarizer()
        
        return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Emotion and Content Pipeline")
    parser.add_argument("--input_dir", default="data/raw", help="Directory containing input audio files")
    parser.add_argument("--whisper", default="base", help="Whisper model size (base, small, medium, large)")
    parser.add_argument("--summarizer", default="Qwen/Qwen2.5-1.5B-Instruct", help="HuggingFace summarization model")
    
    args = parser.parse_args()
    
    pipeline = AudioPipeline(whisper_model=args.whisper, summarizer_model=args.summarizer)
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[!] Error: Directory '{input_dir}' does not exist.")
        exit(1)
        
    supported_extensions = ['.mp3', '.wav', '.m4a']
    audio_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]
    
    if not audio_files:
        print(f"[*] No supported audio files found in {input_dir}")
        exit(0)
        
    print(f"[*] Found {len(audio_files)} audio files. Starting batch processing...")
    
    for audio_file in audio_files:
        expected_output = Path("data/processed") / f"{audio_file.stem}_analysis.json"
        if expected_output.exists():
            print(f"\n>> Skipping: {audio_file.name} (Already processed)")
            continue
            
        print(f"\n>> Processing: {audio_file.name}")
        pipeline.run(audio_file)
        print("-" * 50)
