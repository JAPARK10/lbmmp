from pydub import AudioSegment
from pathlib import Path
import os

class AudioSegmenter:
    def __init__(self, segment_length_ms=1000, output_dir="data/processed/segments"):
        """
        Initializes the segmenter.
        
        Args:
            segment_length_ms (int): Length of each segment in milliseconds.
            output_dir (str): Root directory to save segments.
        """
        self.segment_length_ms = segment_length_ms
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def segment_audio(self, input_file):
        """
        Splits an audio file into strict segments of specified length.
        Pads the final segment with silence if it's shorter than the target length.
        
        Args:
            input_file (str or Path): Path to the audio file to segment.
            
        Returns:
            list: A list of dictionaries containing metadata for each segment.
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        print(f"[*] Segmenting audio: {input_path.name}")
        
        try:
            audio = AudioSegment.from_file(input_path)
        except Exception as e:
            print(f"[-] Error loading audio file with pydub: {e}")
            raise

        duration_ms = len(audio)
        segments_metadata = []
        
        # Create a sub-directory for this specific file's segments
        filename_stem = input_path.stem
        file_segment_dir = self.output_dir / filename_stem
        file_segment_dir.mkdir(parents=True, exist_ok=True)
        
        # Process in chunks
        for start_ms in range(0, duration_ms, self.segment_length_ms):
            end_ms = start_ms + self.segment_length_ms
            
            chunk = audio[start_ms:end_ms]
            
            # Check if padding is needed for the last segment
            if len(chunk) < self.segment_length_ms:
                padding_ms = self.segment_length_ms - len(chunk)
                silence = AudioSegment.silent(duration=padding_ms)
                chunk = chunk + silence
            
            # Generate filename with zero-padded seconds for easy sorting
            start_sec = start_ms // 1000
            end_sec = end_ms // 1000
            chunk_name = f"segment_{start_sec:05d}_{end_sec:05d}.wav"
            chunk_path = file_segment_dir / chunk_name
            
            # Export segment
            chunk.export(chunk_path, format="wav")
            
            segments_metadata.append({
                "path": str(chunk_path.absolute()),
                "start_time": start_ms / 1000,
                "end_time": end_ms / 1000,
                "index": start_ms // self.segment_length_ms
            })
            
        print(f"[+] Created {len(segments_metadata)} segments in {file_segment_dir}")
        return segments_metadata

if __name__ == "__main__":
    # Quick test logic
    import sys
    if len(sys.argv) > 1:
        segmenter = AudioSegmenter()
        segmenter.segment_audio(sys.argv[1])
    else:
        print("Usage: python segmenter.py <path_to_audio_file>")
