import subprocess
import os
import sys  # <-- ADD THIS LINE
from pathlib import Path

class AudioIsolator:
    def __init__(self, output_dir="data/processed/isolated"):
        """
        Initializes the isolator with an output directory.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def isolate_vocals(self, input_file):
        """
        Uses Demucs to isolate vocals from the input file.
        Returns the path to the isolated vocal file.
        
        Args:
            input_file (str or Path): Path to the input audio file.
            
        Returns:
            Path: Path to the isolated vocal WAV file.
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        print(f"[*] Starting vocal isolation for: {input_path.name}")
        
        # Demucs command
        # -n htdemucs: Use the high-quality hybrid transformer model
        # --two-stems=vocals: Only output vocals and 'no_vocals'
        command = [
            sys.executable, "-m", "demucs",  # <-- FIXED TO USE MODULE PATH
            "-n", "htdemucs",
            "--two-stems", "vocals",
            str(input_path.absolute()),
            "-o", str(self.output_dir.absolute())
        ]
        
        try:
            # Run demucs as a subprocess to use GPU if available
            subprocess.run(command, check=True)
            
            # Demucs default output structure:
            # {output_dir}/htdemucs/{filename}/vocals.wav
            filename_stem = input_path.stem
            vocal_path = self.output_dir / "htdemucs" / filename_stem / "vocals.wav"
            
            if vocal_path.exists():
                print(f"[+] Isolated vocals saved to: {vocal_path}")
                return vocal_path
            else:
                # Sometimes demucs replaces spaces with underscores or similar
                # but htdemucs usually keeps the stem name.
                raise FileNotFoundError(f"Could not find isolated vocal file at {vocal_path}")
                
        except subprocess.CalledProcessError as e:
            print(f"[-] Error during Demucs execution: {e}")
            raise
        except Exception as e:
            print(f"[-] An unexpected error occurred: {e}")
            raise

if __name__ == "__main__":
    # Quick test logic
    import sys
    if len(sys.argv) > 1:
        isolator = AudioIsolator()
        isolator.isolate_vocals(sys.argv[1])
    else:
        print("Usage: python isolator.py <path_to_audio_file>")
