from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydub import AudioSegment


class AudioMixer:
    """
    Mixes selected soundtrack regions with the original audiobook narration.

    The music bed is created from selected tracks, then adaptive ducking lowers
    the music when narration is active.
    """

    def __init__(
        self,
        soundtrack_dir: str | Path,
        base_music_gain_db: float = -24.0,
        speech_duck_gain_db: float = -8.0,
        speech_threshold_dbfs: float = -38.0,
        chunk_ms: int = 250,
        fade_ms: int = 1500,
    ):
        self.soundtrack_dir = Path(soundtrack_dir)

        if not self.soundtrack_dir.exists():
            raise FileNotFoundError(f"Soundtrack directory not found: {self.soundtrack_dir}")

        self.base_music_gain_db = base_music_gain_db
        self.speech_duck_gain_db = speech_duck_gain_db
        self.speech_threshold_dbfs = speech_threshold_dbfs
        self.chunk_ms = chunk_ms
        self.fade_ms = fade_ms

    def mix(
        self,
        audiobook_path: str | Path,
        regions_df: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        audiobook_path = Path(audiobook_path)
        output_path = Path(output_path)

        if not audiobook_path.exists():
            raise FileNotFoundError(f"Audiobook file not found: {audiobook_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[*] Loading audiobook: {audiobook_path}")
        narration = AudioSegment.from_file(audiobook_path)
        narration_duration_ms = len(narration)

        print("[*] Building music bed...")
        music_bed = AudioSegment.silent(duration=narration_duration_ms)

        for _, region in regions_df.iterrows():
            start_ms = int(float(region["start_time"]) * 1000)
            end_ms = int(float(region["end_time"]) * 1000)
            duration_ms = max(0, end_ms - start_ms)

            if duration_ms <= 0:
                continue

            soundtrack_path = self.soundtrack_dir / str(region["filename"])

            if not soundtrack_path.exists():
                print(f"[!] Missing soundtrack file, skipping: {soundtrack_path}")
                continue

            music_clip = AudioSegment.from_file(soundtrack_path)
            prepared_clip = self._prepare_music_clip(music_clip, duration_ms)

            music_bed = music_bed.overlay(prepared_clip, position=start_ms)

        print("[*] Applying adaptive ducking...")
        ducked_music = self._apply_ducking(music_bed, narration)

        print("[*] Mixing narration and soundtrack...")
        final_audio = narration.overlay(ducked_music)

        print(f"[*] Exporting enhanced audiobook to: {output_path}")
        final_audio.export(output_path, format=self._infer_format(output_path))

        return output_path

    def _prepare_music_clip(self, clip: AudioSegment, target_duration_ms: int) -> AudioSegment:
        """
        Loops or crops music clip to match target duration.
        Applies base gain and fades.
        """
        if len(clip) == 0:
            return AudioSegment.silent(duration=target_duration_ms)

        repeated = AudioSegment.empty()

        while len(repeated) < target_duration_ms:
            repeated += clip

        repeated = repeated[:target_duration_ms]
        repeated = repeated + self.base_music_gain_db

        fade_duration = min(self.fade_ms, max(0, target_duration_ms // 3))

        if fade_duration > 0:
            repeated = repeated.fade_in(fade_duration).fade_out(fade_duration)

        return repeated

    def _apply_ducking(self, music: AudioSegment, narration: AudioSegment) -> AudioSegment:
        """
        Lowers the soundtrack when narration is active.

        If narration chunk loudness is above speech_threshold_dbfs, apply
        an extra negative gain to the music chunk.
        """
        output = AudioSegment.empty()

        total_ms = len(music)

        for start in range(0, total_ms, self.chunk_ms):
            end = min(start + self.chunk_ms, total_ms)

            music_chunk = music[start:end]
            narration_chunk = narration[start:end]

            narration_is_active = (
                narration_chunk.dBFS != float("-inf")
                and narration_chunk.dBFS > self.speech_threshold_dbfs
            )

            if narration_is_active:
                music_chunk = music_chunk + self.speech_duck_gain_db

            output += music_chunk

        return output

    def _infer_format(self, output_path: Path) -> str:
        suffix = output_path.suffix.lower().replace(".", "")

        if suffix in {"mp3", "wav", "ogg", "flac"}:
            return suffix

        return "mp3"