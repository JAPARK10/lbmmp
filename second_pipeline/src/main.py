from __future__ import annotations

import argparse
from pathlib import Path

from soundtrack_selector import SoundtrackSelector
from audio_mixer import AudioMixer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Second pipeline: emotion-based soundtrack selection and audio mixing."
    )

    parser.add_argument(
        "--analysis",
        required=True,
        help="Path to JSON output from first_pipeline.",
    )

    parser.add_argument(
        "--audiobook",
        required=True,
        help="Path to original audiobook audio file.",
    )

    parser.add_argument(
        "--metadata",
        default="data/metadata/soundtrack_metadata.csv",
        help="Path to soundtrack metadata CSV.",
    )

    parser.add_argument(
        "--soundtrack-dir",
        default="data/soundtracks",
        help="Directory containing soundtrack audio files.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output path for enhanced audiobook. Defaults to output/<original_filename>.",
    )

    parser.add_argument(
        "--assignments-output",
        default="output/selected_soundtracks.csv",
        help="Output CSV with per-segment selected soundtrack.",
    )

    parser.add_argument(
        "--regions-output",
        default="output/soundtrack_regions.csv",
        help="Output CSV with grouped soundtrack regions.",
    )

    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Number of neighbouring segments used for emotion smoothing.",
    )

    parser.add_argument(
        "--min-switch-duration",
        type=float,
        default=8.0,
        help="Minimum number of seconds before allowing a soundtrack switch.",
    )

    parser.add_argument(
        "--switch-threshold",
        type=float,
        default=0.08,
        help="Minimum similarity improvement required to switch tracks.",
    )

    parser.add_argument(
        "--stable-segments",
        type=int,
        default=2,
        help="How many consecutive segments a new candidate should remain stable before switching.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("output") / Path(args.audiobook).name
    assignments_output = Path(args.assignments_output)
    regions_output = Path(args.regions_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments_output.parent.mkdir(parents=True, exist_ok=True)
    regions_output.parent.mkdir(parents=True, exist_ok=True)

    selector = SoundtrackSelector(
        metadata_csv=args.metadata,
        smoothing_window=args.smoothing_window,
        switch_threshold=args.switch_threshold,
        min_switch_duration=args.min_switch_duration,
        stable_segments_required=args.stable_segments,
    )

    assignments_df, regions_df = selector.select(args.analysis)

    assignments_df.to_csv(assignments_output, index=False)
    regions_df.to_csv(regions_output, index=False)

    print(f"[+] Saved per-segment assignments to: {assignments_output}")
    print(f"[+] Saved grouped regions to: {regions_output}")

    mixer = AudioMixer(
        soundtrack_dir=args.soundtrack_dir,
        base_music_gain_db=-8.0,
        speech_duck_gain_db=-4.0,
        speech_threshold_dbfs=-38.0,
        chunk_ms=250,
        fade_ms=1500,
    )

    final_path = mixer.mix(
        audiobook_path=args.audiobook,
        regions_df=regions_df,
        output_path=output_path,
    )

    print("=" * 60)
    print("[SUCCESS] Second pipeline complete.")
    print(f"[SUCCESS] Enhanced audiobook saved to: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()