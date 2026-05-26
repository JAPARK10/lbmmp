from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class SoundtrackSelector:
    """
    Selects soundtrack clips based on segment-level emotion vectors.

    Input:
        analysis_json:
            JSON output from first_pipeline
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        smoothing_window: int = 5,
        switch_threshold: float = 0.08,
        min_switch_duration: float = 8.0,
        stable_segments_required: int = 2,
    ):
        self.metadata_csv = Path(metadata_csv)
        self.smoothing_window = smoothing_window
        self.switch_threshold = switch_threshold
        self.min_switch_duration = min_switch_duration
        self.stable_segments_required = stable_segments_required

        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Soundtrack metadata file not found: {self.metadata_csv}")

        self.tracks_df = pd.read_csv(self.metadata_csv)

        required_columns = {"track_id", "filename"}
        missing = required_columns - set(self.tracks_df.columns)
        if missing:
            raise ValueError(f"Missing required columns in soundtrack metadata: {missing}")

    def load_analysis(self, analysis_json: str | Path) -> pd.DataFrame:
        analysis_path = Path(analysis_json)

        if not analysis_path.exists():
            raise FileNotFoundError(f"Analysis JSON not found: {analysis_path}")

        with open(analysis_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Supports both possible formats:
        # 1) [ {...}, {...} ]
        # 2) { "raw_chunks": [ {...}, {...} ] }
        if isinstance(data, dict) and "raw_chunks" in data:
            data = data["raw_chunks"]

        rows = []

        for item in data:
            timestamp = item.get("timestamp", {})
            emotions = item.get("emotions", {}) or {}

            row = {
                "index": item.get("index"),
                "start_time": float(timestamp.get("start", 0.0)),
                "end_time": float(timestamp.get("end", 0.0)),
                "transcript": item.get("transcript", ""),
                "content_summary": item.get("content_summary", ""),
            }

            for emotion_name, score in emotions.items():
                row[self._normalize_emotion_name(emotion_name)] = float(score)

            rows.append(row)

        if not rows:
            raise ValueError("Analysis JSON is empty or has no valid segments.")

        segments_df = pd.DataFrame(rows)
        segments_df = segments_df.sort_values("start_time").reset_index(drop=True)

        return segments_df

    def select(self, analysis_json: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
            assignments_df:
                One row per segment, with selected soundtrack.

            regions_df:
                Consecutive segments grouped into longer regions with the same soundtrack.
        """
        segments_df = self.load_analysis(analysis_json)

        emotion_columns = self._find_shared_emotion_columns(segments_df)

        if not emotion_columns:
            raise ValueError(
                "No shared emotion columns found between analysis JSON and soundtrack metadata. "
                f"Segment columns: {list(segments_df.columns)}. "
                f"Metadata columns: {list(self.tracks_df.columns)}."
            )

        print(f"[*] Using emotion columns: {emotion_columns}")

        smoothed_df = self._smooth_emotions(segments_df, emotion_columns)
        raw_assignments = self._raw_track_selection(smoothed_df, emotion_columns)
        final_assignments = self._apply_continuity_rules(raw_assignments)
        regions_df = self._build_regions(final_assignments)

        return final_assignments, regions_df

    def _normalize_emotion_name(self, name: str) -> str:
        return name.strip().lower().replace(" ", "_")

    def _find_shared_emotion_columns(self, segments_df: pd.DataFrame) -> List[str]:
        ignored = {
            "index",
            "start_time",
            "end_time",
            "transcript",
            "content_summary",
            "track_id",
            "filename",
            "energy",
            "loopable",
        }

        segment_emotions = {
            col for col in segments_df.columns
            if col not in ignored and pd.api.types.is_numeric_dtype(segments_df[col])
        }

        track_emotions = {
            col for col in self.tracks_df.columns
            if col not in ignored and pd.api.types.is_numeric_dtype(self.tracks_df[col])
        }

        return sorted(segment_emotions.intersection(track_emotions))

    def _smooth_emotions(self, segments_df: pd.DataFrame, emotion_columns: List[str]) -> pd.DataFrame:
        smoothed = segments_df.copy()

        # Centered rolling average reduces isolated emotion spikes.
        for col in emotion_columns:
            smoothed[col] = (
                smoothed[col]
                .rolling(window=self.smoothing_window, center=True, min_periods=1)
                .mean()
            )

        return smoothed

    def _raw_track_selection(self, segments_df: pd.DataFrame, emotion_columns: List[str]) -> pd.DataFrame:
        result = segments_df.copy()

        segment_vectors = result[emotion_columns].fillna(0.0).to_numpy(dtype=float)
        track_vectors = self.tracks_df[emotion_columns].fillna(0.0).to_numpy(dtype=float)

        similarities = cosine_similarity(segment_vectors, track_vectors)

        best_indices = np.argmax(similarities, axis=1)
        best_scores = similarities[np.arange(len(result)), best_indices]

        selected_tracks = self.tracks_df.iloc[best_indices].reset_index(drop=True)

        result["raw_track_id"] = selected_tracks["track_id"].values
        result["raw_filename"] = selected_tracks["filename"].values
        result["raw_similarity"] = best_scores

        # Dominant emotion is useful for debugging and visual inspection.
        result["dominant_emotion"] = result[emotion_columns].idxmax(axis=1)

        return result

    def _apply_continuity_rules(self, assignments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prevents chaotic soundtrack changes.

        Rules:
        1. Keep the current track for at least min_switch_duration seconds.
        2. Only switch if the new candidate stays stable for stable_segments_required segments.
        3. Only switch if the new candidate is sufficiently better than the current one.
        """
        df = assignments_df.copy()

        selected_track_ids = []
        selected_filenames = []
        selected_scores = []

        current_track_id = None
        current_filename = None
        current_track_start = None
        current_score = None

        for i, row in df.iterrows():
            candidate_track = row["raw_track_id"]
            candidate_filename = row["raw_filename"]
            candidate_score = float(row["raw_similarity"])
            start_time = float(row["start_time"])

            if current_track_id is None:
                current_track_id = candidate_track
                current_filename = candidate_filename
                current_track_start = start_time
                current_score = candidate_score

            else:
                current_duration = start_time - float(current_track_start)

                candidate_is_stable = self._candidate_is_stable(df, i, candidate_track)
                candidate_is_better = candidate_score > float(current_score) + self.switch_threshold
                enough_time_passed = current_duration >= self.min_switch_duration

                should_switch = (
                    candidate_track != current_track_id
                    and candidate_is_stable
                    and candidate_is_better
                    and enough_time_passed
                )

                if should_switch:
                    current_track_id = candidate_track
                    current_filename = candidate_filename
                    current_track_start = start_time
                    current_score = candidate_score
                else:
                    # Keep current track, but update score estimate softly.
                    current_score = max(float(current_score), candidate_score)

            selected_track_ids.append(current_track_id)
            selected_filenames.append(current_filename)
            selected_scores.append(current_score)

        df["selected_track_id"] = selected_track_ids
        df["selected_filename"] = selected_filenames
        df["selected_similarity"] = selected_scores

        return df

    def _candidate_is_stable(self, df: pd.DataFrame, start_index: int, candidate_track: str) -> bool:
        end_index = min(start_index + self.stable_segments_required, len(df))

        future_candidates = df.iloc[start_index:end_index]["raw_track_id"].tolist()

        if len(future_candidates) < self.stable_segments_required:
            return True

        return all(track == candidate_track for track in future_candidates)

    def _build_regions(self, assignments_df: pd.DataFrame) -> pd.DataFrame:
        regions = []

        current = None

        for _, row in assignments_df.iterrows():
            track_id = row["selected_track_id"]
            filename = row["selected_filename"]

            if current is None:
                current = {
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "track_id": track_id,
                    "filename": filename,
                    "dominant_emotions": [row["dominant_emotion"]],
                }
                continue

            if track_id == current["track_id"]:
                current["end_time"] = float(row["end_time"])
                current["dominant_emotions"].append(row["dominant_emotion"])
            else:
                current["dominant_emotion"] = self._most_common(current["dominant_emotions"])
                del current["dominant_emotions"]
                regions.append(current)

                current = {
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "track_id": track_id,
                    "filename": filename,
                    "dominant_emotions": [row["dominant_emotion"]],
                }

        if current is not None:
            current["dominant_emotion"] = self._most_common(current["dominant_emotions"])
            del current["dominant_emotions"]
            regions.append(current)

        return pd.DataFrame(regions)

    def _most_common(self, values: List[str]) -> str:
        return pd.Series(values).mode().iloc[0]