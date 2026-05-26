from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Same 5-emotion space used by the visualizer.
TARGET_EMOTION_KEYS = ["joy", "sadness", "anger", "calm", "fear"]

# First pipeline / fused_emotion outputs 8 emotions.
SOURCE_EMOTION_KEYS = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

# Same mapping logic as the visualizer.
EMOTION_MAPPING = {
    "joy": [("happy", 1.0), ("surprised", 0.6)],
    "sadness": [("sad", 1.0)],
    "anger": [("angry", 1.0)],
    "calm": [("calm", 1.0), ("neutral", 0.4)],
    "fear": [("fearful", 1.0), ("disgust", 0.5)],
}

# Same sharpening value used by the visualizer.
SOFTMAX_TEMPERATURE = 0.05


class SoundtrackSelector:
    """
    Selects soundtrack clips based on emotion vectors.

    Supported analysis JSON formats:

    1) New first-pipeline format:
       {
         "sentences": [
           {
             "sentence_index": 0,
             "timestamp": {"start": 0.0, "end": 7.1},
             "transcript": "...",
             "fused_emotion": {
               "angry": ...,
               "calm": ...,
               ...
             }
           }
         ]
       }

    2) Older raw_chunks test format:
       {
         "raw_chunks": [
           {
             "index": 0,
             "timestamp": {"start": 0.0, "end": 4.0},
             "emotions": {...}
           }
         ]
       }

    Internally, the pipeline converts everything to the same 5-emotion space:
    joy, sadness, anger, calm, fear.
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        smoothing_window: int = 5,
        switch_threshold: float = 0.08,
        min_switch_duration: float = 8.0,
        stable_segments_required: int = 2,
        softmax_temperature: float = SOFTMAX_TEMPERATURE,
    ):
        self.metadata_csv = Path(metadata_csv)
        self.smoothing_window = smoothing_window
        self.switch_threshold = switch_threshold
        self.min_switch_duration = min_switch_duration
        self.stable_segments_required = stable_segments_required
        self.softmax_temperature = softmax_temperature

        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Soundtrack metadata file not found: {self.metadata_csv}")

        self.tracks_df = pd.read_csv(self.metadata_csv)
        self.tracks_df.columns = [self._normalize_name(c) for c in self.tracks_df.columns]

        required_columns = {"track_id", "filename"}
        missing = required_columns - set(self.tracks_df.columns)

        if missing:
            raise ValueError(f"Missing required columns in soundtrack metadata: {missing}")

        self.tracks_df = self._prepare_track_emotions(self.tracks_df)

    def load_analysis(self, analysis_json: str | Path) -> pd.DataFrame:
        analysis_path = Path(analysis_json)

        if not analysis_path.exists():
            raise FileNotFoundError(f"Analysis JSON not found: {analysis_path}")

        with open(analysis_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = self._extract_items(data)
        rows = []

        for i, item in enumerate(items):
            timestamp = item.get("timestamp", {})

            # Priority:
            # 1. fused_emotion: new first-pipeline output
            # 2. emotions: older test/raw_chunks format
            fused = item.get("fused_emotion") or item.get("emotions") or {}

            if not fused:
                continue

            emotion_vector = self._to_target_emotions(
                fused,
                use_softmax=True,
            )

            row = {
                "index": item.get("sentence_index", item.get("index", i)),
                "start_time": float(timestamp.get("start", 0.0)),
                "end_time": float(timestamp.get("end", 0.0)),
                "transcript": item.get("transcript", ""),
            }

            for key in TARGET_EMOTION_KEYS:
                row[key] = emotion_vector[key]

            # Useful for debugging.
            normalized_fused = {
                self._normalize_name(k): float(v)
                for k, v in fused.items()
            }

            row["source_dominant_emotion"] = max(
                normalized_fused,
                key=normalized_fused.get,
            )

            rows.append(row)

        if not rows:
            raise ValueError("Analysis JSON is empty or has no valid emotion data.")

        segments_df = pd.DataFrame(rows)
        segments_df = segments_df.sort_values("start_time").reset_index(drop=True)

        return segments_df

    def select(self, analysis_json: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
            assignments_df:
                One row per sentence/segment, with selected soundtrack.

            regions_df:
                Consecutive sentences/segments grouped into longer soundtrack regions.
        """
        segments_df = self.load_analysis(analysis_json)

        print(f"[*] Using target emotion columns: {TARGET_EMOTION_KEYS}")

        smoothed_df = self._smooth_emotions(segments_df)
        raw_assignments = self._raw_track_selection(smoothed_df)
        final_assignments = self._apply_continuity_rules(raw_assignments)
        regions_df = self._build_regions(final_assignments)

        return final_assignments, regions_df

    def _extract_items(self, data):
        if isinstance(data, dict) and "sentences" in data:
            return data["sentences"]

        if isinstance(data, dict) and "raw_chunks" in data:
            return data["raw_chunks"]

        if isinstance(data, list):
            return data

        raise ValueError("Unrecognized analysis JSON format.")

    def _normalize_name(self, name: str) -> str:
        return str(name).strip().lower().replace(" ", "_")

    def _prepare_track_emotions(self, tracks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Allows the soundtrack metadata CSV to use either:

        A) 5 target columns:
           joy, sadness, anger, calm, fear

        B) 8 source columns:
           angry, calm, disgust, fearful, happy, neutral, sad, surprised

        In both cases, the dataframe is converted to the 5-emotion target space.
        """
        result = tracks_df.copy()

        for idx, row in result.iterrows():
            row_dict = row.to_dict()

            emotion_vector = self._to_target_emotions(
                row_dict,
                use_softmax=False,
            )

            for key in TARGET_EMOTION_KEYS:
                result.loc[idx, key] = emotion_vector[key]

        return result

    def _to_target_emotions(self, values: dict, use_softmax: bool) -> dict:
        """
        Converts either 8-emotion or 5-emotion dictionaries into:
        joy, sadness, anger, calm, fear.

        For first-pipeline fused_emotion values, use_softmax=True.
        For manually labelled soundtrack metadata, use_softmax=False.
        """
        clean = {
            self._normalize_name(k): float(v)
            for k, v in values.items()
            if self._can_be_float(v)
        }

        has_target_keys = all(key in clean for key in TARGET_EMOTION_KEYS)

        if has_target_keys:
            target = {
                key: max(0.0, float(clean.get(key, 0.0)))
                for key in TARGET_EMOTION_KEYS
            }
        else:
            target = self._map_8_to_5(clean)

        if use_softmax:
            return self._softmax(target)

        return self._normalize_to_sum(target)

    def _map_8_to_5(self, fused: dict) -> dict:
        mapped = {}

        for target_emotion, sources in EMOTION_MAPPING.items():
            total = 0.0

            for source_emotion, weight in sources:
                total += float(fused.get(source_emotion, 0.0)) * weight

            mapped[target_emotion] = total

        return mapped

    def _softmax(self, values: dict) -> dict:
        keys = list(values.keys())
        raw = np.array([values[k] for k in keys], dtype=float)

        temperature = max(self.softmax_temperature, 1e-6)
        scaled = raw / temperature
        scaled -= scaled.max()

        exp = np.exp(scaled)
        probs = exp / exp.sum()

        return {k: float(p) for k, p in zip(keys, probs)}

    def _normalize_to_sum(self, values: dict) -> dict:
        clean = {
            key: max(0.0, float(values.get(key, 0.0)))
            for key in TARGET_EMOTION_KEYS
        }

        total = sum(clean.values())

        if total <= 0:
            uniform = 1.0 / len(TARGET_EMOTION_KEYS)
            return {key: uniform for key in TARGET_EMOTION_KEYS}

        return {
            key: clean[key] / total
            for key in TARGET_EMOTION_KEYS
        }

    def _can_be_float(self, value) -> bool:
        try:
            float(value)
            return True
        except Exception:
            return False

    def _smooth_emotions(self, segments_df: pd.DataFrame) -> pd.DataFrame:
        smoothed = segments_df.copy()

        for col in TARGET_EMOTION_KEYS:
            smoothed[col] = (
                smoothed[col]
                .rolling(window=self.smoothing_window, center=True, min_periods=1)
                .mean()
            )

        return smoothed

    def _raw_track_selection(self, segments_df: pd.DataFrame) -> pd.DataFrame:
        result = segments_df.copy()

        segment_vectors = result[TARGET_EMOTION_KEYS].fillna(0.0).to_numpy(dtype=float)
        track_vectors = self.tracks_df[TARGET_EMOTION_KEYS].fillna(0.0).to_numpy(dtype=float)

        similarities = cosine_similarity(segment_vectors, track_vectors)

        best_indices = np.argmax(similarities, axis=1)
        best_scores = similarities[np.arange(len(result)), best_indices]

        selected_tracks = self.tracks_df.iloc[best_indices].reset_index(drop=True)

        result["raw_track_id"] = selected_tracks["track_id"].values
        result["raw_filename"] = selected_tracks["filename"].values
        result["raw_similarity"] = best_scores
        result["dominant_emotion"] = result[TARGET_EMOTION_KEYS].idxmax(axis=1)

        return result

    def _apply_continuity_rules(self, assignments_df: pd.DataFrame) -> pd.DataFrame:
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