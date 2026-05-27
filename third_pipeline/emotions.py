"""
emotions.py
-----------
Emotion configuration, colour blending, and timeline interpolation.

Each emotion now carries a `hue_range` (±degrees) so the renderer
can pick a slightly different hue for every individual fibre, producing
the colour "ranges" seen in Anadol's work (joy isn't one yellow —
it ranges from amber to warm orange).
"""

import json
import numpy as np
import colorsys


# ── Emotion definitions ────────────────────────────────────────────────────────
EMOTION_CONFIG = {
    "joy": {
        "color_hsl": (45, 0.98, 0.65),   # warm gold
        "hue_range":  18,                  # ±18° → amber .. orange
        "cohesion":  0.003,
        "separation":0.018,
        "max_speed": 2.6,
        "drift":     np.array([0.0, -0.3]),
        "scatter":   True,
    },
    "sadness": {
        "color_hsl": (220, 0.85, 0.48),  # deep blue
        "hue_range":  22,                  # ±22° → cobalt .. blue-violet
        "cohesion":  0.004,
        "separation":0.006,
        "max_speed": 0.4,
        "drift":     np.array([0.0,  0.4]),
        "scatter":   False,
    },
    "anger": {
        "color_hsl": (4, 0.95, 0.52),    # red-coral
        "hue_range":  14,                  # ±14° → crimson .. deep orange
        "cohesion":  0.001,
        "separation":0.038,
        "max_speed": 2.0,
        "drift":     np.array([0.0,  0.0]),
        "scatter":   True,
    },
    "calm": {
        "color_hsl": (185, 0.75, 0.50),  # cyan-teal
        "hue_range":  16,                  # ±16° → aqua .. soft teal
        "cohesion":  0.008,
        "separation":0.010,
        "max_speed": 0.5,
        "drift":     np.array([0.0,  0.0]),
        "scatter":   False,
    },
    "fear": {
        "color_hsl": (270, 0.70, 0.42),  # mid-purple
        "hue_range":  20,                  # ±20° → indigo .. magenta-violet
        "cohesion":  0.030,
        "separation":0.004,
        "max_speed": 0.3,
        "drift":     np.array([0.0,  0.0]),
        "scatter":   False,
    },
}

EMOTION_KEYS = list(EMOTION_CONFIG.keys())


# ── Colour helpers ─────────────────────────────────────────────────────────────

def hsl_to_rgb255(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def blend_color(emotion_vec: dict, alpha: float = 1.0) -> tuple:
    """Weighted blend of all emotion colours → (R, G, B)."""
    r, g, b = 0.0, 0.0, 0.0
    for key, weight in emotion_vec.items():
        cfg = EMOTION_CONFIG[key]
        h, s, l = cfg["color_hsl"]
        cr, cg, cb = hsl_to_rgb255(h, s, l)
        r += cr * weight; g += cg * weight; b += cb * weight
    return (int(r), int(g), int(b))


def blend_param(emotion_vec: dict, param: str) -> float:
    val = 0.0
    for key, weight in emotion_vec.items():
        val += EMOTION_CONFIG[key][param] * weight
    return val


def blend_drift(emotion_vec: dict) -> np.ndarray:
    drift = np.zeros(2)
    for key, weight in emotion_vec.items():
        drift += EMOTION_CONFIG[key]["drift"] * weight
    return drift


# ── Emotion state interpolation ────────────────────────────────────────────────

def lerp_emotions(current: dict, target: dict, speed: float = 0.025) -> dict:
    return {
        k: current[k] + (target[k] - current[k]) * speed
        for k in EMOTION_KEYS
    }


def get_emotions_at(timeline: list, t: float) -> dict:
    if not timeline:
        return {k: 0.2 for k in EMOTION_KEYS}
    if t >= timeline[-1]["t"]:
        return dict(timeline[-1]["emotions"])
    if t <= timeline[0]["t"]:
        return dict(timeline[0]["emotions"])
    prev, nxt = timeline[0], timeline[-1]
    for i in range(len(timeline) - 1):
        if timeline[i]["t"] <= t < timeline[i + 1]["t"]:
            prev = timeline[i]; nxt = timeline[i + 1]; break
    span  = nxt["t"] - prev["t"]
    alpha = (t - prev["t"]) / span if span > 0 else 0.0
    return {
        k: prev["emotions"][k] + (nxt["emotions"][k] - prev["emotions"][k]) * alpha
        for k in EMOTION_KEYS
    }


# ── Sentence-level JSON support ─────────────────────────────────────────────
# The emotion-recognition pipeline produces 8 emotion classes per sentence;
# we map them to our 5. Each target lists (source_name, weight) pairs.
# Secondary sources use weights < 1.0 because the upstream model tends to
# over-elevate "neutral" and "disgust" — without down-weighting them, calm
# beats joy on excited sentences and fear beats sadness on sad sentences.
EMOTION_MAPPING = {
    "joy":     [("happy",   1.0), ("surprised", 0.6)],
    "sadness": [("sad",     1.0)],
    "anger":   [("angry",   1.0)],
    "calm":    [("calm",    1.0), ("neutral",   0.4)],
    "fear":    [("fearful", 1.0), ("disgust",   0.5)],
}

# Softmax temperature for amplifying the differences in fused_emotion.
# Raw fused values cluster around 0.11–0.17 (very flat); we sharpen them.
#   T = 0.10 → mild amplification (gentle peaks)
#   T = 0.05 → strong amplification (one clearly dominant emotion + minor blend)
#   T = 0.02 → near winner-take-all
SOFTMAX_TEMPERATURE = 0.05


def _map_emotions_8_to_5(fused: dict) -> dict:
    """Aggregate the 8 emotions into our 5 using the weighted mapping."""
    out = {}
    for target, sources in EMOTION_MAPPING.items():
        out[target] = sum(float(fused.get(src, 0.0)) * w for src, w in sources)
    return out


def _softmax(values: dict, temperature: float = SOFTMAX_TEMPERATURE) -> dict:
    """Softmax with temperature → returns dict summing to 1.0."""
    keys = list(values.keys())
    raw  = np.array([values[k] for k in keys], dtype=float)
    scaled = raw / max(temperature, 1e-6)
    scaled -= scaled.max()    # numerical stability
    exp = np.exp(scaled)
    probs = exp / exp.sum()
    return {k: float(p) for k, p in zip(keys, probs)}


def _convert_sentences_to_keyframes(sentences: list) -> list:
    """Sentence list → keyframe list. Each sentence becomes TWO keyframes
    (at start and end) so its emotion is held constant DURING speech and only
    transitions during silence between sentences. Transcript is preserved on
    the start keyframe in case you want to display it on the HUD later."""
    keyframes = []
    for s in sentences:
        fused    = s.get("fused_emotion", {})
        mapped   = _map_emotions_8_to_5(fused)
        emotions = _softmax(mapped)

        ts    = s.get("timestamp", {})
        start = float(ts.get("start", 0.0))
        end   = float(ts.get("end",   start + 1.0))

        keyframes.append({
            "t":          start,
            "emotions":   emotions,
            "transcript": s.get("transcript", ""),
        })
        keyframes.append({"t": end, "emotions": emotions})
    keyframes.sort(key=lambda x: x["t"])
    return keyframes


# ── Timeline loader ────────────────────────────────────────────────────────────

# Where teammates drop the real audiobook JSON files. The latest .json in
# this folder is preferred over the argument passed to load_timeline().
TIMELINE_DIR = "../emotional_timeline"


def _find_latest_timeline():
    """Return the path of the most recently modified .json in TIMELINE_DIR,
    or None if the folder is missing/empty."""
    import os
    if not os.path.isdir(TIMELINE_DIR):
        return None
    jsons = [f for f in os.listdir(TIMELINE_DIR) if f.lower().endswith(".json")]
    if not jsons:
        return None
    paths = [os.path.join(TIMELINE_DIR, f) for f in jsons]
    return max(paths, key=os.path.getmtime)


def load_timeline(fallback_path: str) -> list:
    """Load a timeline JSON. Resolution order:

      1. The most recently modified .json in `emotional_timeline/`
      2. `fallback_path` (typically your dev `fake_timeline.json`)

    Auto-detects two JSON formats:

      A. Sentence-level (new format from the emotion-recognition pipeline):
         { "sentences": [ { "timestamp": {...}, "fused_emotion": {...}, ... } ] }

      B. Pre-baked keyframes (legacy / fake_timeline.json):
         [ { "t": 0.0, "emotions": { "joy": 0.2, ... } }, ... ]
    """
    discovered = _find_latest_timeline()
    if discovered is not None:
        path = discovered
        print(f"[emotions] ▶ TIMELINE FILE: {path}")
    else:
        path = fallback_path
        print(f"[emotions] '{TIMELINE_DIR}/' empty or missing — falling back to {fallback_path}")
        print(f"[emotions] ▶ TIMELINE FILE: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "sentences" in data:
        return _convert_sentences_to_keyframes(data["sentences"])

    if isinstance(data, list):
        data.sort(key=lambda x: x["t"])
        return data

    raise ValueError(f"Unrecognized timeline format in {path}")


def get_total_duration(timeline: list, default: float = 64.0) -> float:
    """Total playback length for a loaded timeline (last keyframe + 1s tail).
    Used by the playback loop and the on-screen timeline bar."""
    if not timeline:
        return default
    return max(kf["t"] for kf in timeline) + 1.0