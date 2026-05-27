<div align="center">

# 🌳 LBMMP — MoodBook

**Audiobook Emotion Pipeline · Learning-Based Multimedia Processing**

**Instituto Superior Técnico · Lisbon · 2026**

*Turn spoken audio into emotion-driven soundtracks and a living visualization.*

---

</div>

## ✨ What It Does

Take any audiobook, podcast, or spoken recording. The pipeline detects its **emotional arc** sentence by sentence and uses that arc in three ways:

- 🎵 **Soundtrack layer** — automatically selects and mixes background music that matches the emotional tone of each sentence
- 🌿 **Living tree** — animates a procedural tree in real time, responding to the detected emotions as the audio plays
- 📊 **Data dashboard** *(optional)* — browser-based view of the analysis JSON for inspection and debugging

A single launcher (`app.py`) chains all three pipelines into one continuous experience: pick a chapter, watch it process, then enjoy the synchronized music + visualization.

---

## 🚀 Quick Start

```bash
# from the repo root
python app.py
```

That's the entire experience: a launcher window opens, you pick an audiobook from the library, watch it process, and the tree visualizer plays in sync with the music-enhanced audio.

For installation details, individual pipeline usage, and troubleshooting, read on.

---

## 📦 Installation

### 1. Python

Python **3.10 or newer** is required. Verify with `python --version`.

### 2. Install Python dependencies

Each pipeline has its own `requirements.txt`. Install them all:

```bash
pip install -r first_pipeline/requirements.txt
pip install -r second_pipeline/requirements.txt
pip install -r third_pipeline/requirements.txt
```

**Key libraries that get installed:**

| Stage | Main dependencies |
|---|---|
| `first_pipeline` | `torch` · `transformers` · `openai-whisper` · `demucs` · `librosa` · `tqdm` |
| `second_pipeline` | `numpy` · `pandas` · `scikit-learn` · `pydub` |
| `third_pipeline` | `pygame` · `numpy` |

> If you only want to run the **visualizer standalone** (using a pre-existing analysis JSON), you only need `third_pipeline/requirements.txt`.

### 3. Install FFmpeg ⚠️ *required*

Whisper (transcription) and pydub (audio mixing) both call out to **FFmpeg**, which is a separate native program — not a Python package. Without it on your system PATH, transcription will silently fail with `[WinError 2]` and the mixer will refuse to load MP3s.

**Windows:**
1. Download `ffmpeg-release-full.7z` from <https://www.gyan.dev/ffmpeg/builds/>
2. Extract to a stable location like `C:\ffmpeg\`
3. Add `C:\ffmpeg\bin` to your **System PATH** (search Windows for "Environment Variables")
4. **Open a new terminal** and verify: `ffmpeg -version` should print version info

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### 4. First-run model downloads (~5 GB total)

On **first execution**, the first_pipeline downloads three AI models from HuggingFace and one from Meta:

| Model | Size | Purpose | Cached to |
|---|---|---|---|
| Demucs `htdemucs` | ~80 MB | Vocal isolation | `~/.cache/torch/hub/checkpoints/` |
| Whisper `base` | ~140 MB | Speech transcription | `~/.cache/whisper/` |
| Wav2Vec2 emotion | ~1.3 GB | Acoustic emotion classification | `~/.cache/huggingface/hub/` |
| Qwen2.5-1.5B-Instruct | ~3 GB | Text-based emotion reasoning | `~/.cache/huggingface/hub/` |

These cache after the first download — subsequent runs are instant. Expect the first run to take 10-20 minutes depending on connection speed.

> 🔥 **University / corporate networks sometimes block HuggingFace.** If you see `NameResolutionError` or `Failed to resolve 'huggingface.co'`, switch to a mobile hotspot for the first run only. Once cached locally, you never need internet again.

### 5. Soundtrack library

For the music mixing to actually produce audio, you need:

- `second_pipeline/data/metadata/soundtrack_metadata.csv` — catalog mapping emotions to music files
- `second_pipeline/data/soundtracks/*.mp3` — the actual music files listed in the metadata

If the metadata lists files that don't exist on disk, the mixer logs `[!] Missing soundtrack file, skipping:` warnings and you'll get silence under those moments. The app falls back to the original audio in this case.

---

## 🎬 How to Use

### Layout your audiobooks

Drop audio files into `first_pipeline/data/raw/`. Two layouts are supported:

```
first_pipeline/data/raw/
├── single_recording.mp3                   ← single-file audiobook
└── war_of_the_worlds/                     ← multi-chapter audiobook (folder)
    ├── chapter_01.mp3
    ├── chapter_02.mp3
    └── chapter_03.mp3
```

Supported formats: `.mp3` · `.wav` · `.m4a` · `.flac` · `.ogg` · `.aac`

### Run it

```bash
python app.py
```

### What happens on screen

| Screen | What it does |
|---|---|
| 🏠 **START** | Title, brief description, "Enter Experience" button |
| 📚 **SELECT** | Lists your library (folder = playlist), browse own file option, scroll wheel works |
| ⚙️ **PROCESSING** | Runs first_pipeline → JSON, then second_pipeline → enhanced audio. Watch the terminal for live progress. Auto-skips first_pipeline if a cached JSON already exists. |
| 🌳 **PLAYBACK** | Tree visualizer driven by the audio's playback position. SPACE pauses, ESC returns home. |
| ✓ **END** | "Chapter complete" — auto-queues next chapter if it's a multi-file audiobook, or returns to library |

The full chain takes 5-15 minutes for a 5-minute audiobook chapter on first analysis (mostly Whisper + LLM inference). On a re-run of an already-analyzed chapter, it skips straight to the mixer (~30 seconds) and then playback.

### Cleanup

The launcher registers an `atexit` hook that deletes `temp/enhanced_audiobook.mp3` when you close the app — no stale audio lingers between sessions. Cached emotion JSONs in `first_pipeline/data/processed/` are kept (deletion would force expensive re-analysis).

---

## 🏛️ Architecture

```
                ┌────────────────────────────────────┐
                │      AUDIO  (audiobook, voice)     │
                └────────────────────────────────────┘
                                  │
                                  ▼
            ┌──────────────────────────────────────────┐
            │   first_pipeline/  —  EMOTION ANALYSIS   │
            │   isolator → segmenter → emotion +       │
            │   transcriber → fused 8-class emotion    │
            └──────────────────────────────────────────┘
                                  │
                     ┌────────────┴────────────┐
                     │   sentence-level JSON   │
                     │  (timestamps + emotions │
                     │  + transcript per line) │
                     └────────────┬────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
┌─────────────────────┐ ┌──────────────────────┐ ┌────────────────────┐
│  second_pipeline/   │ │  third_pipeline/     │ │  first_pipeline/   │
│  🎵 SOUNDTRACK MIX  │ │  🌳 TREE VISUALIZER  │ │  📊 WEB DASHBOARD  │
│  cosine matching →  │ │  pygame, procedural  │ │  visualizer.html   │
│  enhanced audiobook │ │  → live animation    │ │  → in-browser deck │
└─────────────────────┘ └──────────────────────┘ └────────────────────┘
                                  ▲
                                  │
                ┌─────────────────┴─────────────────┐
                │  app.py  (orchestrator at root)   │
                │  state machine: START → SELECT →  │
                │  PROCESSING → PLAYBACK → END      │
                └───────────────────────────────────┘
```

> The JSON file is the contract between stages. Each downstream pipeline reads it independently — you can run any of them without the orchestrator.

---

## 🧩 The Pipelines

### 1️⃣ `first_pipeline/` — Emotion Analysis

Produces the shared analysis JSON consumed by all downstream stages.

| Component | Role |
|---|---|
| `isolator.py` | Vocal separation — Demucs strips background music from the input audio |
| `segmenter.py` | Splits the isolated voice into sentence-level chunks (librosa) |
| `emotion.py` | Wav2Vec2-based acoustic emotion classifier (8 classes) |
| `transcriber.py` | Whisper for transcription + Qwen2.5-1.5B for text-based emotion |
| `pipeline.py` | Orchestrates all components; globally normalizes emotion scores |
| `visualizer.html` | Optional in-browser dashboard for inspecting analysis output |

**Output:** `data/processed/<audiofile>_analysis.json` — one entry per sentence with `acoustic_emotion`, `text_emotion`, and merged `fused_emotion` across 8 classes.

**Standalone use:**

```bash
python first_pipeline/src/pipeline.py --input_file path/to/audio.mp3
```

---

### 2️⃣ `second_pipeline/` — Soundtrack Selection & Mixing

Reads the analysis JSON, picks a background track per sentence via cosine similarity against tagged soundtrack metadata, and produces an enhanced audiobook with voice layered over emotion-matched music plus adaptive ducking (music dips when the narrator speaks).

| Component | Role |
|---|---|
| `soundtrack_selector.py` | Cosine similarity + smoothing + switch-threshold logic to avoid rapid track thrashing |
| `audio_mixer.py` | Overlays selected tracks beneath the original audio with adaptive ducking |
| `main.py` | CLI entry point |

**Standalone use:**

```bash
python second_pipeline/src/main.py \
    --analysis  first_pipeline/data/processed/audiobook_analysis.json \
    --audiobook first_pipeline/data/raw/audiobook.mp3 \
    --metadata  second_pipeline/data/metadata/soundtrack_metadata.csv \
    --soundtrack-dir second_pipeline/data/soundtracks \
    --output    temp/enhanced_audiobook.mp3
```

---

### 3️⃣ `third_pipeline/` — Generative Tree Visualizer

A pygame visualization that animates a living tree whose appearance and behaviour are fully driven by the emotional timeline.

| Emotion | Visual effect |
|---|---|
| ☀️ **Joy** | Warm sun glow, golden pollen drifting, lush canopy, gentle breeze |
| 🌙 **Calm** | Moon rising, blue fireflies wandering, soft slow sway |
| 🌧️ **Sadness** | Branches sag, leaves drop, diagonal rain falls |
| ⚡ **Anger** | Violent gusts strip the canopy, hot embers fly, lightning flashes |
| 👻 **Fear** | Bare dead trees appear, ghostly mist drifts, leaves tremble |

> Effects blend smoothly — a sad-to-fearful transition shows rain easing as mist drifts in and the trunk slowly drains of colour and leaves.

**Standalone use** (uses cached JSON, no audio playback):

```bash
python third_pipeline/main_tree.py
```

**Keyboard controls** (standalone mode):

| Key | Action |
|---|---|
| `1` – `5` | Force an emotion (joy / sad / anger / calm / fear) |
| `0` | Release override and resume the timeline |
| `Space` | Pause / resume |
| `R` | Reset the tree |
| `F` | Toggle fullscreen |
| `Q` | Quit |

> The visualizer auto-loads the **most recent `.json`** found in `emotional_timeline/`. The orchestrator (`app.py`) drops the right one there automatically when you pick a chapter.

---

## 📦 JSON Contract

Every sentence in the analysis file is a dict shaped like this:

```json
{
    "sentence_index": 0,
    "timestamp": { "start": 0.0, "end": 6.78 },
    "transcript": "This is a recording in which at the beginning I'm talking very normal and formal.",
    "acoustic_emotion": {
        "angry": 0.12, "calm": 0.17, "disgust": 0.12, "fearful": 0.11,
        "happy": 0.10, "neutral": 0.16, "sad": 0.11, "surprised": 0.11
    },
    "text_emotion":  { "...same 8 keys..." },
    "fused_emotion": { "...same 8 keys — this is what downstream stages use..." }
}
```

`third_pipeline` collapses 8 → 5 classes for the visualizer:

| Source classes | Visualizer class |
|---|---|
| `happy` + `surprised` | **joy** |
| `calm` + `neutral` | **calm** |
| `fearful` + `disgust` | **fear** |
| `sad` | **sadness** |
| `angry` | **anger** |

A softmax is then applied to amplify differences in the typically flat fused vectors. Mapping weights and softmax temperature are tunable in `third_pipeline/emotions.py`.

---

## 🗂️ Project Structure

```
lbmmp/
├── app.py                     🚀  Main launcher — start here
├── README.md
├── emotional_timeline/         (created by app, holds the active analysis JSON)
├── temp/                       (created by app, holds the generated enhanced audio)
│
├── first_pipeline/             🎙️  Audio → JSON
│   ├── src/
│   │   ├── pipeline.py             entry point, orchestrates the stages
│   │   ├── isolator.py             vocal isolation via Demucs
│   │   ├── segmenter.py            sentence-level audio chunking
│   │   ├── emotion.py              Wav2Vec2 acoustic classifier
│   │   ├── transcriber.py          Whisper + Qwen LLM
│   │   └── visualizer.html         optional in-browser dashboard
│   ├── data/
│   │   ├── raw/                    your audio files / book folders
│   │   └── processed/              cached analysis JSONs (one per audio)
│   └── requirements.txt
│
├── second_pipeline/            🎵  JSON + audio → enhanced audio
│   ├── src/
│   │   ├── main.py
│   │   ├── soundtrack_selector.py
│   │   └── audio_mixer.py
│   ├── data/
│   │   ├── metadata/soundtrack_metadata.csv
│   │   └── soundtracks/            actual music mp3s referenced by metadata
│   └── requirements.txt
│
└── third_pipeline/             🌳  JSON → live animation
    ├── main_tree.py                standalone entry point
    ├── tree.py                     procedural tree + leaf physics
    ├── weather.py                  rain · embers · mist · fireflies · lightning · moon · sun
    ├── emotions.py                 JSON loader + 8→5 mapping + interpolation
    ├── fake_timeline.json          dev fallback timeline
    ├── assets/background.png       optional twilight backdrop
    └── requirements.txt
```

---

## 🐛 Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `ModuleNotFoundError: No module named 'whisper'` | Run `pip install -r first_pipeline/requirements.txt` |
| `ModuleNotFoundError: No module named 'emotions'` | The launcher couldn't find `third_pipeline/` modules. Make sure you're running `python app.py` from the repo root. |
| `[WinError 2] The system cannot find the file specified` during transcription | FFmpeg is not on your PATH. See installation step 3. **Close all terminals and open a new one** after editing PATH. |
| `NameResolutionError: Failed to resolve 'huggingface.co'` | Your network is blocking HuggingFace. Use a mobile hotspot for the first model download — it caches locally afterwards. |
| Endless retries on first run, GPU at 0% | Same as above — the model download is hanging. Kill with Ctrl+C, switch network, retry. |
| `[!] Missing soundtrack file, skipping:` warnings + silent music | A filename in `soundtrack_metadata.csv` doesn't exist in `second_pipeline/data/soundtracks/`. Fix the csv or add the missing mp3. |
| Music too quiet / too loud | Edit `base_music_gain_db` in `second_pipeline/src/main.py` (default -16 dB). Less negative = louder. |
| Second pipeline crashes | The app falls back gracefully to the original audio + visualizer. Check the terminal for the actual error. |
| Audio plays but tree doesn't move | The JSON is empty or all-neutral. Check `first_pipeline/data/processed/<name>_analysis.json` — the `fused_emotion` field should have varying values. |

---

## ⚙️ System Requirements

- **CPU:** any modern x86-64
- **GPU:** optional but strongly recommended (CUDA-capable Nvidia card) — without it, transcription and emotion analysis fall back to CPU and take ~10× longer
- **RAM:** 8 GB minimum, 16 GB comfortable (the LLM phase loads ~3 GB into memory)
- **Disk:** ~6 GB free for model caches + your audio library
- **OS:** Windows 10/11, macOS, or Linux

---

<div align="center">

### 👥 Authors

Course project for *Learning-Based Multimedia Processing*

**Instituto Superior Técnico · Universidade de Lisboa · 2026**

</div>
