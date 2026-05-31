"""
app.py — MoodBook Launcher

Single-window pygame app that orchestrates the full LBMMP experience.

State machine:
    START  →  SELECT  →  PROCESSING  →  PLAYBACK  →  END  →  SELECT
                  ↑                                            │
                  └────────────────────────────────────────────┘

Run with:  python app.py
"""

from cmath import rect
from cmath import rect
import math
import os
import shutil
import subprocess
import sys
from matplotlib import lines
import pygame
from pathlib import Path
import atexit

# ── Configuration ───────────────────────────────────────────────────────
W, H = 1280, 720
FPS  = 60

def _wrap_text(text, font, max_width):
    """Break text into lines that fit within max_width pixels."""
    words = text.split()
    lines, current = [], []
    for word in words:
        current.append(word)
        if font.size(" ".join(current))[0] > max_width:
            if len(current) > 1:
                lines.append(" ".join(current[:-1]))
                current = [word]
    if current:
        lines.append(" ".join(current))
    return lines

# Paths (relative to this file's location)
HERE                  = Path(__file__).parent.resolve()
LIBRARY_DIR           = HERE/ "first_pipeline" / "data" / "raw"
ANALYSIS_OUTPUT_DIR   = HERE/ "first_pipeline" / "data" / "processed"
EMOTIONAL_TIMELINE    = HERE / "emotional_timeline"
ENHANCED_AUDIO_PATH   = HERE / "temp" / "enhanced_audiobook.mp3"

# The tree visualizer modules live in third_pipeline/. Add that folder to
# sys.path so `import emotions / tree / weather` works when running app.py
# from the repo root.
sys.path.insert(0, str(HERE / "third_pipeline"))

# Pipeline CLI templates — adjust if your teammates use different invocations
# Each list is passed to subprocess.Popen; {placeholders} get substituted.
FIRST_PIPELINE_CMD = [
    sys.executable,
    str(HERE / "first_pipeline" / "src" / "pipeline.py"),
    "--input_file", "{audio}",  # <-- Added the flag!
]
# Updated list with explicit metadata path
SECOND_PIPELINE_CMD = [
    sys.executable,
    str(HERE / "second_pipeline" / "src" / "main.py"),
    "--analysis", "{json}",
    "--audiobook", "{audio}",
    "--metadata",       str(HERE / "second_pipeline" / "data" / "metadata" / "soundtrack_metadata.csv"),
    "--soundtrack-dir", str(HERE / "second_pipeline" / "data" / "soundtracks"),
    "--output", "{output}",
]

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac")

# Screen IDs
START      = "start"
SELECT     = "select"
PROCESSING = "processing"
PLAYBACK   = "playback"
END        = "end"

# pygame USEREVENT used as a one-shot signal that music finished playing.
MUSIC_END_EVENT = pygame.USEREVENT + 1

# ── Palette ─────────────────────────────────────────────────────────────
# Anadol-inspired dark canvas with a faint warm undertone (matches the
# tree's twilight world rather than feeling like a sterile gallery).
BG_TOP    = (10,  9, 14)        # near-black, hint of indigo
BG_BOT    = (16, 12, 20)        # marginally warmer at the bottom
TEXT_MAIN = (245, 240, 230)     # off-white, slightly warm
TEXT_DIM  = (140, 135, 145)     # muted gray for metadata
TEXT_FAINT= ( 70,  68,  78)     # darker, for borders and hints
ACCENT    = (220, 195, 130)     # amber gold (= tree's joy emotion)
ERROR_COL = (220, 120, 120)


class App:
    def __init__(self):
        # Pre-init mixer with a small buffer so get_pos() is responsive
        # (default 4096-sample buffer makes audio sync feel laggy)
        pygame.mixer.pre_init(44100, -16, 2, 1024)
        pygame.init()
        atexit.register(self._cleanup_temp_files)
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("MoodBook")
        self.clock  = pygame.time.Clock()

        # Music end signal — fires once when pygame.mixer.music finishes naturally
        pygame.mixer.music.set_endevent(MUSIC_END_EVENT)

        # Fonts
        self.f_brand = pygame.font.SysFont("georgia", 88, italic=True)   # "MoodBook" wordmark
        self.f_title = pygame.font.SysFont("georgia", 32, italic=True)   # screen titles
        self.f_sub   = pygame.font.SysFont("helvetica", 13, bold=True)   # ALL-CAPS subtitles
        self.f_body  = pygame.font.SysFont("helvetica", 14)              # descriptive prose
        self.f_btn   = pygame.font.SysFont("helvetica", 12, bold=True)   # buttons
        self.f_item  = pygame.font.SysFont("helvetica", 15)              # file names
        self.f_tag   = pygame.font.SysFont("helvetica", 10, bold=True)   # [ MICRO TAGS ]

        self._bg = self._make_gradient_bg()

        self.state = START
        self.frame = 0
        self.t = 0.0
        self.mouse_pos = (0, 0)

        # Cross-screen state
        self.selected_audio_path = None    
        self.timeline_json_path  = None    
        self._enter_processing   = False   
        self._enter_playback     = False
        self._library_cache      = []      
        self._scroll_offset      = 0       
        self.playlist            = []      
        self.playlist_index      = 0       
        
        # Processing state (initialized here to prevent 1-frame crashes)
        self._proc_status        = ""
        self._proc_stage         = 0
        self._proc_first         = None
        self._proc_second        = None
        self._proc_err           = None
        self._proc_started_at    = 0.0

        # Playback state (initialized in _playback_init on entry; declared
        # here so attribute accesses during the entry frame don't explode)
        self._pb_timeline    = []
        self._pb_total       = 0.0
        self._pb_tree        = None
        self._pb_weather     = None
        self._pb_bg          = None
        self._pb_tint_surf   = None
        self._pb_gradient    = None
        self._pb_cur_ev      = None
        self._pb_frame       = 0
        self._pb_last_log_s  = -1
        self._pb_paused      = False
        self._pb_audio_loaded= False
        self._pb_pause_at_ms = 0      # ms position when pause was triggered
        self._pb_pause_total_ms = 0   # accumulated paused ms

        self._particles = self._spawn_particles(30)

    # ── One-time setup ──────────────────────────────────────────────────

    def _make_gradient_bg(self):
        surf = pygame.Surface((W, H))
        for y in range(H):
            f = y / (H - 1)
            r = int(BG_TOP[0] + (BG_BOT[0] - BG_TOP[0]) * f)
            g = int(BG_TOP[1] + (BG_BOT[1] - BG_TOP[1]) * f)
            b = int(BG_TOP[2] + (BG_BOT[2] - BG_TOP[2]) * f)
            pygame.draw.line(surf, (r, g, b), (0, y), (W, y))
        return surf

    def _spawn_particles(self, n):
        import random
        ps = []
        for _ in range(n):
            ps.append({
                "x": random.uniform(0, W), "y": random.uniform(0, H),
                "vx": random.uniform(-0.08, 0.08), "vy": random.uniform(-0.05, 0.05),
                "r":  random.uniform(0.4, 1.6),
                "phase": random.uniform(0, 2 * math.pi),
                "phase_speed": random.uniform(0.2, 0.6),
            })
        return ps
    
    def _draw_ghost_button(self, rect, label, hovered, accent=None):
        """Anadol-style button: thin border, fills warm gold on hover."""
        if accent is None: accent = ACCENT
        if hovered:
            pygame.draw.rect(self.screen, accent, rect)
            pygame.draw.rect(self.screen, accent, rect, width=1)
            col = BG_TOP
        else:
            pygame.draw.rect(self.screen, TEXT_MAIN, rect, width=1)
            col = TEXT_MAIN
        txt = self.f_btn.render(label, True, col)
        self.screen.blit(txt, (rect.centerx - txt.get_width() // 2,
                        rect.centery - txt.get_height() // 2))

    def _draw_tag(self, x, y, label, align="left", color=None):
        """Bracketed micro-label like [ ARCHIVE · 14 FILES ]"""
        if color is None: color = TEXT_DIM
        txt = self.f_tag.render(f"[  {label}  ]", True, color)
        if align == "center":
            self.screen.blit(txt, (x - txt.get_width() // 2, y))
        elif align == "right":
            self.screen.blit(txt, (x - txt.get_width(), y))
        else:
            self.screen.blit(txt, (x, y))

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            self.t += dt
            self.frame += 1
            self.mouse_pos = pygame.mouse.get_pos()

            events = pygame.event.get()
            for ev in events:
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    if self.state == START:
                        pygame.quit(); sys.exit()
                    elif self.state != PROCESSING:   # don't abort a running pipeline
                        # Stop any audio that was playing
                        if self.state == PLAYBACK:
                            pygame.mixer.music.stop()
                        self.state = START

            getattr(self, f"_{self.state}_update")(dt, events)
            getattr(self, f"_{self.state}_draw")()
            pygame.display.flip()

    # ── START SCREEN ────────────────────────────────────────────────────

    def _start_update(self, dt, events):
        for p in self._particles:
            p["x"] += p["vx"]; p["y"] += p["vy"]
            p["phase"] += p["phase_speed"] * dt
            if p["x"] < -10: p["x"] = W + 10
            elif p["x"] > W + 10: p["x"] = -10
            if p["y"] < -10: p["y"] = H + 10
            elif p["y"] > H + 10: p["y"] = -10

        for ev in events:
            if ev.type == pygame.MOUSEBUTTONDOWN and self._start_btn_rect().collidepoint(ev.pos):
                self.state = SELECT
            if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_SPACE, pygame.K_RETURN):
                self.state = SELECT


    def _start_draw(self):
        self.screen.blit(self._bg, (0, 0))

        # Subtle dust particles
        part_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        for p in self._particles:
            brightness = (math.sin(p["phase"]) + 1) * 0.5
            if brightness < 0.05: continue
            a = int(brightness * 70); r = max(1, int(p["r"]))
            pygame.draw.circle(part_surf, (240, 230, 220, a), (int(p["x"]), int(p["y"])), r)
        self.screen.blit(part_surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        # Top-left brand tag
        self._draw_tag(48, 36, "MOODBOOK  ·  LBMMP 2026")
        # Top-right meta
        self._draw_tag(W - 48, 36, "IMMERSIVE  AUDIOBOOK  EXPERIENCE",
                    align="right", color=TEXT_FAINT)

        # Wordmark — kept italic Georgia, this is the heart of the brand
        title = self.f_brand.render("MoodBook", True, TEXT_MAIN)
        self.screen.blit(title, (W // 2 - title.get_width() // 2, int(H * 0.30)))

        # Tagline below, all-caps Helvetica
        tagline = self.f_sub.render("AUDIOBOOKS  THAT  BREATHE  WITH  YOU", True, TEXT_DIM)
        self.screen.blit(tagline, (W // 2 - tagline.get_width() // 2, int(H * 0.45)))

        # Thin divider line
        pygame.draw.line(self.screen, TEXT_FAINT,
                     (W // 2 - 30, int(H * 0.50)),
                     (W // 2 + 30, int(H * 0.50)), 1)

        # Description prose, muted
        lines = [
        "MoodBook analyses the emotional arc of the narration,",
        "layers a matching soundtrack underneath, and animates",
        "a living tree that responds to every shift in feeling.",
        ]
        y = int(H * 0.55)
        for line in lines:
            txt = self.f_body.render(line, True, TEXT_DIM)
            self.screen.blit(txt, (W // 2 - txt.get_width() // 2, y)); y += 22

        # Ghost button
        btn = self._start_btn_rect()
        self._draw_ghost_button(btn, "ENTER  EXPERIENCE  ↗", btn.collidepoint(self.mouse_pos))

        # Footer hint
        self._draw_tag(W // 2, H - 36, "PRESS  SPACE  OR  CLICK   ·   Q  TO  QUIT",
                   align="center", color=TEXT_FAINT)


    def _start_btn_rect(self):
        return pygame.Rect(W // 2 - 130, int(H * 0.80), 260, 44)

    # ── SELECT SCREEN ───────────────────────────────────────────────────

    def _scan_library(self):
        if not LIBRARY_DIR.is_dir():
            return []
        
        audiobooks = set()
        # Search for supported audio files
        for ext in AUDIO_EXTENSIONS:
            for f in LIBRARY_DIR.rglob(f"*{ext}"):
                rel = f.relative_to(LIBRARY_DIR)
                audiobooks.add(rel.parts[0]) # Grabs just the folder name (or file name if loose)
            for f in LIBRARY_DIR.rglob(f"*{ext.upper()}"):
                rel = f.relative_to(LIBRARY_DIR)
                audiobooks.add(rel.parts[0])
                
        return sorted(list(audiobooks))
    
    def _select_update(self, dt, events):
        if not self._library_cache:    # FIX: Check if the list is empty!
            self._library_cache = self._scan_library()
            self._scroll_offset = 0

        # Handle mouse wheel scroll for long library lists
        for ev in events:
            if ev.type == pygame.MOUSEWHEEL:
                max_offset = max(0, len(self._library_cache) - self._max_visible_items())
                self._scroll_offset = max(0, min(max_offset, self._scroll_offset - ev.y))

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                # Browse-own-file button
                if self._upload_btn_rect().collidepoint(mx, my):
                    self._open_file_picker()
                    return
                # Library item click
                visible_count = self._max_visible_items()
                for i in range(self._scroll_offset, min(len(self._library_cache),
                                                       self._scroll_offset + visible_count)):
                    rect = self._library_item_rect(i - self._scroll_offset)
                    if rect.collidepoint(mx, my):
                        selected_name = self._library_cache[i]
                        full_path = LIBRARY_DIR / selected_name
                        
                        # --- PLAYLIST LOGIC ---
                        if full_path.is_dir():
                            # If it's a folder, grab all audio files inside it
                            folder_files = []
                            for ext in AUDIO_EXTENSIONS:
                                folder_files.extend(full_path.rglob(f"*{ext}"))
                                folder_files.extend(full_path.rglob(f"*{ext.upper()}"))
                            
                            # Sort them alphanumerically so chapters play in order
                            folder_files = sorted(set(folder_files))
                            self.playlist = [str(f.relative_to(LIBRARY_DIR)).replace("\\", "/") for f in folder_files]
                        else:
                            # If it's a standalone file
                            self.playlist = [selected_name]
                            
                        self.playlist_index = 0
                        self.selected_audio_path = LIBRARY_DIR / self.playlist[0]
                        # ----------------------
                        
                        self._enter_processing = True
                        self.state = PROCESSING
                        return

    def _open_file_picker(self):
        """Open OS-native file dialog. Sets selected_audio_path and transitions."""
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose an audiobook file",
            filetypes=[("Audio files", " ".join(f"*{x}" for x in AUDIO_EXTENSIONS))],
        )
        root.destroy()
        if path:
            self.selected_audio_path = Path(path)
            self._enter_processing = True
            self.state = PROCESSING

    def _max_visible_items(self):
        # Space between top of list (y=200) and the upload button (y=H-110)
        return (H - 110 - 200 - 20) // 44

    def _select_draw(self):
        self.screen.blit(self._bg, (0, 0))

        self._draw_tag(48, 36, "ARCHIVE", color=ACCENT)
        self._draw_tag(W - 48, 36, "ESC  TO  RETURN", align="right", color=TEXT_FAINT)

        # Screen title — italic Georgia for warmth
        title = self.f_title.render("Choose an audiobook", True, TEXT_MAIN)
        self.screen.blit(title, (W // 2 - title.get_width() // 2, 90))

        lib = self._library_cache
        if not lib:
            msg = self.f_body.render(
                f"No audio files found in  {LIBRARY_DIR}", True, TEXT_FAINT)
            self.screen.blit(msg, (W // 2 - msg.get_width() // 2, 220))
        else:
            self._draw_tag(W // 2 - 400, 160,
                       f"LIBRARY  ·  {len(lib)}  FILES", color=TEXT_DIM)

            visible_count = self._max_visible_items()
            for i in range(self._scroll_offset,
                       min(len(lib), self._scroll_offset + visible_count)):
                rect = self._library_item_rect(i - self._scroll_offset)
                hovered = rect.collidepoint(self.mouse_pos)

                # Hover: subtle fill + amber CTA on the right
                if hovered:
                    pygame.draw.rect(self.screen, (24, 20, 32), rect)
                    cta = self.f_tag.render("LOAD  ↗", True, ACCENT)
                    self.screen.blit(cta, (rect.right - cta.get_width() - 18,
                                       rect.centery - cta.get_height() // 2))

                # Bottom border line on every row
                pygame.draw.line(self.screen, TEXT_FAINT,
                             (rect.left, rect.bottom),
                             (rect.right, rect.bottom), 1)

                name_col = TEXT_MAIN if hovered else TEXT_DIM
                name_surf = self.f_item.render(lib[i], True, name_col)
                self.screen.blit(name_surf,
                             (rect.x + 18, rect.centery - name_surf.get_height() // 2))

            # Scroll indicator
            if len(lib) > visible_count:
                scroll_txt = self.f_tag.render(
                    f"{self._scroll_offset + 1}–"
                    f"{min(len(lib), self._scroll_offset + visible_count)}  /  {len(lib)}",
                    True, TEXT_FAINT)
                self.screen.blit(scroll_txt, (W // 2 + 400 - scroll_txt.get_width(), 160))

        # Browse button — ghost style
        btn = self._upload_btn_rect()
        self._draw_ghost_button(btn, "BROWSE  LOCAL  FILE  ↗", btn.collidepoint(self.mouse_pos))


    def _library_item_rect(self, i):
        return pygame.Rect(W // 2 - 400, 200 + i * 44, 800, 40)

    def _upload_btn_rect(self):
        return pygame.Rect(W // 2 - 140, H - 90, 280, 44)
    
    # ── PROCESSING SCREEN ───────────────────────────────────────────────

    def _processing_init(self):
        self._proc_status = "Preparing…"
        self._proc_stage = 0
        self._proc_first = None
        self._proc_second = None
        self._proc_err = None
        self._proc_started_at = self.t

        # Where would first_pipeline put the JSON for this audio?
        audio = Path(self.selected_audio_path)
        expected_json = ANALYSIS_OUTPUT_DIR / f"{audio.stem}_analysis.json"

        print(f"[DEBUG] Looking for file: {expected_json}")
        
        # --- ADD THESE 3 LINES ---
        print(f"[DEBUG] Looking for JSON at: {expected_json}")
        print(f"[DEBUG] Does it exist? {expected_json.exists()}")
        # -------------------------

        if expected_json.exists():
            self._proc_status = "Found cached emotion analysis"
            self._copy_json_to_timeline(expected_json)
            self._start_second_pipeline()
            self._proc_stage = 1
        else:
            print(f"[DEBUG] File not found! Starting pipeline.") # ADD THIS
            self._start_first_pipeline()

    def _start_first_pipeline(self):
        cmd = [arg.format(audio=str(self.selected_audio_path)) for arg in FIRST_PIPELINE_CMD]
        try:
            self._proc_first = subprocess.Popen(
                cmd, 
                stdout=sys.stdout, # CHANGE THIS
                stderr=sys.stderr, # CHANGE THIS
                cwd=str(HERE / "first_pipeline")
            )        
            self._proc_status = "Analysing emotions"
        except FileNotFoundError as e:
            self._proc_err = f"Could not launch first_pipeline: {e}"

    def _start_second_pipeline(self):
        cmd = [arg.format(
              audio=str(self.selected_audio_path),
              json=str(self.timeline_json_path),
              output=str(ENHANCED_AUDIO_PATH),
            ) for arg in SECOND_PIPELINE_CMD]
        ENHANCED_AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Remove any stale enhanced audio from a previous run so we don't
        # accidentally play yesterday's mix if this one fails mid-way.
        if ENHANCED_AUDIO_PATH.exists():
            try: ENHANCED_AUDIO_PATH.unlink()
            except OSError: pass

        print(f"[app] launching second_pipeline → {ENHANCED_AUDIO_PATH.name}")
        try:
            self._proc_second = subprocess.Popen(
                cmd,
                stdout=sys.stdout,        # visible progress (music bed building, ducking, export)
                stderr=sys.stderr,        # visible errors too
                cwd=str(HERE / "second_pipeline"))
            self._proc_status = "Generating soundtrack"
        except FileNotFoundError as e:
            self._proc_err = f"Could not launch second_pipeline: {e}"

    def _copy_json_to_timeline(self, src_path: Path):
        """Wipe emotional_timeline/ and copy the analysis JSON in."""
        EMOTIONAL_TIMELINE.mkdir(parents=True, exist_ok=True)
        for old in EMOTIONAL_TIMELINE.glob("*.json"):
            try: old.unlink()
            except OSError: pass
        dst = EMOTIONAL_TIMELINE / src_path.name
        shutil.copy(src_path, dst)
        self.timeline_json_path = dst
        print(f"[app] timeline JSON ready: {dst.name}")

    def _processing_update(self, dt, events):
        # Initialise once on entry
        if self._enter_processing:
            self._processing_init()
            self._enter_processing = False

        if self._proc_err:
            return

        # Poll first_pipeline
        if self._proc_stage == 0 and self._proc_first is not None:
            ret = self._proc_first.poll()
            if ret is not None:
                if ret == 0:
                    audio = Path(self.selected_audio_path)
                    expected_json = ANALYSIS_OUTPUT_DIR / f"{audio.stem}_analysis.json"
                    if expected_json.exists():
                        self._copy_json_to_timeline(expected_json)
                        self._start_second_pipeline()
                        self._proc_stage = 1
                    else:
                        self._proc_err = (
                            f"first_pipeline finished but no JSON at\n{expected_json}")
                else:
                    err = self._proc_first.stderr.read().decode("utf-8", "ignore")
                    self._proc_err = f"first_pipeline failed (code {ret})\n{err[:400]}"

        # Poll second_pipeline
        elif self._proc_stage == 1 and self._proc_second is not None:
            ret = self._proc_second.poll()
            if ret is not None:
                if ret == 0:
                    print("[app] second_pipeline success — using enhanced audio")
                    self._proc_status = "Ready"
                    self._proc_stage = 2
                    self._playback_init()
                    self.state = PLAYBACK
                else:
                    print(f"[app] second_pipeline failed (code {ret}) — falling back to original audio")
                    self._playback_init()       # ENHANCED_AUDIO_PATH won't exist → playback uses original
                    self.state = PLAYBACK

    def _processing_draw(self):
        self.screen.blit(self._bg, (0, 0))

        self._draw_tag(48, 36, "PROCESSING", color=ACCENT)
        self._draw_tag(W - 48, 36, "NEURAL  SYNTHESIS  IN  PROGRESS",
                   align="right", color=TEXT_FAINT)

        # File name in italic title
        if self.selected_audio_path:
            name = self.f_title.render(Path(self.selected_audio_path).stem, True, TEXT_MAIN)
            self.screen.blit(name, (W // 2 - name.get_width() // 2, int(H * 0.28)))

        # Status line ABOVE the bar
        dots = "." * (1 + int(self.t * 2) % 3)
        if self._proc_err:
            status_text = self._proc_err
            col = ERROR_COL
            font = self.f_body
        else:
            status_text = self._proc_status.upper() + dots
            col = TEXT_DIM
            font = self.f_tag

        for i, line in enumerate(status_text.split("\n")):
            txt = font.render(line, True, col)
            self.screen.blit(txt, (W // 2 - txt.get_width() // 2,
                               int(H * 0.42) + i * (font.get_height() + 4)))

        # Brutalist sliding pulse bar (Gemini's idea — perfect for this aesthetic)
        bar_w, bar_h = 520, 2
        bar_x = W // 2 - bar_w // 2
        bar_y = int(H * 0.50)
        pygame.draw.rect(self.screen, TEXT_FAINT, (bar_x, bar_y, bar_w, bar_h))
        if not self._proc_err:
            pulse_w = 140
            cycle = bar_w + pulse_w
            pulse_x = int((self.t * 260) % cycle) - pulse_w
            p_start = max(bar_x, bar_x + pulse_x)
            p_end   = min(bar_x + bar_w, bar_x + pulse_x + pulse_w)
            if p_end > p_start:
                pygame.draw.rect(self.screen, ACCENT, (p_start, bar_y, p_end - p_start, bar_h))

        # Stage indicators — terminal log style
        stages = [("EMOTION  ANALYSIS",   0),
                ("SOUNDTRACK  MIXING",  1),
                ("READY",               2)]
        y = int(H * 0.62)
        for label, idx in stages:
            done    = self._proc_stage > idx
            running = self._proc_stage == idx and not self._proc_err
            if done:
                prefix, col = "[ DONE ]", TEXT_DIM
            elif running:
                prefix, col = "[ RUN  ]", ACCENT
            else:
                prefix, col = "[ WAIT ]", TEXT_FAINT
            prefix_surf = self.f_tag.render(prefix, True, col)
            label_surf  = self.f_tag.render(label, True, col)
            self.screen.blit(prefix_surf, (W // 2 - 130, y))
            self.screen.blit(label_surf,  (W // 2 - 60, y))
            y += 26

        if self._proc_err:
            self._draw_tag(W // 2, H - 36, "ESC  TO  GO  HOME",
                       align="center", color=TEXT_FAINT)

    # ── PLAYBACK SCREEN ─────────────────────────────────────────────────
    #
    # Tree + weather visualization driven by pygame.mixer.music position.
    # All visual logic is the same as main_tree.py — only the time source
    # changes (audio.get_pos() instead of an internal counter).
    #
    # SPACE      pause/resume the audio
    # ESC        return to START (also stops audio)
    # music end  natural finish → transition to END

    def _playback_init(self):
        """Run once when entering the screen. Loads timeline, sets up tree,
        weather, background; starts the audio file."""
        import emotions
        from tree import EmotionalTree
        from weather import create_weather_system

        # Force the emotions module to look for JSON in OUR folder, by absolute
        # path. Otherwise it uses the hardcoded "../emotional_timeline" which
        # is CWD-relative and unreliable when launching app.py from anywhere.
        emotions.TIMELINE_DIR = str(EMOTIONAL_TIMELINE)

        try:
            # Load your REAL generated JSON instead of the fake one!
            self._pb_timeline = emotions.load_timeline(str(self.timeline_json_path))
        except (Exception) as e:
            print(f"[playback] timeline load failed: {e}")
            self._pb_timeline = [{"t": 0.0,
                                  "emotions": {k: 0.2 for k in emotions.EMOTION_KEYS}}]
        self._pb_total = emotions.get_total_duration(self._pb_timeline)

        # Background image (optional)
        bg_path = HERE / "third_pipeline" / "assets" / "background.png"
        if bg_path.exists():
            try:
                img = pygame.image.load(str(bg_path)).convert()
                if img.get_size() != (W, H):
                    img = pygame.transform.smoothscale(img, (W, H))
                self._pb_bg = img
            except pygame.error as e:
                print(f"[playback] background load failed: {e}")
                self._pb_bg = None
        else:
            self._pb_bg = None
        self._pb_tint_surf = pygame.Surface((W, H), pygame.SRCALPHA)

        # Tree + weather
        self._pb_tree = EmotionalTree(W, H)
        self._pb_tree.draw_ground = (self._pb_bg is None)
        self._pb_weather = create_weather_system(W, H)

        # Pre-render timeline gradient (avoids per-frame scan of timeline)
        self._pb_gradient = self._build_timeline_gradient(emotions)

        # Animation state
        self._pb_cur_ev = {k: 0.2 for k in emotions.EMOTION_KEYS}
        self._pb_frame = 0
        self._pb_last_log_s = -1
        self._pb_paused = False
        self._pb_pause_total_ms = 0
        self._pb_pause_at_ms = 0

        # HUD fonts — lazy-init once
        if not hasattr(self, "_pb_f_sm"):
            self._pb_f_sm = pygame.font.SysFont("inconsolata", 12)
            self._pb_f_lg = pygame.font.SysFont("georgia", 32, italic=True)
            self._pb_f_sub = pygame.font.SysFont("georgia", 20, italic=True)
            self._pb_f_cap = pygame.font.SysFont("georgia", 20, italic=True)

        # Start audio. Prefer the enhanced output from second_pipeline; if
        # that's missing, fall back to the original audiobook file.
        audio_path = ENHANCED_AUDIO_PATH if ENHANCED_AUDIO_PATH.exists() \
                      else self.selected_audio_path
        try:
            pygame.mixer.music.load(str(audio_path))
            pygame.mixer.music.play()
            self._pb_audio_loaded = True
            print(f"[playback] ▶ {Path(audio_path).name}")
        except (pygame.error, TypeError) as e:
            print(f"[playback] audio load failed: {e}")
            self._pb_audio_loaded = False

    def _build_timeline_gradient(self, emotions_mod):
        """Pre-render the colored timeline bar as a static surface."""
        width = W - 48
        surf = pygame.Surface((width, 3))
        if self._pb_total <= 0:
            return surf
        for px in range(width):
            col = emotions_mod.blend_color(
                emotions_mod.get_emotions_at(self._pb_timeline,
                                             px / width * self._pb_total))
            pygame.draw.line(surf, col, (px, 0), (px, 2))
        return surf

    def _playback_time_s(self):
        """Current audio playback position in seconds. Accounts for paused time."""
        if not self._pb_audio_loaded:
            return 0.0
        pos_ms = pygame.mixer.music.get_pos()
        if pos_ms < 0:
            return 0.0
        # get_pos() keeps counting while paused; subtract paused-time offset.
        # While currently paused, freeze at pause-start position.
        if self._pb_paused:
            effective = self._pb_pause_at_ms - self._pb_pause_total_ms
        else:
            effective = pos_ms - self._pb_pause_total_ms
        return max(0.0, effective / 1000.0)

    def _playback_update(self, dt, events):
        import emotions

        if self._enter_playback:
            self._playback_init()
            self._enter_playback = False

        for ev in events:
            # Music finished naturally → END screen
            if ev.type == MUSIC_END_EVENT:
                pygame.mixer.music.stop()
                self.state = END
                return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE and self._pb_audio_loaded:
                    if self._pb_paused:
                        pygame.mixer.music.unpause()
                        # Add elapsed paused time to the running total
                        self._pb_pause_total_ms += (
                            pygame.mixer.music.get_pos() - self._pb_pause_at_ms)
                        self._pb_paused = False
                    else:
                        self._pb_pause_at_ms = pygame.mixer.music.get_pos()
                        pygame.mixer.music.pause()
                        self._pb_paused = True

        # The ESC-handler in run() takes us back to START, but doesn't stop
        # audio. Catch that case here too.
        if self.state == START:
            pygame.mixer.music.stop()
            return

        if self._pb_paused:
            return

        # Drive everything from audio position
        play_t = self._playback_time_s()
        tgt_ev = emotions.get_emotions_at(self._pb_timeline, play_t)
        self._pb_cur_ev = emotions.lerp_emotions(self._pb_cur_ev, tgt_ev, 0.04)

        # One-second emotion log (matches main_tree.py format)
        if int(play_t) > self._pb_last_log_s:
            dom = max(self._pb_cur_ev, key=self._pb_cur_ev.get)
            vals = "  ".join(f"{k[:3]}={self._pb_cur_ev[k]:.2f}"
                             for k in emotions.EMOTION_KEYS)
            print(f"[moodbook] t={play_t:6.1f}s  dom={dom:<8}  {vals}")
            self._pb_last_log_s = int(play_t)

        # Update simulation
        self._pb_tree.update(self._pb_cur_ev, self._pb_frame)
        self._pb_weather.update(self._pb_cur_ev, self._pb_tree, self._pb_frame)
        self._pb_frame += 1

    def _playback_draw(self):
        import emotions
        from tree import blend_leaf_color

        ev = self._pb_cur_ev or {k: 0.2 for k in emotions.EMOTION_KEYS}
        play_t = self._playback_time_s()

        # Background: image + emotion tint, or fall back to gradient sky
        if self._pb_bg is not None:
            self.screen.blit(self._pb_bg, (0, 0))
            tint = self._blend_playback_tint(ev)
            if tint[3] > 0:
                self._pb_tint_surf.fill(tint)
                self.screen.blit(self._pb_tint_surf, (0, 0))
        else:
            self.screen.fill(self._blend_playback_sky(ev))

        # Weather → tree → weather (foreground)
        if self._pb_weather is not None and self._pb_tree is not None:
            self._pb_weather.draw_background(self.screen, ev)
            self._pb_tree.draw(self.screen, ev, self._pb_frame)
            self._pb_weather.draw_foreground(self.screen, ev)

        # HUD: dominant emotion label + weight bars + timeline
        self._draw_playback_hud(ev, play_t)
        self._draw_playback_timeline(play_t)

        # Current sentence transcript (subtitle-style bottom bar)
        text = self._current_transcript(play_t)
        if text:
            self._draw_subtitle(text)

        # Pause overlay
        if self._pb_paused:
            self._draw_pause_overlay()

    # ── Playback helper methods ─────────────────────────────────────────

    _PB_SKY = {
        "joy":     (25, 20, 45), "sadness": (8, 10, 22),
        "anger":   (30, 8, 10),  "calm":    (10, 18, 30),
        "fear":    (12, 10, 18),
    }
    _PB_TINT = {
        "joy":     (255, 170,  70,  35),
        "sadness": ( 55,  75, 160,  55),
        "anger":   (210,  40,  30,  55),
        "calm":    ( 70, 140, 170,  18),
        "fear":    ( 90,  55, 140,  40),
    }

    def _blend_playback_sky(self, ev):
        r = g = b = 0
        for k, w in ev.items():
            sc = self._PB_SKY.get(k, (10, 12, 20))
            r += sc[0] * w; g += sc[1] * w; b += sc[2] * w
        return (int(r), int(g), int(b))

    def _blend_playback_tint(self, ev):
        r = g = b = a = 0.0
        for k, w in ev.items():
            tc = self._PB_TINT.get(k, (0, 0, 0, 0))
            r += tc[0]*w; g += tc[1]*w; b += tc[2]*w; a += tc[3]*w
        return (int(r), int(g), int(b), int(a))

    def _draw_playback_hud(self, ev, play_t):
        import emotions
        from tree import blend_leaf_color
        import colorsys

        # Dominant emotion in big italic at top-right
        dom = max(ev, key=ev.get)
        col = blend_leaf_color(ev)
        lbl = self._pb_f_lg.render(dom, True, col)
        self.screen.blit(lbl, (W - lbl.get_width() - 28, 18))

        # Five thin bars under the label
        bx, by = W - 190, 18 + lbl.get_height() + 8
        for key in emotions.EMOTION_KEYS:
            try:
                h, s, l = emotions.EMOTION_CONFIG[key]["color_hsl"]
                cr, cg, cb = colorsys.hls_to_rgb(h/360, l, s)
                ecol = (int(cr*255), int(cg*255), int(cb*255))
            except (KeyError, AttributeError):
                ecol = (180, 180, 200)
            pygame.draw.rect(self.screen, (20, 20, 30), (bx+44, by, 130, 3), border_radius=1)
            fw = int(130 * ev[key])
            if fw > 0:
                pygame.draw.rect(self.screen, ecol, (bx+44, by, fw, 3), border_radius=1)
            self.screen.blit(self._pb_f_sm.render(key[:3].upper(), True, (60, 58, 72)),
                             (bx, by-1))
            by += 14

        # Bottom-left status line: clock + chapter indicator
        chap = ""
        if self.playlist and len(self.playlist) > 1:
            chap = f"  ·  Ch {self.playlist_index + 1}/{len(self.playlist)}"
        clock_text = f"▶ {play_t:.1f}s / {self._pb_total:.1f}s{chap}"
        self.screen.blit(self._pb_f_sm.render(clock_text, True, (55, 53, 68)),
                         (24, H - 86))
        hint = "SPACE pause  ·  ESC home"
        self.screen.blit(self._pb_f_sm.render(hint, True, (40, 38, 52)),
                         (24, H - 68))



    def _draw_playback_timeline(self, play_t):
        bx, by, bw, bh = 24, H - 44, W - 48, 3
        pygame.draw.rect(self.screen, (14, 14, 24), (bx, by, bw, bh), border_radius=1)
        if self._pb_gradient is not None:
            self.screen.blit(self._pb_gradient, (bx, by))
        if self._pb_total > 0:
            ph = int(bx + (play_t / self._pb_total) * bw)
            pygame.draw.rect(self.screen, (180, 175, 165),
                             (ph-1, by-4, 2, bh+8), border_radius=1)

    def _current_transcript(self, t):
        """Walk timeline keyframes backwards from t and return the most recent
        transcript text. Returns empty string if none."""
        text = ""
        for kf in self._pb_timeline:
            if kf["t"] > t:
                break
            if "transcript" in kf and kf["transcript"]:
                text = kf["transcript"]
        return text

    def _draw_subtitle(self, text):
        if not text:
            return
        lines = _wrap_text(text, self._pb_f_sub, W - 240)[:2]
        if not lines or not any(lines):
            return

        line_h = self._pb_f_sub.get_height()
        total_h = line_h * len(lines) + 6 * (len(lines) - 1)
        ty = H - total_h - 56

        for line in lines:
            # Soft shadow so text stays readable over the bright tree/moon
            shadow = self._pb_f_sub.render(line, True, (0, 0, 0))
            surf   = self._pb_f_sub.render(line, True, (245, 240, 230))
            x = W // 2 - surf.get_width() // 2
            self.screen.blit(shadow, (x + 2, ty + 2))
            self.screen.blit(surf,   (x, ty))
            ty += line_h + 6

    def _draw_pause_overlay(self):
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 100))
        self.screen.blit(veil, (0, 0))
        txt = self._pb_f_lg.render("paused", True, TEXT_MAIN)
        self.screen.blit(txt, (W // 2 - txt.get_width() // 2,
                               H // 2 - txt.get_height() // 2))
        hint = self._pb_f_sm.render("SPACE to resume", True, TEXT_DIM)
        self.screen.blit(hint, (W // 2 - hint.get_width() // 2,
                                H // 2 + txt.get_height() // 2 + 6))
        
    def _cleanup_temp_files(self):
        """Remove the generated enhanced audiobook on exit. Called by atexit
        so it runs no matter how the app closes."""
        # Release pygame's file handle first, otherwise Windows refuses to delete
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass

        if ENHANCED_AUDIO_PATH.exists():
            try:
                ENHANCED_AUDIO_PATH.unlink()
                print(f"[app] cleaned up {ENHANCED_AUDIO_PATH.name}")
            except OSError as e:
                print(f"[app] couldn't delete {ENHANCED_AUDIO_PATH}: {e}")

    # ── END SCREEN ──────────────────────────────────────────────────────

    def _has_next_chapter(self):
        return self.playlist and self.playlist_index < len(self.playlist) - 1

    def _next_btn_rect(self):
        return pygame.Rect(W // 2 - 160, H // 2 - 20, 320, 56)

    def _back_btn_rect(self):
        return pygame.Rect(W // 2 - 160, H // 2 + 60, 320, 56)

    def _end_update(self, dt, events):
        for ev in events:
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                
                # Next Chapter Button Clicked
                if self._has_next_chapter() and self._next_btn_rect().collidepoint(mx, my):
                    self.playlist_index += 1
                    self.selected_audio_path = LIBRARY_DIR / self.playlist[self.playlist_index]
                    self._enter_processing = True
                    self.state = PROCESSING
                    
                # Back to Library Button Clicked
                elif self._back_btn_rect().collidepoint(mx, my):
                    self.state = SELECT

    def _end_draw(self):
        self.screen.blit(self._bg, (0, 0))

        self._draw_tag(48, 36, "SEQUENCE  COMPLETE", color=ACCENT)
        self._draw_tag(W - 48, 36, "ESC  FOR  HOME", align="right", color=TEXT_FAINT)

        # Title — italic
        title = self.f_title.render("Chapter complete", True, TEXT_MAIN)
        self.screen.blit(title, (W // 2 - title.get_width() // 2, int(H * 0.30)))

        # Subtle decoration line
        pygame.draw.line(self.screen, TEXT_FAINT,
                     (W // 2 - 40, int(H * 0.42)),
                     (W // 2 + 40, int(H * 0.42)), 1)

        # Next chapter button
        if self._has_next_chapter():
            next_btn = self._next_btn_rect()
            next_name = Path(self.playlist[self.playlist_index + 1]).name
            display_name = next_name if len(next_name) < 28 else next_name[:25] + "…"
            self._draw_ghost_button(next_btn,
                                f"NEXT  ·  {display_name.upper()}  ↗",
                                next_btn.collidepoint(self.mouse_pos))

        # Back to library — quieter ghost (faint border)
        back_btn = self._back_btn_rect()
        hovered = back_btn.collidepoint(self.mouse_pos)
        if hovered:
            pygame.draw.rect(self.screen, (24, 20, 32), back_btn)
        pygame.draw.rect(self.screen, TEXT_FAINT, back_btn, width=1)
        label_col = TEXT_MAIN if hovered else TEXT_DIM
        txt = self.f_btn.render("BACK  TO  ARCHIVE", True, label_col)
        self.screen.blit(txt, (back_btn.centerx - txt.get_width() // 2,
                           back_btn.centery - txt.get_height() // 2))


    def _next_btn_rect(self):
        return pygame.Rect(W // 2 - 180, int(H * 0.55), 360, 44)

    def _back_btn_rect(self):
        return pygame.Rect(W // 2 - 180, int(H * 0.55) + 60, 360, 44)

if __name__ == "__main__":
    App().run()