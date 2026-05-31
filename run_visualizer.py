import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

# Paths
# Now that the script is in repository/, its parent is repository/
REPOSITORY_DIR = Path(os.path.abspath(__file__)).parent
THIRD_PIPELINE_DIR = REPOSITORY_DIR / "third_pipeline"

# Only search the second_pipeline/output directory for audio
AUDIO_DIR = REPOSITORY_DIR / "second_pipeline" / "output"

# Search for timeline JSONs in first and third pipelines
JSON_DIRS = [
    REPOSITORY_DIR / "first_pipeline" / "data" / "processed",
    REPOSITORY_DIR / "third_pipeline" / "emotional_timeline"
]

# Modern Dark Theme Colors
BG_COLOR = "#1e1e2e"
CARD_COLOR = "#313244"
TEXT_COLOR = "#cdd6f4"
SUBTEXT_COLOR = "#a6adc8"
ACCENT_COLOR = "#89b4fa"
BTN_HOVER = "#b4befe"

def get_valid_pairs():
    # 1. Collect all json files
    json_files = []
    for d in JSON_DIRS:
        if d.exists() and d.is_dir():
            for f in os.listdir(d):
                if f.lower().endswith('.json'):
                    json_files.append(d / f)
                    
    # 2. Collect audio files that have a matching json
    valid_pairs = []
    if AUDIO_DIR.exists() and AUDIO_DIR.is_dir():
        for f in os.listdir(AUDIO_DIR):
            if f.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
                audio_path = AUDIO_DIR / f
                base_audio = os.path.splitext(f)[0]
                
                # Find matching json
                matched_json = None
                for jp in json_files:
                    # Match if the json name contains the audio name or vice versa
                    json_base = jp.name.replace("_analysis.json", "").replace(".json", "")
                    if json_base in base_audio or base_audio in json_base:
                        matched_json = jp
                        break
                        
                if matched_json:
                    valid_pairs.append({
                        "display": f"🎵 {f}",
                        "audio_path": audio_path,
                        "json_path": matched_json
                    })
                    
    return sorted(valid_pairs, key=lambda x: x["display"])

class VisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MoodBook Visualizer")
        self.root.geometry("650x350")
        self.root.configure(bg=BG_COLOR)
        
        title_font = ("Segoe UI", 18, "bold")
        header_font = ("Segoe UI", 12, "bold")
        normal_font = ("Segoe UI", 11)
        
        main_frame = tk.Frame(root, bg=BG_COLOR, padx=30, pady=30)
        main_frame.pack(fill='both', expand=True)

        title_lbl = tk.Label(main_frame, text="MoodBook Pygame Visualizer", font=title_font, bg=BG_COLOR, fg=ACCENT_COLOR)
        title_lbl.pack(pady=(0, 5))
        
        subtitle_lbl = tk.Label(main_frame, text="Select a generated audiobook. Only files with valid emotion data are listed.", font=normal_font, bg=BG_COLOR, fg=SUBTEXT_COLOR)
        subtitle_lbl.pack(pady=(0, 25))

        # Audio Selection Card
        audio_card = tk.Frame(main_frame, bg=CARD_COLOR, padx=20, pady=15, highlightbackground="#45475a", highlightthickness=1)
        audio_card.pack(fill='x', pady=10)
        
        tk.Label(audio_card, text="Enhanced Audiobook", font=header_font, bg=CARD_COLOR, fg=TEXT_COLOR).pack(anchor='w', pady=(0, 5))
        
        self.valid_pairs = get_valid_pairs()
        self.display_names = [p["display"] for p in self.valid_pairs]
        self.audio_var = tk.StringVar()
        
        if self.display_names:
            self.audio_var.set(self.display_names[0])
        else:
            self.audio_var.set("No valid audio/JSON pairs found in output directory")
            
        self.audio_combo = ttk.Combobox(audio_card, textvariable=self.audio_var, values=self.display_names, state="readonly", font=normal_font)
        self.audio_combo.pack(fill='x')

        # Launch Button
        self.launch_btn = tk.Button(main_frame, text="🚀 Launch Pygame", font=("Segoe UI", 12, "bold"), 
                                    bg=ACCENT_COLOR, fg=BG_COLOR, activebackground=BTN_HOVER, 
                                    activeforeground=BG_COLOR, bd=0, padx=20, pady=10, cursor="hand2",
                                    command=self.launch)
        self.launch_btn.pack(pady=25)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox', fieldbackground=BG_COLOR, background=CARD_COLOR, foreground=TEXT_COLOR, arrowcolor=TEXT_COLOR)

    def launch(self):
        selection = self.audio_var.get()
        
        if not selection or "No valid audio/JSON pairs found" in selection:
            messagebox.showerror("Error", "Please select a valid audio file.")
            return
            
        pair = next((p for p in self.valid_pairs if p["display"] == selection), None)
        if not pair:
            messagebox.showerror("Error", "Failed to resolve file paths.")
            return
            
        main_tree_script = THIRD_PIPELINE_DIR / "main_tree.py"
        
        if not main_tree_script.exists():
            messagebox.showerror("Error", f"Could not find {main_tree_script}")
            return
        
        cmd = [
            sys.executable,
            str(main_tree_script),
            "--audio", str(pair["audio_path"]),
            "--timeline", str(pair["json_path"])
        ]
        
        try:
            subprocess.Popen(cmd, cwd=str(THIRD_PIPELINE_DIR))
        except Exception as e:
            messagebox.showerror("Launch Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = VisualizerApp(root)
    root.mainloop()
