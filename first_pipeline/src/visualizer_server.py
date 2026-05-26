import http.server
import socketserver
import os
import json
import webbrowser
import sys
from pathlib import Path

PORT = 8080

# Get the true root directory of the pipeline
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent.resolve()

class EmotionVizHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Force initialization from root directory so paths match data/raw
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def do_GET(self):
        # Route API requests
        if self.path == "/":
            # Redirect root path directly to visualizer file
            self.path = "/src/visualizer.html"
            return super().do_GET()
            
        elif self.path == "/api/tracks":
            try:
                raw_dir = ROOT_DIR / "data" / "raw"
                proc_dir = ROOT_DIR / "data" / "processed"
                
                audio_exts = ('.wav', '.mp3', '.m4a', '.flac')
                tracks = []
                
                if raw_dir.exists():
                    # Find all audio files in raw
                    for file_name in os.listdir(raw_dir):
                        if file_name.lower().endswith(audio_exts):
                            p = Path(file_name)
                            stem = p.stem
                            
                            # Check if analysis exists
                            json_file = f"{stem}_analysis.json"
                            json_path = proc_dir / json_file
                            
                            if json_path.exists():
                                tracks.append({
                                    "name": file_name,
                                    "audio_url": f"data/raw/{file_name}",
                                    "json_url": f"data/processed/{json_file}"
                                })
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(tracks).encode())
                return
                
            except Exception as e:
                self.send_error(500, f"Server Error: {str(e)}")
                return

        # Fall back to standard file server for local files
        return super().do_GET()

def run_server():
    # Check for port availability, fail gracefully
    handler = EmotionVizHandler
    try:
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            print("="*60)
            print(f"🚀 EMOTION PIPELINE VISUALIZER IS READY")
            print(f"📍 URL: http://localhost:{PORT}")
            print(f"📂 Root: {ROOT_DIR}")
            print("="*60)
            print("[*] Launching auto-browser window...")
            
            webbrowser.open(f"http://localhost:{PORT}")
            
            print("\n[*] Press Ctrl+C in terminal to stop server.")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 98 or e.errno == 10048:
             print(f"[!] Port {PORT} is in use. Close the existing server first.")
        else:
             raise e

if __name__ == "__main__":
    run_server()
