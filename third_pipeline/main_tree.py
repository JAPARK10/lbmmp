"""
main_tree.py — MoodBook Tree Visualizer

Controls:
  1=joy  2=sadness  3=anger  4=calm  5=fear
  0=release   SPACE=pause   R=reset   F=fullscreen   Q=quit

Backgrounds:
  Looks for assets/background.png next to this file. If found, it's used
  as the backdrop with a per-emotion colour overlay on top. If missing,
  falls back to the old flat sky colour.
"""

import os
import sys
import math
import pygame
import argparse
import json

from emotions import (load_timeline, get_total_duration, get_emotions_at,
                      lerp_emotions, blend_color, EMOTION_KEYS)
from tree import EmotionalTree, blend_leaf_color
from weather import create_weather_system

TIMELINE_PATH  = "fake_timeline.json"   # fallback used if emotional_timeline/ is empty
SCREEN_W       = 1280
SCREEN_H       = 720
FPS            = 60
LERP_SPEED     = 0.04
BG_IMAGE_PATH  = "assets/background.png"

KEY_MAP = {
    pygame.K_1: "joy",
    pygame.K_2: "sadness",
    pygame.K_3: "anger",
    pygame.K_4: "calm",
    pygame.K_5: "fear",
}


def pure(key):
    v = {k: 0.02 for k in EMOTION_KEYS}
    v[key] = 0.92
    return v


def fake_energy(t):
    return 0.15 + 0.12 * math.sin(t * 2.1) + 0.06 * math.sin(t * 0.37)


# ── Sky colour (fallback when no background image) ──────────────────────────
SKY_COLORS = {
    "joy":     (25, 20, 45),
    "sadness": (8,  10, 22),
    "anger":   (30, 8,  10),
    "calm":    (10, 18, 30),
    "fear":    (12, 10, 18),
}

def blend_sky(ev):
    r, g, b = 0, 0, 0
    for key, w in ev.items():
        sc = SKY_COLORS.get(key, (10, 12, 20))
        r += sc[0] * w; g += sc[1] * w; b += sc[2] * w
    return (int(r), int(g), int(b))


# ── Emotion-driven colour wash that sits on top of the background ───────────
# Each emotion contributes a coloured translucent overlay. Joy warms the
# scene with gold; sadness chills it with blue; anger floods it red; calm
# is nearly invisible; fear adds a purple cast. Alpha is intentionally low
# so the underlying landscape still reads through.
TINT_COLORS = {
    "joy":     (255, 170,  70,  35),   # warm gold
    "sadness": ( 55,  75, 160,  55),   # cold blue
    "anger":   (210,  40,  30,  55),   # red wash
    "calm":    ( 70, 140, 170,  18),   # gentle cyan (subtle)
    "fear":    ( 90,  55, 140,  40),   # purple
}

def blend_tint(ev):
    """Weighted blend of the per-emotion tints. Returns (R, G, B, A) for an overlay."""
    r = g = b = a = 0.0
    for key, w in ev.items():
        tc = TINT_COLORS.get(key, (0, 0, 0, 0))
        r += tc[0] * w; g += tc[1] * w
        b += tc[2] * w; a += tc[3] * w
    return (int(r), int(g), int(b), int(a))


# ── Asset loading ───────────────────────────────────────────────────────────
def load_background(path, W, H):
    """Load and scale background image. Returns None if missing or unreadable."""
    if not os.path.exists(path):
        print(f"[moodbook] no background image at '{path}' — using flat sky")
        return None
    try:
        img = pygame.image.load(path).convert()
        if img.get_size() != (W, H):
            img = pygame.transform.smoothscale(img, (W, H))
        print(f"[moodbook] loaded background '{path}'")
        return img
    except pygame.error as e:
        print(f"[moodbook] failed to load background: {e}")
        return None


# ── HUD ─────────────────────────────────────────────────────────────────────
def draw_hud(screen, f_sm, f_lg, ev, t, playing, override, W, H):
    from emotions import EMOTION_CONFIG
    import colorsys

    dom = max(ev, key=ev.get)
    col = blend_leaf_color(ev)
    lbl = f_lg.render(dom, True, col)
    screen.blit(lbl, (W - lbl.get_width() - 28, 18))

    bx, by = W - 190, 18 + lbl.get_height() + 8
    for key in EMOTION_KEYS:
        h, s, l = EMOTION_CONFIG[key]["color_hsl"]
        cr, cg, cb = colorsys.hls_to_rgb(h/360, l, s)
        ecol = (int(cr*255), int(cg*255), int(cb*255))
        pygame.draw.rect(screen, (20, 20, 30), (bx+44, by, 130, 3), border_radius=1)
        fw = int(130 * ev[key])
        if fw > 0:
            pygame.draw.rect(screen, ecol, (bx+44, by, fw, 3), border_radius=1)
        screen.blit(f_sm.render(key[:3].upper(), True, (160, 158, 172)), (bx, by-1))
        by += 14

    # Close button (Top Left)
    close_rect = pygame.Rect(20, 20, 70, 30)
    pygame.draw.rect(screen, (60, 40, 50), close_rect, border_radius=4)
    close_text = f_sm.render("CLOSE X", True, (240, 200, 200))
    screen.blit(close_text, (close_rect.x + (close_rect.w - close_text.get_width())//2, close_rect.y + (close_rect.h - close_text.get_height())//2))
    
    # Play/Pause button
    btn_rect = pygame.Rect(24, H - 86, 70, 30)
    pygame.draw.rect(screen, (40, 50, 60) if playing else (60, 50, 40), btn_rect, border_radius=4)
    btn_text = f_sm.render("PAUSE" if playing else "PLAY", True, (200, 220, 240))
    screen.blit(btn_text, (btn_rect.x + (btn_rect.w - btn_text.get_width())//2, btn_rect.y + (btn_rect.h - btn_text.get_height())//2))

    ov = f"  [{override}]" if override else ""
    screen.blit(
        f_sm.render(f"t={t:.1f}s{ov}", True, (155, 153, 168)),
        (104, H - 80)
    )
    screen.blit(
        f_sm.render(
            "1-5=emotion  0=release  SPACE=pause  R=reset  F=full",
            True, (120, 118, 132)),
        (24, H - 108))


def draw_timeline(screen, gradient_surf, t, total, W, H):
    bx, by, bw, bh = 24, H - 44, W - 48, 3
    pygame.draw.rect(screen, (14, 14, 24), (bx, by, bw, bh), border_radius=1)
    # Pre-rendered gradient — single blit instead of ~1200 get_emotions_at calls
    screen.blit(gradient_surf, (bx, by))
    ph = int(bx + (t / total) * bw)
    pygame.draw.rect(screen, (180, 175, 165), (ph-1, by-4, 2, bh+8), border_radius=1)


def build_timeline_gradient(timeline, total, width, height=3):
    """Pre-render the colored timeline bar as a static surface so the gradient
    isn't recomputed every frame. This was the single biggest cost in the main
    loop (~1200 get_emotions_at calls per frame, each scanning the timeline)."""
    width = max(1, int(width))
    surf = pygame.Surface((width, height))
    for px in range(width):
        col = blend_color(get_emotions_at(timeline, px / width * total))
        pygame.draw.line(surf, col, (px, 0), (px, height - 1))
    return surf


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MoodBook Tree Visualizer")
    parser.add_argument("--audio", type=str, help="Path to audio file to play", default=None)
    parser.add_argument("--timeline", type=str, help="Path to timeline JSON file", default=None)
    args = parser.parse_args()

    pygame.init()
    if args.audio:
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(args.audio)
            print(f"[moodbook] Loaded audio: {args.audio}")
        except Exception as e:
            print(f"[moodbook] Failed to load audio: {e}")
            args.audio = None

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("MoodBook — Tree")
    clock  = pygame.time.Clock()
    W, H   = SCREEN_W, SCREEN_H

    try:
        f_sm = pygame.font.SysFont("inconsolata", 12)
        f_lg = pygame.font.SysFont("georgia", 32, italic=True)
    except Exception:
        f_sm = pygame.font.SysFont("monospace", 11)
        f_lg = pygame.font.SysFont("serif", 30)

    try:
        if args.timeline:
            print(f"[moodbook] Forcing timeline file: {args.timeline}")
            with open(args.timeline, "r") as f:
                data = json.load(f)
            from emotions import _convert_sentences_to_keyframes
            if isinstance(data, dict) and "sentences" in data:
                timeline = _convert_sentences_to_keyframes(data["sentences"])
            elif isinstance(data, list):
                data.sort(key=lambda x: x["t"])
                timeline = data
            else:
                raise ValueError("Unrecognized timeline format")
        else:
            timeline = load_timeline(TIMELINE_PATH)
        print(f"[moodbook] {len(timeline)} keyframes")
    except Exception as e:
        print(f"[moodbook] Timeline load error: {e}")
        timeline = [{"t": 0, "emotions": {k: 0.2 for k in EMOTION_KEYS}}]
    total_duration = get_total_duration(timeline)
    timeline_gradient = build_timeline_gradient(timeline, total_duration, W - 48)

    # ── Background image (optional) ───────────────────────────────────
    bg_image = load_background(BG_IMAGE_PATH, W, H)
    # Reusable overlay surface for emotion tint
    tint_surf = pygame.Surface((W, H), pygame.SRCALPHA)

    tree = EmotionalTree(W, H)
    # If we have a background image, skip the tree's procedural ground rect
    tree.draw_ground = (bg_image is None)

    weather = create_weather_system(W, H)

    cur_ev     = {k: 0.2 for k in EMOTION_KEYS}
    tgt_ev     = get_emotions_at(timeline, 0.0)
    play_t     = 0.0
    audio_start_offset = 0.0
    playing    = True
    override   = None
    frame      = 0
    fullscreen = False
    dragging_timeline = False

    print("[moodbook] tree — 1-5 emotions, Q quit")

    if args.audio:
        pygame.mixer.music.play(start=0.0)

    while True:
        dt    = clock.tick(FPS) / 1000.0
        frame += 1

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    # check close
                    if pygame.Rect(20, 20, 70, 30).collidepoint(ev.pos):
                        pygame.quit(); sys.exit()
                    # check play/pause
                    if pygame.Rect(24, H - 86, 70, 30).collidepoint(ev.pos):
                        playing = not playing
                        if args.audio:
                            if playing: pygame.mixer.music.unpause()
                            else: pygame.mixer.music.pause()
                    # check timeline
                    bx, by, bw, bh = 24, H - 44, W - 48, 3
                    if pygame.Rect(bx, by - 15, bw, bh + 30).collidepoint(ev.pos):
                        dragging_timeline = True
                        frac = max(0.0, min(1.0, (ev.pos[0] - bx) / bw))
                        play_t = frac * total_duration
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1 and dragging_timeline:
                    dragging_timeline = False
                    bx, by, bw, bh = 24, H - 44, W - 48, 3
                    frac = max(0.0, min(1.0, (ev.pos[0] - bx) / bw))
                    play_t = frac * total_duration
                    if args.audio:
                        audio_start_offset = play_t
                        pygame.mixer.music.play(start=play_t)
                        if not playing:
                            pygame.mixer.music.pause()
            elif ev.type == pygame.MOUSEMOTION:
                if dragging_timeline:
                    bx, by, bw, bh = 24, H - 44, W - 48, 3
                    frac = max(0.0, min(1.0, (ev.pos[0] - bx) / bw))
                    play_t = frac * total_duration
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                elif ev.key == pygame.K_SPACE:
                    playing = not playing
                    if args.audio:
                        if playing:
                            pygame.mixer.music.unpause()
                        else:
                            pygame.mixer.music.pause()
                elif ev.key == pygame.K_r:
                    play_t = 0.0; audio_start_offset = 0.0; playing = True; override = None
                    if args.audio:
                        pygame.mixer.music.play(start=0.0)
                    tree = EmotionalTree(W, H)
                    tree.draw_ground = (bg_image is None)
                    weather = create_weather_system(W, H)
                elif ev.key == pygame.K_0:
                    override = None
                elif ev.key == pygame.K_f:
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
                        W, H = screen.get_size()
                    else:
                        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
                        W, H = SCREEN_W, SCREEN_H
                    # Rescale the background and overlay to the new size
                    bg_image = load_background(BG_IMAGE_PATH, W, H)
                    tint_surf = pygame.Surface((W, H), pygame.SRCALPHA)
                    tree = EmotionalTree(W, H)
                    tree.draw_ground = (bg_image is None)
                    weather = create_weather_system(W, H)
                    timeline_gradient = build_timeline_gradient(timeline, total_duration, W - 48)
                elif ev.key in KEY_MAP:
                    override = KEY_MAP[ev.key]
                    tgt_ev = pure(override)
                    print(f"[moodbook] {override}")

        if playing and not dragging_timeline:
            if args.audio and pygame.mixer.music.get_busy():
                play_t = audio_start_offset + (pygame.mixer.music.get_pos() / 1000.0)
            else:
                if not args.audio:
                    play_t = (play_t + dt) % total_duration
        if override is None:
            tgt_ev = get_emotions_at(timeline, play_t)
        cur_ev = lerp_emotions(cur_ev, tgt_ev, LERP_SPEED)

        # Log emotions once per second (at 60 FPS that's every 60 frames).
        # Shows time, dominant emotion, and all 5 weights so you can verify
        # the timeline JSON is being read correctly.
        if frame % FPS == 0:
            dom  = max(cur_ev, key=cur_ev.get)
            vals = "  ".join(f"{k[:3]}={cur_ev[k]:.2f}" for k in EMOTION_KEYS)
            print(f"[moodbook] t={play_t:6.1f}s  dom={dom:<8}  {vals}")

        # ── Update tree, then weather (weather reads tree's wind + gusts) ─
        tree.update(cur_ev, frame)
        weather.update(cur_ev, tree, frame)

        # ── Render background ────────────────────────────────────────────
        if bg_image is not None:
            screen.blit(bg_image, (0, 0))
            # Emotion-driven colour wash on top
            tint = blend_tint(cur_ev)
            if tint[3] > 0:
                tint_surf.fill(tint)
                screen.blit(tint_surf, (0, 0))
        else:
            screen.fill(blend_sky(cur_ev))

        # ── Background weather: sun glow, low-lying mist ────────────────
        weather.draw_background(screen, cur_ev)

        # ── Tree on top of the backdrop ──────────────────────────────────
        tree.draw(screen, cur_ev, frame)

        # ── Foreground weather: rain, embers, fireflies, motes ──────────
        weather.draw_foreground(screen, cur_ev)

        draw_timeline(screen, timeline_gradient, play_t, total_duration, W, H)
        draw_hud(screen, f_sm, f_lg, cur_ev, play_t, playing, override, W, H)

        pygame.display.flip()


if __name__ == "__main__":
    main()