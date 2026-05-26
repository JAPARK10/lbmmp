"""
weather.py — Per-emotion atmospheric particle effects (v2)

  joy      → warm sun glow + golden pollen motes
  sadness  → diagonal rain streaks (shorter, softer than v1)
  anger    → hot embers rising, blown by wind, OCCASIONAL LIGHTNING on gusts
  calm     → slow drifting BLUE fireflies (was warm yellow in v1)
  fear     → low-lying soft mist (now a real blurred blob, not a hard circle)
             + faint DEAD BARE TREES in the background

weather.update() now takes the tree directly so it can read gust state
(tree.gust_timer, tree.gust_strength) for lightning sync.

Drawing order: draw_background() BEFORE tree (mist, dead trees, sun glow);
draw_foreground() AFTER tree (fireflies, rain, embers, motes, lightning flash).
"""

import math
import numpy as np
import pygame


def _smooth_ramp(value, start, full):
    if value <= start:
        return 0.0
    if value >= full:
        return 1.0
    return (value - start) / (full - start)


class WeatherSystem:
    def __init__(self, W, H):
        self.W = int(W)
        self.H = int(H)

        # Particle pools
        self.raindrops = []
        self.embers    = []
        self.fireflies = []
        self.motes     = []
        self.mist      = []

        # Lightning state
        self.flash_intensity     = 0.0
        self._was_gust_active    = False
        self._double_flash_timer = 0

        # Cached sun-glow and moon surfaces
        self._sun_cache = None
        self._sun_cache_intensity = None
        self._moon_cache = None
        self._moon_cache_intensity = None

        # Reusable overlay surfaces
        self._rain_surf  = pygame.Surface((W, H), pygame.SRCALPHA)
        self._mist_surf  = pygame.Surface((W, H), pygame.SRCALPHA)
        self._ff_surf    = pygame.Surface((W, H), pygame.SRCALPHA)
        self._ember_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        self._mote_surf  = pygame.Surface((W, H), pygame.SRCALPHA)

        # Lightning flash surface (opaque white, alpha varies per frame)
        self._flash_surf = pygame.Surface((W, H))
        self._flash_surf.fill((255, 255, 255))

        # Pre-rendered soft alpha-gradient blob used for mist particles
        self._mist_blob = self._make_mist_blob()

        # Pre-rendered dead trees, baked to a static surface (fear-only effect)
        self._dead_trees_surf = self._make_dead_trees_surface()

    # ── One-time renders ───────────────────────────────────────────────

    def _make_mist_blob(self, size=200):
        """Smooth radial alpha-gradient blob built with numpy.

        Previous approach (stacking concentric circles with low alpha each)
        produced visible ring artefacts because each circle was a discrete
        alpha step. This version computes per-pixel alpha as a continuous
        falloff from centre → edge, so the blob reads as soft fog instead
        of a series of nested circles."""
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        surf.fill((200, 210, 222, 255))  # uniform mist colour, opaque baseline

        # Per-pixel alpha: smooth quadratic falloff from centre to edge
        yy, xx = np.indices((size, size))
        cx = cy = size / 2
        max_r = size / 2 - 4
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        # Power < 1 → wider soft middle; power > 1 → sharper centre.
        # 1.4 gives a natural fog falloff.
        falloff = np.clip(1.0 - dist / max_r, 0.0, 1.0) ** 1.4
        alpha = (falloff * 230).astype(np.uint8)

        # Overwrite the alpha channel (RGB stays as the fill colour)
        pygame.surfarray.pixels_alpha(surf)[:] = alpha.T
        return surf

    def _build_dead_tree(self, root_x, root_y, height):
        """Returns a list of (sx, sy, ex, ey, thickness) for one bare tree."""
        segments = []
        def grow(x, y, angle, length, thick, depth, max_depth=5):
            ex = x + math.sin(angle) * length
            ey = y - math.cos(angle) * length
            segments.append((x, y, ex, ey, max(1, int(thick))))
            if depth >= max_depth or length < 4:
                return
            n_kids = 2 if np.random.random() > 0.3 else 3
            spread = 0.50
            if n_kids == 2:
                offsets = [-spread, spread]
            else:
                offsets = [-spread, 0.0, spread]
            for da in offsets:
                a = angle + da + np.random.uniform(-0.18, 0.18)
                new_len = length * 0.72 * np.random.uniform(0.80, 1.20)
                grow(ex, ey, a, new_len, thick * 0.70, depth + 1, max_depth)

        trunk_thick = height * 0.025
        grow(root_x, root_y, 0.0, height * 0.40, trunk_thick, 0)
        return segments

    def _make_dead_trees_surface(self):
        """Generate 5 bare trees off to the sides, render once to a static
        surface, modulate its alpha per-frame based on fear intensity."""
        surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        color = (24, 26, 38, 255)

        # Pin a seed so trees don't reshape every restart
        rng = np.random.RandomState(7)

        placements = []
        for _ in range(2):
            x = rng.uniform(self.W * 0.04, self.W * 0.30)
            h = rng.uniform(self.H * 0.18, self.H * 0.28)
            placements.append((x, h))
        for _ in range(2):
            x = rng.uniform(self.W * 0.70, self.W * 0.96)
            h = rng.uniform(self.H * 0.18, self.H * 0.28)
            placements.append((x, h))
        # One smaller distant tree
        x = rng.uniform(self.W * 0.20, self.W * 0.80)
        h = rng.uniform(self.H * 0.10, self.H * 0.16)
        placements.append((x, h))

        ground_y = self.H * 0.83
        for rx, height in placements:
            # Each tree's random branches use its own deterministic seed
            np.random.seed(int(rx * 7919) & 0xFFFF)
            segments = self._build_dead_tree(rx, ground_y, height)
            for (sx, sy, ex, ey, thick) in segments:
                pygame.draw.line(surf, color,
                                 (int(sx), int(sy)), (int(ex), int(ey)), thick)
        return surf

    # ── Spawn helpers ──────────────────────────────────────────────────

    @staticmethod
    def _spawn_count(rate):
        if rate <= 0:
            return 0
        n = int(rate)
        if np.random.random() < (rate - n):
            n += 1
        return n

    def _spawn_raindrop(self):
        self.raindrops.append({
            'x':     np.random.uniform(-self.W * 0.15, self.W * 1.1),
            'y':     np.random.uniform(-40, -5),
            'vx':    np.random.uniform(0.6, 1.4),
            'vy':    np.random.uniform(13.0, 18.0),
            'len':   np.random.uniform(4.0, 7.0),   # shorter than v1 (was 8–16)
            'alpha': np.random.uniform(80, 160),
        })

    def _spawn_ember(self):
        self.embers.append({
            'x':    np.random.uniform(self.W * 0.15, self.W * 0.85),
            'y':    np.random.uniform(self.H * 0.65, self.H * 0.85),
            'vx':   np.random.uniform(-0.8, 0.8),
            'vy':   np.random.uniform(-2.5, -0.8),
            'life': 0,
            'max':  np.random.uniform(50, 140),
            'size': np.random.uniform(1.2, 2.6),
        })

    def _spawn_firefly(self):
        self.fireflies.append({
            'x': np.random.uniform(0, self.W),
            'y': np.random.uniform(self.H * 0.30, self.H * 0.85),
            'vx': np.random.uniform(-0.25, 0.25),
            'vy': np.random.uniform(-0.15, 0.15),
            'phase': np.random.uniform(0, 2 * math.pi),
            'phase_speed': np.random.uniform(0.020, 0.045),
        })

    def _spawn_mote(self):
        self.motes.append({
            'x': np.random.uniform(0, self.W),
            'y': np.random.uniform(self.H * 0.10, self.H * 0.70),
            'vx': np.random.uniform(-0.3, 0.5),
            'vy': np.random.uniform(-0.2, 0.4),
            'life': 0,
            'max': np.random.uniform(200, 500),
            'size': np.random.uniform(1, 2.5),
        })

    def _spawn_mist(self):
        self.mist.append({
            'x':       np.random.uniform(-150, self.W + 150),
            # Spread vertically across the whole scene so mist surrounds the tree
            'y':       np.random.uniform(self.H * 0.25, self.H * 0.95),
            'vx':      np.random.uniform(-0.4, 0.4),
            'vy':      np.random.uniform(-0.15, 0.15),    # vertical drift too
            'size':    np.random.uniform(140, 260),
            'opacity': np.random.uniform(0.35, 0.75),
            # Slow pulse — ghosts "breathe" with subtle size variation
            'pulse':       np.random.uniform(0, 2 * math.pi),
            'pulse_speed': np.random.uniform(0.006, 0.015),
        })

    # ── Update ──────────────────────────────────────────────────────────

    def update(self, ev, tree, frame):
        """Reads tree.wind_x, tree.wind_y, tree.gust_timer, tree.gust_strength."""
        joy     = ev.get('joy',     0.0)
        sadness = ev.get('sadness', 0.0)
        anger   = ev.get('anger',   0.0)
        calm    = ev.get('calm',    0.0)
        fear    = ev.get('fear',    0.0)

        wind_x = tree.wind_x
        wind_y = tree.wind_y

        # ── Rain ────────────────────────────────────────────────────────
        rain_intensity = _smooth_ramp(sadness, 0.20, 1.0)
        for _ in range(self._spawn_count(rain_intensity * 6.0)):
            self._spawn_raindrop()
        for r in self.raindrops:
            r['x'] += r['vx'] + wind_x * 0.10
            r['y'] += r['vy']
        self.raindrops = [r for r in self.raindrops
                          if r['y'] < self.H + 20 and -100 < r['x'] < self.W + 100]

        # ── Embers ──────────────────────────────────────────────────────
        ember_intensity = _smooth_ramp(anger, 0.25, 1.0)
        for _ in range(self._spawn_count(ember_intensity * 3.5)):
            self._spawn_ember()
        for e in self.embers:
            e['x'] += e['vx'] + wind_x * 0.20
            e['y'] += e['vy']
            e['vy'] += 0.012
            e['vy'] *= 0.992
            e['life'] += 1
        self.embers = [e for e in self.embers
                       if e['life'] < e['max'] and -60 < e['x'] < self.W + 60]

        # ── Fireflies (calm or joy) ─────────────────────────────────────
        ff_target = int(calm * 28 + joy * 8)
        if len(self.fireflies) < ff_target and np.random.random() < 0.15:
            self._spawn_firefly()
        while len(self.fireflies) > ff_target + 4:
            self.fireflies.pop(np.random.randint(len(self.fireflies)))
        for f in self.fireflies:
            f['vx'] += np.random.uniform(-0.02, 0.02)
            f['vy'] += np.random.uniform(-0.02, 0.02)
            f['vx'] *= 0.96
            f['vy'] *= 0.96
            f['x'] += f['vx']
            f['y'] += f['vy']
            f['phase'] += f['phase_speed']
            f['x'] = f['x'] % self.W
            if f['y'] < self.H * 0.30: f['y'] = self.H * 0.30
            if f['y'] > self.H * 0.85: f['y'] = self.H * 0.85

        # ── Motes (joy) ─────────────────────────────────────────────────
        mote_intensity = _smooth_ramp(joy, 0.20, 1.0)
        for _ in range(self._spawn_count(mote_intensity * 0.4)):
            self._spawn_mote()
        for m in self.motes:
            m['x'] += m['vx'] + wind_x * 0.05
            m['y'] += m['vy']
            m['life'] += 1
        self.motes = [m for m in self.motes
                      if m['life'] < m['max'] and -20 < m['x'] < self.W + 20]

        # ── Mist (fear) ─────────────────────────────────────────────────
        mist_intensity = _smooth_ramp(fear, 0.15, 1.0)
        mist_target = int(mist_intensity * 22)   # was 14 — more to cover full height
        while len(self.mist) < mist_target:
            self._spawn_mist()
        while len(self.mist) > mist_target + 2:
            self.mist.pop(0)
        for m in self.mist:
            # Ghostly wandering — tiny random acceleration each frame
            m['vx'] += np.random.uniform(-0.06, 0.06)
            m['vy'] += np.random.uniform(-0.03, 0.03)
            # Cap velocity (faster than before but still ghostly, not zippy)
            if m['vx'] >  1.5: m['vx'] =  1.5
            elif m['vx'] < -1.5: m['vx'] = -1.5
            if m['vy'] >  0.6: m['vy'] =  0.6
            elif m['vy'] < -0.6: m['vy'] = -0.6

            m['x'] += m['vx'] + wind_x * 0.03
            m['y'] += m['vy']
            m['pulse'] += m['pulse_speed']

            # Horizontal wrap
            if m['x'] < -m['size']:
                m['x'] = self.W + m['size']
            elif m['x'] > self.W + m['size']:
                m['x'] = -m['size']
            # Soft vertical bounds — bounce velocity back if drifting too high or low
            if m['y'] < self.H * 0.20:
                m['vy'] = abs(m['vy']) * 0.5
            elif m['y'] > self.H * 1.00:
                m['vy'] = -abs(m['vy']) * 0.5

        # ── Lightning (anger gust starts) ───────────────────────────────
        gust_active = tree.gust_timer > 0
        # Edge-trigger: gust just started this frame
        if (gust_active and not self._was_gust_active
                and anger > 0.5 and tree.gust_strength > 2.5
                and self.flash_intensity < 0.1):
            if np.random.random() < 0.35:
                self.flash_intensity = 1.0
                # Sometimes schedule a secondary flash 3–8 frames later
                if np.random.random() < 0.5:
                    self._double_flash_timer = np.random.randint(3, 9)
        self._was_gust_active = gust_active

        # Trigger the scheduled secondary flash
        if self._double_flash_timer > 0:
            self._double_flash_timer -= 1
            if self._double_flash_timer == 0:
                self.flash_intensity = max(self.flash_intensity, 0.55)

        # Decay
        self.flash_intensity *= 0.78
        if self.flash_intensity < 0.02:
            self.flash_intensity = 0.0

    # ── Background pass ────────────────────────────────────────────────

    def draw_background(self, screen, ev):
        joy  = ev.get('joy',  0.0)
        fear = ev.get('fear', 0.0)
        calm = ev.get('calm', 0.0)

        # Dead trees fade in with fear, drawn behind everything else
        if fear > 0.15:
            intensity = _smooth_ramp(fear, 0.15, 1.0)
            self._dead_trees_surf.set_alpha(int(200 * intensity))
            screen.blit(self._dead_trees_surf, (0, 0))

        # Moon (calm) — opposite corner from where the sun appears
        if calm > 0.20:
            self._draw_moon(screen, _smooth_ramp(calm, 0.20, 1.0))

        if joy > 0.20:
            self._draw_sun_glow(screen, _smooth_ramp(joy, 0.20, 1.0))

    # ── Foreground pass ────────────────────────────────────────────────

    def draw_foreground(self, screen, ev):
        joy     = ev.get('joy',     0.0)
        sadness = ev.get('sadness', 0.0)
        anger   = ev.get('anger',   0.0)
        calm    = ev.get('calm',    0.0)
        fear    = ev.get('fear',    0.0)

        # Mist drawn FIRST in the foreground pass so it sits over the tree
        # but UNDER any particle effects (rain, embers, fireflies).
        if fear > 0.15:
            self._draw_mist(screen, _smooth_ramp(fear, 0.15, 1.0))

        if self.motes and joy > 0.20:
            self._draw_motes(screen)
        if self.fireflies and (calm > 0.10 or joy > 0.10):
            self._draw_fireflies(screen, calm, joy)
        if self.raindrops and sadness > 0.15:
            self._draw_rain(screen)
        if self.embers and anger > 0.20:
            self._draw_embers(screen)

        # Lightning sits on TOP of everything (still behind the HUD)
        if self.flash_intensity > 0.02:
            self._flash_surf.set_alpha(int(220 * self.flash_intensity))
            screen.blit(self._flash_surf, (0, 0))

    # ── Renderers ──────────────────────────────────────────────────────

    def _draw_sun_glow(self, screen, intensity):
        key = round(intensity, 1)
        if self._sun_cache_intensity != key:
            max_r = int(self.H * 0.55)
            surf = pygame.Surface((max_r * 2, max_r * 2), pygame.SRCALPHA)
            steps = 22
            for i in range(steps, 0, -1):
                r = int(max_r * i / steps)
                a = int(intensity * 14 * (1 - i / steps))
                if a < 1: continue
                pygame.draw.circle(surf, (255, 215, 130, a), (max_r, max_r), r)
            self._sun_cache = surf
            self._sun_cache_intensity = key
        if self._sun_cache is not None:
            cx = int(self.W * 0.72)
            cy = int(self.H * 0.18)
            r = self._sun_cache.get_width() // 2
            screen.blit(self._sun_cache, (cx - r, cy - r),
                        special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_moon(self, screen, intensity):
        """Pale moon disc with a soft halo. Cached, drawn upper-left."""
        key = round(intensity, 1)
        if self._moon_cache_intensity != key:
            halo_r = int(self.H * 0.20)               # soft halo radius
            disc_r = int(self.H * 0.045)              # solid disc radius
            surf = pygame.Surface((halo_r * 2, halo_r * 2), pygame.SRCALPHA)
            # Outer halo: cool blue-white, soft falloff
            steps = 18
            for i in range(steps, 0, -1):
                r = int(halo_r * i / steps)
                a = int(intensity * 10 * (1 - i / steps))
                if a < 1: continue
                pygame.draw.circle(surf, (170, 195, 235, a), (halo_r, halo_r), r)
            # Inner glow ring (brighter, smaller)
            pygame.draw.circle(surf, (210, 225, 245, int(120 * intensity)),
                               (halo_r, halo_r), int(disc_r * 1.6))
            # Solid moon disc on top
            pygame.draw.circle(surf, (240, 240, 250, int(230 * intensity)),
                               (halo_r, halo_r), disc_r)
            self._moon_cache = surf
            self._moon_cache_intensity = key
        if self._moon_cache is not None:
            cx = int(self.W * 0.22)
            cy = int(self.H * 0.16)
            r = self._moon_cache.get_width() // 2
            screen.blit(self._moon_cache, (cx - r, cy - r),
                        special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_mist(self, screen, intensity):
        """Underlying haze + drifting ghost-blobs with breathing pulse."""
        # Thin uniform haze ties the whole scene together as fog atmosphere
        # (without it, the blobs read as isolated ghost orbs)
        self._mist_surf.fill((0, 0, 0, 0))
        haze_alpha = int(55 * intensity)
        self._mist_surf.fill((200, 210, 222, haze_alpha))
        screen.blit(self._mist_surf, (0, 0))

        # Ghost blobs on top — denser concentrations within the fog
        for m in self.mist:
            # Slow size pulse — ghosts "breathe"
            pulse = 1.0 + 0.10 * math.sin(m['pulse'])
            size = int(m['size'] * pulse)
            if size < 8:
                continue
            scaled = pygame.transform.smoothscale(self._mist_blob, (size, size))
            # 200 (was 230) since the haze layer adds its own opacity
            alpha = int(200 * intensity * m['opacity'])
            scaled.set_alpha(alpha)
            screen.blit(scaled, (int(m['x'] - size / 2),
                                 int(m['y'] - size / 2)))

    def _draw_rain(self, screen):
        self._rain_surf.fill((0, 0, 0, 0))
        for r in self.raindrops:
            x, y = int(r['x']), int(r['y'])
            ex = int(x - r['vx'] * r['len'] * 0.35)
            ey = int(y - r['vy'] * r['len'] * 0.35)
            a = int(r['alpha'])
            pygame.draw.line(self._rain_surf, (180, 200, 220, a),
                             (x, y), (ex, ey), 1)
        screen.blit(self._rain_surf, (0, 0))

    def _draw_embers(self, screen):
        self._ember_surf.fill((0, 0, 0, 0))
        for e in self.embers:
            life_frac = 1.0 - e['life'] / e['max']
            r = max(1, int(e['size'] * (0.4 + life_frac * 0.8)))
            a = int(255 * life_frac)
            cr = 255
            cg = int(80 + 140 * life_frac)
            cb = int(20 * life_frac)
            pygame.draw.circle(self._ember_surf, (cr, cg, cb, a),
                               (int(e['x']), int(e['y'])), r)
        screen.blit(self._ember_surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_fireflies(self, screen, calm, joy):
        """Blue during calm, warm yellow during joy, blended for mixes."""
        total = calm + joy + 1e-3
        cf = calm / total
        jf = joy  / total
        halo_r = int(255 * jf + 110 * cf)
        halo_g = int(220 * jf + 200 * cf)
        halo_b = int(140 * jf + 255 * cf)
        core_r = int(255 * jf + 170 * cf)
        core_g = int(250 * jf + 230 * cf)
        core_b = int(200 * jf + 255 * cf)

        self._ff_surf.fill((0, 0, 0, 0))
        for f in self.fireflies:
            brightness = (math.sin(f['phase']) + 1) * 0.5
            if brightness < 0.08:
                continue
            x, y = int(f['x']), int(f['y'])
            core_rad = max(1, int(1 + brightness * 1.8))
            halo_rad = core_rad * 4
            core_a = int(brightness * 220)
            halo_a = int(brightness * 45)
            pygame.draw.circle(self._ff_surf, (halo_r, halo_g, halo_b, halo_a),
                               (x, y), halo_rad)
            pygame.draw.circle(self._ff_surf, (core_r, core_g, core_b, core_a),
                               (x, y), core_rad)
        screen.blit(self._ff_surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_motes(self, screen):
        self._mote_surf.fill((0, 0, 0, 0))
        for m in self.motes:
            life_frac = m['life'] / m['max']
            fade = math.sin(life_frac * math.pi)
            if fade < 0.05:
                continue
            r = max(1, int(m['size']))
            a = int(180 * fade)
            pygame.draw.circle(self._mote_surf, (255, 230, 180, a),
                               (int(m['x']), int(m['y'])), r)
        screen.blit(self._mote_surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)


def create_weather_system(W, H):
    print(f"[weather] {W}×{H} — rain, embers, fireflies, motes, mist, lightning, dead trees")
    return WeatherSystem(W, H)