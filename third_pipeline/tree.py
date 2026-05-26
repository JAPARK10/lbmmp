"""
tree.py — Emotional tree with wind-driven leaves (v3)

Wind is now a first-class physical force. Emotion shapes the wind,
and the wind shapes the tree and leaves.

  calm    : ~no wind. Leaves grow steadily, rarely fall.
  joy     : gentle wandering breeze. Leaves grow fast, occasional drift.
  sadness : ~no wind. Leaves stop growing and fall vertically (gravity only).
  anger   : strong gusting wind, sudden direction swings. Leaves are torn off
            and fly sideways across the canvas.
  fear    : high-frequency jittery wind, no clear direction. Leaves tremble.
"""

import numpy as np
import pygame
import colorsys
import math
from emotions import EMOTION_CONFIG


# ── Emotion → leaf colour mapping ────────────────────────────────────────────
LEAF_COLORS = {
    "joy":     (0.13, 0.95, 0.58),
    "sadness": (0.58, 0.45, 0.38),
    "anger":   (0.02, 0.90, 0.48),
    "calm":    (0.34, 0.70, 0.42),
    "fear":    (0.11, 0.30, 0.72),
}


def blend_leaf_color(ev):
    h, s, l = 0.0, 0.0, 0.0
    for key, w in ev.items():
        ch, cs, cl = LEAF_COLORS.get(key, (0.3, 0.5, 0.4))
        h += ch * w; s += cs * w; l += cl * w
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r*255), int(g*255), int(b*255))


# ── Branch ───────────────────────────────────────────────────────────────────

class Branch:
    def __init__(self, start, angle, length, thickness, depth, max_depth,
                 parent=None, thick_shrink=0.70):
        self.start      = np.array(start, dtype=float)
        self.base_angle = angle
        self.length     = length
        self.thickness  = thickness
        self.depth      = depth
        self.is_tip     = (depth >= max_depth)
        self.parent     = parent

        # Tapered end thickness:
        #   non-tip → matches child's start thickness (smooth taper through joint)
        #   tip     → tapers to a fine point (twig)
        if self.is_tip:
            self.end_thickness = max(0.4, thickness * 0.22)
        else:
            self.end_thickness = max(1.0, thickness * thick_shrink)

        # Organic bend: each branch curves slightly perpendicular to its axis.
        # Smaller branches bend a bit more (younger growth is more flexible).
        max_bend = 0.18 + 0.10 * (depth / max_depth)
        self.bend = float(np.random.uniform(-max_bend, max_bend))

        self.end = self._calc_end(angle)

    def _calc_end(self, angle):
        return self.start + np.array([
            math.sin(angle) * self.length,
            -math.cos(angle) * self.length
        ])


# ── Leaf ─────────────────────────────────────────────────────────────────────

class Leaf:
    ATTACHED = 0
    FALLING  = 1
    GROUND   = 2
    GROWING  = 3

    def __init__(self, x, y, branch_tip):
        self.x = x
        self.y = y
        self.branch = branch_tip
        self.vx = 0.0
        self.vy = 0.0
        self.state = Leaf.GROWING

        # Growth
        self.size       = 0.1
        self.max_size   = np.random.uniform(5.5, 9.5)
        self.grow_speed = np.random.uniform(0.0025, 0.006)

        # Rotation
        self.rotation  = np.random.uniform(0, 2*math.pi)
        self.rot_speed = np.random.uniform(-0.012, 0.012)

        # Personal phase offsets
        self.sway_phase      = np.random.uniform(0, 2*math.pi)
        self.tremble_phase_x = np.random.uniform(0, 2*math.pi)
        self.tremble_phase_y = np.random.uniform(0, 2*math.pi)

        # Baked colour variation (no per-frame strobe)
        self.jitter_r   = int(np.random.uniform(-18, 18))
        self.jitter_g   = int(np.random.uniform(-18, 18))
        self.jitter_b   = int(np.random.uniform(-10, 10))
        self.brightness = np.random.uniform(0.82, 1.08)
        self.aspect     = np.random.uniform(1.6, 2.4)

        # How much THIS leaf is affected by wind (some are looser than others)
        self.wind_factor = np.random.uniform(0.7, 1.3)

        self.alpha    = 255.0
        self.ground_y = 0

    def update(self, ev, frame, wind_x, wind_y, wind_mag):
        joy     = ev.get("joy", 0)
        sadness = ev.get("sadness", 0)
        anger   = ev.get("anger", 0)
        calm    = ev.get("calm", 0)
        fear    = ev.get("fear", 0)

        # ── Compute fall chance up front ────────────────────────────────
        # Thresholds so background "noise" levels of an emotion don't contribute.
        #
        #   sadness  → tree dies in ~30s (2x faster than before; full extinction)
        #   wind     → only STRONG wind (>1.5) tears leaves. Raised from 0.6 so
        #              joy's stronger breeze (now ~1.1 wind_mag) doesn't strip.
        #   fear     → small contribution, mostly tremble
        #   calm     → virtually never falls
        #   joy_dance → a few leaves drop into the breeze to swirl playfully
        gravity_fall   = max(0.0, sadness - 0.1) * 0.0015
        fear_fall      = max(0.0, fear    - 0.2) * 0.00008
        calm_attrition = max(0.0, calm    - 0.1) * 0.000005
        joy_dance      = max(0.0, joy     - 0.5) * 0.00005
        strong_wind    = max(0.0, wind_mag - 1.5)
        wind_tear      = strong_wind * 0.00015
        fall_chance = (gravity_fall + wind_tear + fear_fall
                       + calm_attrition + joy_dance)

        # ── GROWING ──────────────────────────────────────────────────────
        if self.state == Leaf.GROWING:
            # Track the branch tip (same logic as ATTACHED) — otherwise
            # growing leaves stay frozen at their birth position while the
            # tree sways, producing visible "ghost" leaves during anger.
            base_x = self.branch.end[0]
            base_y = self.branch.end[1]
            wind_sway = wind_x * self.wind_factor * 2.5
            wind_lift = wind_y * self.wind_factor * 1.5
            rustle    = math.sin(frame * 0.025 + self.sway_phase) * 1.5
            self.x = base_x + wind_sway + rustle
            self.y = base_y + wind_lift
            if fear > 0.15:
                tx = math.sin(frame * 0.35 + self.tremble_phase_x)
                ty = math.cos(frame * 0.41 + self.tremble_phase_y)
                self.x += tx * fear * 1.2
                self.y += ty * fear * 0.8

            # Joy → fast growth; calm → very gentle; sad/anger → frozen
            boost   = joy * 1.5 + calm * 0.3
            halt    = max(0.0, 1.0 - sadness*1.5 - anger*1.2 - fear*0.4)
            grow_rate = self.grow_speed * (1 + boost) * halt
            if grow_rate > 0:
                self.size = min(self.size + grow_rate, self.max_size)
            if self.size >= self.max_size * 0.95:
                self.state = Leaf.ATTACHED

            # CRITICAL: growing leaves can also fall — so sadness actually
            # strips the WHOLE canopy, not just mature leaves.
            if np.random.random() < fall_chance:
                self.state = Leaf.FALLING
                self.vx = wind_x * self.wind_factor * 0.4 + np.random.uniform(-0.3, 0.3)
                self.vy = np.random.uniform(0.1, 0.4)

        # ── ATTACHED ─────────────────────────────────────────────────────
        elif self.state == Leaf.ATTACHED:
            if np.random.random() < fall_chance:
                self.state = Leaf.FALLING
                self.vx = wind_x * self.wind_factor * 0.4 + np.random.uniform(-0.3, 0.3)
                self.vy = np.random.uniform(0.1, 0.4)

            # Position anchored to branch tip + wind deflection + rustle
            base_x = self.branch.end[0]
            base_y = self.branch.end[1]
            wind_sway = wind_x * self.wind_factor * 2.5
            wind_lift = wind_y * self.wind_factor * 1.5
            rustle    = math.sin(frame * 0.025 + self.sway_phase) * 1.5
            self.x = base_x + wind_sway + rustle
            self.y = base_y + wind_lift

            # Fear adds smooth high-freq tremble on top
            if fear > 0.15:
                tx = math.sin(frame * 0.35 + self.tremble_phase_x)
                ty = math.cos(frame * 0.41 + self.tremble_phase_y)
                self.x += tx * fear * 1.2
                self.y += ty * fear * 0.8

        # ── FALLING ──────────────────────────────────────────────────────
        elif self.state == Leaf.FALLING:
            # Gravity always present
            self.vy += 0.025 + sadness * 0.01

            # Wind continuously pushes flying leaves —
            # this is what makes anger leaves fly across the canvas.
            self.vx += wind_x * 0.04
            self.vy += wind_y * 0.025

            # Flutter
            self.vx += math.sin(frame * 0.06 + self.sway_phase) * 0.12

            # Air resistance + vertical cap
            self.vx *= 0.985
            self.vy = min(self.vy, 2.5)

            self.x += self.vx
            self.y += self.vy
            # Wind spins them as they fly
            self.rotation += self.rot_speed + self.vx * 0.02

            # Slow alpha decay during flight so they hold their colour
            self.alpha -= 0.06

            if self.y >= self.ground_y:
                self.y = self.ground_y
                self.state = Leaf.GROUND
                self.vx = 0
                self.vy = 0

        # ── GROUND ───────────────────────────────────────────────────────
        elif self.state == Leaf.GROUND:
            self.alpha -= 0.12
            # Strong wind drags ground leaves a little
            self.x += wind_x * 0.3


# ── Tree ─────────────────────────────────────────────────────────────────────

class EmotionalTree:
    def __init__(self, screen_w, screen_h):
        self.W = screen_w
        self.H = screen_h
        self.root_x = screen_w / 2
        self.root_y = screen_h * 0.82
        self.ground_y = screen_h * 0.85

        # Geometry
        self.max_depth     = 8
        self.trunk_length  = screen_h * 0.17
        self.trunk_thick   = 9
        self.branch_shrink = 0.74
        self.thick_shrink  = 0.70
        self.spread_angle  = 0.46

        self.branches = []
        self.tips     = []
        self._build(
            start=(self.root_x, self.root_y),
            angle=0,
            length=self.trunk_length,
            thickness=self.trunk_thick,
            depth=0
        )

        # Pre-populate canopy
        self.leaves = []
        for tip in self.tips:
            for _ in range(np.random.randint(4, 8)):
                lf = Leaf(tip.end[0], tip.end[1], tip)
                lf.ground_y = self.ground_y + np.random.uniform(0, 30)
                lf.size = np.random.uniform(lf.max_size * 0.6, lf.max_size)
                if lf.size >= lf.max_size * 0.95:
                    lf.state = Leaf.ATTACHED
                self.leaves.append(lf)

        # ── Wind state ────────────────────────────────────────────────
        # wind_base_*  : smoothed wind from emotion-driven target
        # wind_*       : wind_base + fear jitter (what leaves/branches see)
        self.wind_base_x   = 0.0
        self.wind_base_y   = 0.0
        self.wind_x        = 0.0
        self.wind_y        = 0.0
        self.wind_direction = np.random.uniform(0, 2*math.pi)
        self.wind_noise_t  = 0.0
        self.gust_timer    = 0.0
        self.gust_strength = 1.0

        # Tree sway driven by wind_x
        self.sway_angle = 0.0
        self.sway_vel   = 0.0

        # If a background image is being drawn underneath, set this to False
        # to skip the procedural ground rectangle.
        self.draw_ground = True

    def _build(self, start, angle, length, thickness, depth, parent=None):
        if depth > self.max_depth or length < 3:
            return
        br = Branch(start, angle, length, thickness, depth, self.max_depth,
                    parent=parent, thick_shrink=self.thick_shrink)
        self.branches.append(br)
        if br.is_tip:
            self.tips.append(br)
            return
        n_children = 2 if np.random.random() > 0.3 else 3
        if n_children == 2:
            angles = [angle - self.spread_angle, angle + self.spread_angle]
        else:
            angles = [angle - self.spread_angle, angle,
                      angle + self.spread_angle]
        for a in angles:
            a += np.random.uniform(-0.15, 0.15)
            new_len   = length * self.branch_shrink * np.random.uniform(0.85, 1.15)
            new_thick = max(1, thickness * self.thick_shrink)
            self._build(br.end, a, new_len, new_thick, depth + 1, parent=br)

    # ── Wind update — the core of v3 ────────────────────────────────────
    def _update_wind(self, ev, frame):
        joy     = ev.get("joy", 0)
        sadness = ev.get("sadness", 0)
        anger   = ev.get("anger", 0)
        calm    = ev.get("calm", 0)
        fear    = ev.get("fear", 0)

        # Target wind MAGNITUDE
        # Anger drives BIG wind. Joy is now a meaningful breeze (~1.1) that
        # can spin falling leaves into swirls. Fear adds small base.
        # Calm AND sadness multiplicatively suppress wind (stillness).
        target_mag = (anger * 4.0 + joy * 1.2 + fear * 0.4)
        target_mag *= (1 - calm    * 0.85)
        target_mag *= (1 - sadness * 0.85)
        target_mag  = max(0, target_mag)

        # Direction drift — joy noticeably accelerates the wander so its
        # breeze keeps changing direction, creating the swirling effect.
        self.wind_noise_t += 0.01 + joy * 0.015
        drift  = math.sin(self.wind_noise_t * 0.30)
        drift += math.sin(self.wind_noise_t * 0.13) * 0.5
        self.wind_direction += drift * (0.003 + joy * 0.006)

        # Joy occasionally nudges direction — gentle shifts, makes leaves swirl
        if joy > 0.3 and np.random.random() < 0.012 * joy:
            self.wind_direction += np.random.uniform(-0.6, 0.6) * joy

        # Anger throws direction around violently and often → leaves fly all directions
        if anger > 0.2 and np.random.random() < 0.020 * anger:
            self.wind_direction += np.random.uniform(-1.5, 1.5) * anger

        # Gusts — anger-only. More frequent, longer, much stronger than before
        # so the tree visibly bends during peak gusts (some gusts up to ~3.5x).
        if anger > 0.2 and np.random.random() < 0.040 * anger:
            self.gust_timer    = np.random.uniform(20, 60)
            self.gust_strength = 1.0 + np.random.uniform(1.5, 3.5) * anger
        if self.gust_timer > 0:
            self.gust_timer -= 1
            gust = self.gust_strength
        else:
            gust = 1.0

        # Target wind VECTOR — mostly horizontal, small vertical.
        # Joy gets a slightly bigger vertical share so its falling leaves
        # genuinely lift and swirl rather than just drifting sideways.
        ty_factor = 0.25 + joy * 0.30
        tx = math.cos(self.wind_direction) * target_mag * gust
        ty = math.sin(self.wind_direction) * target_mag * gust * ty_factor

        # Smooth base wind toward target (anger snaps, calm settles)
        smooth = 0.04 + anger * 0.08
        self.wind_base_x += (tx - self.wind_base_x) * smooth
        self.wind_base_y += (ty - self.wind_base_y) * smooth

        # Final wind = base + fear jitter (un-smoothed so it actually trembles)
        self.wind_x = self.wind_base_x
        self.wind_y = self.wind_base_y
        if fear > 0.15:
            self.wind_x += math.sin(frame * 0.80) * fear * 0.6
            self.wind_y += math.cos(frame * 0.93) * fear * 0.4

    def update(self, ev, frame):
        sadness = ev.get("sadness", 0)
        fear    = ev.get("fear", 0)
        calm    = ev.get("calm", 0)

        # ── Wind ────────────────────────────────────────────────────────
        self._update_wind(ev, frame)
        wind_mag = math.hypot(self.wind_x, self.wind_y)

        # ── Tree sway driven by wind_x ──────────────────────────────────
        # Recalibrated curve: less aggressive tanh, larger amplitude cap, so
        # mild winds bend gently while STRONG gusts bend the tree noticeably
        # more — the bend now scales with gust strength.
        #   wind_x =  2 →  7°  (gentle)
        #   wind_x =  6 → 20°
        #   wind_x = 12 → 30°  (strong gust)
        #   wind_x = 30 → 34°  (cap)
        target_sway = math.tanh(self.wind_x * 0.10) * 0.60

        # Calm: imperceptibly slow ambient swing so the tree breathes
        # rather than freezes. Period ~8s, amplitude ~3°.
        target_sway += math.sin(frame * 0.012) * 0.05 * calm

        self.sway_vel += (target_sway - self.sway_angle) * 0.028
        self.sway_vel *= 0.92
        self.sway_angle += self.sway_vel

        # Rebuild branch positions.
        # Walk in order (parents before children — guaranteed by _build's
        # pre-order traversal). Each branch's start follows its parent's end,
        # so when the trunk leans, the whole tree bends as one structure.
        # Distribution of sway: 0.30 at trunk → 1.00 at tips.
        for br in self.branches:
            if br.parent is not None:
                br.start = br.parent.end
            depth_frac = br.depth / self.max_depth
            sway_share = 0.30 + 0.70 * depth_frac
            sway  = self.sway_angle * sway_share
            # Sadness droop: stronger, reaches one level shallower than before
            # so the bend is much more obvious — branches visibly sag downward.
            droop = sadness * 0.45 * depth_frac
            angle = br.base_angle + sway
            if br.depth >= 2:
                sign = 1 if br.base_angle > 0 else -1
                angle += sign * droop
            br.end = br._calc_end(angle)

        # ── Update leaves ───────────────────────────────────────────────
        for lf in self.leaves:
            lf.update(ev, frame, self.wind_x, self.wind_y, wind_mag)

        # ── Regrow ──────────────────────────────────────────────────────
        # Only joy & calm create new leaves. Sadness/anger/fear fully suppress.
        #   joy   → flood of new growth (≈ 2 leaves/sec at peak)
        #   calm  → steady gentle regrow that balances calm's tiny fall rate
        #   sad   → ZERO regrow (the death — tree cannot heal while sad)
        #   anger → ZERO regrow (canopy strips violently)
        #   fear  → near-zero regrow (suppressed but not absolute)
        joy   = ev.get("joy", 0)
        anger = ev.get("anger", 0)
        # joy bumped from 0.150 → 0.25 so the tree refills meaningfully fast
        regrow_base = joy * 0.25 + calm * 0.010
        regrow_gate = max(0.0, 1.0 - sadness*2.0 - anger*1.5 - fear*0.6)
        regrow_rate = regrow_base * regrow_gate
        if np.random.random() < regrow_rate:
            tip = self.tips[np.random.randint(0, len(self.tips))]
            existing = sum(1 for lf in self.leaves
                           if lf.branch is tip
                           and lf.state in (Leaf.ATTACHED, Leaf.GROWING))
            # Per-tip cap scales with joy. Base 8, up to 18 at peak joy — so
            # sustained joy keeps adding leaves, never plateaus to a static cap.
            cap = int(8 + joy * 10)
            if existing < cap:
                lf = Leaf(tip.end[0], tip.end[1], tip)
                lf.ground_y = self.ground_y + np.random.uniform(0, 30)
                self.leaves.append(lf)

        # GC
        self.leaves = [lf for lf in self.leaves if lf.alpha > 5]

    def draw(self, screen, ev, frame):
        # Procedural ground (skip when a background image provides it)
        if self.draw_ground:
            pygame.draw.rect(screen, (16, 20, 14),
                             (0, int(self.ground_y), self.W, self.H - int(self.ground_y)))

        # ── Branches: tapered, slightly curved black polygons ───────────────
        # Each branch is a quadratic Bézier curve, widening from end_thickness
        # at the tip back to thickness at the base, sampled into a polygon.
        bark_color = (0, 0, 0)
        for br in sorted(self.branches, key=lambda b: b.depth):
            self._draw_branch_curve(screen, br, bark_color)

        # Leaves: ground first (behind), then airborne
        leaf_col = blend_leaf_color(ev)
        for lf in self.leaves:
            if lf.state == Leaf.GROUND:
                self._draw_leaf(screen, lf, leaf_col)
        for lf in self.leaves:
            if lf.state != Leaf.GROUND:
                self._draw_leaf(screen, lf, leaf_col)

    def _draw_branch_curve(self, screen, br, color):
        """Draw a single branch as a tapered, curved polygon (quadratic Bézier)."""
        sx, sy = float(br.start[0]), float(br.start[1])
        ex, ey = float(br.end[0]),   float(br.end[1])
        dx, dy = ex - sx, ey - sy
        length = math.hypot(dx, dy)
        if length < 1:
            return

        thick_s = max(0.6, br.thickness)
        thick_e = max(0.3, br.end_thickness)

        # Very thin twigs: aaline is faster and reads cleaner than a sub-pixel polygon
        if max(thick_s, thick_e) < 1.4:
            pygame.draw.aaline(screen, color, (sx, sy), (ex, ey))
            return

        # Perpendicular unit vector (for bend offset)
        nx, ny = -dy / length, dx / length

        # Control point: midpoint pushed sideways by the branch's bend factor.
        mx, my = (sx + ex) * 0.5, (sy + ey) * 0.5
        bend_offset = br.bend * length * 0.30
        cx, cy = mx + nx * bend_offset, my + ny * bend_offset

        # Sample the curve. More samples for longer branches.
        steps = 6 if length < 40 else (10 if length < 90 else 14)
        left, right = [], []
        for i in range(steps + 1):
            t  = i / steps
            mt = 1.0 - t
            # Quadratic Bézier point
            px = mt*mt*sx + 2*mt*t*cx + t*t*ex
            py = mt*mt*sy + 2*mt*t*cy + t*t*ey
            # Tangent (derivative of the Bézier)
            tx = 2*mt*(cx - sx) + 2*t*(ex - cx)
            ty = 2*mt*(cy - sy) + 2*t*(ey - cy)
            tlen = math.hypot(tx, ty) or 1.0
            pnx, pny = -ty / tlen, tx / tlen
            # Thickness lerp from base to tip
            th   = thick_s + (thick_e - thick_s) * t
            half = th * 0.5
            left.append((px + pnx * half, py + pny * half))
            right.append((px - pnx * half, py - pny * half))

        polygon = left + right[::-1]
        if len(polygon) >= 3:
            pygame.draw.polygon(screen, color, polygon)

    def _draw_leaf(self, screen, lf, base_col):
        s = max(1, int(lf.size))
        if s < 2: return
        alpha = int(max(0, min(255, lf.alpha)))
        if alpha < 10: return

        r = min(255, max(0, int((base_col[0] + lf.jitter_r) * lf.brightness)))
        g = min(255, max(0, int((base_col[1] + lf.jitter_g) * lf.brightness)))
        b = min(255, max(0, int((base_col[2] + lf.jitter_b) * lf.brightness)))

        # Leaf bounding box: w (across) × h (along the leaf axis)
        # Real leaves are taller than they are wide → use a tall aspect.
        w = int(s * 1.1)
        h = int(s * lf.aspect * 0.9)
        if w < 3 or h < 4: return

        # Surface large enough to hold the rotated leaf without clipping
        diag = int(math.hypot(w, h)) + 4
        leaf_surf = pygame.Surface((diag, diag), pygame.SRCALPHA)
        cx, cy = diag // 2, diag // 2

        # Asymmetric pointed-leaf polygon (tip at top, rounded base at bottom).
        # Wider in the middle, tapering to both ends — readable as a leaf even at small sizes.
        hw, hh = w * 0.5, h * 0.5
        pts = [
            (cx,              cy - hh * 1.05),   # sharp tip
            (cx + hw * 0.55,  cy - hh * 0.55),
            (cx + hw * 0.95,  cy - hh * 0.10),
            (cx + hw * 0.85,  cy + hh * 0.35),
            (cx + hw * 0.50,  cy + hh * 0.75),
            (cx,              cy + hh * 0.90),   # stem base
            (cx - hw * 0.50,  cy + hh * 0.75),
            (cx - hw * 0.85,  cy + hh * 0.35),
            (cx - hw * 0.95,  cy - hh * 0.10),
            (cx - hw * 0.55,  cy - hh * 0.55),
        ]

        # Shadow underleaf — slightly offset, darker; gives a hint of depth.
        sr = max(0, r - 28); sg = max(0, g - 28); sb = max(0, b - 22)
        shadow_pts = [(p[0] + 1, p[1] + 2) for p in pts]
        pygame.draw.polygon(leaf_surf, (sr, sg, sb, alpha), shadow_pts)
        # Main leaf face
        pygame.draw.polygon(leaf_surf, (r, g, b, alpha), pts)

        rotated = pygame.transform.rotate(leaf_surf, math.degrees(lf.rotation))
        rect = rotated.get_rect(center=(int(lf.x), int(lf.y)))
        screen.blit(rotated, rect)