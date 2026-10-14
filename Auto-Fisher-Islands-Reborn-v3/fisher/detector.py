"""Pixel-pattern detectors for player / water / fish / rod / bobber.

Strategy (no ML needed, tuned to your screenshots):
- player  -> large low-saturation bright blob
- water   -> HSV band for murky pond + largest connected region
- fish    -> dark (low V) elongated blobs *inside* water mask
- rod     -> long brown Hough line (pole) + thin dark line (string)
- bobber  -> small bright blob near string endpoint (+ template if provided)
- bubbles -> handled in bite_detector.py (temporal, not single-frame)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import copy
import cv2
import math
import numpy as np

from .config import FisherConfig
from .illumination import compensate
from .fish_filter import filter_candidates

Point = Tuple[int, int]

@dataclass
class FishDetection:
    center: Point
    bbox: Tuple[int, int, int, int]
    area: float

@dataclass
class FrameAnalysis:
    water_mask: np.ndarray
    player_bbox: Optional[Tuple[int, int, int, int]]
    fishes: List[FishDetection]
    rod_lines: List[Tuple[int, int, int, int]]
    bobber: Optional[Point]
    string_end: Optional[Point] = None
    string_seg: Optional[Tuple[int, int, int, int]] = None
    bobber_score: float = 0.0
    night: bool = False
    lighting: dict | None = None

_TEMPLATES: dict = {}

def _load_templates(template_dir: Path = Path(__file__).parent / "assets" / "templates"):
    global _TEMPLATES
    if _TEMPLATES or not template_dir.exists():
        return _TEMPLATES
    for name in ("bobber", "fish", "bubbles"):
        p = template_dir / f"{name}.png"
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                _TEMPLATES[name] = img
    return _TEMPLATES

def ui_allowed_mask(h: int, w: int, cfg: FisherConfig) -> np.ndarray:
    """255 where detections are allowed, 0 inside Roblox UI zones."""
    m = np.full((h, w), 255, np.uint8)
    top = int(h * cfg.colors.ui_top_frac)
    left = int(w * cfg.colors.ui_left_frac)
    bot = int(h * cfg.colors.ui_bottom_frac)
    if top > 0:
        m[0:top, :] = 0
    if left > 0:
        m[:, 0:left] = 0
    if bot > 0:
        m[h - bot:h, :] = 0
    return m

def _scaled_cfg(cfg: FisherConfig, scale: float, hsv: np.ndarray | None = None,
                allowed: np.ndarray | None = None) -> tuple:
    """Scale area/length thresholds to detection size and auto-pick the
    day/tinted profile from scene saturation. Returns (cfg, is_night)."""
    night = False
    if hsv is not None:
        s = hsv[:, :, 1]
        sample = s[allowed > 0] if allowed is not None else s.ravel()
        if sample.size and float(np.median(sample)) > cfg.colors.tint_s_thresh:
            night = True
    if scale == 1.0 and not night:
        return cfg, False
    cfg2 = copy.copy(cfg)
    c2 = copy.copy(cfg.colors)
    if scale != 1.0:
        s2 = scale * scale
        c2.fish_min_area = max(50, int(cfg.colors.fish_min_area * s2))
        c2.fish_max_area = max(200, int(cfg.colors.fish_max_area * s2))
        c2.player_min_area = max(200, int(cfg.colors.player_min_area * s2))
        c2.bobber_min_area = max(4, int(cfg.colors.bobber_min_area * s2))
        c2.bobber_max_area = max(16, int(cfg.colors.bobber_max_area * s2))
        c2.rod_min_length = max(30, int(cfg.colors.rod_min_length * scale))
        c2.rod_max_dist_from_player = int(cfg.colors.rod_max_dist_from_player * scale)
        c2.bobber_max_dist_from_line = int(cfg.colors.bobber_max_dist_from_line * scale)
        c2.tip_search_radius = max(40, int(cfg.colors.tip_search_radius * scale))
    # Lighting is corrected in pixel space. Never substitute a fixed
    # "night" rod-saturation threshold that can erase the entire pole.
    cfg2.colors = c2
    return cfg2, night

def detect_water(hsv: np.ndarray, cfg: FisherConfig) -> np.ndarray:
    c = cfg.colors
    lower = np.array([c.water_h_low, c.water_s_low, c.water_v_low], np.uint8)
    upper = np.array([c.water_h_high, c.water_s_high, c.water_v_high], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask

def detect_player(bgr: np.ndarray, hsv: np.ndarray, cfg: FisherConfig,
                  allowed: np.ndarray | None = None, tinted: bool = False,
                  rod_lines: list | None = None):
    # White avatar is cool-gray in shadows; warm sand and white reflections
    # must not join it into one giant component. All working coordinates are
    # normalized to a 1280-wide frame by the live/replay runners.
    hh, ww = hsv.shape[:2]
    H, S, V = cv2.split(hsv)
    grid=hsv[::8,::8]
    sample=grid[allowed[::8,::8]>0] if allowed is not None else grid.reshape(-1,3)
    scene_s=float(np.median(sample[:,1])) if sample.size else 0
    scene_v=float(np.median(sample[:,2])) if sample.size else 0
    warm_mask=(S<min(60,scene_s*.55))&(V>max(75,scene_v*.95))
    # Warm scenes: do not reduce the white avatar to just its least-tinted
    # feet. Prefer the full neutral, upright body before estimating exposure.
    if scene_s>80:
        cool=warm_mask.astype(np.uint8)*255
    else:
        cool = (((H >= 75) | (S < 16)) & (S < 70) & (V > 125)).astype(np.uint8) * 255
    if allowed is not None:
        cool = cv2.bitwise_and(cool, allowed)
    cool = cv2.morphologyEx(cool, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cc, _ = cv2.findContours(cool, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in cc:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if (area > 450 and 1.15 < bh / max(1, bw) < 4.5
                and .25 * ww < x + bw / 2 < .75 * ww
                and .32 * hh < y + bh / 2 < .82 * hh
                and .04 * hh < bh < .45 * hh):
            score = area / (1 + abs(x + bw / 2 - ww / 2) / 100)
            candidates.append((score, (x, y, bw, bh)))
    if candidates:
        return max(candidates)[1]
    # Tinted scenes: the day ranges match the whole terrain, so skip them
    # entirely and go shape-based (compact blob, edge-gated).
    if tinted:
        return _player_compact(hsv, cfg, allowed)
    c = cfg.colors
    ends = []
    if rod_lines:
        for (x1, y1, x2, y2) in rod_lines:
            ends += [(x1, y1), (x2, y2)]
    def contains_rod(x, y, w, h) -> bool:
        m = 15
        return any(x - m <= ex <= x + w + m and y - m <= ey <= y + h + m
                   for ex, ey in ends)
    mask = cv2.inRange(hsv,
                       np.array([0, 0, c.player_v_min], np.uint8),
                       np.array([179, c.player_s_max, 255], np.uint8))
    if allowed is not None:
        mask = cv2.bitwise_and(mask, mask, mask=allowed)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    best_rod = None
    best_rod_area = 0.0
    for cnt in cnts:
        a = cv2.contourArea(cnt)
        if a < c.player_min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if (w > 200 or h > 300 or h < w * 1.15
                or not .25 * hsv.shape[1] < x + w / 2 < .75 * hsv.shape[1]):
            continue  # walls/shops, not an avatar
        if a > best_area:
            best_area = a
            best = (x, y, w, h)
        if contains_rod(x, y, w, h) and a > best_rod_area:
            best_rod_area = a
            best_rod = (x, y, w, h)
    if best_rod is not None:
        return best_rod  # the avatar holds the pole; sand doesn't
    if best is not None:
        return best
    # Fallback for tinted light (lantern at night turns the white avatar
    # orange): brightest large blob in the central band of the frame.
    h, w = hsv.shape[:2]
    fb = cv2.inRange(hsv, np.array([0, 0, 150], np.uint8),
                     np.array([179, 130, 255], np.uint8))
    if allowed is not None:
        fb = cv2.bitwise_and(fb, fb, mask=allowed)
    fb[:, :int(w * 0.2)] = 0
    fb[:, int(w * 0.8):] = 0
    fb = cv2.morphologyEx(fb, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    cnts, _ = cv2.findContours(fb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        a = cv2.contourArea(cnt)
        if a < max(1500, c.player_min_area * 0.4):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if (w > 200 or h > 300 or h < w * 1.15
                or not .25 * hsv.shape[1] < x + w / 2 < .75 * hsv.shape[1]):
            continue
        if a > best_area:
            best_area = a
            best = (x, y, w, h)
        if contains_rod(x, y, w, h) and a > best_rod_area:
            best_rod_area = a
            best_rod = (x, y, w, h)
    if best_rod is not None:
        return best_rod
    if best is not None:
        return best
    # Last resort, tinted scenes only: the avatar is mid-tone orange like the
    # sand, so select by SHAPE — compact, tall, central, not touching edges.
    if tinted:
        return _player_compact(hsv, cfg, allowed)
    return None  # (x,y,w,h) or None

def _player_compact(hsv: np.ndarray, cfg: FisherConfig,
                    allowed: np.ndarray | None = None):
    h, w = hsv.shape[:2]
    m = cv2.inRange(hsv, np.array([0, 0, 90], np.uint8),
                    np.array([179, 140, 175], np.uint8))
    if allowed is not None:
        m = cv2.bitwise_and(m, m, mask=allowed)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0.0
    for cnt in cnts:
        a = cv2.contourArea(cnt)
        if not (2000 <= a <= 50000):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if x <= 5 or y <= 5 or x + bw >= w - 5 or y + bh >= h - 5:
            continue  # sprawling terrain touches frame edges
        if bh < bw * 0.9:
            continue  # avatar stands taller than wide
        if not (w * 0.25 <= x + bw / 2.0 <= w * 0.75):
            continue
        fill = a / max(1.0, bw * bh)
        if fill < 0.40:
            continue
        score = fill * min(1.0, a / 8000.0)
        if score > best_score:
            best_score, best = score, (x, y, bw, bh)
    return best

def _ellipse_coverage(cnt, center, axes) -> float:
    pts = cnt.reshape(-1, 2).astype(float)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    bins = np.floor((ang + np.pi) / (2 * np.pi) * 36).astype(int)
    return float(len(np.unique(bins))) / 36.0

def detect_fish_outline(gray: np.ndarray, cfg: FisherConfig,
                        water_mask: np.ndarray,
                        allowed: np.ndarray | None = None,
                        player_bbox=None) -> List[FishDetection]:
    """Fish with no color contrast (gray fish in gray-green water) still own
    an outline: long elliptical edge curves. Keep curves that cover much of
    a fitted ellipse's circumference (ripple arcs don't)."""
    H, W = gray.shape[:2]
    gb = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gb, 40, 120)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[FishDetection] = []
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        if peri < 120 or len(cnt) < 8:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if not (25 <= w <= 350 and 25 <= h <= 350 and w * h <= 60000):
            continue
        try:
            (cx, cy), (aw, bh), _ = cv2.fitEllipse(cnt)
        except cv2.error:
            continue
        if min(aw, bh) < 12:
            continue
        area = float(np.pi * aw * bh / 4.0)
        if area > 10000:
            continue  # pond-scale curves, not fish
        cov = _ellipse_coverage(cnt, (cx, cy), (aw, bh))
        if cov < 0.45:
            continue
        pts = cnt.reshape(-1, 2).astype(float)
        dx, dy = (pts[:, 0] - cx) / max(1.0, aw / 2.0), (pts[:, 1] - cy) / max(1.0, bh / 2.0)
        resid = float(np.mean(np.abs(np.sqrt(dx * dx + dy * dy) - 1.0)))
        if resid > 0.30:
            continue
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < W and 0 <= iy < H):
            continue
        if water_mask[iy, ix] == 0:
            continue
        if allowed is not None and allowed[iy, ix] == 0:
            continue
        if player_bbox is not None:
            px, py, pw, ph = player_bbox
            if px - 8 <= ix <= px + pw + 8 and py - 8 <= iy <= py + ph + 8:
                continue  # avatar's own outline, not a fish
        out.append(FishDetection(center=(ix, iy),
                                 bbox=(int(cx - aw / 2), int(cy - bh / 2),
                                       int(aw), int(bh)),
                                 area=area))
    out.sort(key=lambda f: f.area, reverse=True)
    return out[:6]

def detect_fish_stringband(gray: np.ndarray, seg, cfg: FisherConfig,
                           water_mask: np.ndarray,
                           allowed: np.ndarray | None = None,
                           player_bbox=None) -> List[FishDetection]:
    """Outline search in a corridor around the string: the global edge map
    merges a gray fish's outline into surroundings, but locally it stands
    out. Targets exactly the bite-relevant fish (the one ON the line)."""
    H, W = gray.shape[:2]
    x1, y1, x2, y2 = seg
    m = 120
    rx0, ry0 = max(0, min(x1, x2) - m), max(0, min(y1, y2) - m)
    rx1, ry1 = min(W, max(x1, x2) + m), min(H, max(y1, y2) + m)
    if rx1 - rx0 < 40 or ry1 - ry0 < 40:
        return []
    sub = gray[ry0:ry1, rx0:rx1]
    gb = cv2.GaussianBlur(sub, (5, 5), 0)
    edges = cv2.Canny(gb, 40, 120)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[FishDetection] = []
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        if peri < 100 or len(cnt) < 8:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if not (25 <= w <= 350 and 25 <= h <= 350):
            continue
        try:
            (cx, cy), (aw, bh), _ = cv2.fitEllipse(cnt)
        except cv2.error:
            continue
        if min(aw, bh) < 12:
            continue
        area = float(np.pi * aw * bh / 4.0)
        if area > 10000:
            continue  # pond-scale curves, not fish
        cov = _ellipse_coverage(cnt, (cx, cy), (aw, bh))
        if cov < 0.40:
            continue
        pts = cnt.reshape(-1, 2).astype(float)
        dx, dy = (pts[:, 0] - cx) / max(1.0, aw / 2.0), (pts[:, 1] - cy) / max(1.0, bh / 2.0)
        resid = float(np.mean(np.abs(np.sqrt(dx * dx + dy * dy) - 1.0)))
        if resid > 0.35:
            continue
        gx, gy = int(rx0 + cx), int(ry0 + cy)
        if water_mask[gy, gx] == 0:
            continue
        if allowed is not None and allowed[gy, gx] == 0:
            continue
        if player_bbox is not None:
            px, py, pw, ph = player_bbox
            if px - 8 <= gx <= px + pw + 8 and py - 8 <= gy <= py + ph + 8:
                continue
        out.append(FishDetection(center=(gx, gy),
                                 bbox=(int(rx0 + cx - aw / 2), int(ry0 + cy - bh / 2),
                                       int(aw), int(bh)),
                                 area=area))
    out.sort(key=lambda f: f.area, reverse=True)
    return out[:4]

def detect_fish(bgr: np.ndarray, hsv: np.ndarray, water_mask: np.ndarray, cfg: FisherConfig,
                allowed: np.ndarray | None = None, player_bbox=None) -> List[FishDetection]:
    c = cfg.colors
    # Dilate water slightly so fish swimming at the shoreline edge still count.
    wm = cv2.dilate(water_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    # No whole-screen fallback: a lost water mask must not classify trees,
    # avatar shadows and UI as fish. A separate tackle-local search below
    # recovers warm/dark pond silhouettes without global false positives.
    # Fish = dark pixels in the water region. The fish body stays much darker
    # than water in all light (core V~45 at dusk vs water V~110), so a tight
    # absolute cap separates them; keep it tight or fish merge with water.
    absdark = cv2.inRange(hsv, np.array([0, 0, 0], np.uint8),
                          np.array([179, 255, c.fish_v_max], np.uint8))
    dark = cv2.bitwise_and(absdark, absdark, mask=wm)
    if allowed is not None:
        dark = cv2.bitwise_and(dark, dark, mask=allowed)
    dark = cv2.medianBlur(dark, 5)
    # Open FIRST (erase speckle), close SECOND (bridge thin cuts such as the
    # string crossing a fish). Close-then-open self-defeats on those bridges.
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[FishDetection] = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (c.fish_min_area <= area <= c.fish_max_area):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / (max(1, min(w, h)))
        if aspect < c.fish_min_aspect:
            continue
        if area < 600 and aspect < c.fish_small_aspect:
            continue  # small round specks are ripple noise, not fish
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / max(1.0, hull_area)
        if solidity < 0.55:  # fish shadow is a solid oval; reject speckles
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        if player_bbox is not None:
            px, py, pw, ph = player_bbox
            if px - 8 <= cx <= px + pw + 8 and py - 8 <= cy <= py + ph + 8:
                continue
        out.append(FishDetection(center=(cx, cy), bbox=(x, y, w, h), area=area))
    out.sort(key=lambda f: f.area, reverse=True)
    return out

def _rod_mask(hsv: np.ndarray, cfg: FisherConfig) -> np.ndarray:
    c = cfg.colors
    return cv2.inRange(hsv,
                       np.array([c.rod_h_low, c.rod_s_min, c.rod_v_min], np.uint8),
                       np.array([c.rod_h_high, 255, c.rod_v_max], np.uint8))

def probe_pole(hsv: np.ndarray, cfg: FisherConfig,
               player_bbox=None, allowed: np.ndarray | None = None) -> dict:
    """Pole search with diagnostics. Returns {"pole", "candidates", "brown_frac"}."""
    c = cfg.colors
    mask = _rod_mask(hsv, cfg)
    if allowed is not None:
        mask = cv2.bitwise_and(mask, mask, mask=allowed)
    brown_frac = float((mask > 0).mean())
    # The pole stands IN brownish water, so it is often one contour with the
    # pond. A tall vertical opening keeps thin vertical structures (pole)
    # while erasing the broad water area.
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25)))
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15)))
    cnts, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pcx = pcy = None
    if player_bbox is not None:
        px, py, pw, ph = player_bbox
        pcx, pcy = px + pw / 2.0, py + ph / 2.0
    cands = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cands.append((x, y, w, h))
    cands.sort(key=lambda r: r[3], reverse=True)
    best, best_h = None, 0.0
    for (x, y, w, h) in cands:
        if h < c.rod_min_length or h < 2.0 * max(1, w):
            continue
        if w > 60:  # pole is thin; wide blobs are terrain
            continue
        cx, cy = x + w / 2.0, y + h / 2.0
        if pcx is not None and math.hypot(cx - pcx, cy - pcy) > c.rod_max_dist_from_player * 1.5:
            continue
        if h > best_h:
            best_h = h
            best = (int(cx), int(y), int(cx), int(y + h))
    return {"pole": best, "candidates": cands[:8], "brown_frac": brown_frac}

def _pole_by_contour(hsv: np.ndarray, cfg: FisherConfig,
                     player_bbox=None, allowed: np.ndarray | None = None):
    """Find the rod pole as a tall narrow brown blob (see probe_pole)."""
    return probe_pole(hsv, cfg, player_bbox, allowed)["pole"]

def detect_rod(bgr: np.ndarray, hsv: np.ndarray, cfg: FisherConfig,
               player_bbox=None, allowed: np.ndarray | None = None):
    """Pole attached to the avatar, using color and local chroma ridges.

    All expensive color/line work is confined to the hands/pole ROI.
    """
    if player_bbox is None:
        return [],None
    px,py,pw,ph=player_bbox;h,w=hsv.shape[:2]
    x0,x1=max(0,px-180),min(w,px+pw+200)
    y0,y1=max(0,py-240),min(h,py+int(ph*.8))
    roi=hsv[y0:y1,x0:x1]
    if roi.size==0:
        return [],None
    H,S,V=roi[:,:,0],roi[:,:,1],roi[:,:,2];c=cfg.colors
    mask=cv2.inRange(roi,np.array([c.rod_h_low,c.rod_s_min,c.rod_v_min],np.uint8),
                     np.array([min(c.rod_h_high,18),255,c.rod_v_max],np.uint8))
    red=cv2.cvtColor(bgr[y0:y1,x0:x1],cv2.COLOR_BGR2LAB)[:,:,1].astype(np.float32)
    ridge=red-cv2.GaussianBlur(red,(0,0),3)
    ridge_mask=((ridge>2)&(S>30)&(H<30)&(V>25)).astype(np.uint8)*255
    if allowed is not None:
        permitted=allowed[y0:y1,x0:x1]
        mask=cv2.bitwise_and(mask,permitted)
        ridge_mask=cv2.bitwise_and(ridge_mask,permitted)
    sources=[cv2.Canny(mask,50,150),ridge_mask]
    pooled=[]
    for source in sources:
        lines=cv2.HoughLinesP(source,1,np.pi/180,threshold=20,
                             minLineLength=max(45,int(ph*.5)),maxLineGap=18)
        if lines is not None:
            pooled.extend(lines.reshape(-1,4).tolist())
    support_mask=cv2.dilate(mask,np.ones((5,5),np.uint8))
    candidates=[]
    def rectdist(point):
        x,y=point
        return math.hypot(max(px-x,0,x-(px+pw)),max(py-y,0,y-(py+ph)))
    for ax,ay,bx,by in pooled:
        a=(ax+x0,ay+y0);b=(bx+x0,by+y0)
        top,butt=(a,b) if a[1]<b[1] else (b,a)
        dx,dy=butt[0]-top[0],butt[1]-top[1]
        length=math.hypot(dx,dy)
        angle=math.degrees(math.atan2(abs(dy),abs(dx)+1e-6))
        if (angle<48 or rectdist(butt)>28 or top[1]>py-.08*ph
                or butt[1]<py+.15*ph or butt[1]>py+.85*ph):
            continue
        count=max(2,int(length))
        xs=np.rint(np.linspace(top[0],butt[0],count)).astype(int)
        ys=np.rint(np.linspace(top[1],butt[1],count)).astype(int)
        support=support_mask[ys-y0,xs-x0]>0
        smooth=np.convolve(support.astype(float),np.ones(8)/8,mode='valid')
        good=np.flatnonzero(smooth>.65)
        if not good.size:
            continue
        first=max(0,int(good[0]));last=min(count-1,int(good[-1])+7)
        top=(int(xs[first]),int(ys[first]));butt=(int(xs[last]),int(ys[last]))
        length=math.hypot(butt[0]-top[0],butt[1]-top[1])
        if length<max(45,ph*.5):
            continue
        candidates.append((length-2*rectdist(butt),(*top,*butt)))
    if not candidates:
        return [],None
    pole=max(candidates)[1]
    return [pole],(pole[2],pole[3])


def pt_seg_dist(px, py, x1, y1, x2, y2) -> float:
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return float(np.hypot(px - x1, py - y1))
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return float(np.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))

def near_rod(pt, rod_lines, px: float = 12.0, end_margin: float = 25.0) -> bool:
    """True if a point hugs the MIDDLE of a detected pole line (pole rings /
    guides live there). Points near a line's endpoints (tip, string end) are
    allowed — real bobbers hang there."""
    if pt is None or not rod_lines:
        return False
    for (x1, y1, x2, y2) in rod_lines:
        if pt_seg_dist(pt[0], pt[1], x1, y1, x2, y2) <= px:
            if min(float(np.hypot(pt[0] - x1, pt[1] - y1)),
                   float(np.hypot(pt[0] - x2, pt[1] - y2))) > end_margin:
                return True
    return False

def rod_tip(rod_lines: list, player_bbox=None):
    """Rod tip = endpoint farthest from the player (string attaches here)."""
    if not rod_lines:
        return None
    x1, y1, x2, y2 = rod_lines[0]
    if player_bbox is not None:
        qx, qy, qw, qh = player_bbox
        pcx, pcy = qx + qw / 2.0, qy + qh / 2.0
        d1 = float(np.hypot(x1 - pcx, y1 - pcy))
        d2 = float(np.hypot(x2 - pcx, y2 - pcy))
        return (x1, y1) if d1 > d2 else (x2, y2)
    return (x1, y1) if y1 > y2 else (x2, y2)

def merge_collinear(lines, ang_tol: float = 8.0, off_tol: float = 12.0):
    """Join Hough fragments lying on one line (mask gaps split the string).
    Groups by angle + lateral offset, each group spans its extreme endpoints.
    """
    groups = []
    for (x1, y1, x2, y2) in lines:
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length == 0:
            continue
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        off = abs(x1 * dy - y1 * dx) / length
        placed = False
        for g in groups:
            da = abs(ang - g["ang"])
            if min(da, 180.0 - da) < ang_tol and abs(off - g["off"]) < off_tol:
                g["segs"].append((x1, y1, x2, y2))
                placed = True
                break
        if not placed:
            groups.append({"ang": ang, "off": off, "segs": [(x1, y1, x2, y2)]})
    merged = []
    for g in groups:
        (x1, y1, x2, y2) = g["segs"][0]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy)) or 1.0
        ux, uy = dx / length, dy / length
        projs = []
        for (ax, ay, bx, by) in g["segs"]:
            projs += [ax * ux + ay * uy, bx * ux + by * uy]
        lo, hi = min(projs), max(projs)
        # anchor point: first seg start projected back out
        ox, oy = g["segs"][0][0], g["segs"][0][1]
        base = ox * ux + oy * uy
        merged.append((int(ox + ux * (lo - base)), int(oy + uy * (lo - base)),
                       int(ox + ux * (hi - base)), int(oy + uy * (hi - base))))
    return merged

def detect_string(hsv: np.ndarray, rod_lines: list, cfg: FisherConfig,
                  allowed: np.ndarray | None = None, player_bbox=None,gray=None):
    """Trace only the reddish line connected to the pole tip.

    Do not merge dark water edges: those used to extend the endpoint deep
    past the real bobber. Missing line pixels are tolerated briefly.
    """
    tip = rod_tip(rod_lines, player_bbox)
    if tip is None or player_bbox is None:
        return None
    H,S,V = cv2.split(hsv)
    mask = ((H>=3)&(H<=24)&(S>=30)&(V>=30)&(V<=235)).astype(np.uint8)*255
    if allowed is not None:
        mask = cv2.bitwise_and(mask, allowed)
    search = np.zeros_like(mask)
    hh,ww=mask.shape
    search[max(0,tip[1]-20):min(hh,tip[1]+400),
           max(0,tip[0]-320):min(ww,tip[0]+320)] = 255
    mask = cv2.bitwise_and(mask,search)
    ls = cv2.HoughLinesP(cv2.Canny(mask,50,150),1,np.pi/180,
                         threshold=18,minLineLength=45,maxLineGap=8)
    # Gradient segments detect dark/thin strings even when warm ground and
    # string share a hue. LSD endpoints are not extrapolated through ripples.
    color_lines=[] if ls is None else [tuple(map(float,l)) for l in ls.reshape(-1,4)]
    if gray is None:
        gray=cv2.cvtColor(cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR),cv2.COLOR_BGR2GRAY)
    x0,x1=max(0,tip[0]-320),min(ww,tip[0]+320)
    y0,y1=max(0,tip[1]-45),min(hh,tip[1]+450)
    detected=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray[y0:y1,x0:x1])[0]
    edge_lines=[]
    if detected is not None:
        for ax,ay,bx,by in detected.reshape(-1,4):
            edge_lines.append((ax+x0,ay+y0,bx+x0,by+y0))
    px,py,pw,ph=player_bbox
    pcx=px+pw/2
    candidates=[]
    for l,edge_source in [(l,False) for l in color_lines]+[(l,True) for l in edge_lines]:
        x1,y1,x2,y2=map(int,l)
        top,bottom=((x1,y1),(x2,y2)) if y1<y2 else ((x2,y2),(x1,y1))
        dx,dy=bottom[0]-top[0],bottom[1]-top[1]
        length=math.hypot(dx,dy)
        if dy<=35 or abs(dx)/dy>1.5:
            continue
        # Cast string slopes away from the hands, whereas the pole returns
        # towards them. Reject the pole itself and unrelated distant lines.
        if dx*(tip[0]-pcx)<=0 or abs(bottom[0]-pcx)<pw*.7:
            continue
        dist=abs(dx*(tip[1]-top[1])-dy*(tip[0]-top[0]))/length
        if dist>(42 if edge_source else 22) or math.hypot(top[0]-tip[0],top[1]-tip[1])>(160 if edge_source else 95):
            continue
        ux,uy=dx/length,dy/length
        # Trace from segment start along its own fitted ray (parallel
        # edge lines can lie a few pixels off the estimated pole tip).
        misses=0; last=bottom; h,w=mask.shape
        if edge_source:
            # A fish can cut the string into two visible gradient segments.
            # Join only short-gap, tightly collinear LSD fragments linked to
            # this pole-connected segment, never arbitrary dark water edges.
            endpoint=length
            for _ in range(3):
                best=endpoint
                for ax,ay,bx,by in edge_lines:
                    vx,vy=bx-ax,by-ay;frag=math.hypot(vx,vy)
                    if frag<12 or abs(vx*uy-vy*ux)/frag>.075:
                        continue
                    off=max(abs((ax-top[0])*uy-(ay-top[1])*ux),
                            abs((bx-top[0])*uy-(by-top[1])*ux))
                    if off>5:
                        continue
                    p1=(ax-top[0])*ux+(ay-top[1])*uy
                    p2=(bx-top[0])*ux+(by-top[1])*uy
                    near,far=min(p1,p2),max(p1,p2)
                    if far>endpoint and near<=endpoint+50 and far<=endpoint+160:
                        best=max(best,far)
                if best==endpoint:
                    break
                endpoint=best
            last=(round(top[0]+ux*endpoint),round(top[1]+uy*endpoint))
        for t in ([] if edge_source else range(max(0, int(length)-4), min(500, int(length)+200))):
            xx,yy=round(top[0]+ux*t),round(top[1]+uy*t)
            if not (2<=xx<w-2 and 2<=yy<h-2):
                break
            hit=mask[yy-2:yy+3,xx-2:xx+3].any()
            if hit:
                last=(xx,yy);misses=0
            else:
                misses+=1
                if misses>=12:
                    break
        if last is None:
            continue
        traced=math.hypot(last[0]-tip[0],last[1]-tip[1])
        if traced<120 or last[1]<py+ph*.45:
            continue
        candidates.append((traced-4*dist+(15 if edge_source else 0), last,top,bottom))
    if not candidates:
        return None
    _,bottom,line_top,line_bottom=max(candidates,key=lambda c:c[0])
    # The pole/string junction is a more reliable tip than the longest
    # Hough endpoint. Intersect their fitted axes; reject distant solutions.
    rx1,ry1,rx2,ry2=rod_lines[0]
    pole=np.array([rx2-rx1,ry2-ry1],float)
    thread=np.array([line_bottom[0]-line_top[0],line_bottom[1]-line_top[1]],float)
    matrix=np.column_stack((pole,-thread))
    if abs(float(np.linalg.det(matrix)))>1:
        t,_=np.linalg.solve(matrix,np.array(line_top,float)-np.array([rx1,ry1],float))
        joint=np.array([rx1,ry1],float)+t*pole
        if np.hypot(joint[0]-tip[0],joint[1]-tip[1])<60 and joint[1]<py:
            tip=(round(float(joint[0])),round(float(joint[1])))
    return {"top":tip,"bottom":bottom,"seg":(*tip,*bottom)}


def _bobber_by_color(bgr: np.ndarray, hsv: np.ndarray, cfg: FisherConfig,
                     allowed: np.ndarray | None = None):
    c = cfg.colors
    mask = cv2.inRange(hsv,
                       np.array([0, 0, c.bobber_v_min], np.uint8),
                       np.array([179, c.bobber_s_max, 255], np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0.0
    H, W = mask.shape[:2]
    for cnt in cnts:
        a = cv2.contourArea(cnt)
        if not (c.bobber_min_area <= a <= c.bobber_max_area):
            continue
        per = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * a / max(1.0, per * per)
        if circ < 0.35:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        pt = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if allowed is not None and allowed[min(H - 1, pt[1]), min(W - 1, pt[0])] == 0:
            continue  # inside Roblox UI bars (sidebar icons, hotbar, top menu)
        score = circ * min(1.0, a / 150.0)
        if score > best_score:
            best, best_score = pt, score
    return best, best_score

def probe_bobber(hsv: np.ndarray, cfg: FisherConfig,
                 allowed: np.ndarray | None = None) -> dict:
    """White-blob diagnostics: why is / isn't a bobber found?"""
    c = cfg.colors
    mask = cv2.inRange(hsv,
                       np.array([0, 0, c.bobber_v_min], np.uint8),
                       np.array([179, c.bobber_s_max, 255], np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape[:2]
    info = []
    for cnt in cnts:
        a = cv2.contourArea(cnt)
        per = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * a / max(1.0, per * per)
        M = cv2.moments(cnt)
        pt = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] else None
        if not (c.bobber_min_area <= a <= c.bobber_max_area):
            reason = f"area {a:.0f} out of [{c.bobber_min_area},{c.bobber_max_area}]"
        elif circ < 0.35:
            reason = f"circ {circ:.2f}<0.35"
        elif (allowed is not None and pt is not None
              and allowed[min(H - 1, pt[1]), min(W - 1, pt[0])] == 0):
            reason = "in UI zone"
        else:
            reason = "OK"
        info.append({"area": round(float(a), 1), "circ": round(float(circ), 2),
                     "pt": pt, "reason": reason})
    info.sort(key=lambda d: -d["area"])
    return {"white_frac": round(float((mask > 0).mean()), 5), "blobs": info[:12],
            "n_blobs": len(info)}

def _red_mask(hsv: np.ndarray, cfg: FisherConfig) -> np.ndarray:
    c = cfg.colors
    lo1 = np.array([0, c.red_s_min, c.red_v_min], np.uint8)
    hi1 = np.array([c.red_h_halfwidth, 255, 255], np.uint8)
    lo2 = np.array([179 - c.red_h_halfwidth, c.red_s_min, c.red_v_min], np.uint8)
    hi2 = np.array([179, 255, 255], np.uint8)
    return cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2))

def _bobber_near_tip(hsv: np.ndarray, cfg: FisherConfig, tip,
                     allowed: np.ndarray | None = None,
                     rod_lines: list | None = None):
    """Search a small disc around the rod tip for the bobber.

    Returns (pt, score, detail). The white dot usually touches the pole tip
    or the red trailer, so shape rules are relaxed and proximity to the tip
    decides. Red trailer pixels nearby boost confidence. Dots hugging a pole
    line are guides/rings, not the bobber.
    """
    c = cfg.colors
    H, W = hsv.shape[:2]
    r = c.tip_search_radius
    tx, ty = int(np.clip(tip[0], 0, W - 1)), int(np.clip(tip[1], 0, H - 1))
    yy, xx = np.ogrid[:H, :W]
    roi = ((xx - tx) ** 2 + (yy - ty) ** 2 <= r * r)
    white = cv2.inRange(hsv,
                        np.array([0, 0, c.bobber_v_min], np.uint8),
                        np.array([179, c.bobber_s_max, 255], np.uint8))
    white = cv2.bitwise_and(white, white, mask=roi.astype(np.uint8) * 255)
    if allowed is not None:
        white = cv2.bitwise_and(white, white, mask=allowed)
    red = _red_mask(hsv, cfg)
    red = cv2.bitwise_and(red, red, mask=roi.astype(np.uint8) * 255)
    if allowed is not None:
        red = cv2.bitwise_and(red, red, mask=allowed)
    red_frac = float((red > 0).mean())
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_key = None, None
    for cnt in cnts:
        a = cv2.contourArea(cnt)
        if not (c.bobber_min_area <= a <= c.bobber_max_area * 2):
            continue
        per = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * a / max(1.0, per * per)
        if circ < c.tip_min_circ:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        pt = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        d = float(np.hypot(pt[0] - tx, pt[1] - ty))
        if d > r:
            continue
        if near_rod(pt, rod_lines):
            continue
        key = (d, -circ)  # closest to tip wins, circularity breaks ties
        if best_key is None or key < best_key:
            best_key = key
            red_bonus = 0.25 if red_frac * (H * W) > c.red_min_area else 0.0
            best = (pt, min(1.0, circ + 0.35 + red_bonus),
                    f"tip d={d:.0f} circ={circ:.2f} red_px={red_frac * H * W:.0f}")
    if best is None:
        return None, 0.0, f"no white blob in r={r} of tip; red_px={red_frac * H * W:.0f}"
    return best

def _bobber_plumb(hsv: np.ndarray, cfg: FisherConfig, tip,
                  allowed: np.ndarray | None = None,
                  water_mask: np.ndarray | None = None,
                  rod_lines: list | None = None):
    """Bobber hangs BELOW the tip (gravity): search a vertical corridor under
    it. Works when the string itself is camouflaged — geometry only.
    Dots on a pole line are guides, not the bobber.

    Returns (pt, score, detail)."""
    c = cfg.colors
    H, W = hsv.shape[:2]
    tx, ty = int(tip[0]), int(tip[1])
    x0, x1 = max(0, tx - 80), min(W, tx + 80)
    y0, y1 = max(0, ty + 30), min(H, ty + 450)
    if x1 <= x0 or y1 <= y0:
        return None, 0.0, "corridor empty"
    white = cv2.inRange(hsv,
                        np.array([0, 0, c.bobber_v_min], np.uint8),
                        np.array([179, c.bobber_s_max, 255], np.uint8))
    sub = white[y0:y1, x0:x1]
    sub_allowed = (allowed[y0:y1, x0:x1] > 0) if allowed is not None else np.ones_like(sub, bool)
    sub_water = (water_mask[y0:y1, x0:x1] > 0) if water_mask is not None else np.ones_like(sub, bool)
    cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red = _red_mask(hsv, cfg)
    best, best_score = None, -1e9
    for cnt in cnts:
        a = cv2.contourArea(cnt)
        if not (c.bobber_min_area <= a <= c.bobber_max_area * 2):
            continue
        per = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * a / max(1.0, per * per)
        if circ < 0.15:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        lx, ly = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        gx, gy = x0 + lx, y0 + ly
        if not sub_allowed[ly, lx] or not sub_water[ly, lx]:
            continue
        if near_rod((gx, gy), rod_lines):
            continue
        # Lowest + centered + red-trailer nearby wins (bobber = line's end).
        rbox = red[max(0, gy - 25):gy + 25, max(0, gx - 25):gx + 25]
        red_px = float((rbox > 0).sum())
        score = (gy - ty) - 1.5 * abs(gx - tx) + (40.0 if red_px > c.red_min_area else 0.0)
        if score > best_score:
            best_score = score
            best = ((gx, gy), min(1.0, 0.45 + circ),
                    f"plumb dy={gy - ty:.0f} dx={abs(gx - tx):.0f} circ={circ:.2f} red={red_px:.0f}")
    if best is None:
        return None, 0.0, "no white blob in plumb corridor"
    return best

def _bobber_by_template(bgr: np.ndarray, cfg: FisherConfig):
    if not cfg.use_templates:
        return None, 0.0
    tpl = _load_templates().get("bobber")
    if tpl is None:
        return None, 0.0
    gray, tgray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < tgray.shape[0] or gray.shape[1] < tgray.shape[1]:
        return None, 0.0
    res = cv2.matchTemplate(gray, tgray, cv2.TM_CCOEFF_NORMED)
    _, mx, _, ml = cv2.minMaxLoc(res)
    if mx < cfg.template_thresh:
        return None, float(mx)
    th, tw = tgray.shape[:2]
    return (ml[0] + tw // 2, ml[1] + th // 2), float(mx)

def detect_bobber(bgr: np.ndarray, hsv: np.ndarray,
                  rod_lines: list, cfg: FisherConfig,
                  allowed: np.ndarray | None = None,
                  water_mask: np.ndarray | None = None,
                  player_bbox=None, string_end=None):
    """Recognize a compact pale float near the traced endpoint.

    Shaded bobbers are gray (~V170), not V255. If the fish covers the
    float, return None instead of labeling the line-end as a seen bobber.
    """
    if string_end is None:
        return None,0.0
    x,y=string_end;h,w=hsv.shape[:2];r=24
    x0,y0=max(0,x-r),max(0,y-r);x1,y1=min(w,x+r),min(h,y+r)
    patch=hsv[y0:y1,x0:x1]
    if patch.size==0:
        return None,0.0
    v=patch[:,:,2];floor=max(140,float(np.median(v))+20)
    mask=((v>floor)&(patch[:,:,1]<80)).astype(np.uint8)*255
    cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    choices=[]
    for c in cs:
        area=cv2.contourArea(c)
        bx,by,bw,bh=cv2.boundingRect(c)
        if not 2<=area<=180 or max(bw,bh)>25 or max(bw,bh)>4*min(bw,bh):
            continue
        point=(x0+bx+bw//2,y0+by+bh//2)
        dist=math.hypot(point[0]-x,point[1]-y)
        if dist>22:
            continue
        if allowed is not None and allowed[point[1],point[0]]==0:
            continue
        score=.8-dist/80
        choices.append((score,point))
    tpl,score=_bobber_by_template(bgr,cfg)
    if tpl is not None and score>=cfg.template_thresh and math.hypot(tpl[0]-x,tpl[1]-y)<30:
        choices.append((score,tpl))
    if not choices:
        return None,0.0
    score,point=max(choices)
    return point,float(score)


def verify_bobber(bobber, rod_lines, string_seg) -> bool:
    """A bobber without tackle is a glint: require a rod line or the string
    segment within 30px (string-end hits already sit on the string)."""
    if bobber is None:
        return False
    if string_seg is not None:
        if pt_seg_dist(bobber[0], bobber[1], *string_seg) <= 30.0:
            return True
    if not rod_lines:
        return False
    return min(pt_seg_dist(bobber[0], bobber[1], *r) for r in rod_lines) <= 30.0

def analyze_frame(bgr: np.ndarray, cfg: FisherConfig) -> FrameAnalysis:
    # Downscale huge windowed captures (your window is 2560x1334) so the
    # area/length thresholds stay resolution-independent and 30fps is reachable.
    max_w = getattr(cfg, "detect_width", 1280)
    h0, w0 = bgr.shape[:2]
    scale = 1.0
    if w0 > max_w:
        scale = max_w / float(w0)
        small = cv2.resize(bgr, (max_w, int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        small = bgr
    raw_hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    sh,sw=small.shape[:2]
    raw_allowed=ui_allowed_mask(sh,sw,cfg)
    seed=detect_player(small,raw_hsv,cfg,raw_allowed)
    small,lighting=compensate(small,seed,raw_allowed,cfg.adaptive_lighting,raw_hsv)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sh, sw = small.shape[:2]
    allowed = ui_allowed_mask(sh, sw, cfg)  # fractions are scale-free
    # Day/night profile + scale-aware thresholds follow the frame.
    dcfg=cfg
    night=lighting["scene_v"]<125 and lighting["scene_s"]>80
    water_small = detect_water(hsv, dcfg)
    # Pole first (unanchored): the avatar is the bright blob holding it.
    player = seed if not lighting["corrected"] else detect_player(small, hsv, dcfg, allowed, night)
    if player is not None:
        ground_y = max(int(sh * .38), player[1] - int(player[3] * .15))
        water_small[:ground_y] = 0
    else:
        water_small[:int(sh * .45)] = 0
    water_small = cv2.bitwise_and(water_small, allowed)
    if cfg.pond_roi is not None:
        rx,ry,rw,rh=cfg.pond_roi
        pond=np.zeros_like(water_small)
        pond[round(ry*sh):round((ry+rh)*sh),round(rx*sw):round((rx+rw)*sw)]=255
        water_small=cv2.bitwise_and(water_small,pond)
    fishes = detect_fish(small, hsv, water_small, dcfg, allowed, player)
    # Outline channel for camouflaged (gray-on-gray) fish; dedupe overlap.
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    for of in detect_fish_outline(gray_small, dcfg, water_small, allowed, player):
        if all(float(np.hypot(of.center[0] - f.center[0],
                              of.center[1] - f.center[1])) > 40.0 for f in fishes):
            fishes.append(of)
    # String-band channel needs the segment: computed below with rods; run a
    # second pass after the string is known (cheap, small ROI).
    fishes.sort(key=lambda f: f.area, reverse=True)
    rods, pole_butt = detect_rod(small, hsv, dcfg, player, allowed)
    if player is None and pole_butt is not None:
        # Avatar untinted-unfindable but the pole is known: the player holds
        # its butt end, so seed the box below it. (Only from the contour
        # pole — Hough lines may sit on the string, whose "butt" is the bobber.)
        bx, by = pole_butt
        pw, ph = 130, 200
        qx = max(0, min(sw - pw, int(bx - pw / 2)))
        qy = max(0, min(sh - ph, int(by + 10)))
        player = (qx, qy, pw, ph)
    string = detect_string(hsv, rods, dcfg, allowed, player,gray_small)
    if string is not None and rods:
        rods[0]=(*string['top'],rods[0][2],rods[0][3])
    string_end = string["bottom"] if string else None
    string_seg = string["seg"] if string else None
    if string_end is not None:
        # Warm/night water can leave the HSV band even after correction.
        # A traced cast provides a small, explicit tackle neighbourhood;
        # search relative darkness there rather than widening all terrain.
        ex,ey=string_end
        local=np.zeros_like(water_small)
        cv2.circle(local,(ex,ey),140,255,-1)
        local=cv2.bitwise_and(local,allowed)
        if cfg.pond_roi is not None:
            local=cv2.bitwise_and(local,pond)
        nearby=hsv[:,:,2][local>0]
        if nearby.size:
            relative=copy.copy(dcfg);relative.colors=copy.copy(dcfg.colors)
            relative.colors.fish_v_max=min(dcfg.colors.fish_v_max,
                max(35,int(np.median(nearby)*.68)))
            for of in detect_fish(small,hsv,local,relative,allowed,player):
                if all(math.hypot(of.center[0]-f.center[0],of.center[1]-f.center[1])>40 for f in fishes):
                    fishes.append(of)
    if string_seg is not None:
        for of in detect_fish_stringband(gray_small, string_seg, dcfg,
                                         water_small, allowed, player):
            if all(float(np.hypot(of.center[0] - f.center[0],
                                  of.center[1] - f.center[1])) > 40.0 for f in fishes):
                fishes.append(of)
        fishes.sort(key=lambda f: f.area, reverse=True)
    bobber, score = detect_bobber(small, hsv, rods, dcfg, allowed, water_small,
                                  player, string_end)
    if string_end is None:
        bobber, score = None, 0.0
    elif bobber is not None and math.hypot(bobber[0]-string_end[0], bobber[1]-string_end[1]) > 40:
        bobber, score = None, 0.0
    if bobber is not None and not verify_bobber(bobber, rods, string_seg):
        bobber, score = None, 0.0  # glint with no tackle behind it
    fishes=filter_candidates(gray_small,fishes,allowed,water_small,player,
                             bobber or string_end,cfg.fish_search_radius,
                             cfg.fish_min_contrast,cfg.fish_min_fill,cfg.pond_roi)
    if scale != 1.0:
        inv = 1.0 / scale
        # upscale water mask to full res (cheap nearest) for consistent overlay
        water = cv2.resize(water_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
        def up_pt(p):
            return (int(p[0] * inv), int(p[1] * inv)) if p else None
        player = tuple(int(v * inv) for v in player) if player else None
        fishes = [FishDetection(center=up_pt(f.center),
                                bbox=tuple(int(v * inv) for v in f.bbox),
                                area=f.area * inv * inv) for f in fishes]
        rods = [tuple(int(v * inv) for v in r) for r in rods]
        bobber = up_pt(bobber)
        string_end = up_pt(string_end)
        string_seg = tuple(int(v * inv) for v in string_seg) if string_seg else None
    else:
        water = water_small
    return FrameAnalysis(water_mask=water, player_bbox=player,
                         fishes=fishes, rod_lines=rods,
                         bobber=bobber, string_end=string_end,
                         string_seg=string_seg,
                         bobber_score=score, night=night,lighting=lighting)
