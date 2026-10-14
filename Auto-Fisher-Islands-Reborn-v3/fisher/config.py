"""Shared configuration at 1280-wide working resolution, with adaptive lighting."""
from dataclasses import dataclass, field

@dataclass
class ColorRanges:
    # HSV ranges (OpenCV: H 0-179, S 0-255, V 0-255)
    # Fish shadow: very dark oval in water (images 1,3,4,5 show near-black silhouette)
    # Sunlit fish read up to V~100, so the day cap admits them; shape gates
    # (aspect/solidity) reject the extra speckle this admits.
    fish_v_max: int = 100
    fish_min_area: int = 300
    fish_small_aspect: float = 1.5  # blobs under 600px must be clearly elongated
    fish_max_area: int = 30000
    fish_min_aspect: float = 1.2

    # Player avatar: flat white/light-gray Roblox blocks (shaded parts go
    # grayish, so saturation tolerance is generous)
    player_s_max: int = 60
    player_v_min: int = 170
    player_min_area: int = 4000

    # Water: murky pond AND bright sunlit shallows (V runs to ~240).
    water_h_low: int = 20
    water_h_high: int = 85
    water_s_low: int = 15
    water_s_high: int = 110
    water_v_low: int = 90
    water_v_high: int = 245

    # Rod wood brown: pole is REDDER and more saturated than terrain
    # (terrain H~20 tightly; pole reaches H 13; pole S hits 125+ vs terrain
    # ~108). Hue cap + saturation floor separate them ~100:1.
    rod_h_low: int = 4
    rod_h_high: int = 24
    rod_s_min: int = 35
    rod_v_min: int = 30
    rod_v_max: int = 235
    rod_min_length: int = 90

    # Bobber: small near-white dot at end of line (+ faint red trailer in img 2,5)
    bobber_s_max: int = 60
    bobber_v_min: int = 200
    bobber_min_area: int = 8
    bobber_max_area: int = 600
    # Max distance (px at detection scale) a bobber candidate may be from the
    # rod tip before it's rejected (global fallback only; string-end gate
    # below is what normally decides).
    bobber_max_dist_from_line: int = 300
    # Fishing STRING: thin dark line from pole to bobber. The bobber hangs at
    # the string's far end — often hundreds of px from the pole tip — so the
    # string endpoint, not the pole tip, is the true bobber anchor.
    string_v_max: int = 120
    string_min_length: int = 60
    # Long diagonal casts run ~50°; precision comes from the top-near-pole
    # gate, not steepness.
    string_min_steep_deg: float = 30.0
    string_tip_dist: int = 50        # string top must start near the pole
    string_bobber_radius: int = 110  # bobber must be within this of string end
    bite_touch_px: int = 45         # fish-center to string distance = ON the line
    # Tip-anchored search: the bobber always hangs near the rod tip, so when a
    # rod is known we search this radius around the tip with relaxed shape
    # rules (the dot merges with the pole/line, killing circularity).
    tip_search_radius: int = 130
    tip_min_circ: float = 0.20
    # Red trailer cue (faint red line at the bobber in your screenshots)
    red_h_halfwidth: int = 8       # H in [0,h] or [179-h,179]
    red_s_min: int = 100
    red_v_min: int = 60
    red_min_area: int = 5

    # Bite bubbles: sudden white splash in bobber ROI
    bite_white_v_min: int = 215
    bite_white_s_max: int = 60

    # Roblox UI exclusion zones (fractions of frame): top menu bar, left
    # icon sidebar, bottom hotbar. Detections inside these are ignored —
    # this kills the false "bobber" on the white sidebar icons.
    ui_top_frac: float = 0.11
    ui_left_frac: float = 0.04
    ui_bottom_frac: float = 0.16

    # Rod must be steep (near-vertical pole, not flat sand ripples) and near
    # the player. Distances/angles are evaluated at detection scale.
    rod_min_steep_deg: float = 45.0
    rod_max_dist_from_player: int = 350

    # Legacy night settings retained for compatibility, not applied by the
    # v2 detection pipeline. Avatar-referenced correction now handles warm/dim light.
    # Night profile (lantern light tints everything orange and drops V).
    # Active when median frame SATURATION exceeds tint_s_thresh — tint, not
    # brightness, is what breaks detection (broken scenes read S~108,
    # working ones S~78).
    tint_s_thresh: float = 95.0
    night_player_s_max: int = 130
    night_player_v_min: int = 110
    night_rod_s_min: int = 110
    night_rod_h_high: int = 18
    night_rod_v_min: int = 25
    night_fish_v_max: int = 90
    night_bobber_v_min: int = 140

@dataclass
class FisherConfig:
    adaptive_lighting: bool = True  # bounded neutral-reference correction
    pond_roi: tuple | None = None   # normalized [x,y,width,height] optional pond constraint
    monitor_index: int = 1          # used only when capture_mode="monitor"
    capture_mode: str = "window"    # "window" (recommended for windowed play) or "monitor"
    window_title_sub: str = "Roblox"  # substring matched against window title
    region: str = ""                # manual override "x,y,w,h" (screen px); "" = auto
    follow_window: bool = True      # re-resolve window rect every N frames (handles moves)
    follow_every_n_frames: int = 30
    detect_width: int = 1280        # downscale wider captures to this for detection
    fish_search_radius: int = 380   # cast-centred fish context, working pixels
    fish_min_contrast: int = 10
    fish_min_fill: float = .22
    max_vision_age_s: float = .35    # never act on ancient asynchronous results
    fps: float = 60.0               # capture loop rate
    click_cooldown_s: float = 1.2
    cast_settle_s: float = 0.8      # ignore the cast's own splash before bite watch
    reel_delay_s: float = 0.05      # delay after bite before click (near-instant)
    bite_confirm_frames: int = 2    # consecutive positive frames to reel
    bite_roi_radius: int = 40       # px box around bobber watched for bubbles
    bite_motion_thresh: float = 0.012   # fraction of ROI pixels changed
    bite_white_jump: float = 0.012      # extra white-pixel fraction signaling splash
    bite_edge_jump: float = 0.008   # Canny-density jump from bubble rings (measured ~0.013)
    bite_edge_level: float = 0.008  # edge density ABOVE session baseline = sustained churn
    bite_fish_px: int = 90          # fish must be this close to the anchor for a bite
    bite_touch_px: int = 45         # fish-center to string distance = ON the line
    bite_motion_veto: float = 0.35  # above this ROI motion = camera jerk, never a bite
    bite_relative_jump: float = 0.010  # local bright-residual increase
    bite_relative_v_min: int = 125
    bite_timeout_s: float = 45.0    # recast if no bite
    nobobber_timeout_s: float = 8.0  # recast if no bobber latched (missed cast)
    fish_near_bobber_px: int = 250  # informational only (log when fish close)
    debug_view: bool = True
    save_debug_frames: bool = False
    colors: ColorRanges = field(default_factory=ColorRanges)

    # Optional template matching (drop PNGs into assets/templates/):
    #   bobber.png, fish.png, bubbles.png
    use_templates: bool = True
    template_thresh: float = 0.65
