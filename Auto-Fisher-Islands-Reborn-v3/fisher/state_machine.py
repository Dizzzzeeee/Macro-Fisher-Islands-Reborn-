"""Deterministic, screen-only cast / wait / reel state machine.

All frames and distance thresholds use 1280-wide working coordinates.
Time is injectable for replay/tests. No mouse or desktop imports here.
"""
import time
from collections import deque
from enum import Enum, auto
import numpy as np
from .config import FisherConfig
from .detector import analyze_frame
from .bite_detector import BiteDetector


class State(Enum):
    SEEK = auto()
    SETTLING = auto()
    WAITING = auto()
    COOLDOWN = auto()
    PAUSED = auto()


class AutoFisher:
    def __init__(self, cfg: FisherConfig, controller=None, observe_only=False):
        self.cfg = cfg
        self.controller = controller  # compatibility; step never performs input
        self.observe_only = observe_only
        self.state = State.SEEK
        self.bite = BiteDetector(cfg)
        self.now = 0.0
        self.t_state = 0.0
        self.bobber = None
        self.anchor_verified = False
        self.casts = 0
        self.catches_attempted = 0
        self.last_info = {}
        self.t_fresh = 0.0
        self.fish_latch = None
        self.t_fish = -10.0
        self.ready_since = None
        self.line_since = None
        self.roi_center = None
        self.t_last_action = -100.0
        self.failed_casts = 0
        self.fish_history = deque()
        self.stable_fish_until = -10.0
        self._last_analysis_time = None

    def _set(self, state):
        self.state, self.t_state = state, self.now
        self.ready_since = self.line_since = None

    def reset(self):
        """Discard stale anchors/baselines after focus loss/camera change."""
        self.bobber = self.fish_latch = self.roi_center = None
        self.anchor_verified = False
        self.t_fish = -10.0
        self.fish_history.clear()
        self.stable_fish_until = -10.0
        self._last_analysis_time = None
        self.bite.reset()
        self.failed_casts = 0
        self._set(State.SEEK)

    def pause(self):
        self.reset()
        self._set(State.PAUSED)

    def time_in_state(self):
        return self.now - self.t_state

    def fish_to_bobber(self, analysis):
        if not analysis.fishes or self.bobber is None:
            return None
        return min(float(np.hypot(f.center[0]-self.bobber[0],
                                  f.center[1]-self.bobber[1])) for f in analysis.fishes)

    def step(self, frame_bgr, now=None, analysis=None, analysis_time=None, tackle_visible_now=None):
        self.now = time.monotonic() if now is None else float(now)
        a = analyze_frame(frame_bgr, self.cfg) if analysis is None else analysis
        observed_at=self.now if analysis_time is None else min(self.now,float(analysis_time))
        age=max(0.,self.now-observed_at)
        new_observation=(analysis_time is None or self._last_analysis_time is None
                         or observed_at>self._last_analysis_time+1e-6)
        fresh = a.bobber or a.string_end
        if tackle_visible_now is False:
            fresh=None
        info = {"state": self.state.name, "fish": len(a.fishes),
                "vision_age_ms":round(age*1000,2),"analysis_new":new_observation,
                "anchor_source": "bobber" if a.bobber else "line-end" if fresh else "none"}
        action = "none"
        if age>self.cfg.max_vision_age_s:
            self.bite.reset()
            info.update(next_state=self.state.name,note='vision result stale; no action',bobber=self.bobber)
            self.last_info=info
            return action,a,info
        if (analysis_time is not None and self._last_analysis_time is not None
                and observed_at<self._last_analysis_time-1e-6):
            info.update(next_state=self.state.name,note='out-of-order vision; no action',bobber=self.bobber)
            self.last_info=info
            return action,a,info
        if new_observation:self._last_analysis_time=observed_at
        water = float((a.water_mask > 0).mean()) > .02
        ready = a.player_bbox is not None and bool(a.rod_lines) and water
        if self.state == State.PAUSED:
            info['note'] = 'paused; re-arm to continue'
        elif self.state == State.SEEK:
            if fresh is not None:
                self.ready_since = None
                if self.line_since is None:
                    self.line_since = self.now
                if self.now - self.line_since >= .15:
                    self.bobber, self.t_fresh = fresh, (self.now if tackle_visible_now is True else observed_at)
                    self.anchor_verified = a.bobber is not None
                    self.bite.reset()
                    self._set(State.SETTLING)
                    info['note'] = 'existing line; no cast click'
            else:
                self.line_since = None
                # No arbitrary cast when a leaning pole suggests the line is
                # out but occluded. Upright pose and repeated readiness needed.
                pole_upright = False
                if a.rod_lines:
                    x1,y1,x2,y2 = a.rod_lines[0]
                    pole_upright = abs(y2-y1) > 8 * abs(x2-x1)
                if ready and pole_upright and not self.observe_only:
                    if self.ready_since is None:
                        self.ready_since = self.now
                    if (self.now-self.ready_since >= .35
                            and self.now-self.t_last_action >= self.cfg.click_cooldown_s):
                        action = 'cast'
                        self.casts += 1
                        self.t_last_action = self.now
                        self.bobber = self.fish_latch = None
                        self.anchor_verified = False
                        self.bite.reset()
                        self.t_fresh = self.now
                        self._set(State.SETTLING)
                else:
                    self.ready_since = None
        elif self.state in (State.SETTLING, State.WAITING):
            if fresh is not None:
                if tackle_visible_now is True:
                    self.t_fresh=self.now
                elif new_observation:
                    self.t_fresh=observed_at
                if self.bobber is None:
                    self.bobber = fresh
                    self.anchor_verified = a.bobber is not None
                elif new_observation:
                    dx,dy=fresh[0]-self.bobber[0],fresh[1]-self.bobber[1]
                    if self.anchor_verified and a.bobber is None:
                        # An occluding fish shortens the visible string. Never
                        # drag a previously observed float up to that fragment.
                        pass
                    elif not self.anchor_verified and a.bobber is not None and np.hypot(dx,dy)<60:
                        self.bobber=fresh
                        self.anchor_verified=True
                        self.bite.reset()
                    elif np.hypot(dx,dy)>75:
                        # A new cast/camera move invalidates temporal comparison.
                        self.bobber=fresh
                        self.fish_latch=None
                        self.bite.reset()
                        self._set(State.SETTLING)
                    elif self.now - self.t_fish > .8:
                        self.bobber=(round(self.bobber[0]+.25*dx),round(self.bobber[1]+.25*dy))
                    # Freeze anchor once a fish approaches: changing the ROI
                    # by even one pixel paints false bright fish-edge ghosts.
            if self.state == State.SETTLING and self.time_in_state() >= self.cfg.cast_settle_s:
                self._set(State.WAITING)
            if self.state == State.WAITING:
                anchor = self.bobber
                d = self.fish_to_bobber(a)
                if new_observation and d is not None and d < self.cfg.bite_fish_px:
                    self.t_fish = observed_at
                    self.fish_latch = min(a.fishes, key=lambda f: np.hypot(
                        f.center[0]-anchor[0],f.center[1]-anchor[1])).center
                fish_near = ((d is not None and d < self.cfg.bite_fish_px)
                             or (self.fish_latch is not None and self.now-self.t_fish < .8))
                # Fish arrival changes silhouettes/occludes the tackle, but is
                # not a bite. Require a stationary fish over a short window
                # before interpreting local bright residuals as bubbles.
                if new_observation and d is not None and d < self.cfg.bite_fish_px:
                    self.fish_history.append((observed_at, self.fish_latch))
                while self.fish_history and self.fish_history[0][0] < self.now - .55:
                    self.fish_history.popleft()
                stable = False
                if len(self.fish_history) >= 3 and self.now-self.fish_history[0][0] >= .4:
                    pts = np.array([p for _,p in self.fish_history], dtype=float)
                    stable = float(np.hypot(np.ptp(pts[:,0]),np.ptp(pts[:,1]))) < 12
                if stable and new_observation:
                    self.stable_fish_until = observed_at + .5
                fish_near = fish_near and (stable or self.now < self.stable_fish_until)
                info['stable_fish'] = bool(fish_near)
                self.roi_center = anchor  # never move the bite ROI to arbitrary fish mid-line
                bitten, metrics = self.bite.update(frame_bgr, anchor, fish_near, now=self.now)
                info.update(metrics)
                info['fish_dist'] = round(d,1) if d is not None else None
                info['tackle_age']=round(self.now-self.t_fresh,3)
                if bitten and self.now-self.t_fresh>.1:
                    # A caught-fish appearance after a manual reel is not a
                    # bubble bite. No late clicks without recent cast evidence.
                    bitten=False
                    self.bite.reset()
                    info['note']='late brightness flash; no recent tackle'
                if bitten:
                    action='reel' 
                    self.catches_attempted += 1
                    self.failed_casts=0
                    self.t_last_action=self.now
                    self._set(State.COOLDOWN)
            # A vanished line after an observed manual reel is not a recast
            # request. Return to SEEK and require upright rod evidence again.
            if action == 'none' and self.now-self.t_fresh > (.65 if self.observe_only else self.cfg.nobobber_timeout_s):
                self.failed_casts += 1
                self.reset_anchor()
                self._set(State.SEEK if self.failed_casts < 3 else State.PAUSED)
                info['note']='tackle lost; waiting for upright rod'
            elif action == 'none' and self.time_in_state() > self.cfg.bite_timeout_s:
                # Clicking blindly would toggle a live cast, not safely recast.
                self.pause()
                info['note']='bite timeout; paused for manual inspection'
        elif self.state == State.COOLDOWN:
            if self.time_in_state() >= self.cfg.click_cooldown_s:
                # The live game should remove the line after the reel. In
                # offline observe mode do not count a sustained animation twice.
                if fresh is None:
                    self.reset_anchor()
                    self._set(State.SEEK)
                elif self.time_in_state() > self.cfg.nobobber_timeout_s and not self.observe_only:
                    self.pause()
                    info['note']='line remains after reel; paused'
        info['next_state'] = self.state.name
        info['bobber'] = self.bobber
        info['anchor_verified'] = self.anchor_verified
        info["lighting"]=a.lighting if hasattr(a,"lighting") else None
        self.last_info=info
        return action,a,info

    def reset_anchor(self):
        self.bobber=self.fish_latch=self.roi_center=None
        self.anchor_verified=False
        self.t_fish=-10.0
        self.fish_history.clear()
        self.stable_fish_until=-10.0
        self.bite.reset()
