"""Local temporal bubble detector; never interprets fish proximity alone as a bite.

A fish can be visually occluded during a splash. The caller briefly latches
its proximity. Bubble pixels are not always pure white: shaded gray bubbles
are detected as positive local brightness changes against a short-term
background. A sudden large camera/anchor change resets comparison history.
"""
import cv2
import numpy as np
from .config import FisherConfig


class BiteDetector:
    def __init__(self, cfg: FisherConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.prev_roi = None
        self.background = None
        self.prev_anchor = None
        self.streak = 0
        self.edge_base = None
        self.white_base = None
        self.samples = 0
        self.last_sample_time = None

    def roi_box(self, frame_shape, anchor):
        if anchor is None:
            return None
        h,w = frame_shape[:2]
        x,y = map(int,anchor)
        r=self.cfg.bite_roi_radius
        if x<0 or y<0 or x>=w or y>=h:
            return None
        return max(0,x-r),max(0,y-r),min(w,x+r),min(h,y+r)

    def update(self, frame_bgr, anchor, fish_near=False, now=None):
        info={'motion':0.0,'bright_change':0.0,'edge_level':0.0,
              'streak':0,'fish_near':bool(fish_near)}
        box=self.roi_box(frame_bgr.shape,anchor)
        if box is None:
            self.reset()
            return False,info
        # Keep the same background time constant at 30/60 FPS. Legacy callers
        # without a clock retain the original 30 FPS adaptation.
        dt=1/30 if now is None or self.last_sample_time is None else max(0.,min(.5,now-self.last_sample_time))
        self.last_sample_time=now
        gray_alpha=1-(1-.22)**(dt*30)
        edge_alpha=1-(1-.1)**(dt*30)
        x0,y0,x1,y1=box
        roi=frame_bgr[y0:y1,x0:x1]
        if roi.size==0:
            self.reset()
            return False,info
        small=cv2.resize(roi,(80,80),interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        gray=cv2.GaussianBlur(gray,(3,3),0)
        hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
        edges=cv2.Canny(gray,30,80)
        edge=float((edges>0).mean())
        moved=(self.prev_anchor is not None and np.hypot(
            anchor[0]-self.prev_anchor[0],anchor[1]-self.prev_anchor[1])>5)
        self.prev_anchor=anchor
        if self.prev_roi is None or moved:
            self.prev_roi=gray
            self.background=gray.astype(np.float32)
            self.edge_base=edge
            self.samples=0
            self.streak=0
            info['reset']='new anchor' if moved else 'learning'
            return False,info
        median_shift=float(np.median(gray))-float(np.median(self.prev_roi))
        motion=float((cv2.absdiff(gray,self.prev_roi)>18).mean())
        self.prev_roi=gray
        self.samples+=1
        if motion>self.cfg.bite_motion_veto:
            self.background=gray.astype(np.float32)
            self.edge_base=edge
            self.samples=self.streak=0
            info.update(motion=round(motion,4),reset='large motion')
            return False,info
        if abs(median_shift)>=10:
            # Uniform exposure/lighting changes are not local bubble events.
            self.background=gray.astype(np.float32)
            self.edge_base=edge
            self.samples=self.streak=0
            info.update(reset='illumination change',median_shift=round(median_shift,2))
            return False,info
        # Central-only brightness increases. The fishing line and bobber
        # already belong to the background and cannot trigger a sustained bite.
        yy,xx=np.ogrid[:80,:80]
        central=(xx-40)**2+(yy-40)**2 < 30**2
        # Bounded local exposure adaptation: shaded bubble flashes have
        # less absolute contrast, while bright daylight retains v1 thresholds.
        local_level=float(np.median(gray))
        dim=self.cfg.adaptive_lighting and local_level<110
        contrast=max(10.,min(18.,18*local_level/130)) if dim else 18.
        brightness_floor=max(45.,min(70.,local_level*.65)) if dim else 70.
        vote_floor=self.cfg.bite_relative_jump*(max(.65,min(1.,local_level/130)) if dim else 1.)
        bright=((gray.astype(np.float32)-self.background)>contrast)
        bright &= (hsv[:,:,1]<70) & (hsv[:,:,2]>brightness_floor) & central
        # Gray bubbles brighten the DARK fish surface. A fish sliding away
        # merely reveals mid-gray water and must not count as a bubble.
        bright &= (((gray<120)&(self.background<100)) | (gray>160))
        bright_fraction=float(bright.mean())
        # Require compact bright components, not one tiny glint or the
        # disappearance of a whole dark fish contour.
        n,lab,stats,_=cv2.connectedComponentsWithStats(bright.astype(np.uint8),8)
        compact=sum(int(st[cv2.CC_STAT_AREA]) for st in stats[1:]
                    if 3<=st[cv2.CC_STAT_AREA]<=(350 if dim else 180)
                    and .4<=st[cv2.CC_STAT_WIDTH]/max(1,st[cv2.CC_STAT_HEIGHT])<=2.5)
        compact_fraction=compact/6400
        edge_level=edge-self.edge_base
        positive=(fish_near and self.samples>=5
                  and compact_fraction>vote_floor
                  and compact_fraction<.12
                  and edge>self.cfg.bite_edge_level)
        if positive:
            self.streak+=1  # freeze background while confirming the splash
        else:
            self.streak=0   # strictly consecutive, not accumulating old votes
            cv2.accumulateWeighted(gray,self.background,gray_alpha)
            self.edge_base=(1-edge_alpha)*self.edge_base+edge_alpha
        info.update(motion=round(motion,4),bright_change=round(bright_fraction,4),
                    compact_change=round(compact_fraction,4),
                    contrast_threshold=round(contrast,2),vote_threshold=round(vote_floor,5),
                    edge_level=round(edge_level,4),edge=round(edge,4),streak=self.streak)
        bitten=self.streak>=self.cfg.bite_confirm_frames
        if bitten:
            self.streak=0
        return bitten,info
