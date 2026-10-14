"""Avatar-referenced photometric correction (vision only; never alters capture).

The supplied avatar is white/gray. Its brightest low-saturation body pixels
provide a local neutral reference under warm lamps and changing exposure.
Gains are bounded. Without a usable reference, no correction is applied.
"""
import cv2
import numpy as np


def compensate(bgr, player_bbox, allowed=None, enabled=True, hsv=None):
    if hsv is None:
        hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    grid=hsv[::8,::8]
    sample=grid[allowed[::8,::8]>0] if allowed is not None else grid.reshape(-1,3)
    scene_v=float(np.median(sample[:,2])) if sample.size else 0
    scene_s=float(np.median(sample[:,1])) if sample.size else 0
    info={'scene_v':round(scene_v,1),'scene_s':round(scene_s,1),
          'gains_bgr':[1.,1.,1.],'corrected':False,'reference':'none'}
    if not enabled or player_bbox is None:
        return bgr,info
    x,y,w,h=player_bbox
    patch=bgr[y:y+h,x:x+w];ph=hsv[y:y+h,x:x+w]
    if patch.size==0:
        return bgr,info
    mask=(ph[:,:,1]<65)&(ph[:,:,2]>65)
    if np.count_nonzero(mask)<80:
        return bgr,info
    values=ph[:,:,2][mask]
    mask &= ph[:,:,2]>=np.percentile(values,55)
    if np.count_nonzero(mask)<40:
        return bgr,info
    neutral=np.median(patch[mask],axis=0).astype(float)
    # Preserve the previously calibrated daylight appearance. Low exposure
    # or a warm neutral reference invokes correction; no binary night profile.
    warm=neutral[2]>neutral[0]*1.06
    dim=float(neutral.max())<165
    info['reference']='avatar'
    if not (warm or dim):
        return bgr,info
    gains=np.clip(200/np.maximum(neutral,60),.7,2.5)
    table=np.clip(np.arange(256)[:,None]*gains[None,:],0,255).astype(np.uint8).reshape(256,1,3)
    corrected=cv2.LUT(bgr,table)
    info.update(gains_bgr=[round(float(g),3) for g in gains],corrected=True)
    return corrected,info
