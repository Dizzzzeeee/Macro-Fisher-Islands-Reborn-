"""Reject scenery/outline-only fish candidates using fishing context and fill."""
import cv2
import numpy as np


def filter_candidates(gray, fishes, allowed, water, player=None, anchor=None,
                      radius=380, min_contrast=10, min_fill=.22, pond_roi=None):
    h,w=gray.shape;kept=[]
    for f in fishes:
        cx,cy=f.center;x,y,bw,bh=f.bbox
        if not (0<=cx<w and 0<=cy<h) or allowed[cy,cx]==0 or bw<5 or bh<5:
            continue
        if anchor is not None and np.hypot(cx-anchor[0],cy-anchor[1])>radius:
            continue
        if pond_roi is not None:
            rx,ry,rw,rh=pond_roi
            if not (rx*w<=cx<(rx+rw)*w and ry*h<=cy<(ry+rh)*h):continue
        if max(bw,bh)/max(1,min(bw,bh))>6:
            continue  # long strings, shoreline edges, not a filled fish body
        if player is not None:
            px,py,pw,ph=player
            # Exclude avatar edges/nearby limb shadows before checking texture.
            margin=max(8,int(pw*.5))
            if px-margin<=cx<=px+pw+margin and py-8<=cy<=py+ph+12:continue
        x0,y0=max(0,x),max(0,y);x1,y1=min(w,x+bw),min(h,y+bh)
        if x1-x0<5 or y1-y0<5:continue
        pad=max(6,min(18,int(min(bw,bh)*.2)))
        ax,ay=max(0,x0-pad),max(0,y0-pad);bx,by=min(w,x1+pad),min(h,y1+pad)
        patch=gray[ay:by,ax:bx]
        ring=np.ones(patch.shape,bool);ring[y0-ay:y1-ay,x0-ax:x1-ax]=False
        ring &= allowed[ay:by,ax:bx]>0
        if ring.sum()<20:continue
        background=float(np.median(patch[ring]))
        yy,xx=np.ogrid[y0:y1,x0:x1]
        inner=((xx-cx)/(max(3,bw*.40)))**2+((yy-cy)/(max(3,bh*.40)))**2<=1
        body=gray[y0:y1,x0:x1][inner]
        if body.size<20:continue
        contrast=background-float(np.percentile(body,40))
        fill=float((body<background-min_contrast).mean())
        if contrast<min_contrast or fill<min_fill:continue
        # Away from a cast, do not promote a dark shape with no water context.
        if anchor is None and pond_roi is None:
            if float((water[ay:by,ax:bx][ring]>0).mean())<.2:continue
        kept.append(f)
    return kept
