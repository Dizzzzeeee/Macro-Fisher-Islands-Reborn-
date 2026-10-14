"""Current-frame cast evidence for a cached line; no whole-scene recognition."""
import cv2
import numpy as np


def line_visible_now(frame, segment):
    if segment is None:return False
    x1,y1,x2,y2=segment;dx,dy=x2-x1,y2-y1
    length=float(np.hypot(dx,dy))
    if length<100:return False
    h,w=frame.shape[:2];ux,uy=dx/length,dy/length;nx,ny=-uy,ux
    # Only the upper/middle cast: the fish can cover the bottom at bite time.
    t=np.linspace(.14,.60,40)
    xs=x1+dx*t;ys=y1+dy*t
    x0=max(0,int(xs.min())-12);xx=min(w,int(xs.max())+13)
    y0=max(0,int(ys.min())-12);yy=min(h,int(ys.max())+13)
    if xx<=x0 or yy<=y0:return False
    gray=cv2.cvtColor(frame[y0:yy,x0:xx],cv2.COLOR_BGR2GRAY)
    def sample(offset):
        x=np.rint(xs+nx*offset).astype(int)-x0
        y=np.rint(ys+ny*offset).astype(int)-y0
        return gray[np.clip(y,0,gray.shape[0]-1),np.clip(x,0,gray.shape[1]-1)].astype(float)
    centre=np.minimum.reduce([sample(-2),sample(0),sample(2)])
    background=(sample(-7)+sample(7))/2
    hits=background-centre>5
    # Several supported samples, not one random glint or fish pixel.
    return bool(hits.mean()>=.35 and np.convolve(hits.astype(int),np.ones(8,int),mode='valid').max()>=5)
