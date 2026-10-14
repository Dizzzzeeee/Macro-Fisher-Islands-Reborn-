"""Capture the selected game window. Never silently fall back to a desktop."""
import numpy as np
from mss import mss
from . import window as winutil


def parse_region(value):
    if not value:
        return None
    try:
        x,y,w,h=map(int,value.split(','))
    except ValueError:
        raise ValueError('region must be x,y,width,height') from None
    if w<=0 or h<=0:
        raise ValueError('region width and height must be positive')
    return {'left':x,'top':y,'width':w,'height':h}


class ScreenCapturer:
    def __init__(self, monitor_index=1,mode='window',window_title_sub='Roblox',
                 region=None,follow_window=True,follow_every=30):
        self.mode=mode
        self.window_title_sub=window_title_sub
        self.hwnd=None
        self.manual_region=region
        self.window_title=''
        if mode=='window':
            winutil.require_windows()
            winutil.enable_dpi_awareness()
        self.sct=mss()
        try:
            if mode=='window':
                self.hwnd,self.window_title=winutil.find_window(window_title_sub)
                rect=winutil.get_client_rect_on_screen(self.hwnd)
                if rect is None:
                    raise RuntimeError(f'Game window "{window_title_sub}" not found or minimized; no desktop fallback.')
                self.region=region or self._region(rect)
            elif mode=='monitor':
                if not 1<=monitor_index<len(self.sct.monitors):
                    raise ValueError(f'Monitor index must be 1..{len(self.sct.monitors)-1}')
                self.region=region or dict(self.sct.monitors[monitor_index])
            else:
                raise ValueError('mode must be window or monitor')
        except Exception:
            self.sct.close()
            raise
        self.offset_x,self.offset_y=self.region['left'],self.region['top']
        self._last_rect=None

    @staticmethod
    def _region(rect):
        x,y,w,h=rect
        return {'left':x,'top':y,'width':w,'height':h}

    def refresh(self):
        if self.mode=='window':
            rect=winutil.get_client_rect_on_screen(self.hwnd)
            if rect is None:
                raise RuntimeError('Game window was closed or minimized; capture stopped.')
            if self.manual_region is None:
                self.region=self._region(rect)
                self.offset_x,self.offset_y=rect[:2]
        return self.region

    def grab(self):
        self.refresh()
        return np.asarray(self.sct.grab(self.region))[:,:,:3].copy()

    def is_active(self):
        return self.mode=='window' and winutil.is_foreground(self.hwnd)

    def center_screen_coords(self):
        self.refresh()
        return self.to_screen_coords(self.region['width']//2,self.region['height']//2)

    def to_screen_coords(self,x,y):
        return self.region['left']+int(x),self.region['top']+int(y)

    def close(self):
        self.sct.close()
