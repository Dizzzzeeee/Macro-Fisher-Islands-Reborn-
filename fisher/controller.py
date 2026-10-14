"""Click guard: opt-in, armed, verified foreground window, in-bounds target."""
import time


class Controller:
    def __init__(self,cooldown_s=1.2,capturer=None,enabled=False,stop_requested=None):
        self.cooldown_s=cooldown_s
        self.capturer=capturer
        self.enabled=enabled
        self.armed=False
        self._last=-100.0
        self.reason='not armed'
        self.stop_requested=stop_requested or (lambda:False)
        self._mouse=None
        if enabled:
            import pyautogui
            pyautogui.FAILSAFE=True
            pyautogui.PAUSE=.01
            self._mouse=pyautogui

    def can_click(self):
        return time.monotonic()-self._last>=self.cooldown_s

    def click(self,x=None,y=None,force=False):
        if self.stop_requested():
            self.armed=False
            self.reason='stop/pause requested'
            return False
        if not self.enabled or not self.armed:
            self.reason='clicking disabled or paused'
            return False
        if self.capturer is None or not self.capturer.is_active():
            self.armed=False
            self.reason='game lost focus; re-arm with F8'
            return False
        rect=self.capturer.refresh()
        if x is None or y is None:
            x,y=self.capturer.center_screen_coords()
        if not (rect['left']<=x<rect['left']+rect['width']
                and rect['top']<=y<rect['top']+rect['height']):
            self.reason='target outside capture'
            return False
        if not force and not self.can_click():
            self.reason='click cooldown'
            return False
        # Recheck immediately before dispatch; never bring another app into focus.
        if not self.capturer.is_active():
            self.armed=False
            self.reason='focus changed before click'
            return False
        if self.stop_requested():
            self.armed=False
            self.reason='stop/pause requested'
            return False
        self._mouse.click(int(x),int(y))
        self._last=time.monotonic()
        self.reason='sent'
        return True
