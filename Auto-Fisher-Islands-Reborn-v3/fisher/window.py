"""Windows window lookup with explicit 64-bit-safe ctypes signatures."""
import ctypes
import sys
from ctypes import wintypes

user32=None
if sys.platform=='win32':
    user32=ctypes.WinDLL('user32',use_last_error=True)
    ENUMPROC=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
    user32.EnumWindows.argtypes=[ENUMPROC,wintypes.LPARAM]
    user32.EnumWindows.restype=wintypes.BOOL
    for name in ('IsWindow','IsWindowVisible','IsIconic'):
        getattr(user32,name).argtypes=[wintypes.HWND]
        getattr(user32,name).restype=wintypes.BOOL
    user32.GetWindowTextW.argtypes=[wintypes.HWND,wintypes.LPWSTR,ctypes.c_int]
    user32.GetWindowTextW.restype=ctypes.c_int
    user32.GetClientRect.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.restype=wintypes.BOOL
    user32.ClientToScreen.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype=wintypes.BOOL
    user32.GetForegroundWindow.restype=wintypes.HWND


def require_windows():
    if user32 is None:
        raise RuntimeError('Live window capture/clicking requires Windows. Use replay.py on other systems.')


def enable_dpi_awareness():
    if user32 is not None:
        try:
            user32.SetProcessDpiAwarenessContext.argtypes=[ctypes.c_void_p]
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError,OSError):
            user32.SetProcessDPIAware()


def list_windows():
    require_windows()
    out=[]
    @ENUMPROC
    def callback(hwnd,_):
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            buf=ctypes.create_unicode_buffer(1024)
            user32.GetWindowTextW(hwnd,buf,1024)
            title=buf.value.strip()
            if title:
                out.append((hwnd,title))
        return True
    user32.EnumWindows(callback,0)
    return out


def find_window(substring):
    matches=[(hwnd,title) for hwnd,title in list_windows() if substring.lower() in title.lower()]
    foreground=user32.GetForegroundWindow()
    for hwnd,title in matches:
        if hwnd==foreground:
            return hwnd,title
    if len(matches)>1:
        raise RuntimeError('Multiple matching windows. Focus the intended game or use a more specific --window title.')
    return matches[0] if matches else (None,'')


def get_client_rect_on_screen(hwnd):
    require_windows()
    if not hwnd or not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
        return None
    rect=wintypes.RECT();point=wintypes.POINT()
    if not user32.GetClientRect(hwnd,ctypes.byref(rect)) or not user32.ClientToScreen(hwnd,ctypes.byref(point)):
        return None
    w,h=rect.right-rect.left,rect.bottom-rect.top
    if w<50 or h<50:
        return None
    return point.x,point.y,w,h


def is_foreground(hwnd):
    return user32 is not None and hwnd is not None and user32.GetForegroundWindow()==hwnd and get_client_rect_on_screen(hwnd) is not None
