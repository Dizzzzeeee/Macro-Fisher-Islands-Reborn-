"""Latest-frame background vision. Bounded memory, no input/control operations."""
import threading,time
from dataclasses import dataclass
from .detector import analyze_frame


@dataclass(frozen=True)
class VisionResult:
    analysis: object
    captured_at: float
    scan_ms: float
    generation: int


class VisionWorker:
    def __init__(self,cfg,fps=10,analyzer=analyze_frame):
        self.cfg=cfg;self.period=1/fps;self.analyzer=analyzer
        self._condition=threading.Condition();self._pending=None;self._result=None
        self._generation=0;self._stop=False;self._error=None
        self._thread=threading.Thread(target=self._run,name='vision-latest-frame',daemon=True)
        self._thread.start()

    def submit(self,frame,captured_at):
        with self._condition:
            if self._stop:return
            # Only newest pending frame survives; never accumulate lag frames.
            self._pending=(frame.copy(),float(captured_at),self._generation)
            self._condition.notify()

    def result(self):
        with self._condition:
            if self._error is not None:raise RuntimeError(f'Vision worker failed: {self._error}')
            return self._result

    def reset(self):
        with self._condition:
            self._generation+=1;self._pending=self._result=None
            self._condition.notify()

    def _run(self):
        next_scan=0.
        while True:
            with self._condition:
                while not self._stop:
                    wait=next_scan-time.monotonic()
                    if self._pending is not None and wait<=0:break
                    self._condition.wait(timeout=max(.001,wait) if self._pending is not None else None)
                if self._stop:return
                frame,stamp,generation=self._pending;self._pending=None
            start=time.monotonic()
            try:
                analysis=self.analyzer(frame,self.cfg)
            except Exception as error:
                with self._condition:self._error=error;self._stop=True;self._condition.notify_all()
                return
            result=VisionResult(analysis,stamp,(time.monotonic()-start)*1000,generation)
            with self._condition:
                if not self._stop and generation==self._generation:self._result=result
            next_scan=start+self.period

    def close(self):
        with self._condition:self._stop=True;self._pending=None;self._condition.notify_all()
        self._thread.join(timeout=2)
