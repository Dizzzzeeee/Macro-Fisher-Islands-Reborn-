"""Desktop-independent video replay and frame normalization."""
import cv2


def working_frame(frame, width=1280):
    h,w=frame.shape[:2]
    if w==width:
        return frame
    return cv2.resize(frame,(width,round(h*width/w)),interpolation=cv2.INTER_AREA if w>width else cv2.INTER_LINEAR)


def video_frames(path, rate=30):
    capture=cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    fps=capture.get(cv2.CAP_PROP_FPS)
    if fps<=0:
        capture.release()
        raise ValueError('Video has no usable frame rate')
    index=0;next_sample=0.0
    try:
        while True:
            ok,frame=capture.read()
            if not ok:
                break
            timestamp=index/fps
            index+=1
            if timestamp+1e-6<next_sample:
                continue
            next_sample+=1/rate
            yield timestamp,working_frame(frame)
    finally:
        capture.release()
