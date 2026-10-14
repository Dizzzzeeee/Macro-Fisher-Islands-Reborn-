import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from fisher.config import FisherConfig
from fisher.video import video_frames
from fisher.detector import analyze_frame
from fisher.state_machine import AutoFisher
from fisher.fast_tackle import line_visible_now
cv2.setNumThreads(1)
import argparse
parser=argparse.ArgumentParser(description='Offline v3 short-bite regression; never sends input.')
parser.add_argument('--day',type=Path,required=True,help='Original 45-second daylight recording')
parser.add_argument('--night',type=Path,required=True,help='Original 139-second night recording')
parser.add_argument('--output',type=Path,default=Path('validation/v3/short-replays.json'))
args=parser.parse_args()
clips=[('night',args.night,5),('day',args.day,12)]
rows=[]
for label,path,end in clips:
    cfg=FisherConfig();sync=AutoFisher(cfg,observe_only=True);cached=AutoFisher(cfg,observe_only=True)
    events=[];cache_events=[];pending=[];latest=None;frames=0
    for t,frame in video_frames(path,30):
        if t>end:break
        a=analyze_frame(frame,cfg)
        action,_,_=sync.step(frame,t,a)
        if action!='none':events.append({'time':round(t,4),'action':action})
        # Deterministic 10Hz scene results with 80ms synthetic processing lag;
        # fresh ROI checks at 60Hz using duplicated 30FPS video frames.
        if frames%3==0:pending.append((t+.08,t,a))
        for stamp in (t,t+1/60):
            while pending and pending[0][0]<=stamp:
                _,source,analysis=pending.pop(0);latest=(source,analysis)
            if latest:
                source,analysis=latest
                visible=line_visible_now(frame,analysis.string_seg) if analysis.string_seg else None
                act,_,_=cached.step(frame,stamp,analysis,analysis_time=source,tackle_visible_now=visible)
                if act!='none':cache_events.append({'time':round(stamp,4),'action':act})
        frames+=1
    rows.append({'clip':label,'seconds_checked':end,'sample_fps':30,'frames':frames,'events':events,'simulated_cached_events':cache_events,'simulation_note':'10Hz cached geometry, synthetic 80ms scan delay, 60Hz checks on duplicated 30FPS video; not a live benchmark.'})
    print(label,events,cache_events,flush=True)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(rows,indent=2))
