"""Recheck bite/state logic against cached replay detections, with no desktop I/O.

The cache must come from replay.py for this exact video at a constant sample
rate. Water masks/fish boxes are not cached: overlay shows actual cached fish
centres, player, rod, line and float markers only, with no water segmentation.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from fisher.profile import load_profile
from fisher.detector import FrameAnalysis,FishDetection
from fisher.state_machine import AutoFisher
from fisher.video import video_frames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',type=Path);p.add_argument('cache',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path);p.add_argument('--overlay',action='store_true')
    args=p.parse_args();cv2.setNumThreads(1)
    rows=[json.loads(line) for line in args.cache.read_text().splitlines() if line.strip()]
    if len(rows)<2:raise ValueError('Need at least two cached frames')
    rate=round(1/float(np.median(np.diff([r['time'] for r in rows]))),3)
    meta=cv2.VideoCapture(str(args.video));source_fps=meta.get(cv2.CAP_PROP_FPS);meta.release()
    if source_fps>0:rate=min(rate,source_fps)
    args.output.mkdir(parents=True,exist_ok=True)
    engine=AutoFisher(load_profile(args.config),observe_only=True)
    events=[];writer=None;processed=0
    try:
        with (args.output/'frames.jsonl').open('w') as log:
            for row,(timestamp,frame) in zip(rows,video_frames(args.video,rate)):
                if abs(timestamp-row['time'])>.002:
                    raise ValueError('Video/cache timestamps differ; use the original replay sampling rate')
                rod=[tuple(l) for l in row['rod']];end=row.get('line_end')
                light=row.get('lighting')
                a=FrameAnalysis(water_mask=np.zeros(frame.shape[:2],np.uint8),
                    player_bbox=tuple(row['player']) if row['player'] else None,
                    fishes=[FishDetection(tuple(pt),(0,0,0,0),0) for pt in row['fish_centers']],
                    rod_lines=rod,bobber=tuple(row['visible_bobber']) if row.get('visible_bobber') else None,
                    string_end=tuple(end) if end else None,
                    string_seg=(*rod[0][:2],*end) if end and rod else None,lighting=light)
                action,_,info=engine.step(frame,now=timestamp,analysis=a)
                log.write(json.dumps({'time':round(timestamp,4),'action':action,**info})+'\n')
                processed+=1
                if action!='none':events.append({'time':round(timestamp,4),'action':action,'metrics':info})
                if args.overlay or action!='none':
                    vis=frame.copy()
                    if a.player_bbox:
                        x,y,w,h=a.player_bbox;cv2.rectangle(vis,(x,y),(x+w,y+h),(255,255,255),2)
                    for x1,y1,x2,y2 in rod:cv2.line(vis,(x1,y1),(x2,y2),(30,120,230),3)
                    if a.string_seg:
                        x1,y1,x2,y2=a.string_seg;cv2.line(vis,(x1,y1),(x2,y2),(230,60,230),2)
                    if a.bobber:cv2.circle(vis,a.bobber,7,(0,255,255),2)
                    for f in a.fishes:cv2.circle(vis,f.center,5,(50,50,255),2)
                    box=engine.bite.roi_box(frame.shape,engine.roi_center)
                    if box:cv2.rectangle(vis,box[:2],box[2:],(40,255,80),2)
                    cv2.rectangle(vis,(0,0),(vis.shape[1],90),(22,22,22),-1)
                    cv2.putText(vis,f'{timestamp:.2f}s | {info["next_state"]} | {action.upper()}',(15,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(90,255,160),2)
                    cv2.putText(vis,'Cached detector review | fish centres only | no water mask',(15,50),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,230,230),1)
                    if light:cv2.putText(vis,f'Light V={light["scene_v"]} | correction={light["corrected"]} | gains={light["gains_bgr"]}',(15,73),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,230,230),1)
                    if action!='none':cv2.imwrite(str(args.output/f'{action}_{timestamp:.2f}.jpg'),vis)
                    if args.overlay:
                        if writer is None:
                            writer=cv2.VideoWriter(str(args.output/'overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),rate,(frame.shape[1],frame.shape[0]))
                            if not writer.isOpened():raise RuntimeError('Cannot create review video')
                        writer.write(vis)
        if processed!=len(rows):raise ValueError('Video ended before the cache')
        summary={'frames':processed,'sample_fps':rate,'events':events,
                 'note':'Final temporal logic rechecked on raw video frames with cached detector geometry. No clicks. Overlay omits uncached water masks and fish bounding boxes. Predictions are not confirmed catches.'}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print('Rechecked',processed,'frames; predicted events:',[(e['time'],e['action']) for e in events])
    finally:
        if writer:writer.release()


if __name__=='__main__':main()
