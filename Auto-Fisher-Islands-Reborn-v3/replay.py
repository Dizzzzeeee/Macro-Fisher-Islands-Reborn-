"""Offline observer. No capture, mouse, keyboard, or Windows imports."""
import argparse,json,time
from pathlib import Path
import cv2
from fisher.profile import load_profile
from fisher.state_machine import AutoFisher
from fisher.video import video_frames
from fisher.debug_view import draw_overlay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',type=Path)
    p.add_argument('--no-adaptive-lighting',action='store_true',help='disable automatic exposure/color correction')
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,default=Path('replay-output'))
    p.add_argument('--fps',type=float,default=30)
    p.add_argument('--overlay',action='store_true',help='save annotated MP4')
    args=p.parse_args()
    cv2.setNumThreads(1)
    if not 1<=args.fps<=120:
        p.error('--fps must be 1..120')
    args.output.mkdir(parents=True,exist_ok=True)
    meta=cv2.VideoCapture(str(args.video))
    source_fps=meta.get(cv2.CAP_PROP_FPS)
    meta.release()
    output_fps=min(args.fps,source_fps) if source_fps>0 else args.fps
    cfg=load_profile(args.config)
    if args.no_adaptive_lighting:
        cfg.adaptive_lighting=False
    engine=AutoFisher(cfg,observe_only=True)
    events=[]; frames=0;counts={'player':0,'rod':0,'fish':0,'bobber':0,'line_end':0};writer=None
    start=time.perf_counter()
    try:
        with (args.output/'frames.jsonl').open('w') as log:
            for timestamp,frame in video_frames(args.video,args.fps):
                action,a,info=engine.step(frame,now=timestamp)
                frames+=1
                for k,found in [('player',a.player_bbox is not None),('rod',bool(a.rod_lines)),('fish',bool(a.fishes)),('bobber',a.bobber is not None),('line_end',a.string_end is not None)]:
                    counts[k]+=int(found)
                row={'time':round(timestamp,4),'action':action,**info,'player':a.player_bbox,
                     'rod':a.rod_lines,'fish_centers':[f.center for f in a.fishes],
                     'line_end':a.string_end,'visible_bobber':a.bobber}
                log.write(json.dumps(row)+'\n')
                if action!='none':
                    events.append({'time':round(timestamp,4),'action':action,'metrics':info})
                    cv2.imwrite(str(args.output/f'{action}_{timestamp:.2f}.jpg'),draw_overlay(frame,a,engine.bite.roi_box(frame.shape,engine.roi_center),f'{timestamp:.2f}s {action}'))
                if args.overlay:
                    if writer is None:
                        writer=cv2.VideoWriter(str(args.output/'overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),output_fps,(frame.shape[1],frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError('Cannot create overlay video')
                    writer.write(draw_overlay(frame,a,engine.bite.roi_box(frame.shape,engine.roi_center),f'{timestamp:.2f}s {info["next_state"]} {action}'))
        summary={'frames':frames,'processing_seconds':round(time.perf_counter()-start,2),
                 'detections_in_frames':counts,'events':events,
                 'note':'Observed video; reel events are predictions, not confirmed catches. No input was sent.'}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2))
    finally:
        if writer is not None:
            writer.release()


if __name__=='__main__':
    main()
