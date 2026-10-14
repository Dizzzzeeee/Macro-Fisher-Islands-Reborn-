"""Windows screen-only fishing assistant. Dry-run by default; F8 arms, F9 stops."""
import argparse,json,threading,time
from pathlib import Path
import cv2
from fisher.profile import load_profile
from fisher.state_machine import AutoFisher,State
from fisher.video import working_frame
from fisher.debug_view import draw_overlay
from fisher.detector import analyze_frame
from fisher.vision_worker import VisionWorker
from fisher.fast_tackle import line_visible_now


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--no-adaptive-lighting',action='store_true',help='disable automatic exposure/color correction')
    p.add_argument('--config',type=Path,help='JSON calibration profile')
    p.add_argument('--enable-clicks',action='store_true',help='allow mouse input; still starts paused')
    p.add_argument('--no-click',action='store_true',help='explicit vision-only mode (the default)')
    p.add_argument('--no-view',action='store_true')
    p.add_argument('--window',default='Roblox',help='unique game window title substring')
    p.add_argument('--mode',choices=['window','monitor'],default='window')
    p.add_argument('--monitor',type=int,default=1)
    p.add_argument('--region',default='',help='screen-pixel x,y,width,height; dry-run only')
    p.add_argument('--fps',type=float,default=60,help='target fresh-frame/bite loop FPS')
    p.add_argument('--vision-fps',type=float,default=10,help='background full-scene scans per second (5..30)')
    p.add_argument('--overlay-fps',type=float,default=10,help='maximum overlay redraw FPS')
    p.add_argument('--sync-vision',action='store_true',help='v2-style blocking full-scene loop for troubleshooting')
    p.add_argument('--aim',default='',help='cast aim as normalized x,y fractions (e.g. 0.64,0.70)')
    p.add_argument('--list-windows',action='store_true')
    p.add_argument('--log',type=Path,default=Path('logs/session.jsonl'))
    args=p.parse_args()
    if not args.window.strip():
        p.error("--window must not be empty")
    if not 1<=args.fps<=120:
        p.error('--fps must be 1..120')
    if not 5<=args.vision_fps<=30 or not 1<=args.overlay_fps<=60:
        p.error("--vision-fps must be 5..30; --overlay-fps must be 1..60")
    if args.enable_clicks and args.no_click:
        p.error('choose --enable-clicks or --no-click, not both')
    if args.enable_clicks and (args.mode!='window' or args.region):
        p.error('live clicks require full verified window capture; omit --region / --mode monitor')
    args.aim_point=None
    if args.aim:
        try:
            x,y=map(float,args.aim.split(','))
            if not (.05<x<.95 and .12<y<.83):
                raise ValueError()
            args.aim_point=x,y
        except ValueError:
            p.error('--aim needs two fractions inside the game area, e.g. 0.64,0.70')
    return args


def main():
    args=parse_args()
    cv2.setNumThreads(1)
    # Desktop dependencies are imported only for live mode. replay.py and
    # the vision/state tests work on Linux without a display or mouse library.
    from fisher import window as winutil
    if args.list_windows:
        for hwnd,title in winutil.list_windows():
            print(f'{hwnd}: {title}')
        return
    from fisher.capture import ScreenCapturer,parse_region
    from fisher.controller import Controller
    cap=ScreenCapturer(args.monitor,args.mode,args.window,parse_region(args.region))
    cfg=load_profile(args.config)
    if args.no_adaptive_lighting:
        cfg.adaptive_lighting=False
    cfg.fps=args.fps
    stopped=threading.Event();toggle=threading.Event()
    ctrl=Controller(cfg.click_cooldown_s,cap,enabled=args.enable_clicks,
                    stop_requested=lambda:stopped.is_set() or toggle.is_set())
    engine=AutoFisher(cfg)
    worker=None if args.sync_vision else VisionWorker(cfg,args.vision_fps)
    listener=None;log=None
    try:
        from pynput import keyboard
        def pressed(key):
            if key==keyboard.Key.f8:
                toggle.set()
            elif key==keyboard.Key.f9:
                stopped.set()
        listener=keyboard.Listener(on_press=pressed)
        listener.start()
        args.log.parent.mkdir(parents=True,exist_ok=True)
        log=args.log.open('a',buffering=1)
        print('Vision-only' if not args.enable_clicks else 'Mouse input allowed; PAUSED until F8 with game focused.')
        print('Equip rod, stand at pond, keep camera still. F8 arm/pause; F9 emergency stop; Q quits overlay.')
        enabled_run=not args.enable_clicks
        last_shape=None;last_region=None;observations=0
        next_view=0.;next_metrics=time.monotonic()+5;last_metrics_at=time.monotonic();metrics_frames=0
        latest_timing={}
        def reset_tracking():
            engine.reset()
            if worker is not None:worker.reset()
        while not stopped.is_set():
            t0=time.monotonic()
            if toggle.is_set():
                toggle.clear()
                if args.enable_clicks:
                    ctrl.armed=not ctrl.armed and cap.is_active()
                    enabled_run=ctrl.armed
                    print('ARMED' if ctrl.armed else 'PAUSED (focus the game to arm)')
                else:
                    enabled_run=not enabled_run
                reset_tracking()
            if args.enable_clicks and ctrl.armed and not cap.is_active():
                ctrl.armed=False;enabled_run=False;reset_tracking()
                print('Game lost focus. PAUSED; F8 to re-arm.')
            raw=cap.grab()
            captured_at=time.monotonic()
            capture_ms=(captured_at-t0)*1000
            frame=working_frame(raw,cfg.detect_width)
            region=tuple(cap.region[k] for k in ('left','top','width','height'))
            if last_shape is not None and (raw.shape!=last_shape or region!=last_region):
                reset_tracking()
                if args.enable_clicks:
                    ctrl.armed=False;enabled_run=False
                print('Window moved/resized; state reset. Re-arm with F8.')
            last_shape,last_region=raw.shape,region
            if not enabled_run:
                engine._set(State.PAUSED)
            scan_ms=0.;decision_start=time.monotonic()
            if worker is None:
                start_scan=time.monotonic();a=analyze_frame(frame,cfg)
                scan_ms=(time.monotonic()-start_scan)*1000
                analysis_at=captured_at;tackle_now=None
            else:
                worker.submit(frame,captured_at)
                result=worker.result()
                if result is None:
                    # Warm-up after launch/reset; no old analysis or clicks.
                    time.sleep(max(0,1/args.fps-(time.monotonic()-t0)))
                    continue
                a=result.analysis;analysis_at=result.captured_at;scan_ms=result.scan_ms
                tackle_now=line_visible_now(frame,a.string_seg) if a.string_seg is not None else None
            action,a,info=engine.step(frame,now=time.monotonic(),analysis=a,
                                     analysis_time=analysis_at,tackle_visible_now=tackle_now)
            latest_timing={'capture_ms':round(capture_ms,2),
                           'vision_scan_ms':round(scan_ms,2),
                           'vision_age_ms':info.get('vision_age_ms',0),
                           'decision_ms':round((time.monotonic()-decision_start)*1000,2),
                           'async_vision':worker is not None}
            if action!='none':
                sent=False
                if args.enable_clicks and enabled_run:
                    target=None
                    if action=='reel' and engine.bobber is not None:
                        bx,by=engine.bobber
                        target=cap.to_screen_coords(bx*raw.shape[1]/frame.shape[1],by*raw.shape[0]/frame.shape[0])
                    elif args.aim_point:
                        x,y=args.aim_point
                        target=cap.to_screen_coords(round(x*raw.shape[1]),round(y*raw.shape[0]))
                    else:
                        # Cast toward water to the side of the avatar. The
                        # default center could aim at the player/sand.
                        if a.player_bbox and a.rod_lines:
                            px,py,pw,ph=a.player_bbox
                            tipx=a.rod_lines[0][0]
                            side=1 if tipx>=px+pw/2 else -1
                            xx=max(50,min(frame.shape[1]-50,px+pw/2+side*180))
                            yy=min(frame.shape[0]*.8,py+ph*.95)
                            if a.water_mask[round(yy),round(xx)]>0:
                                target=cap.to_screen_coords(xx*raw.shape[1]/frame.shape[1],yy*raw.shape[0]/frame.shape[0])
                    if target:
                        dispatched_at=time.monotonic()
                        latest_timing['capture_to_dispatch_ms']=round((dispatched_at-captured_at)*1000,2)
                        sent=ctrl.click(*target,force=action=='reel')
                        latest_timing['click_call_ms']=round((time.monotonic()-dispatched_at)*1000,2)
                    else:
                        ctrl.reason='no safe cast target; specify --aim'
                    if not sent:
                        if ctrl.reason=='stop/pause requested':toggle.clear()
                        ctrl.armed=False;enabled_run=False;reset_tracking()
                        print(f'Action suppressed; paused: {ctrl.reason}')
                event={'time':time.time(),'action':action,'sent':sent,'metrics':info,'timing':latest_timing}
                log.write(json.dumps(event)+'\n')
                print(f'{action}: {"sent" if sent else "dry-run/suppressed"}; {info}')
            if engine.state==State.PAUSED and enabled_run:
                ctrl.armed=False;enabled_run=False
                print(f'Engine paused: {info.get("note","inspect game and re-arm")}')
            if not args.no_view:
                text=f'{"ARMED" if ctrl.armed else "DRY-RUN" if enabled_run else "PAUSED"} {info["next_state"]} casts={engine.casts} reel attempts={engine.catches_attempted}'
                if time.monotonic()>=next_view:
                    text+=f' | vision age {latest_timing["vision_age_ms"]:.0f}ms'
                    vis=draw_overlay(frame,a,engine.bite.roi_box(frame.shape,engine.roi_center),text)
                    cv2.imshow('Auto-Fisher | F8 arm/pause | F9 stop | Q quit',vis)
                    next_view=time.monotonic()+1/args.overlay_fps
                k=cv2.waitKey(1)&0xff
                if k in (ord('q'),27):
                    stopped.set()
                elif k==ord('t'):
                    Path('snapshots').mkdir(exist_ok=True)
                    cv2.imwrite(f'snapshots/snap_{int(time.time())}.png',raw)
            observations+=1;metrics_frames+=1
            if time.monotonic()>=next_metrics:
                stamp=time.monotonic()
                diagnostic={'time':time.time(),'type':'timing','loop_fps':round(metrics_frames/(stamp-last_metrics_at),2),
                            'target_fps':args.fps,**latest_timing}
                log.write(json.dumps(diagnostic)+'\n')
                last_metrics_at=stamp;next_metrics=stamp+5;metrics_frames=0
            time.sleep(max(0,1/args.fps-(time.monotonic()-t0)))
    finally:
        ctrl.armed=False
        if worker is not None:worker.close()
        if listener:
            listener.stop()
        if log:
            log.close()
        cap.close()
        cv2.destroyAllWindows()


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped.')
    except Exception as error:
        raise SystemExit(f'Stopped safely: {error}')
