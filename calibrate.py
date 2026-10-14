"""Inspect a screenshot, constrain the pond, or create a bobber template."""
import argparse
from pathlib import Path
import cv2
from fisher.detector import analyze_frame
from fisher.debug_view import draw_overlay
from fisher.video import working_frame
from fisher.profile import load_profile,save_profile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--test',type=Path,help='screenshot to inspect; no desktop needed')
    source.add_argument('--capture',action='store_true',help='take a game-window screenshot')
    p.add_argument('--window',default='Roblox')
    p.add_argument('--no-adaptive-lighting',action='store_true',help='disable automatic exposure/color correction')
    p.add_argument('--config',type=Path)
    p.add_argument('--save-config',type=Path,default=Path('profile.json'))
    p.add_argument('--output',type=Path,default=Path('calibration-preview.jpg'))
    p.add_argument('--view',action='store_true',help='open annotated preview')
    p.add_argument('--select-pond',action='store_true',help='drag rectangle covering ONLY fishable water')
    p.add_argument('--crop-template',choices=['bobber'],help='drag tight crop of visible bobber, excluding mouse/UI')
    p.add_argument('--tune',action='store_true',help='tune HSV and bite sensitivity; S saves, Q exits')
    args=p.parse_args();cv2.setNumThreads(1)
    cfg=load_profile(args.config)
    if args.no_adaptive_lighting:
        cfg.adaptive_lighting=False
    if args.capture:
        from fisher.capture import ScreenCapturer
        cap=ScreenCapturer(window_title_sub=args.window)
        try:
            raw=cap.grab()
        finally:
            cap.close()
        cv2.imwrite('calibration-source.png',raw)
    else:
        raw=cv2.imread(str(args.test))
        if raw is None:
            p.error(f'Cannot read {args.test}')
    frame=working_frame(raw)
    if args.select_pond or args.crop_template:
        x,y,w,h=map(int,cv2.selectROI('Drag rectangle; Enter accepts; Esc cancels',frame,False,False))
        cv2.destroyAllWindows()
        if not w or not h:
            raise SystemExit('Selection canceled; nothing changed.')
        if args.select_pond:
            cfg.pond_roi=(x/frame.shape[1],y/frame.shape[0],w/frame.shape[1],h/frame.shape[0])
            save_profile(cfg,args.save_config)
            print(f'Saved pond constraint: {args.save_config}')
        else:
            crop=frame[y:y+h,x:x+w]
            if min(w,h)<3 or max(w,h)>100 or cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).std()<5:
                raise SystemExit('Choose a textured, tight crop (3..100 pixels), not a flat-color dot or whole scene.')
            dest=Path(__file__).parent/'fisher/assets/templates/bobber.png'
            dest.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(dest),crop)
            print(f'Saved bobber template: {dest}')
    if args.tune:
        title='Calibration | S save | Q quit'
        cv2.namedWindow(title)
        for label,value,limit in [('fish V',cfg.colors.fish_v_max,255),('water H low',cfg.colors.water_h_low,179),
                ('water H high',cfg.colors.water_h_high,179),('rod S min',cfg.colors.rod_s_min,255),
                ('bubble x10000',round(cfg.bite_relative_jump*10000),1000),
                ('confirm frames',cfg.bite_confirm_frames,10)]:
            cv2.createTrackbar(label,title,value,limit,lambda value:None)
        while True:
            cfg.colors.fish_v_max=max(1,cv2.getTrackbarPos('fish V',title))
            cfg.colors.water_h_low=cv2.getTrackbarPos('water H low',title)
            cfg.colors.water_h_high=max(cfg.colors.water_h_low,cv2.getTrackbarPos('water H high',title))
            cfg.colors.rod_s_min=cv2.getTrackbarPos('rod S min',title)
            cfg.bite_relative_jump=max(.0001,cv2.getTrackbarPos('bubble x10000',title)/10000)
            cfg.bite_confirm_frames=max(1,cv2.getTrackbarPos('confirm frames',title))
            a=analyze_frame(frame,cfg)
            cv2.imshow(title,draw_overlay(frame,a,None,'S saves profile. Bubble sensitivity must be checked in replay.'))
            key=cv2.waitKey(30)&255
            if key==ord('s'):
                save_profile(cfg,args.save_config);print(f'Saved {args.save_config}')
            if key in (ord('q'),27):
                break
        cv2.destroyAllWindows()
    a=analyze_frame(frame,cfg)
    preview=draw_overlay(frame,a,None,'Preview only; no mouse input')
    if not cv2.imwrite(str(args.output),preview):
        raise RuntimeError(f'Cannot write {args.output}')
    print(f'Player: {a.player_bbox}; rod: {a.rod_lines}; fish: {[f.center for f in a.fishes]}')
    print(f'Visible bobber: {a.bobber}; estimated endpoint: {a.string_end}; saved {args.output}')
    if args.view:
        cv2.imshow('Preview (any key closes)',preview);cv2.waitKey(0);cv2.destroyAllWindows()


if __name__=='__main__':
    main()
