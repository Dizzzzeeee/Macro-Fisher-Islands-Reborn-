"""Small JSON calibration profiles; never execute code from a profile."""
import json,math
from pathlib import Path
from .config import FisherConfig

FIELDS={'fish_search_radius':(150,800),'fish_min_contrast':(3,40),'fish_min_fill':(.1,.7),
        'bite_relative_jump':(.0001,.2), 'bite_confirm_frames':(1,10),
        'cast_settle_s':(.1,10), 'bite_timeout_s':(5,300),
        'bite_roi_radius':(15,120), 'bite_fish_px':(15,250)}
COLORS={'fish_v_max':(1,255), 'fish_min_area':(5,30000), 'fish_max_area':(10,100000),
        'water_h_low':(0,179),'water_h_high':(0,179),'water_s_low':(0,255),
        'water_s_high':(0,255),'water_v_low':(0,255),'water_v_high':(0,255),
        'rod_h_low':(0,179),'rod_h_high':(0,179),'rod_s_min':(0,255),
        'rod_v_min':(0,255),'rod_v_max':(0,255)}


def load_profile(path=None):
    cfg=FisherConfig()
    if path is None:
        return cfg
    data=json.loads(Path(path).read_text())
    if not isinstance(data,dict) or set(data)-set(FIELDS)-{'colors','pond_roi','adaptive_lighting'}:
        raise ValueError('Profile contains unsupported settings')
    def apply(obj,values,ranges):
        if not isinstance(values,dict) or set(values)-set(ranges):
            raise ValueError('Unsupported color settings')
        for key,value in values.items():
            low,high=ranges[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f'{key} must be numeric and within {low}..{high}')
            if isinstance(getattr(obj,key),int) and not float(value).is_integer():
                raise ValueError(f'{key} must be a whole number')
            setattr(obj,key,int(value) if isinstance(getattr(obj,key),int) else float(value))
    apply(cfg,{k:v for k,v in data.items() if k in FIELDS},FIELDS)
    apply(cfg.colors,data.get('colors',{}),COLORS)
    adaptive=data.get('adaptive_lighting',True)
    if not isinstance(adaptive,bool):
        raise ValueError('adaptive_lighting must be true or false')
    cfg.adaptive_lighting=adaptive
    roi=data.get('pond_roi')
    if roi is not None:
        if (not isinstance(roi,list) or len(roi)!=4 or any(
                isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in roi)):
            raise ValueError('pond_roi must be [x,y,width,height] fractions')
        x,y,w,h=roi
        if w<=0 or h<=0 or x+w>1 or y+h>1:
            raise ValueError('pond_roi must be a positive rectangle within the image')
        cfg.pond_roi=tuple(roi)
    return cfg


def save_profile(cfg,path):
    data={key:getattr(cfg,key) for key in FIELDS}
    data['colors']={key:getattr(cfg.colors,key) for key in COLORS}
    data['pond_roi']=cfg.pond_roi
    data['adaptive_lighting']=cfg.adaptive_lighting
    Path(path).write_text(json.dumps(data,indent=2)+'\n')
