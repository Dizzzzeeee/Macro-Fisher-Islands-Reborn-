"""Overlay painter for tuning: water, player, fish, rod, bobber, ROI."""
import cv2
from .detector import FrameAnalysis

def draw_overlay(frame, analysis: FrameAnalysis, bite_roi=None, state_text: str = ""):
    vis = frame.copy()
    water=analysis.water_mask>0
    if water.any():
        tint=vis.copy();tint[water]=(190,130,30)
        vis=cv2.addWeighted(vis,.88,tint,.12,0)
        cv2.putText(vis, "blue tint: candidate water", (12, vis.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX,.55,(255,210,130),1)
    if analysis.player_bbox:
        x, y, w, h = analysis.player_bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(vis, "player", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for f in analysis.fishes:
        x, y, w, h = f.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (60, 60, 60), 2)
        cv2.circle(vis, f.center, 4, (0, 0, 255), -1)
        cv2.putText(vis, "fish", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for (x1, y1, x2, y2) in analysis.rod_lines:
        cv2.line(vis, (x1, y1), (x2, y2), (42, 100, 180), 2)
    if analysis.rod_lines:
        cv2.putText(vis, "rod", (analysis.rod_lines[0][0], analysis.rod_lines[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (42, 100, 180), 2)
    if analysis.string_end:
        cv2.circle(vis, analysis.string_end, 5, (255, 0, 255), 2)
        cv2.putText(vis, "line-end (estimate)", (analysis.string_end[0] + 8, analysis.string_end[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    if getattr(analysis, "string_seg", None):
        x1, y1, x2, y2 = analysis.string_seg
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
    if analysis.bobber:
        cv2.circle(vis, analysis.bobber, 8, (0, 255, 255), 2)
        cv2.circle(vis, analysis.bobber, 2, (0, 255, 255), -1)
        cv2.putText(vis, f"bobber {analysis.bobber_score:.2f}",
                    (analysis.bobber[0] + 10, analysis.bobber[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    if bite_roi:
        x0, y0, x1, y1 = bite_roi
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 1)
    if analysis.lighting:
        light=analysis.lighting
        cv2.putText(vis,f"Light V={light['scene_v']:.0f} | auto={light['corrected']} | gains={light['gains_bgr']}",
                    (12,52),cv2.FONT_HERSHEY_SIMPLEX,.52,(220,240,255),1)
    if state_text:
        cv2.putText(vis, state_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return vis
