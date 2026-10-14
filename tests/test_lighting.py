"""Warm/dim footage and bounded correction regressions, not accuracy claims."""
import unittest
from pathlib import Path
import cv2
import numpy as np
from fisher.config import FisherConfig
from fisher.detector import analyze_frame
from fisher.illumination import compensate
from fisher.bite_detector import BiteDetector

FIXTURES=Path(__file__).parent/'fixtures'


class LightingTests(unittest.TestCase):
    def test_full_warm_avatar_and_connected_pole(self):
        for ts in (0,40,80,130):
            with self.subTest(ts=ts):
                frame=cv2.imread(str(FIXTURES/f'night_{ts:03d}.png'))
                a=analyze_frame(frame,FisherConfig())
                self.assertIsNotNone(a.player_bbox)
                x,y,w,h=a.player_bbox
                self.assertTrue(610<x<640 and 480<y<525 and h>140)
                self.assertTrue(a.rod_lines)
                tipx,tipy,bx,by=a.rod_lines[0]
                self.assertTrue(710<tipx<755 and 385<tipy<445)
                self.assertTrue(670<bx<710 and 540<by<590)
                self.assertTrue(a.lighting['corrected'])

    def test_night_float_or_endpoint_recovered(self):
        for ts in (0,40,80,130):
            with self.subTest(ts=ts):
                a=analyze_frame(cv2.imread(str(FIXTURES/f'night_{ts:03d}.png')),FisherConfig())
                self.assertIsNotNone(a.string_end)
                self.assertTrue(740<a.string_end[0]<870 and 580<a.string_end[1]<720)

    def test_no_reference_keeps_pixels_unchanged(self):
        frame=np.full((100,100,3),70,np.uint8)
        out,info=compensate(frame,None)
        self.assertTrue(np.array_equal(frame,out))
        self.assertFalse(info['corrected'])

    def test_disabled_correction_keeps_pixels_unchanged(self):
        frame=np.full((100,100,3),(95,105,120),np.uint8)
        out,info=compensate(frame,(10,10,50,70),enabled=False)
        self.assertTrue(np.array_equal(frame,out))
        self.assertFalse(info['corrected'])

    def test_gains_bounded_and_neutralized(self):
        frame=np.full((100,100,3),(95,105,120),np.uint8)
        out,info=compensate(frame,(10,10,50,70))
        self.assertTrue(info['corrected'])
        self.assertLessEqual(max(info['gains_bgr']),2.5)
        self.assertLess(int(np.ptp(out[30,30].astype(int))),3)

    def test_daylight_reference_is_preserved(self):
        frame=np.full((100,100,3),210,np.uint8)
        out,info=compensate(frame,(10,10,50,70))
        self.assertFalse(info['corrected'])
        self.assertTrue(np.array_equal(frame,out))

    def test_exposure_shift_resets_bite_history(self):
        cfg=FisherConfig();b=BiteDetector(cfg)
        base=np.full((180,180,3),130,np.uint8)
        cv2.ellipse(base,(90,90),(27,15),0,0,360,(55,55,55),-1)
        for _ in range(8):b.update(base,(90,90),True)
        brighter=(base.astype(np.int16)+12).clip(0,255).astype(np.uint8)
        bitten,info=b.update(brighter,(90,90),True)
        self.assertFalse(bitten)
        self.assertEqual(info.get('reset'),'illumination change')

    def test_dim_bubbles_use_bounded_local_thresholds(self):
        b=BiteDetector(FisherConfig());base=np.full((180,180,3),84,np.uint8)
        cv2.ellipse(base,(90,90),(27,15),0,0,360,(38,38,38),-1)
        for _ in range(8):b.update(base,(90,90),True)
        splash=base.copy()
        for x,y in ((80,85),(92,83),(103,94),(88,97)):
            cv2.circle(splash,(x,y),4,(90,90,90),2)
        hit,info=b.update(splash,(90,90),True)
        self.assertFalse(hit)
        self.assertTrue(b.update(splash,(90,90),True)[0])
        self.assertGreaterEqual(info['contrast_threshold'],10)
        self.assertLess(info['contrast_threshold'],18)

    def test_dimmed_tinted_daylight_pole_survives(self):
        day=cv2.imread(str(FIXTURES/'gameplay_09.png'))
        gains=np.array([.55,.67,.78])
        tinted=(day*gains).clip(0,255).astype(np.uint8)
        a=analyze_frame(tinted,FisherConfig())
        self.assertIsNotNone(a.player_bbox)
        self.assertTrue(a.rod_lines)


if __name__=='__main__':unittest.main()
