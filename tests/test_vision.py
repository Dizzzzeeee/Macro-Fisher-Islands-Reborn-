"""Regression fixtures sampled from the user-supplied recording.

These are a few known scenes, not an independent accuracy benchmark.
"""
import unittest
from pathlib import Path
import cv2
import numpy as np
from fisher.config import FisherConfig
from fisher.detector import analyze_frame

FIXTURES=Path(__file__).parent/'fixtures'


class VisionTests(unittest.TestCase):
    def test_known_avatar_and_pole(self):
        for t in (0,9,21):
            with self.subTest(t=t):
                frame=cv2.imread(str(FIXTURES/f'gameplay_{t:02d}.png'))
                self.assertIsNotNone(frame)
                a=analyze_frame(frame,FisherConfig())
                self.assertIsNotNone(a.player_bbox)
                x,y,w,h=a.player_bbox
                self.assertTrue(610<=x<=640 and 300<=y<=330)
                self.assertTrue(a.rod_lines)

    def test_known_cast_endpoint(self):
        for t,expected in ((9,(820,468)),(21,(782,463))):
            with self.subTest(t=t):
                a=analyze_frame(cv2.imread(str(FIXTURES/f'gameplay_{t:02d}.png')),FisherConfig())
                self.assertIsNotNone(a.string_end)
                self.assertLess(np.hypot(a.string_end[0]-expected[0],a.string_end[1]-expected[1]),35)

    def test_held_upright_is_not_a_cast_line(self):
        a=analyze_frame(cv2.imread(str(FIXTURES/'gameplay_00.png')),FisherConfig())
        self.assertIsNone(a.string_end)

    def test_empty_scene_has_no_player_or_tackle(self):
        a=analyze_frame(np.zeros((669,1280,3),np.uint8),FisherConfig())
        self.assertIsNone(a.player_bbox)
        self.assertFalse(a.rod_lines)
        self.assertIsNone(a.bobber)
        self.assertIsNone(a.string_end)

    def test_pond_constraint_limits_segmentation(self):
        cfg=FisherConfig(pond_roi=(.6,.5,.2,.2))
        a=analyze_frame(cv2.imread(str(FIXTURES/'gameplay_09.png')),cfg)
        self.assertFalse(a.water_mask[:round(669*.5)].any())
        self.assertFalse(a.water_mask[:,:round(1280*.6)].any())


if __name__=='__main__':unittest.main()
