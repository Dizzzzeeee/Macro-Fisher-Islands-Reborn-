import json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
import cv2
import numpy as np
from fisher.config import FisherConfig
from fisher.detector import FrameAnalysis,FishDetection
from fisher.state_machine import AutoFisher,State
from fisher.bite_detector import BiteDetector
from fisher.controller import Controller
from fisher.profile import load_profile,save_profile
from fisher.video import working_frame


def analysis(line=False,player=True,rod=True,water=True):
    return FrameAnalysis(water_mask=np.full((200,400),255 if water else 0,np.uint8),
        player_bbox=(180,90,30,70) if player else None,fishes=[],
        rod_lines=[(210,40,210,125)] if rod else [],bobber=None,
        string_end=(300,130) if line else None,string_seg=(210,40,300,130) if line else None)


class StateTests(unittest.TestCase):
    def setUp(self):
        self.cfg=FisherConfig(cast_settle_s=.1,bite_timeout_s=5,nobobber_timeout_s=1)
        self.engine=AutoFisher(self.cfg)
        self.frame=np.zeros((200,400,3),np.uint8)
        self.engine.bite.update=Mock(return_value=(False,{}))

    def step(self,t,a):
        return self.engine.step(self.frame,t,a)[0]

    def test_no_cast_without_player(self):
        for t in (0,.5,2):self.assertEqual(self.step(t,analysis(player=False)),'none')

    def test_no_cast_without_water(self):
        for t in (0,.5,2):self.assertEqual(self.step(t,analysis(water=False)),'none')

    def test_no_cast_without_equipped_rod(self):
        for t in (0,.5,2):self.assertEqual(self.step(t,analysis(rod=False)),'none')

    def test_upright_rod_needs_stable_readiness(self):
        self.assertEqual(self.step(0,analysis()),'none')
        self.assertEqual(self.step(.36,analysis()),'cast')
        self.assertEqual(self.step(.4,analysis()),'none')
        self.assertEqual(self.engine.casts,1)

    def test_existing_line_never_casts(self):
        for t in (0,.2,.4,.6):self.assertEqual(self.step(t,analysis(line=True)),'none')
        self.assertEqual(self.engine.state,State.WAITING)
        self.assertEqual(self.engine.casts,0)

    def test_visible_line_timeout_pauses_not_recasts(self):
        for t in (0,.2,.4,6):self.assertEqual(self.step(t,analysis(line=True)),'none')
        self.assertEqual(self.engine.state,State.PAUSED)

    def test_one_reel_per_event(self):
        a=analysis(line=True)
        for t in (0,.2,.4):self.step(t,a)
        self.engine.bite.update=Mock(return_value=(True,{}))
        self.assertEqual(self.step(.5,a),'reel')
        for t in (.6,.9,2):self.assertEqual(self.step(t,a),'none')
        self.assertEqual(self.engine.catches_attempted,1)

    def test_observer_never_casts(self):
        self.engine.observe_only=True
        for t in (0,.5,2):self.assertEqual(self.step(t,analysis()),'none')

    def test_moving_fish_not_bite_ready(self):
        a=analysis(line=True)
        for t in (0,.2,.4):self.step(t,a)
        for t,x in ((.5,260),(.6,270),(.7,280),(.8,290),(.9,300),(1,310),(1.1,320)):
            a.fishes=[FishDetection((x,130),(x-10,120,20,20),300)]
            self.step(t,a)
        self.assertFalse(self.engine.last_info['stable_fish'])


class BiteTests(unittest.TestCase):
    def setUp(self):
        self.detector=BiteDetector(FisherConfig())
        self.base=np.full((180,180,3),130,np.uint8)
        cv2.ellipse(self.base,(90,90),(27,15),0,0,360,(55,55,55),-1)
        self.anchor=(90,90)
        for _ in range(8):self.detector.update(self.base,self.anchor,True)

    def test_static_fish_and_float_do_not_trigger(self):
        for _ in range(15):self.assertFalse(self.detector.update(self.base,self.anchor,True)[0])

    def test_bubble_requires_nearby_fish(self):
        splash=self.splash()
        for _ in range(5):self.assertFalse(self.detector.update(splash,self.anchor,False)[0])

    def splash(self):
        splash=self.base.copy()
        for x,y in ((80,85),(92,83),(103,94),(88,97)):
            cv2.circle(splash,(x,y),4,(200,200,200),2)
        return splash

    def test_bubbles_confirm_over_two_frames(self):
        self.assertFalse(self.detector.update(self.splash(),self.anchor,True)[0])
        self.assertTrue(self.detector.update(self.splash(),self.anchor,True)[0])

    def test_large_camera_change_veto(self):
        white=np.full_like(self.base,240)
        bit,info=self.detector.update(white,self.anchor,True)
        self.assertFalse(bit);self.assertEqual(info.get('reset'),'large motion')

    def test_anchor_jump_relearns(self):
        bit,info=self.detector.update(self.splash(),(110,90),True)
        self.assertFalse(bit);self.assertEqual(info.get('reset'),'new anchor')

    def test_no_anchor_clears_history(self):
        self.assertFalse(self.detector.update(self.splash(),None,True)[0])
        self.assertIsNone(self.detector.background)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.cap=Mock()
        self.cap.is_active.return_value=True
        self.cap.refresh.return_value={'left':100,'top':100,'width':400,'height':300}
        self.ctrl=Controller(capturer=self.cap,enabled=False)
        self.ctrl._mouse=Mock()

    def test_dry_run_cannot_click(self):
        self.ctrl.armed=True
        self.assertFalse(self.ctrl.click(200,200,force=True))
        self.ctrl._mouse.click.assert_not_called()

    def test_unarmed_cannot_click(self):
        self.ctrl.enabled=True
        self.assertFalse(self.ctrl.click(200,200,force=True))
        self.ctrl._mouse.click.assert_not_called()

    def test_focus_loss_disarms(self):
        self.ctrl.enabled=self.ctrl.armed=True
        self.cap.is_active.return_value=False
        self.assertFalse(self.ctrl.click(200,200,force=True))
        self.assertFalse(self.ctrl.armed)
        self.ctrl._mouse.click.assert_not_called()

    def test_out_of_bounds_target_rejected(self):
        self.ctrl.enabled=self.ctrl.armed=True
        self.assertFalse(self.ctrl.click(50,50,force=True))
        self.ctrl._mouse.click.assert_not_called()

    def test_valid_reel_and_cast_cooldown(self):
        self.ctrl.enabled=self.ctrl.armed=True
        self.assertTrue(self.ctrl.click(200,200))
        self.assertFalse(self.ctrl.click(200,200))
        self.assertTrue(self.ctrl.click(200,200,force=True))
        self.assertEqual(self.ctrl._mouse.click.call_count,2)


class ProfileTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cfg.json';cfg=FisherConfig();cfg.pond_roi=(.5,.4,.4,.4)
            save_profile(cfg,p);loaded=load_profile(p)
            self.assertEqual(loaded.pond_roi,cfg.pond_roi)
            self.assertEqual(loaded.bite_confirm_frames,2)

    def test_invalid_profile_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cfg.json'
            for data in ({'bite_confirm_frames':0},{'bite_relative_jump':'bad'},
                         {'pond_roi':[.9,.9,.5,.5]},{'unknown':True}):
                p.write_text(json.dumps(data))
                with self.assertRaises(ValueError):load_profile(p)

    def test_frame_normalization_preserves_aspect(self):
        frame=working_frame(np.zeros((720,1920,3),np.uint8))
        self.assertEqual(frame.shape,(480,1280,3))


if __name__=='__main__':unittest.main()

class OcclusionTests(unittest.TestCase):
    def test_visible_float_not_dragged_to_short_string(self):
        e=AutoFisher(FisherConfig(cast_settle_s=.1),observe_only=True)
        a=analysis(line=True);a.bobber=(300,130);frame=np.zeros((200,400,3),np.uint8)
        e.step(frame,0,a);e.step(frame,.2,a)
        self.assertTrue(e.anchor_verified)
        a.bobber=None;a.string_end=(295,90)
        e.step(frame,.4,a)
        self.assertEqual(e.bobber,(300,130))

    def test_leaning_pole_without_string_does_not_cast(self):
        e=AutoFisher(FisherConfig());a=analysis()
        a.rod_lines=[(220,40,200,125)]
        frame=np.zeros((200,400,3),np.uint8)
        for t in (0,.5,1):self.assertEqual(e.step(frame,t,a)[0],'none')

class LateFlashTests(unittest.TestCase):
    def test_no_late_reel_after_tackle_disappears(self):
        e=AutoFisher(FisherConfig(cast_settle_s=.1))
        a=analysis(line=True);frame=np.zeros((200,400,3),np.uint8)
        for t in (0,.2,.4):e.step(frame,t,a)
        e.bite.update=Mock(return_value=(True,{}))
        a.string_end=None;a.string_seg=None
        action,_,info=e.step(frame,.7,a)
        self.assertEqual(action,'none')
        self.assertIn('no recent tackle',info['note'])
