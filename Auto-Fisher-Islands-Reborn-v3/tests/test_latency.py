import threading,time,unittest
from unittest.mock import Mock
import cv2
import numpy as np
from fisher.config import FisherConfig
from fisher.detector import FrameAnalysis,FishDetection
from fisher.fish_filter import filter_candidates
from fisher.fast_tackle import line_visible_now
from fisher.vision_worker import VisionWorker
from fisher.state_machine import AutoFisher,State


def wait_for(fn,timeout=2):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        value=fn()
        if value:return value
        time.sleep(.003)
    raise AssertionError('Timed out waiting for test worker')


def data():
    return FrameAnalysis(np.full((200,400),255,np.uint8),(180,90,30,70),
        [FishDetection((300,130),(290,120,20,20),300)],[(210,40,210,125)],None,
        string_end=(300,130),string_seg=(210,40,300,130))


class FishFilterTests(unittest.TestCase):
    def setUp(self):
        self.gray=np.full((220,500),140,np.uint8)
        self.allowed=np.full_like(self.gray,255);self.water=self.allowed.copy()
        self.fish=FishDetection((80,90),(55,77,50,26),800)

    def run_filter(self,**kwargs):
        return filter_candidates(self.gray,[self.fish],self.allowed,self.water,**kwargs)

    def test_filled_shadow_kept(self):
        cv2.ellipse(self.gray,(80,90),(25,13),0,0,360,40,-1)
        self.assertEqual(len(self.run_filter(anchor=(100,100))),1)

    def test_outline_only_ripple_rejected(self):
        cv2.ellipse(self.gray,(80,90),(25,13),0,0,360,40,2)
        self.assertFalse(self.run_filter())

    def test_string_speckle_not_filled_body(self):
        cv2.line(self.gray,(57,89),(103,89),40,1)
        self.assertFalse(self.run_filter())

    def test_faraway_scenery_rejected(self):
        cv2.ellipse(self.gray,(80,90),(25,13),0,0,360,40,-1)
        self.assertFalse(self.run_filter(anchor=(450,90),radius=150))

    def test_avatar_shadow_rejected(self):
        cv2.ellipse(self.gray,(80,90),(25,13),0,0,360,40,-1)
        self.assertFalse(self.run_filter(player=(55,50,50,50)))

    def test_pond_constraint_respected(self):
        cv2.ellipse(self.gray,(80,90),(25,13),0,0,360,40,-1)
        self.assertFalse(self.run_filter(pond_roi=(.5,.1,.4,.8)))

    def test_no_water_context_without_cast_rejected(self):
        cv2.ellipse(self.gray,(80,90),(25,13),0,0,360,40,-1)
        self.water[:]=0
        self.assertFalse(self.run_filter())


class FastTackleTests(unittest.TestCase):
    def test_visible_string_supported(self):
        image=np.full((300,400,3),130,np.uint8)
        cv2.line(image,(160,50),(250,230),(45,45,45),2)
        self.assertTrue(line_visible_now(image,(160,50,250,230)))

    def test_missing_string_rejected(self):
        self.assertFalse(line_visible_now(np.full((300,400,3),130,np.uint8),(160,50,250,230)))

    def test_fish_at_bottom_does_not_impersonate_string(self):
        image=np.full((300,400,3),130,np.uint8)
        cv2.ellipse(image,(250,230),(30,15),0,0,360,(45,45,45),-1)
        self.assertFalse(line_visible_now(image,(160,50,250,230)))

    def test_empty_and_short_geometry_rejected(self):
        image=np.full((300,400,3),130,np.uint8)
        self.assertFalse(line_visible_now(image,None))
        self.assertFalse(line_visible_now(image,(100,100,120,130)))

    def test_real_day_and_night_casts(self):
        from pathlib import Path
        from fisher.detector import analyze_frame
        for name in ('gameplay_09','night_000','night_040','night_080','night_130'):
            image=cv2.imread(str(Path(__file__).parent/'fixtures'/f'{name}.png'))
            a=analyze_frame(image,FisherConfig())
            self.assertTrue(line_visible_now(image,a.string_seg),name)


class WorkerTests(unittest.TestCase):
    def test_capture_submission_is_copied(self):
        started=threading.Event();release=threading.Event()
        def analyze(f,cfg):
            started.set();release.wait(2);return int(f[0,0,0])
        worker=VisionWorker(FisherConfig(),30,analyze)
        try:
            frame=np.full((2,2,3),7,np.uint8);worker.submit(frame,time.monotonic())
            self.assertTrue(started.wait(2));frame[:]=99;release.set()
            self.assertEqual(wait_for(worker.result).analysis,7)
        finally:release.set();worker.close()

    def test_pending_queue_keeps_latest_frame(self):
        started=threading.Event();release=threading.Event();seen=[]
        def analyze(f,cfg):
            n=int(f[0,0,0]);seen.append(n)
            if n==0:started.set();release.wait(2)
            return n
        worker=VisionWorker(FisherConfig(),30,analyze)
        try:
            worker.submit(np.zeros((2,2,3),np.uint8),time.monotonic());self.assertTrue(started.wait(2))
            for n in range(1,10):worker.submit(np.full((2,2,3),n,np.uint8),time.monotonic())
            release.set();wait_for(lambda:worker.result() and worker.result().analysis==9)
            self.assertEqual(seen,[0,9])
        finally:release.set();worker.close()

    def test_reset_discards_inflight_results(self):
        started=threading.Event();release=threading.Event()
        def analyze(f,cfg):started.set();release.wait(2);return 1
        worker=VisionWorker(FisherConfig(),30,analyze)
        try:
            worker.submit(np.zeros((2,2,3),np.uint8),time.monotonic());self.assertTrue(started.wait(2))
            worker.reset();release.set();time.sleep(.06)
            self.assertIsNone(worker.result())
        finally:release.set();worker.close()

    def test_worker_failure_surfaces(self):
        def fail(f,cfg):raise ValueError('test failure')
        worker=VisionWorker(FisherConfig(),30,fail)
        try:
            worker.submit(np.zeros((2,2,3),np.uint8),time.monotonic())
            def failed():
                try:worker.result()
                except RuntimeError:return True
                return False
            wait_for(failed)
            with self.assertRaisesRegex(RuntimeError,'test failure'):worker.result()
        finally:worker.close()

    def test_poll_does_not_wait_for_heavy_scan(self):
        started=threading.Event();release=threading.Event()
        def slow(f,cfg):started.set();release.wait(2);return 1
        worker=VisionWorker(FisherConfig(),30,slow)
        try:
            worker.submit(np.zeros((2,2,3),np.uint8),time.monotonic());self.assertTrue(started.wait(2))
            self.assertIsNone(worker.result())  # returns while analyzer is still blocked
            self.assertFalse(release.is_set())
        finally:release.set();worker.close()


class CachedStateTests(unittest.TestCase):
    def setUp(self):
        self.e=AutoFisher(FisherConfig());self.e.now=0;self.e._set(State.WAITING)
        self.e.bobber=(300,130);self.frame=np.zeros((200,400,3),np.uint8);self.a=data()
        self.e.bite.update=Mock(return_value=(False,{}))

    def test_same_cache_not_counted_as_new_fish_history(self):
        for t in (0,.02,.04,.1,.2):self.e.step(self.frame,t,self.a,analysis_time=0,tackle_visible_now=True)
        self.assertEqual(len(self.e.fish_history),1)
        self.assertFalse(self.e.last_info['stable_fish'])

    def test_fresh_distinct_observations_establish_stability(self):
        for t in (0,.2,.4):self.e.step(self.frame,t,self.a,analysis_time=t,tackle_visible_now=True)
        self.assertTrue(self.e.last_info['stable_fish'])

    def test_stale_analysis_cannot_reel(self):
        self.e.bite.update=Mock(return_value=(True,{}))
        act,_,info=self.e.step(self.frame,1,self.a,analysis_time=0,tackle_visible_now=True)
        self.assertEqual(act,'none');self.assertIn('stale',info['note']);self.e.bite.update.assert_not_called()

    def test_cached_cast_does_not_refresh_timestamp_without_pixel_proof(self):
        self.e.step(self.frame,.1,self.a,analysis_time=0)
        self.e.step(self.frame,.2,self.a,analysis_time=0)
        self.assertEqual(self.e.t_fresh,0)

    def test_current_line_proof_refreshes_tackle(self):
        self.e.step(self.frame,.2,self.a,analysis_time=0,tackle_visible_now=True)
        self.assertEqual(self.e.t_fresh,.2)

    def test_out_of_order_analysis_rejected(self):
        self.e.step(self.frame,.2,self.a,analysis_time=.2)
        act,_,info=self.e.step(self.frame,.3,self.a,analysis_time=.1)
        self.assertEqual(act,'none');self.assertIn('out-of-order',info['note'])



class PendingStopTests(unittest.TestCase):
    def test_stop_request_cannot_dispatch_a_click(self):
        from fisher.controller import Controller
        cap=Mock();cap.is_active.return_value=True
        ctrl=Controller(capturer=cap,enabled=False,stop_requested=lambda:True)
        ctrl.enabled=ctrl.armed=True;ctrl._mouse=Mock()
        self.assertFalse(ctrl.click(100,100,force=True))
        ctrl._mouse.click.assert_not_called()
        self.assertFalse(ctrl.armed)


class BackgroundTimingTests(unittest.TestCase):
    def test_adaptation_same_at_thirty_and_sixty_fps(self):
        from fisher.bite_detector import BiteDetector
        backgrounds=[]
        for fps in (30,60):
            detector=BiteDetector(FisherConfig())
            detector.update(np.full((200,200,3),130,np.uint8),(100,100),now=0)
            for i in range(1,int(.2*fps)+1):
                detector.update(np.full((200,200,3),138,np.uint8),(100,100),now=i/fps)
            backgrounds.append(detector.background)
        np.testing.assert_allclose(backgrounds[0],backgrounds[1],atol=.001)


if __name__=='__main__':unittest.main()
