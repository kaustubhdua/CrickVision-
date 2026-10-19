import numpy as np
from crickvision.shot import ShotBoundaryDetector

def test_same_frame_is_not_cut():
    d = ShotBoundaryDetector(.3)
    f = np.zeros((80,120,3), dtype=np.uint8)
    assert d.update(f)[0] is False
    cut, score = d.update(f.copy())
    assert not cut and score == 0

def test_large_change_is_cut():
    d = ShotBoundaryDetector(.2)
    d.update(np.zeros((80,120,3), dtype=np.uint8))
    cut, score = d.update(np.full((80,120,3),255,dtype=np.uint8))
    assert cut and score > .2
