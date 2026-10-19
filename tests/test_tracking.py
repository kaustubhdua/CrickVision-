from crickvision.detector import PersonDetection
from crickvision.tracker import IoUTracker, iou

def box(a,b,c,d):
    return PersonDetection(a,b,c,d,0.9)

def test_iou_identical():
    assert iou(box(0,0,10,10), box(0,0,10,10)) == 1.0

def test_tracker_associates_overlap():
    t = IoUTracker(iou_threshold=.2)
    a = t.update([box(0,0,20,40)])[0]
    b = t.update([box(1,0,21,40)])[0]
    assert a.track_id == b.track_id and b.age == 2

def test_reset_starts_new_identity():
    t = IoUTracker()
    a = t.update([box(0,0,20,40)])[0]
    t.reset()
    b = t.update([box(0,0,20,40)])[0]
    assert a.track_id != b.track_id

def test_tracks_age_out():
    t = IoUTracker(max_missed=1)
    t.update([box(0,0,20,40)])
    t.update([])
    t.update([])
    assert not t._tracks
