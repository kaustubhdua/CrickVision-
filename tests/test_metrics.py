from crickvision.detector import PersonDetection
from crickvision.metrics import detection_metrics

def box(x1,y1,x2,y2):
    return PersonDetection(x1,y1,x2,y2,0.9)

def test_perfect_detection_metrics():
    result = detection_metrics([box(0,0,10,10)], [box(0,0,10,10)])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (1,0,0)
    assert result.precision == result.recall == result.f1 == 1.0

def test_false_positive_and_miss():
    result = detection_metrics([box(0,0,10,10)], [box(30,30,40,40)])
    assert result.true_positives == 0
    assert result.false_positives == result.false_negatives == 1
    assert result.f1 == 0.0

def test_empty_lists():
    result = detection_metrics([], [])
    assert result.precision == result.recall == result.f1 == 1.0
