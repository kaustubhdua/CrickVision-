import numpy as np
from crickvision.team import TeamColourConfig, classify_team_colour

def test_unknown_without_config():
    frame = np.zeros((100,100,3), dtype=np.uint8)
    assert classify_team_colour(frame,(10,10,90,95),())[0] == "Unknown"

def test_red_kit():
    frame = np.zeros((100,100,3), dtype=np.uint8)
    frame[25:60,25:75] = (0,0,255)
    red = TeamColourConfig("Red",(0,100,80),(10,255,255))
    name, score = classify_team_colour(frame,(10,5,90,95),(red,))
    assert name == "Red" and score > .12
