import hypothesis
from score.scoring_functions import get_f1_score

def test_manual():
    assert get_f1_score(0,0) == 0
    assert lambda:get_f1_score(123123123,12312321) == ValueError
    assert lambda:get_f1_score(-123123123,-12312321) == ValueError
    assert get_f1_score(0.12312312312312312323423423,0.2342342342342342342342344244) == 0.16140510258157317

