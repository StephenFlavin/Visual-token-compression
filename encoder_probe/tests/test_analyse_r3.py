"""analyse_r3.py: exact tests and the R3-1 numbers quoted in docs/ANALYSIS.md (committed rows, no torch)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import analyse_r3 as A  # noqa: E402


def test_exact_tests():
    assert math.isclose(A.binom_ge(0, 10, 0.3), 1.0)
    assert math.isclose(A.binom_ge(10, 10, 0.5), 0.5 ** 10)
    assert math.isclose(A.fisher_greater(0, 5, 5, 5), 1.0)       # real worst possible: p = 1
    assert math.isclose(A.fisher_greater(5, 5, 0, 5), 1 / 252)   # real best possible: 1 / C(10, 5)
    assert A.lev("kitten", "sitting") == 3 and A.lev("", "abc") == 3


def test_r31_numbers_match_docs():
    res = {d["enc"]: d for d in A.analyse(A.load())}
    p8, p5, moon = res["latin_p8"], res["latin_p5"], res["rgb3_lg_pb_h5"]
    assert round(p8["gain"], 3) == 0.546 and round(p5["gain"], 3) == 0.099 and abs(moon["gain"]) < 0.001
    assert (p5["mc"], p5["b_mc"]) == (18, 9) and 0.015 < p5["p_mc"] < 0.025
    assert round(p8["bits_delivered"], 1) == 12.7 and round(p5["bits_delivered"], 1) == 5.8
    assert round(moon["bits_needed"]) == 270
    assert res["lg_pb_h5"]["gap_step"] == 500    # its blind run died at step 500
