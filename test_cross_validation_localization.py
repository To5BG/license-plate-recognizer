import pytest
import math
from cross_validation_localization import shoelaceArea, intersect, evaluate_single_box

shoelaceTests = [
    # rectangle
    ([(0, 0), (10, 0), (10, 10), (0, 10)], 100),
    # random quad
    ([(0, 0), (5, 10), (10, 15), (5, 5)], 25), 
    # random quad #2
    ([(7, 11), (4, 12), (1, 21), (3, 5)], 26)
]

intersectionTests = [
    # no intersection
    ([(0, 0), (10, 0), (10, 10), (0, 10)], [(10, 10), (10, 11), (11, 11), (11, 10)], [(0, 0), (0, 0), (0, 0), (0, 0)]),
    # complete overlap
    ([(0, 0), (5, 0), (5, 5), (0, 5)], [(0, 0), (5, 0), (5, 5), (0, 5)], [(0, 0), (5, 0), (5, 5), (0, 5)]), 
    # some overlap
    ([(0, 0), (0, 2), (2, 2), (2, 0)], [(1, 1), (3, 1), (3, 3), (1, 3)], [(1, 1), (1, 2), (2, 2), (2, 1)]),
    # random
    ([(3, 3), (2, 6), (0, 5), (3, 1)], [(1, 1), (2, 0), (3, 7), (1, 4)], [(1, 4), (1, 11 / 3), (2.28, 1.96), (2.6, 4.2), (19 / 9, 17 / 3)]),
    # random #2
    ([(225, 375), (228, 314), (452, 325), (449, 386)], [(226, 316), (449, 325), (447, 377), (230, 372)], 
        [(227.126213592233, 331.7669902912621), (227.89787234042552, 316.07659574468084), (449, 325), (447, 377), (230, 372)]),
    # random #3
    ([(211, 285), (222, 248), (372, 292), (361, 329)], [(220, 246), (369, 288), (357, 327), (211, 285)],
        [(211, 285), (222, 248), (368.1194731890875, 290.86171213546567), (357, 327)])
]

evaluationTests = [
    # no intersection
    ([(0, 0), (10, 0), (10, 10), (0, 10)], [(10, 10), (10, 11), (11, 11), (11, 10)], (0, 0)),
    # complete overlap
    ([(0, 0), (5, 0), (5, 5), (0, 5)], [(0, 0), (5, 0), (5, 5), (0, 5)], (1, 1)), 
    # some overlap
    ([(0, 0), (0, 2), (2, 2), (2, 0)], [(1, 1), (3, 1), (3, 3), (1, 3)], (0, 1 / 7)),
    # random
    ([(3, 3), (2, 6), (0, 5), (3, 1)], [(1, 1), (2, 0), (3, 7), (1, 4)], (0, 1438 / 4637)),
    # random2
    ([(0, 0), (110, 2), (115, 50), (1, 50)], [(2, 1), (112, 5), (113, 50), (0, 44)], (1, 0.886890206))
]

class TestEvaluation:

    @pytest.mark.parametrize("box,expected", shoelaceTests)
    def test_shoelaceArea(self, box, expected):
        assert shoelaceArea(box) == expected

    @pytest.mark.parametrize("box1,box2,expected", intersectionTests)
    def test_intersection(self, box1, box2, expected):
        assert set(intersect(box1, box2)) == set(expected)
    
    @pytest.mark.parametrize("box1,box2,expected", evaluationTests)
    def test_evaluation(self, box1, box2, expected):
        res = evaluate_single_box(box1, box2)
        assert res[0] == expected[0]
        assert math.isclose(res[1], expected[1])