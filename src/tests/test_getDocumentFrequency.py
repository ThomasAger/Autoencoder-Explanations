from unittest import TestCase
from util.classifier_util import getDocumentFrequency
from hypothesis.strategies import composite
from hypothesis.strategies import lists
from hypothesis.strategies import tuples
from util.io import isInt
import numpy as np

@composite
def test_getDocumentFrequency(draw, elements=lists(max_size=50)):
    x = draw(lists(elements, max_size=100))
    assert len(getDocumentFrequency(x) == len(x))
    assert len(getDocumentFrequency(x) > len(x[0]))
    assert isInt(getDocumentFrequency(x)[0])
    assert len(np.nonzero(getDocumentFrequency(x))) == 0