import pytest
from app import app
import json

test_data = [
    ("100700",{'0': 0}),
    ("100703",{'0': 0}),
    ("204", {'0': -1}),
]

def test_index():
    assert (app.index() == "App, model and data loaded ...")

@pytest.mark.parametrize("SK_ID_CURR, expected", test_data)
def test_get_score(SK_ID_CURR, expected):
    response = app.test_client().get(f'/id_score/?SK_ID_CURR={SK_ID_CURR}')
    res = json.loads(response.data.decode('utf-8')).get("score")
    assert(res==expected)


test_n_data=[ (1,{"0":0}), (2, {"0":0,"1":1}) ]
@pytest.mark.parametrize("n, expected", test_n_data)
def test_get_features(n, expected):
    response = app.test_client().get(f'/features/?n={n}')
    res = json.loads(response.data.decode('utf-8')).get("n")
    assert (res == expected)


