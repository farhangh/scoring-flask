import pytest
from app import app, index
import json



def test_index():
    assert (index() == "App, model and data loaded ...")



test_data = [
    ("100700",0),
    ("100703",0),
    ("204", -1),
]
@pytest.mark.parametrize("SK_ID_CURR, expected", test_data)
def test_get_score(SK_ID_CURR, expected):
    response = app.test_client().get(f'/id_score/?id={SK_ID_CURR}')
    res = json.loads(response.data.decode('utf-8')).get("score")
    assert(res==expected)



test_featues = [
    ("1", ["AMT_CREDIT"]),
    ("3", ["AMT_CREDIT","DAYS_BIRTH","NAME_EDUCATION_TYPE_Secondary / secondary special"]),
    ("5", ["AMT_CREDIT","DAYS_BIRTH","NAME_EDUCATION_TYPE_Secondary / secondary special",
         "AMT_ANNUITY","REGION_RATING_CLIENT_W_CITY"]) ]

@pytest.mark.parametrize("n, expected", test_featues)
def test_get_features(n, expected):
    response = app.test_client().get(f'/features/?n={n}')
    res = json.loads(response.data.decode('utf-8')).get("n")
    assert (res == expected)


