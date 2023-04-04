import pytest
import app
import json

def test_index():
    assert (app.index() == "App, model and data loaded ...")

@pytest.mark.parametrize("SK_ID_CURR")
def test_get_score(SK_ID_CURR):
    response = app.app.test_client().get(f'/id_score/?SK_ID_CURR={SK_ID_CURR}')
    res = json.loads(response.data.decode('utf-8')).get("score")
    assert(res=={"0":-1})
    #print(res)

@pytest.mark.parametrize("n")
def test_get_features(n):
    response = app.app.test_client().get(f'/features/?n={n}')
    res = json.loads(response.data.decode('utf-8')).get("n")
    assert (res == {"0":0,"1":1})


if __name__ == "__main__":
    test_index()
    test_get_score(1)
    test_get_features(2)