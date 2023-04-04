import pandas as pd
from flask import Flask, request, jsonify
from p7scoring.feature_extraction import get_features
from p7scoring.models import read_data, load_model

from sklearn.preprocessing import StandardScaler

import json

#data = read_data(path="../data/p7_data.csv")
data = read_data()
features = get_features(data)

data = data[features+ ["SK_ID_CURR", "TARGET"]].copy()

X = data[features].values
X_sc = StandardScaler().fit_transform(X)

#model = load_model(path="../data/best_lr_t.pkl")
model = load_model()


# instantiate Flask object
app = Flask(__name__)

@app.route("/")
def index():
    return "APP loaded, model and data loaded ..."


# test local : http://127.0.0.1:5000/id_score/?SK_ID_CURR=100700
@app.route('/id_score/')
def get_score():
    id = int(request.args.get('SK_ID_CURR'))
    id_data = data[data['SK_ID_CURR']==id]
    idx = id_data.index
    if(id_data.shape[0]==0):
        score = pd.DataFrame(data=[-1], columns=["score"])
    else:
        score = pd.DataFrame(data=[int(model.predict(X_sc[idx])[0])], columns=["score"])

    return jsonify( json.loads(score.to_json()) )

#http://127.0.0.1:5000/features/?n=5
@app.route("/features/")
def get_features():
    n = int(request.args.get('n'))
    df = pd.DataFrame({"n":range(n)})
    return jsonify(json.loads(df.to_json()))



@app.route('/ids/')
def ids_list():
    # Extract list of all the 'SK_ID_CURR' ids in the X_test dataframe
    customers_id_list = data["SK_ID_CURR"].sort_values()
    # Convert Series to JSON
    customers_id_list_json = json.loads(customers_id_list.to_json())
    # Returning the processed data
    # jsonify is a helper method provided by Flask to properly return JSON data.
    return jsonify(customers_id_list_json)


# info for a given costumer
# test local : http://127.0.0.1:5000/id_data/?SK_ID_CURR=100700
@app.route('/id_data/')
def id_data():  # selected_id
    selected_id_customer = int(request.args.get('SK_ID_CURR'))
    features = data[data['SK_ID_CURR']==selected_id_customer].drop(columns=["TARGET"]).loc[:,:]
    status = data.loc[data['SK_ID_CURR']==selected_id_customer, "TARGET"]
    # Convert pd.Series to JSO
    features_json = json.loads(features.to_json())
    status_json = json.loads(status.to_json())
    # Returning the processed data
    return jsonify({'status': status_json, 'data': features_json})


##########################
if __name__ == "__main__":
    app.run()

