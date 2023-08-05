import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt

from flask import Flask, request, jsonify, make_response, Response
from p7scoring.feature_extraction import get_features
from p7scoring.models import read_data, load_model

from sklearn.preprocessing import StandardScaler

from lime.lime_tabular import LimeTabularExplainer

import json

import base64
from io import BytesIO

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
    return "App, model and data loaded ..."


# test local : http://127.0.0.1:5000/id_score/?id=100700
@app.route('/id_score/')
def get_score():
    id = int(request.args.get('id'))
    id_data = data[data['SK_ID_CURR']==id]
    idx = id_data.index
    if(id_data.shape[0]==0):
        score = {"score":-1}
    else:
        score = {"score":int(model.predict(X_sc[idx])[0])}

    return jsonify( json.loads(json.dumps(score)) )

#http://127.0.0.1:5000/features/?n=5
@app.route("/features/")
def get_features():
    n_features = int(request.args.get('n'))
    g_importance = df_g_importance(n_features)
    features = g_importance["feature"].tolist()
    dct = {"n": features}
    return jsonify( json.loads(json.dumps(dct)) )



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
# test local : http://127.0.0.1:5000/id_data/?id=100700
@app.route('/id_data/')
def id_data():
    # selected_id
    selected_id_customer = int(request.args.get('id'))
    features = data[data['SK_ID_CURR']==selected_id_customer].drop(columns=["TARGET"]).loc[:,:]
    status = data.loc[data['SK_ID_CURR']==selected_id_customer, "TARGET"]
    # Convert pd.Series to JSO
    features_json = json.loads(features.to_json())
    status_json = json.loads(status.to_json())
    # Returning the processed data
    return jsonify({'status': status_json, 'data': features_json})

# test local : http://127.0.0.1:5000/lime/?id=100700&n=5
@app.route('/lime/')
def plot_l_importance(data=data):
    id = int(request.args.get('id'))
    n_features = int(request.args.get('n'))

    if data[data["SK_ID_CURR"]==id].empty:
        return 0
    else:
        data = data.reset_index(drop=True)
        explainer=LimeTabularExplainer(X_sc,
                                       mode="classification",
                                       class_names=["O.K.", "Risky"],
                                       feature_names=features)
        idx = data[data["SK_ID_CURR"]==id].index
        data_instance = X_sc[idx].reshape(len(features),)
        explanation = explainer.explain_instance(data_instance, model.predict_proba, num_features=n_features)

        exp_html = explanation.as_html()
        response = make_response(exp_html)

        return response


def df_g_importance(n_feat):
    # Returns the feature importance dataframe for fist n_feat important features
    coef = list(model.coef_[0])
    g_importance = pd.DataFrame(zip(features, coef), columns=["feature", "importance"])
    g_importance = g_importance.sort_values("importance", ascending=False, key=abs)
    return g_importance.head(n_feat)


# test local : http://127.0.0.1:5000/global/?n=10
@app.route('/global/')
def plot_g_importance():

    n_features = int(request.args.get('n'))
    g_importance = df_g_importance(n_features).sort_values("importance", ascending=False)

    sns.set(font_scale=1.)
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    sns.barplot(data=g_importance, x="importance", y="feature")
    plt.xticks(rotation=25)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    info = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{info}'/>"




def dist_per_axis(fig, ax, features, df_target, df_instance):
    """
    Makes a loop over all figure axes and plot the distribution
    :param ax: ndarray
        axe pyplot object
    :param features: list
        List of selected fetures
    :param df_target: dataframe
        dataframe for either OK class(0) or Risky class(1)
    :param df_instance: dataframe
        Data for the given customer
    """
    sns.set(font_scale=.5)

    target = df_target["TARGET"].unique().tolist()[0]
    fig_title = fig.suptitle("Loan O.K. distributions", fontsize=25) if target == 0 \
        else fig.suptitle("Distributions for a Risky allocation", fontsize=25)
    plt.title(fig_title)

    for i, feat in enumerate(features):
        axis = ax[int(i/2), i%2]
        hist = axis.hist(df_target[feat], bins=30)
        axis.set_xlabel(feat, fontsize=18)
        axis.tick_params(axis='both', which='major', labelsize=16)
        axis.tick_params(axis='both', which='minor', labelsize=10)
        # Marking the instance location on the distribution
        axis.plot([df_instance[feat]] * 2, [0, hist[0].max()/3.], c="r", linewidth=4)

    return fig

def plot_dist(df, features, target, uid):
    """
    Plot the distribution of the given features for the given target class in the webpage.
    :param df: datafram
        Whole dataset
    :param features: list
        List of selected features for a given customer (by explain_instance)
    :param target: int
        0 for OK, 1 for Risky
    :param uid: int
        Customer's id
    """

    data_target = df[df["TARGET"] == target]
    data_instance = df[df["SK_ID_CURR"] == uid]
    if len(data_instance)==0:
        return "No data available."

    n_row = int(round(len(features) / 2 +.1))
    fig, ax = plt.subplots(n_row, 2, figsize=(10, int(1.4*len(features))), constrained_layout=True)
    fig = dist_per_axis(fig, ax, features, data_target, data_instance)
    return fig

# https://stackoverflow.com/questions/70083434/combine-two-matplotlib-figures-side-by-side-high-quality
def combine_figures(fig1, fig2, n_feat):
    c1 = fig1.canvas
    c2 = fig2.canvas
    c1.draw()
    c2.draw()
    a1 = np.array(c1.buffer_rgba())
    a2 = np.array(c2.buffer_rgba())
    a = np.hstack((a1, a2))

    backend = mpl.get_backend()
    mpl.use(backend)
    fig, ax = plt.subplots(figsize=(10, int(1.4*n_feat)))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    ax.matshow(a)

    return fig

# test local : http://127.0.0.1:5000/dists/?id=100700&n=10
@app.route('/dists/')
def plot_class_dist(df=data):
    n_features = int(request.args.get('n'))
    uid = int(request.args.get('id'))

    g_importance = df_g_importance(n_features)
    features = g_importance["feature"].tolist()

    fig1 = plot_dist(df, features, 1, uid)
    fig0 = plot_dist(df, features, 0, uid)
    fig = combine_figures(fig1, fig0, n_features)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    info = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{info}'/>"

    # from matplotlib.backends.backend_agg import FigureCanvasAgg
    # import io
    # output = io.BytesIO()
    # FigureCanvasAgg(fig_0).print_png(output)
    # return Response(output.getvalue(), mimetype='image/png')



# test local : http://127.0.0.1:5000/dists2/?id=100700&n=10
@app.route('/dists2/')
def plot_class_dist_2(df=data, features=features):
    """
    Plots the distribution of the given features for the target class separately in the
    "Distribution of characteristics" menu.
    :param df: datafram
        Whole dataset
    :param features: list
        List of selected features for a given customer (by explain_instance)
    :param uid: int
        Customer's id
    :param b_dist: button obj
        "Distribution of characteristics" menu
    """
    n_features = int(request.args.get('n'))
    uid = int(request.args.get('id'))

    g_importance = df_g_importance(n_features)
    features = g_importance["feature"].tolist()

    fig_1 = plot_dist(df, features, 1, uid)
    fig_0 = plot_dist(df, features, 0, uid)

    import io
    from base64 import encodebytes
    encoded_imges = []
    for fig in [fig_0, fig_1]:
        byte_arr = io.BytesIO()
        fig.savefig(byte_arr, format='PNG') # convert the image to byte array
        encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
        encoded_imges.append(encoded_img)

    return jsonify({'result': encoded_imges})








##########################
if __name__ == "__main__":
    app.run()

