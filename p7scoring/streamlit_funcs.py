import json
import requests
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import streamlit as st


url = "http://127.0.0.1:5000/id_score/?SK_ID_CURR="


def st_title():
    """
    App title
    """
    st.title("Customer loan eligibility app")

@st.cache(suppress_st_warning=True)
def st_id():
    """
    Read the customer's id
    :return: int
    """
    id = st.number_input('Client id:',
                          min_value=100002, max_value=111633,
                          value=100700, step=1)
    # URL of the sk_id API
    score_url = url + str(id)
    # Requesting the api
    response = requests.get(score_url)
    # Convert from JSON format to Python dict
    content = json.loads(response.content.decode('utf-8'))["score"]["0"]
    if content>-1:
        status = "Accepted" if content==0 else "Refused"
    else:
        status = "No data available."
    st.markdown(f'<h1 style="color:#33ff33;'f'font-size:16px;">{"Loan status:&emsp;"}{status}</h1>',
                unsafe_allow_html=True)

    return id

def st_buttons(id):
    """
    Defines the web page's menus
    :param id: int
        Customer's id
    :return: button objects
    """
    with st.sidebar:
        b_importance = st.button("Global contribution of characteristics ")
        b_dist = st.button("Characteristics comparison: id_"+str(id))
        b_loc_importance = st.button("Decision details: id_"+str(id))
        #n_features=st.number_input('Number of features:', min_value=2, max_value=20, value=10, step=1)
        n_features = st.slider('Number of features:', min_value=1, max_value=20, value=10, step=1)
    return b_importance, b_loc_importance, b_dist, n_features



def plot_g_importance(model, features, b_importance):
    """
    Plots the general feature importance.
    :param model: scikit-learn machine learning model (pretrained)
        This is a logistic regression model for now
    :param features: list
        list of features
    :param b_importance: button obj
        Characteristics global contribution menu
    :return: feature importance plot
    """
    coef = list(model.coef_[0])
    model_importance = pd.DataFrame(zip(features, coef), columns=["feature", "importance"])
    model_importance = model_importance.sort_values("importance", ascending=False)

    if b_importance:
        st.write("### Global feature importance")
        sns.set(font_scale=1.5)
        fig = plt.figure(figsize=(15, 25))
        sns.barplot(data=model_importance, x="importance", y="feature")
        st.write(fig)



def get_Xtrain(df, features):
    """
    Solit the data to train and test sets
    :param df: dataframe
        whole dataset
    :param features: list
        list of selected columns as features
    :return: ndarray, ndarray
        Training set
        Standard scaled data for the features
    """
    y = df["TARGET"].values
    X = df[features].values
    X_sc = StandardScaler().fit_transform(X)
    X_t = train_test_split(X_sc, y, test_size=0.3)[0]
    return X_t, X_sc


def display_explanation(exp, b_loc_importance):
    """
    Display the local feature importance in the webpage
    :param exp: explain_instance obj
        local feature importance computed for a data instance
    :param b_loc_importance: button obj
        Loan allocation menu
    """
    if b_loc_importance:
        st.write("### Loan allocation possibility")
        exp_html = exp.as_html()
        white_bg = "<style>:root {background-color: white;}</style>"
        text_html = exp_html+white_bg
        st.components.v1.html(html=text_html, height=700)


def select_features(exp):
    """
    :param exp: explain_instance obj
        local feature importance computed for a data instance
    :return: list
     First (10) important features selected by explain_instance
    """
    s_features = [feature.split("<")[0] for feature, value in exp.as_list()]
    s_features = [feature.split(">")[0].strip() for feature in s_features]
    return s_features



def plot_l_importance(df, model, features, b_loc_importance, id, num_features):
    """
    Displays the results of the explain_instance in the Loan allocation menu
    :param df: dataframe
        The whole dataset
    :param model: ML model
        Fitted ML model
    :param features: list
        list of selected columns as features
    :param b_loc_importance: button obj
        Loan allocation menu
    :param id: int
        Customer's id
    :return: list
        First (10) important features selected by explain_instance
    """

    if df[df["SK_ID_CURR"]==id].empty:
        st.write("No data available.")
        return 0
    else:
        X_train, X_sc = get_Xtrain(df, features)
        explainer=LimeTabularExplainer(X_train,
                                       mode="classification",
                                       class_names=["O.K.", "Risky"],
                                       feature_names=features)
        idx = df[df["SK_ID_CURR"]==id].index
        data_instance = X_sc[idx].reshape(len(features),)
        explanation = explainer.explain_instance(data_instance, model.predict_proba, num_features=num_features)
        sel_features = select_features(explanation)

        display_explanation(explanation, b_loc_importance)

        return sel_features


def dist_per_axis(ax, features, df_target, df_instance):
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
    for i, feat in enumerate(features):
        axis = ax[int(round(i/2+.1)), i % 2]
        hist = axis.hist(df_target[feat], bins=30, log=True)
        axis.set_xlabel(feat, fontsize=18)
        axis.tick_params(axis='both', which='major', labelsize=16)
        axis.tick_params(axis='both', which='minor', labelsize=10)

        # Marking the instance location on the distribution
        axis.plot([df_instance[feat]] * 2, [0, hist[0].max()/3.], c="r", linewidth=4)


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
        st.write("No data available.")
        return 0

    n_row = int(round(len(features) / 2 +.1))
    fig, ax = plt.subplots(n_row, 2, figsize=(15, 20), constrained_layout=True)
    fig_title = fig.suptitle("Loan O.K. distributions", fontsize=25) if target == 0 \
        else fig.suptitle("Distributions for a Risky allocation", fontsize=25)

    dist_per_axis(ax, features, data_target, data_instance)
    st.write(fig)


def plot_class_dist(df, features, uid, b_dist):
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
    if(b_dist):
        plot_dist(df, features, 0, uid)
        plot_dist(df, features, 1, uid)

