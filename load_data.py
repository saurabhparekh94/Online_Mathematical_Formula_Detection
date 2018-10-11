"""
File Name: load_data.py
@author : Tappan Ajmera (tpa7999@g.rit.edu)
@author : Saurabh Parekh (sbp4709@g.rit.edu)
"""

import os
import pickle

import bs4 as bs
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

from feature_extractor import FeatureExtractor
from preprocessing import Preprocessing


def extract_features(x, y, pen, ar, testing=False, key=None):
    """
    Extracts features for symbol classifier
    :param x: X coords
    :param y: Y
    :param pen: number of strokes
    :param ar: aspect ratio
    :param testing: boolean
    :param key: labels
    :return: feature vector
    """
    fe = FeatureExtractor()
    region_vecx, region_vecy, position_fx, position_fy, position_lx, position_ly, position_fx_h, position_fy_h, \
    position_lx_h, position_ly_h = fe.crossing(
        x, y)

    sh_x, sh_y = fe.find_sharp(x, y)
    num_sh = len(sh_x) - 2
    leng_line = fe.line_length(x, y)
    fuzzy_histogram = fe.fuzzy_histog(x, y)

    cov_xy = fe.covariance_xy(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    aspect = ar

    if testing:
        feature_vector = [cov_xy, mean_x, mean_y, aspect, pen, num_sh,
                          leng_line] + region_vecx + region_vecy + position_fx + \
                         position_fy + position_lx + position_ly + position_fx_h + position_fy_h + position_lx_h + position_ly_h + fuzzy_histogram

    else:
        feature_vector = [cov_xy, mean_x, mean_y, aspect, pen, num_sh,
                          leng_line] + region_vecx + region_vecy + position_fx + \
                         position_fy + position_lx + position_ly + position_fx_h + position_fy_h + position_lx_h + position_ly_h + fuzzy_histogram + [
                             key]

    return feature_vector


def get_strokes(filename):
    """
    Gets strokes from inkml file
    :param filename: inkml file
    :return: stroke dictionary
    """
    file = open(filename)
    soup = bs.BeautifulSoup(file, 'html.parser')
    trace_id = soup.find_all('trace')
    strokes = {}
    for id in trace_id:

        x = []
        y = []
        traces = id.text.strip().split(',')
        for coords in traces:
            trace_parts = coords.strip().split(' ')
            x.append(float(trace_parts[0]))
            y.append(float(trace_parts[1]))

        strokes[int(id['id'])] = {'x': x, 'y': y}

    return strokes


def read_files(directory):
    """
    reads inkml file, extracts features and saves it to csv
    :param directory:
    :return:
    """

    files = os.listdir(directory)
    print(len(files))
    pre = Preprocessing()
    feature_matrix = []
    total = len(files)
    completed = 0
    gt_c = 0
    for file in files[0:]:
        print("Processing file : ", file, " Remaining files : ", total-completed, " Completed files : ", completed)
        f = open(os.path.join(directory, file))
        soup = bs.BeautifulSoup(f, 'html.parser')
        trace_groups = soup.find_all('tracegroup')

        for tracegroup in trace_groups[1:]:
            traceview = tracegroup.find_all('traceview')
            trace_id = []
            for t in traceview:
                trace_id.append(t['tracedataref'])

            gt = tracegroup.annotation.text
            gt_c += 1
            X = []
            Y = []

            for id in trace_id:
                traces = soup.findAll("trace", {'id': id})
                for trace in traces:
                    coords = trace.text.strip().split(",")
                    x = []
                    y = []
                    for coord in coords:
                        trace_parts = coord.strip().split(' ')
                        x.append(float(trace_parts[0]))
                        y.append(float(trace_parts[1]))

                    X.extend(x)
                    Y.extend(y)

            X, Y = pre.dopreprocess(x=X, y=Y)
            ar = pre.get_aspect(X, Y)
            pen = len(trace_id)
            feature_matrix.append(extract_features(X, Y, pen, ar, key=gt))
        completed += 1

    df = pd.DataFrame(feature_matrix)
    print("Shape of Matrix ", df.shape, " Total Ground truths in file", gt_c)
    name = directory.strip().split("/")[0]
    df.to_csv(name + ".csv", index=False)


def train_model(filename):
    c = [str(i) for i in range(82)]
    dtypes = {i: float for i in c}
    dtypes[82] = str
    train_data = pd.read_csv(filename)
    X_train = train_data.as_matrix(c)
    y_train = train_data['82'].astype(str)

    print("Training model RandomForest...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf.fit(X_train, y_train)
    print("Model Trained, saving...")
    joblib.dump(rf, "rf_classifier_new.pkl", protocol=pickle.HIGHEST_PROTOCOL)


def accuracy_model(filename):
    c = [str(i) for i in range(82)]
    test_data = pd.read_csv(filename)
    X_test = test_data.as_matrix(c)
    y_test = test_data['82'].astype(str)
    clf = joblib.load("rf_classifier_100.pkl")
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_true=y_test, y_pred=y_pred, normalize=True)
    print("Accuracy of model is : ", score * 100)

