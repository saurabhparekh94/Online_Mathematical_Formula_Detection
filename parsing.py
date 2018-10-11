"""
File Name: parsing.py
@author : Tappan Ajmera (tpa7999@g.rit.edu)
@author : Saurabh Parekh (sbp4709@g.rit.edu)
"""

import math
import os
import pickle

import bs4 as bs
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

from preprocessing import Preprocessing
from symbol import Symbol


def check_overlap(unblocked_range, b_min, b_max):
    '''
    Given a range, checkes if it overlaps with a 360 degree view of current symbol
    :param unblocked_range: vector containing how much is visible
    :param b_min: start of blocking range
    :param b_max: end of blocking range
    :return:
    '''
    can_see = False
    adjusted_max = False
    adjusted_min = False
    if int(b_min) == 360:
        b_min = 359
        adjusted_min = True
    if int(b_max) == 360:
        b_max = 359
        adjusted_max = True

    if unblocked_range[int(b_min)] == 0 or unblocked_range[int(b_max)] == 0:

        if adjusted_max:
            b_max = 360
        if adjusted_min:
            b_min = 360

        unblocked_range[int(b_min): int(b_max) + 1] = 1
        can_see = True

    return can_see, unblocked_range


def calculate_angle(symbol, x_c, y_c):
    """
    Find convext hull points and calculate angle between centroid of symbol 1 to hull points of symbol 2
    :param symbol: symbol object
    :param x_c: x centroid of symbol under consideration
    :param y_c: y centroid of symbol under consideration
    :return:
    """
    x = []
    y = []
    v2 = np.array([1, 0]).reshape(2, 1)
    for points in symbol.x:
        for point in points:
            x.append(point)

    for points in symbol.y:
        for point in points:
            y.append(point)

    points = np.column_stack((x, y))

    if len(points) < 3:
        hull_points = points
    elif max(y) == min(y):
        hull_points = points
    elif max(x) == min(x):
        hull_points = points
    else:
        try:
            hull = ConvexHull(points)
            x = points[hull.vertices, 0]
            y = points[hull.vertices, 1]
            hull_points = np.column_stack((x, y))
        except:
            hull_points = points

    bar_angles = []

    for point in hull_points:
        xn, yn = point[0], point[1]
        v1 = np.array([[xn - x_c, yn - y_c]]).reshape(1, 2)
        magv1 = np.sqrt(np.sum(np.square(v1)))
        magv2 = np.sqrt(np.sum(np.square(v2)))

        if magv1 == 0:
            magv1 = 1
        angle = math.acos(v1.dot(v2) / (magv1 * magv2))
        if yn < y_c:
            angle = 2 * math.pi - angle
        deg = math.degrees(angle)
        bar_angles.append(deg)

    return min(bar_angles), max(bar_angles)


def centroid(x, y):
    """
    Finds centroid of a given symbol
    :param x: x coordinates
    :param y: y coordinates
    :return: returns centroid
    """

    X = []
    Y = []
    for val in x:
        X += val

    for val in y:
        Y += val

    X_max = max(X)
    Y_max = max(Y)
    X_min = min(X)
    Y_min = min(Y)

    x_c = (X_max + X_min) / 2
    y_c = (Y_max + Y_min) / 2

    return x_c, y_c


def euclidean(x_c, y_c, curr_x_c, curr_y_c):
    """
    calculates euclidean distance between two points
    :param x_c: x coord of p2
    :param y_c: y coord of p2
    :param curr_x_c: x coord of p1
    :param curr_y_c: y coord of p1
    :return:
    """
    euclid = math.sqrt(((curr_x_c - x_c) ** 2) + ((curr_y_c - y_c) ** 2))

    return euclid


def sort_symbols(x_c, y_c, symbols, idx):
    """
    Sorts symbol list based on distance between the centroids
    """
    distance_list = []
    for i in range(len(symbols)):
        if i != idx:
            sym = symbols[i]
            curr_x_c, curr_y_c = centroid(sym.x, sym.y)
            distance = euclidean(x_c, y_c, curr_x_c, curr_y_c)
            distance_list.append([distance, sym])

    sorted_symbols = sorted(distance_list, key=lambda tup: tup[0])

    return sorted_symbols


def line_of_sight(symbol_list, clf):
    """
    Creates a line of sight graph from given symbol list
    :param symbol_list: list of symbols
    :param clf: classifier to predict relations
    :return: line of sight graph matrix
    """
    graph_mat = [[0 for _ in range(len(symbol_list))] for _ in range(len(symbol_list))]
    label_mat = [[0 for _ in range(len(symbol_list))] for _ in range(len(symbol_list))]
    total_symbols = len(symbol_list)

    sym_to_index = {}
    index_to_sym = {}
    feature_matrix = []
    feature_track = []  # keeps track of which index of feature_matrix represents which symbol pair

    # create mapping of symbol-idx and idx-symbol
    for idx in range(total_symbols):
        sym_to_index[symbol_list[idx]], index_to_sym[idx] = idx, symbol_list[idx]

    if total_symbols > 1:

        for idx in range(total_symbols):
            curr = symbol_list[idx]
            unblocked_range = np.zeros(360)

            #find centroid of symbol
            x_c, y_c = centroid(curr.x, curr.y)

            #sort symbols based on current
            sorted_bycurr = sort_symbols(x_c, y_c, symbol_list, idx)

            for val in sorted_bycurr:
                symbol = val[1]

                #calculate blocking angle range
                bar_min, bar_max = calculate_angle(symbol, x_c, y_c)

                #check overlap
                can_see, unblocked_range = check_overlap(unblocked_range, bar_min, bar_max)

                if can_see:

                    #create a graph if symbol can be seen
                    other_idx = sym_to_index[val[1]]
                    graph_mat[idx][other_idx] = index_to_sym[other_idx].symbol

                    writing_slope = curr.writing_slope(symbol)
                    writing_curve = curr.writing_curvature(symbol)
                    bb_dist = curr.distance_between_box(symbol)
                    distance, horizontal_ofsset, vertical_distance = curr.distance_between_average_centres(symbol)
                    max_point_pair = curr.maximal_point_distance(symbol)
                    feature_matrix.append(
                        [writing_slope, writing_curve, bb_dist, distance, horizontal_ofsset, vertical_distance,
                         max_point_pair])
                    feature_track.append([idx, other_idx])

        #get probabilities of prediction
        probas = clf.predict_proba(feature_matrix)
        labels = clf.predict(feature_matrix)

        #create graph and label graph matrix
        for i in range(len(probas)):
            s1_idx = feature_track[i][0]
            s2_idx = feature_track[i][1]
            graph_mat[s1_idx][s2_idx] = max(probas[i])
            label_mat[s1_idx][s2_idx] = labels[i]

    return graph_mat, label_mat


# Section below is for training a model
def train(ink_dir, lg_dir):

    """
    This function is used for training model

    :param ink_dir:
    :param lg_dir:
    :return:
    """
    lg_files = os.listdir(lg_dir)
    pre = Preprocessing()

    feature_matrix = []
    targets = []
    c = 0
    total = len(lg_files)
    for file in lg_files:
        print(file, total - c, c)
        symbols = {}

        with open(lg_dir + "/" + file) as f:
            for line in f:
                if line.startswith("O"):
                    filt_line = line.strip().split(",")
                    symbols[filt_line[1].strip()] = [filt_line[2], filt_line[4:]]

        inkml_file = file.replace(".lg", ".inkml")

        with open(ink_dir + "/" + inkml_file) as f:
            soup = bs.BeautifulSoup(f, 'html.parser')
            for key in symbols:
                label = symbols[key][0]
                strokes = symbols[key][1]
                id_list = []
                X = []
                Y = []
                for id in strokes:
                    st_id = id.strip()
                    trace = soup.findAll("trace", {'id': st_id})

                    coords = trace[0].text.strip().split(",")
                    x = []
                    y = []
                    for coord in coords:
                        trace_parts = coord.strip().split(' ')
                        x.append(float(trace_parts[0]))
                        y.append(float(trace_parts[1]))

                    X.append(x)
                    Y.append(y)
                    id_list.append(st_id)
                X, Y = pre.dopreprocess(x=X, y=Y, parser=True)
                symbols[key] = Symbol(label=label, x=X, y=Y, stroke_id=id_list)

        # relations section
        with open(lg_dir + "/" + file) as f:
            for line in f:
                if line.startswith("EO"):
                    filt_line = line.strip().split(",")
                    sym1 = symbols[filt_line[1].strip()]
                    sym2 = symbols[filt_line[2].strip()]
                    relation = filt_line[3].strip()

                    writing_slope = sym1.writing_slope(sym2)
                    writing_curve = sym1.writing_curvature(sym2)
                    bb_dist = sym1.distance_between_box(sym2)
                    distance, horizontal_ofsset, vertical_distance = sym1.distance_between_average_centres(sym2)
                    max_point_pair = sym1.maximal_point_distance(sym2)
                    feature_matrix.append(
                        [writing_slope, writing_curve, bb_dist, distance, horizontal_ofsset, vertical_distance,
                         max_point_pair])
                    targets.append(relation)

        c += 1

    print("Shape of Training matrix")
    print(len(feature_matrix), "x", len(feature_matrix[0]))
    print("Unique labels : ", np.unique(targets))

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf.fit(X=feature_matrix, y=targets)
    joblib.dump(rf, "relation_classifier_bonus.pkl", protocol=pickle.HIGHEST_PROTOCOL)

    rf = joblib.load("relation_classifier_bonus.pkl")

    score = accuracy_score(y_true=targets, y_pred=rf.predict(feature_matrix), normalize=True)

    print("accuracy of model is :", (score * 100))
