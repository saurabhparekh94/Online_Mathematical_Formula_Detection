"""
File Name: Segmentation.py
@author : Tappan Ajmera (tpa7999@g.rit.edu)
@author : Saurabh Parekh (sbp4709@g.rit.edu)
"""

import math
from operator import itemgetter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from symbol import Symbol


def euclidiean(distances):
    """
    Find euclidiean distance
    :param distances:
    :return:
    """
    euclidiean_matrix = []
    for x in range(len(distances)):
        euclidiean_dist = []
        i = 0
        while i < x:
            euclidiean_dist.append(0.0)
            i += 1
        for y in range(x, len(distances)):
            sum_of_square = math.pow(distances[x][1] - distances[y][1], 2) + math.pow(distances[x][2] - distances[y][2],
                                                                                      2)
            dist = math.sqrt(sum_of_square)
            euclidiean_dist.append(dist)
        euclidiean_matrix.append(euclidiean_dist)

    return euclidiean_matrix


def find_centroid(strokes):
    """
    Find centroid of all the strokes
    :param strokes:
    :return: centroids
    """
    centroid_dict = {}
    for id in strokes:
        dict = strokes.get(id)
        x = dict['x']
        y = dict['y']

        centroidx = sum(x) / len(x)
        centroidy = sum(y) / len(y)

        centroid_dict[id] = [centroidx, centroidy]

    return centroid_dict


def mst(strokes):
    """
    Creates a minimum spanning tree from the given strokes
    :param strokes:
    :return: minimum spanning tree matrix
    """
    centroid_dict = find_centroid(strokes)
    for_euclidean = []
    for key in centroid_dict:
        x_y_point = centroid_dict.get(key)
        for_euclidean.append([int(key), x_y_point[0], x_y_point[1]])

    for_euclidean = sorted(for_euclidean, key=itemgetter(0))

    euclidiean_matrix = euclidiean(for_euclidean)
    euclidiean_matrix = np.asarray(euclidiean_matrix)

    euclidiean_matrix = csr_matrix(euclidiean_matrix)
    minimum_spanning = minimum_spanning_tree(euclidiean_matrix)

    minimum_spanning_matrix = minimum_spanning.toarray()  # .astype(int)

    return minimum_spanning_matrix


def do_segmentation(file, lg_dir, spanning_tree, traces, classifier, pre, extract_features, class_count):
    """
    Performs segmentation on given MST created from Strokes.
    Algorithm:
        Create a graph from MST matrix
        Perform DFS to find 4 closest connected strokes
        Create combinations of the strokes from DFS
        Send to classifier to predict which combination has highest confidence. This forms a segment
        Update the 'taken' dictionary with all the strokes that are taken by a segment
        create lg file
    :param file: inkml file
    :param lg_dir: directory to store lg files
    :param spanning_tree: MST matrix
    :param traces: strokes in the file
    :param classifier: classifier for prediction of symbols
    :param pre: preprocessing object
    :param extract_features: feature extractor object
    :param class_count: dictionary to maintain count of symbol types
    :return: None
    """

    taken = {i: False for i in range(len(spanning_tree))}
    file_write_data = ''
    connected_dict = {}

    # creates a dicitionary with node : 4 closest nodes in MST
    for i, row in enumerate(spanning_tree):
        non_zero = np.nonzero(row)
        connected_dict[i] = []
        k = i + 4
        for val in non_zero[0]:
            if val <= k:
                connected_dict[i].append(val)

    # Perform DFS and update the existing neighbors
    for node in connected_dict:
        if not taken[node]:
            consider = connected_dict[node]
            for val in consider:
                if val == node + 1:
                    stack = [val]
                    while stack:
                        curr = stack.pop()
                        neighbors = connected_dict[curr]
                        for child in neighbors:
                            if child == curr + 1 or child == curr + 2:
                                if child not in consider:
                                    consider.append(child)
                                    stack.append(child)
                                    if len(consider) >= 3:
                                        stack = []
                                        break
                                if len(consider) >= 3:
                                    stack = []
                                    break
                            if len(consider) >= 3:
                                stack = []
                                break
                else:
                    break

            connected_dict[node] = consider

    # create structure like {stroke_id : {stroke_id_combo: {x : coordinates, y : coordinates}}}
    strokes_seg = {}
    for node in connected_dict:
        strokes = {}

        seg_key = str(node)
        X = traces[node]['x']
        Y = traces[node]['y']
        strokes[seg_key] = {'x': X, 'y': Y}

        for val in connected_dict[node]:
            seg_key += "/" + str(val)
            X = X + traces[val]['x']
            Y = Y + traces[val]['y']
            strokes[seg_key] = {'x': X, 'y': Y}

        strokes_seg[node] = strokes

    # create a small dataset to be sent for prediction by classifier
    feature_frame = []
    seg_combo = []
    for key in strokes_seg:
        for k in strokes_seg[key]:
            curr_seg = strokes_seg[key][k]
            x = curr_seg['x']
            y = curr_seg['y']
            x, y = pre.dopreprocess(x=x, y=y)
            ar = pre.get_aspect(x=x, y=y)
            num_strokes = k.strip().split("/")
            pen = len(num_strokes)
            feature_mat = np.asarray(extract_features(x, y, pen, ar, testing=True))
            feature_frame.append(feature_mat)
            seg_combo.append([key, k])

    # Send all possible segmentations for classification and obtain probabilities
    y_pred = classifier.predict(feature_frame)
    y_proba = classifier.predict_proba(feature_frame)

    first = True
    curr_max = -10000
    max_combo = ''
    max_label = ''
    sid = ''
    probas = {}
    i = 0
    last_cond = False

    symbol_list = []  # list will be sent to parsing
    while i < (len(y_proba)):
        if not taken[seg_combo[i][0]]:
            if first:
                sid = seg_combo[i][0]
                temp_max = max(y_proba[i])
                if temp_max > curr_max:
                    curr_max = temp_max
                    max_combo = seg_combo[i][1]
                    max_label = y_pred[i]
                first = False
                last_cond = False
            else:
                temp_sid = seg_combo[i][0]
                if temp_sid == sid:
                    temp_max = max(y_proba[i])
                    if temp_max > curr_max:
                        curr_max = temp_max
                        max_combo = seg_combo[i][1]
                        max_label = y_pred[i]
                    last_cond = False
                else:
                    probas[max_combo] = [max_label, curr_max]
                    curr_max = -1000
                    first = True
                    stroke_ids = max_combo.split("/")
                    id_for_sym = []
                    x_in_symbol = []
                    y_in_symbol = []
                    for id in stroke_ids:
                        taken[int(id)] = True

                        # create structure for parsing
                        id_for_sym.append(int(id))
                        x_in_symbol.append(traces[int(id)]['x'])
                        y_in_symbol.append(traces[int(id)]['y'])


                    x_in_symbol, y_in_symbol = pre.dopreprocess(x = x_in_symbol, y = y_in_symbol, parser=True)
                    label_ct = class_count[max_label]
                    class_count[max_label] += 1

                    if max_label == ",":
                        max_label = "COMMA"

                    sym_obj = Symbol(x=x_in_symbol, y=y_in_symbol, label=max_label, sym_ct=label_ct,
                                     stroke_id=id_for_sym)
                    symbol_list.append(sym_obj)

                    last_cond = True
                    i -= 1
        i += 1

        if i == len(y_proba):

            if not last_cond:
                probas[max_combo] = [max_label, curr_max]
                curr_max = -1000
                first = True
                stroke_ids = max_combo.split("/")

                id_for_sym = []
                x_in_symbol = []
                y_in_symbol = []

                for id in stroke_ids:
                    taken[int(id)] = True

                    # create structure for parsing
                    id_for_sym.append(int(id))
                    x_in_symbol.append(traces[int(id)]['x'])
                    y_in_symbol.append(traces[int(id)]['y'])

                label_ct = class_count[max_label]
                class_count[max_label] += 1

                x_in_symbol, y_in_symbol = pre.dopreprocess(x=x_in_symbol, y=y_in_symbol, parser=True)

                if max_label == ",":
                    max_label = "COMMA"
                sym_obj = Symbol(x=x_in_symbol, y=y_in_symbol, label=max_label, sym_ct=label_ct, stroke_id=id_for_sym)
                symbol_list.append(sym_obj)
                last_cond = True

    return symbol_list

