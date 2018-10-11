"""
File Name: main.py
@author : Tappan Ajmera (tpa7999@g.rit.edu)
@author : Saurabh Parekh (sbp4709@g.rit.edu)
"""

import os
import sys
import time
import traceback

import bs4 as bs
from sklearn.externals import joblib

from DiGraph import edmonds
from Segmentation import *
from file_writer import write_to_lg
from load_data import get_strokes, extract_features
from preprocessing import Preprocessing
from parsing import  line_of_sight


def stroke_parser(dir):
    """
    This function runs the whole chain : segmentation , classification , parsing
    :param dir: inkml directory
    :return: None
    """
    start = time.time()
    print("Time Started : ", start)
    files = os.listdir(dir)
    print("Loading classifiers")

    seg_clf = joblib.load('rf_classifier_100.pkl')
    rel_clf = joblib.load('relation_classifier4.pkl')
    print("Classifiers loaded")
    pre = Preprocessing()
    lg_dir = dir.strip().split("/")[0] + "_output_lg"

    if not os.path.exists(lg_dir):
        os.mkdir(lg_dir)
    count = 0

    for file in files:
        print("Processing file name : ", file, " Files processed : ", count, " Files remaining : ", len(files) - count)
        strokes = get_strokes(dir + "/" + file)
        class_count = {label: 1 for label in seg_clf.classes_}
        tree = mst(strokes)
        symbol_list = do_segmentation(file, lg_dir, tree, strokes, seg_clf, pre, extract_features, class_count)
        graph, labels = line_of_sight(symbol_list, rel_clf)
        relations = edmonds(graph)
        write_to_lg(file=file, symbol_list=symbol_list, labels=labels, relations=relations, lg_dir=lg_dir)

        count += 1

    print("System executed in ", (time.time() - start) / 60, " minutes.")


def perfectly_segmented_parser(ink_dir, bonus=False):
    """
    This is a parser for perfectly segmented symbols
    :param ink_dir: inkml directory
    :param bonus: boolean for bonus
    :return:
    """
    start = time.time()

    lg_dir = dir.strip().split("/")[0] + "_output_lg"

    if not os.path.exists(lg_dir):
        os.mkdir(lg_dir)

    ink_files = os.listdir(ink_dir)

    if bonus:
        print("Loaded Bonus classifier")
        clf = joblib.load("relation_classifier_bonus.pkl")
    else:
        print("Loaded relationship classifier")
        clf = joblib.load('relation_classifier4.pkl')
    pre = Preprocessing()
    total = len(ink_files)
    c = 0
    gt_c = 0

    for file in ink_files:
        print("Processing file : ", file, " Files remaining : ", total - c, " Files completed : ", c)

        f = open(os.path.join(ink_dir, file))

        soup = bs.BeautifulSoup(f, 'html.parser')
        trace_groups = soup.find_all('tracegroup')
        symbol_list = []

        #loop to isolate symbols
        for tracegroup in trace_groups[1:]:
            traceview = tracegroup.find_all('traceview')
            trace_id = []

            #loop to get strokes in a single symbol
            for t in traceview:
                trace_id.append(t['tracedataref'])

            gt = tracegroup.annotation.text
            gt_c += 1
            X = []
            Y = []

            #extract stroke coordinates
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

                    X.append(x)
                    Y.append(y)

            X, Y = pre.dopreprocess(x=X, y=Y, parser=True)
            if gt == ",":
                gt = "COMMA"
            sym_obj = Symbol(x=X, y=Y, label=gt, stroke_id=trace_id)
            symbol_list.append(sym_obj)

        symbol_count = {}

        #Run through list of symbols to get their count
        for sym in symbol_list:
            if sym.symbol not in symbol_count:
                symbol_count[sym.symbol] = 1
                sym.sym_ct = symbol_count[sym.symbol]
            else:
                symbol_count[sym.symbol] += 1
                sym.sym_ct = symbol_count[sym.symbol]

        #perform line of sight
        graph, labels = line_of_sight(symbol_list, clf)
        #run edmonds on los graph
        relations = edmonds(graph)

        #write result to lg
        write_to_lg(file=file, symbol_list=symbol_list, relations=relations, labels=labels, lg_dir=lg_dir)

        c += 1
    print("System executed in ", (time.time() - start) / 60, " minutes.")




if __name__ == "__main__":
    try:
        dir = sys.argv[1]
        choice = sys.argv[2]

        if choice == "perfect":
            if len(sys.argv) > 3:
                perfectly_segmented_parser(dir, bonus=True)
            else:
                perfectly_segmented_parser(dir)
        elif choice == "segment":
            stroke_parser(dir)

    except:
        traceback.print_exc()
        print("Usage")
        print("python main.py path_to_inkml 'segment/perfect' [bonus]")
