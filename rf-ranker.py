from sklearn.linear_model import SGDClassifier, SGDRegressor, \
    LogisticRegressionCV, LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.metrics import classification_report
import logging
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import csv
import math
from sklearn import utils
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    AdaBoostRegressor, RandomForestRegressor, ExtraTreesClassifier
from treeinterpreter import treeinterpreter as ti
import random
import operator
import sys

'''
Usage: 
For training:
python rf-ranker.py train path_to_train_data.tsv

For testing:
python rf-ranker.py test path_to_test_data.tsv"

The code is based on the following paper:
More Accurate Question Answering on Freebase
Hannah Bast, Elmar Haussmann
http://ad-publications.informatik.uni-freiburg.de/CIKM_freebase_qa_BH_2015.pdf

You can find the original code at:
https://github.com/elmar_haussmann/aqqu

However, we re-implemented the code for easier accessibility.


Written by:
Mohamed Yahya and Abdalghani Abujabal


The file containing data points (feature vectors) must have the following schema:
Each data point must have two rows: a header row and a content row

For taining:
object_id \t data_point_id \t label \t feature_1 \t ..... \t feature_n
train_object_1 \t 1 \t 0.4 \t 1.0 \t .... \t 0.5

object_id \t data_point_id \t label \t feature_1 \t ..... \t feature_n
train_object_1 \t 2 \t 0.8 \t 0.0 \t .... \t 100.0

For testing:
object_id \t data_point_id \t feature_1 \t ..... \t feature_n
test_object_1 \t 1 \t 0.9 \t .... \t 0.5


Note: a set of different data_point_ids share the same object_id.

'''

vec = None
label_encoder = None
decision_tree = None

# True: training mode
# False: testing mode
train = None

# file containing data points either for training or testing
input_file = None




def load_model():
    model_file = get_model_filename()
    try:
        [model, label_enc,vec] \
                = joblib.load(model_file)
        correct_index = label_enc.transform([1])[0]
        print ("Loaded scorer model from %s" % model_file)
    except IOError:
        print ("Model file %s could not be loaded." % model_file)
        raise
    return model, label_enc, correct_index, vec


def get_model_filename():
    ''' Return the model file name '''
    model_filename = "RandomForestClassifierDiff_train_detailed"
    model_base_dir = "models"
    model_file = "%s/%s.model" % (model_base_dir, model_filename)
    return model_file

def load_data(f):
    # group_feautres_map = {object id, list of features vectors}
    group_feautres_map = {}
    # group_labels_map = {object id, list of labels of feature vectors}
    group_labels_map = {}
    # group_point_id_map = {object id, list of ids of feature vectors}
    group_point_id_map = {}
    current_object_id = None
    # list of maps. Each map represents a feature vector {feature_name, value}
    group_features = []
    # list of labels for feature vectors
    group_labels = []
    # list of feature vector ids
    group_ids = []
    lines_read = 0
    while True:
        header = f.readline().split('\t')
        content = f.readline().split('\t')

        if len(header) < 2 :
            break

        lines_read += 2
        object_id = content[0]
        data_point_id = content[1]
        start_index = 2
        label = -1
        if train:
            label = float(content[2])
            start_index = 3

        features_map = {}
        for i in range(start_index, len(content)):
            features_map[header[i]] = float(content[i])

        if object_id != current_object_id :
            if current_object_id != None:
                group_feautres_map[current_object_id] = group_features
                group_labels_map[current_object_id] = group_labels
                group_point_id_map[current_object_id] = group_ids

            group_features = []
            group_labels = []
            group_ids = []
            current_object_id = object_id
            group_features.append(features_map)
            group_labels.append(label)
            group_ids.append(data_point_id)
        else:
            group_features.append(features_map)
            group_labels.append(label)
            group_ids.append(data_point_id)


    # add the last object
    group_feautres_map[object_id] = group_features
    group_labels_map[object_id] = group_labels
    group_point_id_map[object_id] = group_ids
    return group_feautres_map, group_point_id_map, group_labels_map

def feature_diff(features_a, features_b):
    keys = set(features_a.keys() + features_b.keys())
    diff = dict()
    for k in keys:
        v_a = features_a.get(k, 0.0)
        v_b = features_b.get(k, 0.0)
        diff[k + "_a"] = v_a
        diff[k + "_b"] = v_b
        diff[k] = v_a - v_b
    return diff


def construct_pair_examples_best(group_features, group_labels):
    labels = []
    features = []
    # Find candidate with highest label value
    max_label_idx = group_labels.index(max(group_labels)) # first one arbitrary
    max_label = group_labels[max_label_idx]

    for idx, features_at_idx in enumerate(group_features):
        if max_label_idx != idx :
            if group_labels[idx] != max_label: # avoid making preferences for pairs with same label value
                diff = feature_diff(group_features[max_label_idx], features_at_idx)
                labels.append(1)
                features.append(diff)
                diff = feature_diff(features_at_idx, group_features[max_label_idx])
                features.append(diff)
                labels.append(0)
    return features, labels


def preprocess(group_feautres_map, group_labels_map):
    pair_features = []
    pair_labels = []

    for object_id, group_features in group_feautres_map.iteritems():
        group_labels = group_labels_map[object_id]
        testfoldpair_features, testfoldpair_labels = construct_pair_examples_best(group_features, group_labels)
        pair_features.extend(testfoldpair_features)
        pair_labels.extend(testfoldpair_labels)
    labels = label_encoder.fit_transform(pair_labels)
    X = vec.fit_transform(pair_features)
    X, labels = utils.shuffle(X, labels, random_state=999)
    print "Finished preprocessing"
    return X, labels


def train_classifier(X, labels):
    decision_tree.fit(X, labels)
    print "Finished training"

    pred = decision_tree.predict(X)
    print "Finished prediction"

    print ("F_1 score on train: %.4f" % metrics.f1_score(labels, pred, pos_label=1))
    print ("Classification report:\n" + classification_report(labels, pred))


def save_model(label_encoder,model,vec):
    print ("Writing model to %s." % get_model_filename())
    joblib.dump([model, label_encoder,vec], get_model_filename())
    print "Model saved"
  

def compare_pair(x_features, y_features):
    res = None
    if res is None:
        diff = feature_diff(x_features, y_features)
        X = vec.transform(diff)
        decision_tree.n_jobs = -1
        p = decision_tree.predict(X)
        c = label_encoder.inverse_transform(p)
        res = c[0]
    if res == 1:
        return 1
    else:
        return -1


def rank(group_feautres_map):
    ranked_results = {}
    for object_id, group_features in group_feautres_map.iteritems():
        ranked_candidates = sorted(group_features,
                                   cmp=compare_pair,
                                   key=lambda x: x,
                                   reverse=True)
        ranked_results[object_id] = ranked_candidates
    return ranked_results


if __name__ == '__main__':
    if len(sys.argv) <> 3:
        print "usage: python rf-ranker.py train path_to_train_data.tsv  or \n python rf-ranker.py test path_to_test_data.tsv"
        sys.exit()
    else:
        if sys.argv[1] == "train":
            train = True
            input_file = sys.argv[2]
        elif sys.argv[1] == "test":
            train = False
            input_file = sys.argv[2]
        else:
            print "usage: python rf-ranker.py train path_to_train_data.tsv  or \n python rf-ranker.py test path_to_test_data.tsv"
            sys.exit()

        f = open(input_file, 'r')

        # load data
        group_feautres_map, group_point_id_map, group_labels_map = load_data(f)

    if train:
        # initialize the random forest classifier
        # n_estimators: The number of trees in the forest.
        # n_jobs: The number of jobs to run in parallel. If -1, then the number of jobs is set to the number of cores.
        label_encoder = LabelEncoder()
        decision_tree = RandomForestClassifier(class_weight='auto',
                                           random_state=999,
                                           n_jobs=-1,
                                           n_estimators=90)
        vec = DictVectorizer(sparse=True)


        print group_point_id_map
        print group_feautres_map
        print group_labels_map
        X, labels = preprocess(group_feautres_map, group_labels_map)
  		
        # train classifier
        train_classifier(X, labels)

        # save model
        save_model(label_encoder, decision_tree, vec)
    else:
        # load trained model
        decision_tree, label_encoder, correct_index, vec = load_model()

        # rank candidates
        ranked_results = rank(group_feautres_map)
         
        # print results
        for object_id, ranked_candidates in ranked_results.iteritems():
            print object_id
            group_features = group_feautres_map[object_id]
            group_ids = group_point_id_map[object_id]
            rank = 1
            for ranked_candidate in ranked_candidates:
                idx = group_features.index(ranked_candidate)
                query_id = group_ids[idx]
                print query_id, rank
                rank = rank + 1




