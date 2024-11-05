import argparse
import csv
from itertools import islice
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier


def parse_options():
    parser = argparse.ArgumentParser(description='Malware Detection.')
    parser.add_argument('-d', '--dir', help='The path of a dir contains benign and malware feature csv.', required=True, type=str)
    parser.add_argument('-o', '--out', help='The dir_path of output', required=True, type=str)
    args = parser.parse_args()

    return args


def feature_extraction_all(feature_csv):
    features = []

    with open(feature_csv, 'r') as f:
        data = csv.reader(f)
        for line in islice(data, 1, None):
            try:
                feature = [float(i) for i in line[2:]]
                features.append(feature)
            except Exception as e:
                # print(line[0:2])
                # print(e)
                pass
    print('len:')
    print(len(features))
    return features


def obtain_dataset(dir_path):

    clone_featureCSV_1 = dir_path + 'type-1_sim.csv'
    clone_featureCSV_2 = dir_path + 'type-2_sim.csv'
    clone_featureCSV_3 = dir_path + 'type-3_sim.csv'
    clone_featureCSV_4 = dir_path + 'type-4_sim.csv'
    clone_featureCSV_5 = dir_path + 'type-5_sim.csv'
    clone_featureCSV_6 = dir_path + 'type-6_sim.csv'
    nonclone_featureCSV = dir_path + 'noclone_sim.csv'

    Vectors = []
    Labels = []

    clone_features1 = feature_extraction_all(clone_featureCSV_1)
    clone_features2 = feature_extraction_all(clone_featureCSV_2)
    clone_features3 = feature_extraction_all(clone_featureCSV_3)
    clone_features4 = feature_extraction_all(clone_featureCSV_4)
    clone_features5 = feature_extraction_all(clone_featureCSV_5)
    clone_features6 = feature_extraction_all(clone_featureCSV_6)
    nonclone_features = feature_extraction_all(nonclone_featureCSV)

    Vectors.extend(clone_features1)
    Labels.extend([1 for i in range(len(clone_features1))])
    Vectors.extend(clone_features2)
    Labels.extend([1 for i in range(len(clone_features2))])
    Vectors.extend(clone_features3)
    Labels.extend([1 for i in range(len(clone_features3))])
    Vectors.extend(clone_features4)
    Labels.extend([1 for i in range(len(clone_features4))])
    Vectors.extend(clone_features5)
    Labels.extend([1 for i in range(len(clone_features5))])
    Vectors.extend(clone_features6)
    Labels.extend([1 for i in range(len(clone_features6))])

    Vectors.extend(nonclone_features)
    Labels.extend([0 for i in range(len(nonclone_features))])

    print('len of Vectors:')
    print(len(Vectors))
    print('len of Labels:')
    print(len(Labels))

    return Vectors, Labels


def random_features(vectors, labels):
    Vec_Lab = []

    for i in range(len(vectors)):
        vec = vectors[i]
        lab = labels[i]
        vec.append(lab)
        Vec_Lab.append(vec)

    random.shuffle(Vec_Lab)

    return [m[:-1] for m in Vec_Lab], [m[-1] for m in Vec_Lab]


def tag_random_features(vectors, labels):
    Vec_Lab = []

    for i in range(len(vectors)):
        vec = vectors[i]
        lab = labels[i]
        vec.append(lab)
        Vec_Lab.append(vec)

    random.shuffle(Vec_Lab)

    Vectors_token = [m[0:3] for m in Vec_Lab]
    Vectors_tree = [m[3:6] for m in Vec_Lab]
    Vectors_graph = [m[6:9] for m in Vec_Lab]

    lab = [m[-1] for m in Vec_Lab]

    return Vectors_token, Vectors_tree, Vectors_graph, lab


def eleven_5(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=5)

    Precisions1 = []
    Recalls1 = []
    F1s1 = []

    j = 1
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]


        clf1 = KNeighborsClassifier(n_neighbors=5)
        clf1.fit(train_X, train_Y)
        joblib.dump(clf1, 'clf_knn_5.pkl')
        y_pred1 = clf1.predict(test_X)
        #print("5NN")


        clf2 = KNeighborsClassifier(n_neighbors=1)
        clf2.fit(train_X, train_Y)
        joblib.dump(clf2, 'clf_knn_1.pkl')
        y_pred2 = clf2.predict(test_X)
        #print("1NN")


        clf3 = KNeighborsClassifier(n_neighbors=3)
        clf3.fit(train_X, train_Y)
        joblib.dump(clf3, 'clf_knn_3.pkl')
        y_pred3 = clf3.predict(test_X)
        #print("3NN")


        clf4 = tree.DecisionTreeClassifier()
        clf4.fit(train_X, train_Y)
        joblib.dump(clf4, 'clf_decision_tree.pkl')
        y_pred4 = clf4.predict(test_X)
        #print("DT")


        clf5 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=64), random_state=0)
        clf5.fit(train_X, train_Y)
        joblib.dump(clf5, 'clf_adaboost.pkl')
        y_pred5 = clf5.predict(test_X)
        #print("ADABOOST")


        clf6 = GradientBoostingClassifier(max_depth=64, random_state=0)
        clf6.fit(train_X, train_Y)
        joblib.dump(clf6, 'clf_gdbt.pkl')
        y_pred6 = clf6.predict(test_X)
        #print("GDBT")


        clf8 = GaussianNB()
        clf8.fit(train_X, train_Y)
        joblib.dump(clf8, 'clf_GaussianNB.pkl')
        y_pred8 = clf8.predict(test_X)
        #print("GaussianNB")


        clf9 = LogisticRegression()
        clf9.fit(train_X, train_Y)
        joblib.dump(clf9, 'clf_LogisticRegression.pkl')
        y_pred9 = clf9.predict(test_X)
        #print("LogisticRegression")


        clf10 = NearestCentroid()
        clf10.fit(train_X, train_Y)
        joblib.dump(clf10, 'clf_NearestCentroid.pkl')
        y_pred10 = clf10.predict(test_X)
        #print("NearestCentroid")


        clf11 = RidgeClassifier()
        clf11.fit(train_X, train_Y)
        joblib.dump(clf11, 'clf_RidgeClassifier.pkl')
        y_pred11 = clf11.predict(test_X)
        #print("RidgeClassifier")


        clf12 = QuadraticDiscriminantAnalysis()
        clf12.fit(train_X, train_Y)
        joblib.dump(clf12, 'clf_QuadraticDiscriminantAnalysis.pkl')
        y_pred12 = clf12.predict(test_X)
        #print("QuadraticDiscriminantAnalysis")

        y_pred = [0 for i in range(len(y_pred11))]

        for i in range(len(y_pred11)):
            sum = 0
            sum += y_pred1[i]
            sum += y_pred2[i]
            sum += y_pred3[i]
            sum += y_pred4[i]
            sum += y_pred5[i]
            sum += y_pred6[i]
            sum += y_pred8[i]
            sum += y_pred9[i]
            sum += y_pred10[i]
            sum += y_pred11[i]
            sum += y_pred12[i]
            if sum >= 6:
                y_pred[i] = 1

        y_pred = np.array(y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        print(f1, precision, recall)
        Precisions1.append(precision)
        Recalls1.append(recall)
        F1s1.append(f1)

        j += 1
        break
    print(np.mean(F1s1), np.mean(Precisions1), np.mean(Recalls1))

    #return np.mean(Precisions)


def Predict(vectors, labels):
    #Vectors, Labels = random_features(vectors, labels)
    test_X = np.array(vectors)
    test_Y = np.array(labels)

    lr1 = joblib.load("clf_knn_1.pkl")
    y_pred1 = lr1.predict(test_X)

    lr2 = joblib.load("clf_knn_3.pkl")
    y_pred2 = lr2.predict(test_X)

    lr3 = joblib.load("clf_knn_5.pkl")
    y_pred3 = lr3.predict(test_X)

    lr4 = joblib.load("clf_decision_tree.pkl")
    y_pred4 = lr4.predict(test_X)

    lr5 = joblib.load("clf_adaboost.pkl")
    y_pred5 = lr5.predict(test_X)

    lr6 = joblib.load("clf_gdbt.pkl")
    y_pred6 = lr6.predict(test_X)

    lr7 = joblib.load("clf_GaussianNB.pkl")
    y_pred7 = lr7.predict(test_X)

    lr8 = joblib.load("clf_LogisticRegression.pkl")
    y_pred8 = lr8.predict(test_X)

    lr9 = joblib.load("clf_NearestCentroid.pkl")
    y_pred9 = lr9.predict(test_X)

    lr10 = joblib.load("clf_RidgeClassifier.pkl")
    y_pred10 = lr10.predict(test_X)

    lr11 = joblib.load("clf_QuadraticDiscriminantAnalysis.pkl")
    y_pred11 = lr11.predict(test_X)


    y_pred = [0 for i in range(len(y_pred11))]

    for i in range(len(y_pred11)):
        sum = 0
        sum += y_pred1[i]
        sum += y_pred2[i]
        sum += y_pred3[i]
        sum += y_pred4[i]
        sum += y_pred5[i]
        sum += y_pred6[i]
        sum += y_pred7[i]
        sum += y_pred8[i]
        sum += y_pred9[i]
        sum += y_pred10[i]
        sum += y_pred11[i]

        if sum >= 6:
            y_pred[i] = 1

    y_pred = np.array(y_pred)
    print(y_pred)
    print(test_Y)
    precision = precision_score(y_true=test_Y, y_pred=y_pred)
    recall = recall_score(y_true=test_Y, y_pred=y_pred)
    print(precision, recall)
    print(y_pred1)
    print(y_pred2)
    print(y_pred3)
    print(y_pred4)
    print(y_pred5)
    print(y_pred6)
    print(y_pred7)
    print(y_pred8)
    print(y_pred9)
    print(y_pred10)
    print(y_pred11)


def main1():
    dir_path = '/home/data4T/wym/fsl/precision/output/'
    Vectors, Labels = obtain_dataset(dir_path)

    #vectors_token, vectors_tree, vectors_graph, labels = tag_random_features(Vectors, Labels)
    vectors, labels = random_features(Vectors, Labels)

    eleven_5(vectors, labels)
    # eleven_5(vectors_token, labels)
    # print('token')
    # eleven_5(vectors_tree, labels)
    # print('tree')
    # eleven_5(vectors_graph, labels)
    # print('graph')
    #Predict(vectors, labels)


if __name__ == '__main__':
    main1()

