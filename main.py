import copy as cp
import numpy as np
from core.Proposed import BEDCOE
from core.DenStream import DenStream
from core.funcs_utility import distance
from evaluate.evaluation_online import pf_online, confusion_online
from evaluation import pf_report
import os
import scipy.io as sio
from sklearn import preprocessing
import random


class my_norm_scaler:
    def __init__(self, norm_name="z_score"):
        self.norm_name = norm_name  # by default z-score
        if self.norm_name.lower() == "min_max".lower():
            self.norm_scaler = preprocessing.MinMaxScaler()
        elif self.norm_name.lower() == "z_score".lower():
            self.norm_scaler = preprocessing.StandardScaler()

    def my_fit(self, XX):
        my_norm = self.norm_scaler.fit(XX)
        return my_norm

    def my_transform(self, xx):
        if xx.ndim == 1:  # if xx contains only 1 data sample
            xx = np.expand_dims(xx, 0)
        xx_trans = self.norm_scaler.transform(xx)
        return xx_trans


def experiments_repeat(X, y, clf_name, cluster_name, nb_pre_train, nb_test, nb_repeat=10, nb_base=10, center_info=None):
    nb_all = nb_pre_train + nb_test
    prediction_all_seed = np.empty([nb_repeat, nb_all])
    for re in range(nb_repeat):
        prediction_all_seed[re, :] = experiments_one_run(X, y, center_info, clf_name=clf_name,
                                                         cluster_name=cluster_name, nb_pre_train=nb_pre_train,
                                                         nb_test=nb_test, nb_base=nb_base, seed=re)

    return prediction_all_seed


def experiments_one_run(X, y, center_info=None, clf_name="bedcoe", cluster_name="denstream_our", nb_pre_train=500,
                        nb_test=5000, nb_base=10, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    # prepare
    clf_name = clf_name.lower()
    cluster_name = cluster_name.lower()

    # print title info
    print("classifier name: ", clf_name)
    print("pre_train: ", nb_pre_train, "test: ", nb_test)

    """prepare data stream"""
    nb_all = nb_pre_train + nb_test
    class_num = len(np.unique(y))
    X_train = X[0:nb_pre_train, :]
    y_train = y[0:nb_pre_train]
    X_test = X[nb_pre_train:nb_all, :]
    y_test = y[nb_pre_train:nb_all]

    # normalization
    norm_scaler = my_norm_scaler(norm_name="min_max")
    norm_scaler.my_fit(X_train)
    X_train = norm_scaler.my_transform(X_train)

    test_len = nb_test

    """pre-train"""
    classes = np.unique(y)

    if clf_name.lower() == 'bedcoe':
        center_ori = np.zeros([class_num, np.shape(X_train)[-1]])
        for c in range(class_num):
            c_idx = np.where(y_train == c)
            X_train_c = X_train[c_idx, :]
            if X_train_c.ndim == 3:
                X_train_c = np.squeeze(X_train_c, 0)
            center_ori[c, :] = np.mean(X_train_c, 0)
        std = np.zeros(class_num)

        # parameter for denstream
        dist = distance(X_train, X_train)
        dist_sort = np.sort(dist, 1)[:, 1:]  # delete 0
        dist_mean = np.mean(dist_sort)
        idx = np.where(dist_sort <= dist_mean)
        radius = 0.5 * np.mean(dist_sort[idx])
        eps = std + radius

        """cluster: initialization"""
        if cluster_name == 'denstream_our':
            clust = DenStream()
        cluster_all = [cp.deepcopy(clust) for _ in range(class_num)]
        for c in range(class_num):
            if cluster_name == 'denstream_our':
                cluster_all[c] = DenStream(lambd=0.1, eps=eps[c])

        for c in range(class_num):
            c_idx = np.where(y_train == c)
            X_train_c = X_train[c_idx, :]
            if X_train_c.ndim == 3:
                X_train_c = np.squeeze(X_train_c, 0)
            y_train_c = y_train[c_idx]

            """cluster: pre_train"""
            if cluster_name == 'denstream_our':
                cluster_all[c].partial_fit(X_train_c, y_train_c)
                if c == 0:
                    center = cluster_all[c].p_micro_cluster_centers
                    if len(center) == 0:
                        center = center_ori[c, :]
                        center = np.expand_dims(center, 0)
                    center_label = c * np.ones(np.shape(center)[0])
                else:
                    center_tmp = cluster_all[c].p_micro_cluster_centers
                    if len(center_tmp) == 0:
                        center_tmp = center_ori[c, :]
                        center_tmp = np.expand_dims(center_tmp, 0)
                    center = np.r_[center, center_tmp]
                    center_label = np.r_[center_label, c * np.ones(np.shape(center_tmp)[0])]

        center_info = np.c_[center, center_label]
        train_info = [y_train, center_info]
        if clf_name.lower() == 'bedcoe':
            classifier = BEDCOE(classes=classes, mode=0, random_state=seed, n_estimators=nb_base)

        classifier.partial_fit(X_train, train_info, classes=classes)
    else:
        raise Exception("Undefined clf_name=%s. Existing clf_names include %s"
                        % (clf_name, "bedcoe"))

    print('tree:' + str(classifier.n_estimators))
    """test then update"""
    prediction_save = np.empty(nb_all)
    S = np.zeros([class_num])
    N = np.zeros([class_num])
    for i in range(nb_pre_train):
        train_1data = X_train[i, :]
        train_1label = y_train[i]
        if train_1data.ndim == 1:  # if xx contains only 1 X_org sample
            train_1data = np.expand_dims(train_1data, 0)
            train_1label = np.expand_dims(train_1label, 0)
        train_1pre = classifier.predict(train_1data)
        prediction_save[i] = train_1pre
        _, _, S, N = pf_online(S, N, train_1label, train_1pre)

    cf = np.zeros([class_num, class_num])
    recall = np.zeros([test_len, class_num])
    gmean = np.zeros([test_len])

    for t in range(test_len):
        test_1data = X_test[t, :]
        test_1label = y_test[t]
        if test_1data.ndim == 1:  # if xx contains only 1 X_org sample
            test_1data = np.expand_dims(test_1data, 0)
            test_1label = np.expand_dims(test_1label, 0)
        # normalization
        test_1data = norm_scaler.my_transform(test_1data)

        # predict
        test_1pre = classifier.predict(test_1data)
        prediction_save[t + nb_pre_train] = test_1pre
        # evaluation
        recall[t, :], gmean[t], S, N = pf_online(S, N, test_1label, test_1pre)
        cf = confusion_online(cf, test_1label, test_1pre)

        if clf_name == 'bedcoe':
            for c in range(class_num):
                c_idx = np.where(test_1label == c)
                test_1data_c = test_1data[c_idx, :]
                if test_1data_c.ndim == 3:
                    test_1data_c = np.squeeze(test_1data_c, 0)
                test_1label_c = test_1label[c_idx]

                """cluster: update"""
                if cluster_name == 'denstream_our':
                    if len(test_1label_c) != 0:
                        cluster_all[c].partial_fit(test_1data_c, test_1label_c)
                    if c == 0:
                        center = cluster_all[c].p_micro_cluster_centers
                        step = cluster_all[c].t
                        if len(center) == 0:
                            center = center_ori[c, :]
                            center = np.expand_dims(center, 0)
                            step = 0
                            step = np.expand_dims(step, 0)
                        center_label = c * np.ones(np.shape(center)[0])
                    else:
                        center_tmp = cluster_all[c].p_micro_cluster_centers
                        step_tmp = cluster_all[c].t
                        if len(center_tmp) == 0:
                            center_tmp = center_ori[c, :]
                            center_tmp = np.expand_dims(center_tmp, 0)
                            step_tmp = 0
                            step_tmp = np.expand_dims(step_tmp, 0)
                        center = np.r_[center, center_tmp]
                        center_label = np.r_[center_label, c * np.ones(np.shape(center_tmp)[0])]

            center_info = np.c_[center, center_label]
            my_test_1info = [test_1label, center_info]
            classifier.partial_fit(test_1data, my_test_1info, classes=classes)

        else:
            classifier.partial_fit(test_1data, test_1label, classes=classes)

    return prediction_save


if __name__ == "__main__":
    clf_set = ['bedcoe']
    cluster_name = 'denstream_our'

    nb_repeat = 10

    # load real data
    # data_set = ['synthetic', 'abrupt', 'gradual', 'incremental', 'incremental-abrupt', 'incremental-reoccurring', 'elec',
    #             'luxembourg', 'noaa', 'ozone', 'ecoli', 'dermatology', 'pageblocks', 'thyroid', 'yeast', 'chess',
    #             'keystroke', 'outdoor', 'powersupply', 'rialto']
    """Due to the upload limitation, only data sets in ['synthetic'(Gaussian), 'chess', 'outdoor', 'rialto'] 
        are available in this code version."""
    data_set = ['chess']

    for data_name in data_set:
        save_path_base = './results_save/'
        save_path_data_name = save_path_base + data_name + '/'
        if not os.path.exists(save_path_data_name):
            os.mkdir(save_path_data_name)

        file_name = './data/' + data_name + '/' + 'data.mat'
        data = sio.loadmat(file_name)
        X = data['X']
        y = data['y']
        nb_pre_train = data['nb_pre_train']
        X = X.astype(np.double)
        y = y.astype(np.double)
        X = np.squeeze(X)
        y = np.squeeze(y)
        nb_pre_train = np.squeeze(nb_pre_train)

        class_num = len(np.unique(y))
        class_size = np.zeros(class_num)
        for i in range(class_num):
            class_size[i] = len(np.argwhere(y == i))

        for clf_name in clf_set:
            nb_all, _ = np.shape(X)
            nb_test = nb_all - nb_pre_train
            prediction_all_seed = experiments_repeat(X, y, clf_name=clf_name, cluster_name=cluster_name,
                                                     nb_pre_train=nb_pre_train, nb_test=nb_test, nb_repeat=nb_repeat)

            # save
            save_path_classifier = save_path_data_name + str(clf_name) + '_' + cluster_name + '/'
            if not os.path.exists(save_path_classifier):
                os.mkdir(save_path_classifier)
            file_name = save_path_classifier + 'prediction.mat'
            sio.savemat(file_name, {'nb_pre_train': nb_pre_train, 'prediction_all_seed': prediction_all_seed})
            pf_report(data_set)
