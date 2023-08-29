import numpy as np
import matplotlib.pyplot as plt
from evaluate.evaluation_online import pf_online
import scipy.io as sio


def pf_report(data_set):
    clf_set = ['bedcoe']
    label_set = ['BEDCOE']
    color_set = ['red']

    ncol = 1

    fontsize = 15
    legendsize = 13
    limsize = 10

    cluster_name = 'denstream_our'
    nb_repeat = 10
    save_path_base = './results_save/'

    gmean_all = np.zeros([len(data_set), len(clf_set)])
    class_wise_acc_all = np.zeros([len(data_set), len(clf_set)])
    gmean_std_all = np.zeros([len(data_set), len(clf_set)])
    acc_std_all = np.zeros([len(data_set), len(clf_set)])

    data_id = 0
    for data_name in data_set:
        print(data_name)
        # save
        save_path_data_name = save_path_base + data_name + '/'

        # load
        file_name = './data/' + data_name + '/' + 'data.mat'
        data = sio.loadmat(file_name)
        X = data['X']
        y = data['y']
        nb_pre_train = data['nb_pre_train']
        X = np.squeeze(X)
        y = np.squeeze(y)
        nb_all = len(y)
        nb_pre_train = np.squeeze(nb_pre_train)
        class_num = len(np.unique(y))

        class_size = np.zeros(class_num)
        for i in range(class_num):
            class_size[i] = len(np.argwhere(y == i))

        clf_id = 0

        figure1 = plt.figure()
        ax1 = plt.axes()
        plt.xlim(1, nb_all)
        plt.ylim(0, 1)
        plt.ylabel('G-mean', fontsize=fontsize)
        plt.xlabel('Time step', fontsize=fontsize)
        plt.tick_params(labelsize=limsize)
        plt.grid()

        figure2 = plt.figure()
        ax2 = plt.axes()
        plt.xlim(1, nb_all)
        plt.ylim(0, 1)
        plt.ylabel('Class-wise accuracy', fontsize=fontsize)
        plt.xlabel('Time step', fontsize=fontsize)
        plt.tick_params(labelsize=limsize)
        plt.grid()

        for clf_name in clf_set:
            # load
            save_path_classifier = save_path_data_name + str(clf_name) + '_' + cluster_name + '/'
            file_name = save_path_classifier + 'prediction.mat'
            results = sio.loadmat(file_name)
            nb_pre_train = results['nb_pre_train']
            prediction_all_seed = results['prediction_all_seed']

            nb_repeat = np.shape(prediction_all_seed)[0]

            recall = np.zeros([nb_repeat, nb_all - 1, class_num])
            gmean = np.zeros([nb_repeat, nb_all - 1])
            for re in range(nb_repeat):
                S = np.zeros([class_num])
                N = np.zeros([class_num])
                for t in range(nb_all):
                    t = t - 1
                    if t >= 0:
                        test_1data = X[t + 1, :]
                        test_1label = y[t + 1]
                        test_1pre = prediction_all_seed[re, t + 1]
                        if test_1data.ndim == 1:  # if xx contains only 1 X_org sample
                            test_1data = np.expand_dims(test_1data, 0)
                            test_1label = np.expand_dims(test_1label, 0)
                            test_1pre = np.expand_dims(test_1pre, 0)
                        recall[re, t, :], gmean[re, t], S, N = pf_online(S, N, test_1label, test_1pre)

            # across seed
            gmean_avg = np.nanmean(gmean, 0)
            gmean_std = np.nanstd(gmean, 0)
            recall_avg = np.nanmean(recall, 0)
            recall_std = np.nanstd(recall, 0)

            # across step
            nb_pre_train = np.squeeze(nb_pre_train)

            gmean_avg_mean = np.nanmean(gmean_avg[nb_pre_train - 1:])  # [nb_pre_train - 1:]
            recall_avg_mean = np.nanmean(recall_avg[nb_pre_train - 1:, :], 0)  # [nb_pre_train - 1:, :]
            print('recall: ', recall_avg_mean)
            print('class-wise acc: ', np.nanmean(recall_avg_mean))
            print('gmean: ', gmean_avg_mean)

            gmean_all[data_id, clf_id] = gmean_avg_mean
            class_wise_acc_all[data_id, clf_id] = np.nanmean(recall_avg_mean)
            gmean_std_all[data_id, clf_id] = np.nanstd(np.nanmean(gmean[:, nb_pre_train - 1:], 1))
            temp = np.nanmean(recall, 2)
            acc_std_all[data_id, clf_id] = np.nanstd(np.nanmean(temp[:, nb_pre_train - 1:], 1))

            plot_x = range(nb_all - 1)
            plot_y = gmean_avg  # [nb_pre_train - 1:nb_all]
            ax1.plot(plot_x, plot_y, color=color_set[clf_id], label=label_set[clf_id])
            ax1.legend(fontsize=legendsize, ncol=ncol)

            plot_y2 = np.nanmean(recall_avg, 1)
            ax2.plot(plot_x, plot_y2, color=color_set[clf_id], label=label_set[clf_id])
            ax2.legend(fontsize=legendsize, ncol=ncol)

            clf_id = clf_id + 1

        plt.show()


if __name__ == "__main__":
    # load real data
    # data_set = ['synthetic', 'abrupt', 'gradual', 'incremental', 'incremental-abrupt', 'incremental-reoccurring', 'elec',
    #             'luxembourg', 'noaa', 'ozone', 'ecoli', 'dermatology', 'pageblocks', 'thyroid', 'yeast', 'chess',
    #             'keystroke', 'outdoor', 'powersupply', 'rialto']
    """Due to the upload limitation, only data sets in ['synthetic'(Gaussian), 'chess', 'outdoor', 'rialto'] 
        are available in this code version."""
    data_set = ['chess']
    pf_report(data_set)
