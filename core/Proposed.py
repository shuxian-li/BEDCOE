import copy as cp

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.utils.utils import *
from skmultiflow.utils import check_random_state


class BEDCOE(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    def __init__(self, base_estimator=HoeffdingTreeClassifier(), n_estimators=10, theta_imb=0.9, mode=0, classes=None,
                 random_state=None):
        super().__init__()
        # default values
        self.ensemble = None
        self.actual_n_estimators = None
        self._random_state = None
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.__configure()

        self.classes = None
        self.class_num = len(classes)
        self.rho = np.ones(self.class_num)/self.class_num
        self.theta_imb = theta_imb  # decay
        self.mode = mode  # 0 for over-sampling and others for under-sampling

    def __configure(self):
        if hasattr(self.base_estimator, "reset"):
            self.base_estimator.reset()
        self.actual_n_estimators = self.n_estimators
        self.ensemble = [cp.deepcopy(self.base_estimator) for _ in range(self.actual_n_estimators)]
        self._random_state = check_random_state(self.random_state)

    def reset(self):
        self.__configure()
        return self

    def partial_fit(self, X, info, classes=None, sample_weight=None):
        y = info[0]
        center_info = info[1]
        if self.classes is None:
            if classes is None:
                raise ValueError("The first partial_fit call should pass all the classes.")
            else:
                self.classes = classes

        if self.classes is not None and classes is not None:
            if set(self.classes) == set(classes):
                pass
            else:
                raise ValueError("The classes passed to the partial_fit function differ from those passed earlier.")

        self.__adjust_ensemble_size()

        r, _ = get_dimensions(X)
        for j in range(r):
            for c_idx in range(self.class_num):
                self.rho[c_idx] = self.theta_imb * self.rho[c_idx] + (1 - self.theta_imb) * (
                    1 if y[j] == c_idx else 0)

            rho_min = np.min(self.rho)
            rho_max = np.max(self.rho)
            if self.mode == 0:
                lambda_poisson = rho_max / self.rho  # over-sampling
            else:
                lambda_poisson = rho_min / self.rho  # under-sampling

            for i in range(self.actual_n_estimators):  # all base learners
                """plus"""
                proba = self.ensemble[i].predict_proba([X[j]])
                if np.shape(proba)[1] == self.class_num:
                    prob_tmp = proba[0, int(y[j])]
                    proba[0, int(y[j])] = -1
                    Weight = np.exp(np.max(proba) - prob_tmp + 1)
                else:
                    Weight = 1

                a = (i + 1) / self.actual_n_estimators
                k = int(np.round(Weight * self._random_state.poisson(lambda_poisson[int(y[j])])))  # core
                if k > 0:
                    self.ensemble[i].partial_fit([X[j]], [y[j]], classes, sample_weight)
                    k = k - 1

                '''generate k - 1 synthetic samples'''
                idx = np.where(center_info[:, -1] == y[j])
                center_1class = center_info[idx, 0:-1]
                center_1class = np.squeeze(center_1class)
                if center_1class.ndim == 1:
                    center_1class = np.expand_dims(center_1class, 0)
                nb_center_1class = np.shape(center_1class)[0]

                point_1class = np.r_[[X[j]], center_1class]
                nb_point_1class = nb_center_1class + 1
                distance = np.linalg.norm(X[j] - point_1class, axis=1) + 1e-16
                sim = (1 / distance[1:]) ** 1 / np.sum((1 / distance[1:]) ** 1)
                sim = np.r_[1, sim]
                sim[0] = sim[0] * a
                sim[1:] = sim[1:] * (1 - a)
                if sim.ndim == 0:
                    sim = np.expand_dims(sim, 0)
                if k > 0:
                    for b in range(k):
                        for c in range(nb_point_1class):
                            point_tmp = point_1class[c, :]
                            rnd = self._random_state.uniform(0, 1)
                            if c == 0:
                                syn_sample = sim[c] * (X[j] + rnd * (point_tmp - X[j]))
                            else:
                                syn_sample = syn_sample + sim[c] * (X[j] + rnd * (point_tmp - X[j]))

                        """plus"""
                        proba = self.ensemble[i].predict_proba([syn_sample])
                        if np.shape(proba)[1] == self.class_num:
                            prob_tmp = proba[0, int(y[j])]
                            proba[0, int(y[j])] = -1
                            weight = np.exp(np.max(proba) - prob_tmp + 1)
                        else:
                            weight = 1

                        syn_k = int(np.round(weight * self._random_state.poisson(1)))
                        for syn_b in range(syn_k):
                            self.ensemble[i].partial_fit([syn_sample], [y[j]], classes, sample_weight)

        return self

    def __adjust_ensemble_size(self):
        if len(self.classes) != len(self.ensemble):
            if len(self.classes) > len(self.ensemble):
                for i in range(len(self.ensemble), len(self.classes)):
                    self.ensemble.append(cp.deepcopy(self.base_estimator))
                    self.actual_n_estimators += 1

    def predict(self, X):
        r, c = get_dimensions(X)
        proba = self.predict_proba(X)
        predictions = []
        if proba is None:
            return None
        for i in range(r):
            predictions.append(np.argmax(proba[i]))
        return np.asarray(predictions)

    def predict_proba(self, X):
        proba = []
        r, c = get_dimensions(X)
        try:
            for i in range(self.actual_n_estimators):
                partial_proba = self.ensemble[i].predict_proba(X)
                if len(partial_proba[0]) > max(self.classes) + 1:
                    raise ValueError("The number of classes in the base learner is larger than in the ensemble.")

                if len(proba) < 1:
                    for n in range(r):
                        proba.append([0.0 for _ in partial_proba[n]])

                for n in range(r):
                    for l in range(len(partial_proba[n])):
                        try:
                            proba[n][l] += partial_proba[n][l]
                        except IndexError:
                            proba[n].append(partial_proba[n][l])
        except ValueError:
            return np.zeros((r, 1))
        except TypeError:
            return np.zeros((r, 1))

        # normalizing probabilities
        sum_proba = []
        for l in range(r):
            sum_proba.append(np.sum(proba[l]))
        aux = []
        for i in range(len(proba)):
            if sum_proba[i] > 0.:
                aux.append([x / sum_proba[i] for x in proba[i]])
            else:
                aux.append(proba[i])
        return np.asarray(aux)