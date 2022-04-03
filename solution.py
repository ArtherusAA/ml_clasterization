import numpy as np

import sklearn
import sklearn.metrics
from sklearn.cluster import KMeans

def dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def silhouette_score(x, labels):
    '''
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray y: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    '''
    s = np.zeros(labels.size)
    d = np.zeros(labels.size)
    un_labels = np.unique(labels)

    for i, object in enumerate(x):
        label = labels[i]
        for j, t_obj in enumerate(x):
            t_lbl = labels[j]
            if t_lbl == label:
                s[i] += dist(object, t_obj)
        if np.sum(labels == label) == 1:
            s[i] = 0.0
        else:
            s[i] /= (np.sum(labels == label) - 1)
        t_min = None
        for c in un_labels:
            if c != label:
                cur_val = 0.0
                for j, obj in enumerate(x):
                    if labels[j] == c:
                        cur_val += dist(obj, object)
                cur_val /= np.sum(labels == c)
                if t_min is None:
                    t_min = cur_val
                if t_min > cur_val:
                    t_min = cur_val
        if t_min is not None:
            d[i] = t_min
    sil = np.zeros(labels.size)
    for i in range(s.size):
        if np.abs(s[i]) <= 0.00001:
            sil[i] = 0.0
        else:
            sil[i] = (d[i] - s[i]) / np.max(d[i], s[i])
    return np.average(sil)


def bcubed_score(true_labels, predicted_labels):
    '''
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    '''

    # Ваш код здесь:＼(º □ º l|l)/

    return score


class KMeansClassifier(sklearn.base.BaseEstimator):
    def __init__(self, n_clusters):
        '''
        :param int n_clusters: Число кластеров которых нужно выделить в обучающей выборке с помощью алгоритма кластеризации
        '''
        super().__init__()
        self.n_clusters = n_clusters

        # Ваш код здесь:＼(º □ º l|l)/

    def fit(self, data, labels):
        '''
            Функция обучает кластеризатор KMeans с заданым числом кластеров, а затем с помощью
        self._best_fit_classification восстанавливает разметку объектов

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов обучающей выборки
        :param np.ndarray labels: Непустой одномерный массив. Разметка обучающей выборки. Неразмеченные объекты имеют метку -1.
            Размеченные объекты могут иметь произвольную неотрицательную метку. Существует хотя бы один размеченный объект
        :return KMeansClassifier
        '''
        # Ваш код здесь:＼(º □ º l|l)/

        return self

    def predict(self, data):
        '''
        Функция выполняет предсказание меток класса для объектов, поданных на вход. Предсказание происходит в два этапа
            1. Определение меток кластеров для новых объектов
            2. Преобразование меток кластеров в метки классов с помощью выученного преобразования

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов
        :return np.ndarray: Предсказанные метки класса
        '''
        # Ваш код здесь:＼(º □ º l|l)/

        return predictions

    @staticmethod
    def _best_fit_classification(cluster_labels, true_labels):
        '''
        :param np.ndarray cluster_labels: Непустой одномерный массив. Предсказанные метки кластеров.
            Содержит элементы в диапазоне [0, ..., n_clusters - 1]
        :param np.ndarray true_labels: Непустой одномерный массив. Частичная разметка выборки.
            Неразмеченные объекты имеют метку -1. Размеченные объекты могут иметь произвольную неотрицательную метку.
            Существует хотя бы один размеченный объект
        :return
            np.ndarray mapping: Соответствие между номерами кластеров и номерами классов в выборке,
                то есть mapping[idx] -- номер класса для кластера idx
            np.ndarray predicted_labels: Предсказанные в соответствии с mapping метки объектов

            Соответствие между номером кластера и меткой класса определяется как номер класса с максимальным числом объектов
        внутри этого кластера.
            * Если есть несколько классов с числом объектов, равным максимальному, то выбирается метка с наименьшим номером.
            * Если кластер не содержит размеченных объектов, то выбирается номер класса с максимальным числом элементов в выборке.
            * Если же и таких классов несколько, то также выбирается класс с наименьшим номером
        '''
        # Ваш код здесь:＼(º □ º l|l)/

        return mapping, predicted_labels
