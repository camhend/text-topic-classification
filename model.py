import numpy as np
from abc import ABC, abstractmethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectPercentile, chi2, mutual_info_classif
from sklearn.utils.class_weight import compute_sample_weight
from sklearn import set_config
import pickle

def configure_model(config):
    tfidf_kwargs = {
        'norm': config['tfidf_norm'],
        'use_idf': config['tfidf_use_idf'],
        'sublinear_tf': config['tfidf_sublinear_tf']
    }
    if config['model'] == 'knn':
        knn_kwargs = {
            'weights': config['knn_weights'], 
            'metric': config['knn_metric'], 
            'n_jobs': config['n_jobs']
        }
        return KNN(n_neighbors=config['knn_n_neighbors'],
                   transform_name=config['transform'],
                   knn_kwargs=knn_kwargs,
                   tfidf_kwargs=tfidf_kwargs,
                   feature_percentile=config['feature_percentile'])
                # algorithm: leaving default 'brute_force' since fitting on sparse input will override the setting of this parameter, using brute force.
                    # leaf_size: same reason
    elif config['model'] =='cnb':
        return ComplementNaiveBayes(
            alpha=config['cnb_alpha'],
            norm=config['cnb_norm'],
            transform_name=config['transform'],
            tfidf_kwargs=tfidf_kwargs,
            class_weight=config['class_weight'],
            feature_percentile=config['feature_percentile'])
    elif config['model'] == 'logreg':
        logreg_kwargs = {
            'C': config['logreg_C'],
            # 'class_weight': config['class_weight'],
            'solver': config['logreg_solver'],
            'l1_ratio': config['logreg_l1_ratio'],
            'max_iter': config['max_iter'],
            'n_jobs': config['n_jobs']
        }
        return LogReg(
            config['logreg_penalty'], 
            logreg_kwargs=logreg_kwargs, 
            transform_name=config['transform'],
            tfidf_kwargs=tfidf_kwargs,
            class_weight=config['class_weight'],
            feature_percentile=config['feature_percentile'],
            root=config['root'])
    elif config['model'] == 'linearSVC':
        svc_kwargs = {
            "loss": config['svc_loss'],
            "C": config['svc_C'],
            "max_iter": config['max_iter'],
        }
        return LinearSVCModel(
            penalty=config['svc_penalty'],
            svc_kwargs=svc_kwargs,
            transform_name=config['transform'],
            tfidf_kwargs=tfidf_kwargs,
            feature_percentile=config['feature_percentile'],
            class_weight=config['class_weight'])


def configure_class_weight(class_weight, y, root=2):
    if class_weight == 'inv_sqrt':
        class_weight = compute_inv_power_weights(y, root)
    return class_weight  

def compute_inv_power_weights(y, root):
    classes, counts = np.unique(y, return_counts=True)
    weights = {cls: 1.0 / np.power(count, (1/root)) for cls, count in zip(classes, counts)}
    return weights 

def load_model(config):
    if config['model'] == 'knn':
        return KNN.load(config['model_path'])
    elif config['model'] == 'cnb':
        return ComplementNaiveBayes.load(config['model_path'])
    elif config['model'] == 'logreg':
        return LogReg.load(config['model_path'])
    elif config['model'] == 'linearSVC':
        return LinearSVCModel.load(config['model_path'])

class BaseModelClassifier(ABC):
    def __init__(self, tfidf_kwargs=None):
        self.tfidf_kwargs = tfidf_kwargs or {}
        pass

    def create_pipeline(self, model_name, model, transform_name, feature_percentile=100) -> Pipeline:
        if feature_percentile == 100:
            feature_select=None
        else:
            feature_select = SelectPercentile(score_func=chi2, percentile=feature_percentile)
        if transform_name == 'tf-idf':
            transform = TfidfTransformer(**self.tfidf_kwargs)
        elif transform_name == 'maxabs':
            transform = MaxAbsScaler()
        else:
            transform = None
        return Pipeline(steps=[ 
            ("feature select", feature_select),
            (transform_name, transform), 
            (model_name, model)
        ])

    @abstractmethod
    def train(self, X, y, model_path=None, sample_weight=None):
        """Train the model. Save if a path to save the model is given."""
        pass

    @abstractmethod
    def predict(X):
        """Predict class labels"""
        pass
    
    @abstractmethod
    def predict_proba(X):
        """Predict class probabilities"""
        pass

    def save(self, model_path):
        with open(model_path + '.pickle', 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(self, model_path):
        with open(model_path, 'rb') as file:
            instance = pickle.load(file)
        return instance

class KNN(BaseModelClassifier):
    def __init__(self, n_neighbors, transform_name='none', knn_kwargs=None, tfidf_kwargs=None, feature_percentile=100):
        super().__init__(tfidf_kwargs)
        self.knn_kwargs = knn_kwargs or {}
        self.knn = KNeighborsClassifier(n_neighbors, **knn_kwargs)
        self.clf: Pipeline = self.create_pipeline("knn", self.knn, transform_name, feature_percentile)

    def train(self, X, y, model_path=None):
        self.clf.fit(X, y)
        if model_path:
            self.save(model_path) 

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

class ComplementNaiveBayes(BaseModelClassifier):
    def __init__(self, alpha=1.0, norm=False, transform_name='none', tfidf_kwargs=None, class_weight=None, feature_percentile=100):
        super().__init__(tfidf_kwargs)
        set_config(enable_metadata_routing=False)
        self.cnb = ComplementNB(alpha=alpha, norm=norm)
        self.clf: Pipeline = self.create_pipeline("cnb", self.cnb, transform_name, feature_percentile)
        self.class_weight = class_weight

    def train(self, X, y, model_path=None):
        class_weight = configure_class_weight(self.class_weight, y, root=2)
        sample_weight = compute_sample_weight(class_weight, y)
        self.clf.fit(X, y, cnb__sample_weight=sample_weight)
        if model_path:
            self.save(model_path) 

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

class LogReg(BaseModelClassifier):
    def __init__(self, penalty, logreg_kwargs=None, transform_name='none', tfidf_kwargs=None, feature_percentile=100, class_weight=None, root=2):
        super().__init__(tfidf_kwargs)
        self.logreg = LogisticRegression(penalty, verbose=True, **logreg_kwargs)
        self.clf: Pipeline = self.create_pipeline("logreg", self.logreg, transform_name, feature_percentile)
        self.class_weight = class_weight
        self.root = root

    def train(self, X, y, model_path=None):
        class_weight = configure_class_weight(self.class_weight, y, self.root)
        self.clf.set_params(logreg__class_weight=class_weight)
        self.clf.fit(X, y)
        if model_path:
            self.save(model_path) 

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

class LinearSVCModel(BaseModelClassifier):
    def __init__(self, penalty, svc_kwargs=None, transform_name='none', tfidf_kwargs=None, feature_percentile=100, class_weight=None):
        super().__init__(tfidf_kwargs)
        self.svc = LinearSVC(penalty=penalty, verbose=True, **svc_kwargs)
        self.clf: Pipeline = self.create_pipeline("svc", self.svc, transform_name, feature_percentile)
        self.class_weight = class_weight

    def train(self, X, y, model_path=None):
        class_weight = configure_class_weight(self.class_weight, y, root=2)
        self.clf.set_params(svc__class_weight=class_weight)
        self.clf.fit(X, y)
        if model_path:
            self.save(model_path)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)