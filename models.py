import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


class FixedProbaClassifier(BaseEstimator, ClassifierMixin):
    """
    scikit-learn compatible classifier that:
     - predict_proba(X) returns [[1-p1, p1], ...] for every row
     - predict(X) returns random 0/1 sampled with P(1) = p1 per row
     
    Parameters
    p1 : float, default=0.2
        Probability of class 1 (must be in [0,1]).
    random_state : int, RandomState instance or None, default=None
        For reproducible random draws in predict().
    """
    def __init__(self, p1, random_state=None):
        self.p1 = float(p1)
        self.random_state = random_state

    
    def fit(self, X, y=None, sample_weight=None):
        """
        Fit method required by scikit-learn API.
        sample_weight is accepted for compatibility but not used.
        """
        # Validating p1
        if not (0.0 <= self.p1 <= 1.0):
            raise ValueError("p1 must be between 0 and 1.")

        # Storing classes_ attribute (required by sklearn)
        self.classes_ = np.array([0, 1])

        return self

    def predict_proba(self, X):
        n = len(X)
        p1 = float(self.p1)
        p0 = 1.0 - p1
        # two columns: P(0), P(1)
        return np.column_stack([np.full(n, p0), np.full(n, p1)])

    def predict(self, X):
        """Random draw per-sample: 1 with probability p1, else 0."""
        n = len(X)
        rng = check_random_state(self.random_state)
        # returns array of 0/1 integers
        return rng.binomial(1, self.p1, size=n).astype(int)
    

from sklearn.ensemble import HistGradientBoostingClassifier

MODELS = {
    'Baseline_always_zero': DummyClassifier(strategy="constant", constant=0),
    'Logistic Regression': LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, n_jobs=-1),
    'MLP': MLPClassifier(hidden_layer_sizes=(128,), random_state=42)
}