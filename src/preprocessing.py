import numpy as np


def signed_log1p(X):
    """sign(x) * log(1 + |x|): shrinks huge values but keeps the sign."""
    return np.sign(X) * np.log1p(np.abs(X))
