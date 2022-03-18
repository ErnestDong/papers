import pandas as pd
import numpy as np
from src.config import *
from src.markowitz import Markowitz
import scipy


# portofolio = data * proportion.T.values[0]
def var(
    data,
    proportion,
    days=10,
    quantile=0.99,
    **kwargs,
):
    cov_matrix = data.cov()
    weights = proportion.T.values[0]
    std = np.sqrt(weights.dot(cov_matrix).dot(weights.T))
    mean = weights.dot(data.mean())
    return (
        scipy.stats.norm.ppf(1 - quantile, mean, std)
        * np.sqrt(days)
        * initial_investment
    )


if __name__ == "__main__":
    _, data = prepare_data()
    np.testing.assert_almost_equal(
        scipy.stats.normaltest(data.values)[1],
        np.zeros(len(data.columns)),
        err_msg="not normal distribution",
    )
    proportion = Markowitz(data).solveMinVar(expected_return)
    print(var(data, proportion))
    print(var(data, proportion, days=60))
