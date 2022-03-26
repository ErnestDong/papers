import numpy as np
import pandas as pd

from src.config import prepare_data, initial_investment, expected_return
from src.markowitz import Markowitz


def var(
    data,
    proportion,
    days=10,
    quantile=0.99,
    **kwargs,
):

    _revenue = data.dot(proportion).sort_values(by=0)
    var_1 = _revenue.quantile(1 - quantile).values[0]
    return np.sqrt(days) * var_1 * initial_investment


if __name__ == "__main__":
    _, data = prepare_data()
    proportion = Markowitz(data).solveMinVar(expected_return)
    print(var(data, proportion))
    print(var(data, proportion, days=60))
