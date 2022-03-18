import pandas as pd
import numpy as np
from src.config import *
from src.markowitz import Markowitz


def __GBM(s0, mu, sigma, T, n):
    delta_t = T / n
    simulated_price = {0: s0}
    for k in range(n):
        start_price = simulated_price[k]
        epsilon = np.random.normal()
        end_price = start_price + start_price * (
            mu * delta_t + sigma * epsilon * np.sqrt(delta_t)
        )
        end_price = max(0, end_price)
        simulated_price[k + 1] = end_price
    return simulated_price[n]


def var(data, proportion, days=10, quantile=0.99, **kwargs):
    assert "n1" in kwargs, "simulate number n1 is required"  # VaR 模拟生成次数
    assert "n2" in kwargs, "simulate number n2 is required"  # 计算收益的迭代次数
    assert "price" in kwargs, "price of each stock is required"
    stocklist = kwargs["price"]
    meanlist = data.mean()
    stdlist = data.tail(10).std()
    lost = []
    proportion = proportion.T
    initial = sum(proportion[i].values * stocklist[i] for i in stocklist)
    for _ in range(kwargs["n1"]):
        simpricecg = {}
        for i in stocklist:
            simpricecg[i] = __GBM(
                stocklist[i], meanlist[i], stdlist[i], 1, kwargs["n2"]
            )
        result = sum(proportion[i].values * simpricecg[i] for i in stocklist)
        lost.append(1 - result / initial)
    return np.quantile(lost, 1 - quantile) * np.sqrt(days) * initial_investment


if __name__ == "__main__":
    comp, data = prepare_data()
    proportion = Markowitz(data).solveMinVar(expected_return)
    print(var(data, proportion, n1=1000, n2=100, price={i: comp[i][1] for i in comp}))
