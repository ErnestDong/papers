"""
利用 markowitz 有效前沿计算最优持仓分布
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import linalg

from src.config import prepare_data


class Markowitz:
    def __init__(self, returns):
        self.returns = returns
        self.cov = returns.cov()
        self.company = returns.columns

    def solveMinVar(self, expected_return):
        cov = np.array(self.cov)
        mean = np.array(self.returns.mean())
        row1 = np.append(
            np.append(cov.swapaxes(0, 1), [mean], axis=0), [np.ones(len(mean))], axis=0
        ).swapaxes(0, 1)
        row2 = list(np.ones(len(mean)))
        row2.extend([0, 0])
        row3 = list(mean)
        row3.extend([0, 0])
        A = np.append(row1, np.array([row2, row3]), axis=0)
        b = np.append(np.zeros(len(mean)), [1, expected_return], axis=0)
        results = linalg.solve(A, b)

        return pd.DataFrame(results[:-2], index=self.company)

    def calVar(self, portion):
        portion = portion.values
        return np.dot(np.dot(portion.T, self.cov), portion)[0]

    def plotFrontier(self):
        expected_return = [x / 100000 for x in range(-500, 1000)]
        variance = list(
            map(
                lambda x: self.calVar(self.solveMinVar(x)),
                expected_return,
            )
        )
        sns.set()
        plt.plot(variance, expected_return)
        plt.xlabel("Variance")
        plt.ylabel("Expected Return")
        plt.title("Efficient Frontier")
        plt.show()


if __name__ == "__main__":
    companies, data_pct, _ = prepare_data()
    markowitz = Markowitz(data_pct)
    # markowitz.plotFrontier()
    print(markowitz.solveMinVar(0.05 / 365).to_dict())
    # print(data)
