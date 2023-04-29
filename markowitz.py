import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import linalg


class DataLoader:
    def __init__(self, path="lib/data.xlsx"):
        data_source = pd.ExcelFile(path)

        df = pd.DataFrame()
        for sheet in data_source.sheet_names:
            df[sheet] = data_source.parse(sheet, index_col=2)["收盘价"]
        self.price = df.dropna().sort_index().tail(501)
        self.r = self.price.pct_change().dropna()

class Markowitz:

    def __init__(self, returns:pd.DataFrame):
        self.returns = returns
        self.cov = returns.cov()
        self.company = returns.columns

    def solveMinVar(self, expected_return:float):
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

        return pd.DataFrame({"weights":results[:-2]}, index=self.company)

    def calVar(self, portion:pd.DataFrame):
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
    stock_info = DataLoader()
    initial_investment = 10_000_000
    expected_return = 0.05
    data = stock_info.r
    markowitz = Markowitz(data)
    weights = markowitz.solveMinVar(expected_return)
    print(weights*initial_investment)

