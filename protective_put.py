#%%
import datetime
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from markowitz import DataLoader, Markowitz
from pricing import euro_put

stock_info = DataLoader()
initial_investment = 10_000_000
expected_return = 0.10
markowitz = Markowitz(stock_info.r)
weights = markowitz.solveMinVar(expected_return)
print(weights*initial_investment)
option_etf = (
    pd.read_excel("./lib/300etf.xlsx",index_col="日期")[
        ["收盘价"]
    ].dropna().sort_index().rename(columns={"收盘价": "value"})
)
option_etf.index = pd.to_datetime(option_etf.index)
option_etf["pct"] = option_etf.pct_change()
# %%
# portofolio: 股票数量=权重*初始投资/股票价格
portofolio = weights.rename(columns={"weights":"stock"})*initial_investment/stock_info.price.T[[datetime.datetime(2023,1,3)]].rename(columns=lambda x:"stock")
fund = pd.concat(
    [
        tmp:=stock_info.price.dot(portofolio).rename(columns=lambda x:"value"),
        tmp.pct_change().rename(columns=lambda x: "pct"),
    ],
    axis=1,
)

# %%
months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
class Month:
    def __init__(
        self,
        df: pd.DataFrame,
        optiondf: Optional[pd.DataFrame],
        warning=0.98,
        current_month=1,
        init=initial_investment,
    ):
        df.sort_index(inplace=True)
        self.month = current_month
        # 一年内的基金历史
        self.histfund = df[
            (df.index >= datetime.datetime(2022, current_month, 1))
            & (df.index < datetime.datetime(2023, current_month, 1))
        ]
        # 当月基金净值走势
        self.funddf = df[
            (df.index >= datetime.datetime(2023, current_month, 1))
            & (df.index < datetime.datetime(2023, current_month + 1, 1))
        ]
        # 一年内的 300 ETF 走势
        if optiondf is not None:
            self.optiondf = option_etf[
                (option_etf.index >= datetime.datetime(2022, current_month, 1))
                & (option_etf.index < datetime.datetime(2023, current_month, 1))
            ]
        else:
            self.optiondf = optiondf
        # 预警线
        self.warning = warning
        # 月初初始投资
        self.init = init

    def beta(self):
        assert self.optiondf is not None
        beta = (
            self.histfund["pct"].cov(self.optiondf["pct"]) / self.optiondf["pct"].var()
        )
        return beta

    def start_month(self):
        assert self.optiondf is not None
        s0 = self.optiondf["value"].values[0]
        sigma = self.optiondf["pct"].std()
        protective_put_price = euro_put(
            s0=s0, strike=s0 * self.warning, sigma=sigma, t=months[self.month - 1]
        )
        beta = self.beta()
        # \beta \cdot Value_{etf} = Value_{portofolio} = Value_{init}-cost
        # cost = p \cdot units_{etf}=p \cdot Value_{etf}/price_{etf}
        hedge_cost = (
            self.init
            / (beta * self.optiondf["value"].values[0] + protective_put_price)
            * protective_put_price
        )
        return hedge_cost

    def end_month(self):
        if self.optiondf is not None:
            hedge_cost = self.start_month()
            tmp = (
                (self.init - hedge_cost)
                / self.funddf["value"].values[0]
                * self.funddf["value"]
            )
            final = (self.init - hedge_cost) * self.warning
            # tmp.iloc[-1] = final
            return tmp.map(lambda x: x if x > final else final)
        else:
            tmp = self.init / self.funddf["value"].values[0] * self.funddf["value"]
            return tmp
current_with_option = [pd.Series(initial_investment)]
current_without_option = [pd.Series(initial_investment)]
for month in range(1, 5):
    month1 = Month(
        fund, option_etf, current_month=month, init=current_with_option[-1].values[-1]
    )
    month2 = Month(
        fund,
        optiondf=None,
        current_month=month,
        init=current_without_option[-1].values[-1],
    )
    current_with_option.append(month1.end_month())
    current_without_option.append(month2.end_month())

with_option = pd.concat(current_with_option[1:])
without_option = pd.concat(current_without_option[1:])
benchmark = option_etf[option_etf.index >= datetime.datetime(2022, 1, 1)]
result = pd.DataFrame(
    {
        "with_option": with_option,
        "without_option": without_option,
        "300etf": benchmark["value"]
        / benchmark["value"].values[0]
        * initial_investment,
    }
).dropna()
result.index.name = None
result.plot()
plt.savefig("./result.png")

# %%
