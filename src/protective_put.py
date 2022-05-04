#%%
# %cd ..
#%%
from src.config import expected_return, initial_investment, prepare_data
from src.markowitz import Markowitz
from src.pricing import model
import pandas as pd
from functools import partial
import numpy as np
import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
logging.basicConfig(level=logging.INFO)
companies, data_pct, data = prepare_data()
portofolio = (
    Markowitz(data_pct)
    .solveMinVar(expected_return / 252)
    .rename(columns={0: "portofolio"})
)
option_etf = (
    pd.read_excel("./lib/options/沪深300行情统计.xlsx", index_col=0)[["收盘价"]]
    .sort_index()
    .rename(columns={"收盘价": "value"})
)
option_etf.index = pd.to_datetime(option_etf.index)
option_etf["value"] = option_etf["value"].apply(lambda x: float(x.replace(",", "")))
option_etf["pct"] = option_etf.pct_change()
fund = pd.concat(
    [
        data.dot(portofolio).rename(columns={"portofolio": "value"}),
        data_pct.dot(portofolio).rename(columns={"portofolio": "pct"}),
    ],
    axis=1,
)
# %%
euro_put = partial(
    model, rfrate=expected_return / 252, ngrid=15, method=min, path=lambda x: x[-1]
)
months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


class Month:
    def __init__(
        self,
        df: pd.DataFrame,
        optiondf: pd.DataFrame | None,
        warning=0.98,
        month=1,
        init=initial_investment,
    ):
        df = df.sort_index()
        self.month = month
        self.histfund = df[
            (df.index >= datetime.datetime(2021, month, 1))
            & (df.index < datetime.datetime(2022, month, 1))
        ]
        self.funddf = df[
            (df.index >= datetime.datetime(2022, month, 1))
            & (df.index < datetime.datetime(2022, month + 1, 1))
        ]
        if optiondf is not None:
            self.optiondf = option_etf[
                (option_etf.index >= datetime.datetime(2021, month, 1))
                & (option_etf.index < datetime.datetime(2022, month, 1))
            ]
        else:
            self.optiondf = optiondf
        self.warning = warning
        self.init = init

    def beta(self):
        beta = (
            self.histfund["pct"].cov(self.optiondf["pct"]) / self.optiondf["pct"].var()
        )
        return beta

    def start_month(self):
        s0 = self.optiondf["value"].values[0]
        sigma = self.optiondf["pct"].std()
        protective_put_price = euro_put(
            s0=s0, strike=s0 * self.warning, sigma=sigma, t=months[self.month - 1]
        )
        beta = self.beta()
        # \beta \cdot Value_{etf} = Value_{portofolio} = Value_{init}-cost
        # cost = p \cdot units_{etf}=p \cdot Value_{etf}/price_{etf}
        # so cost is
        hedge_cost = (
            self.init
            / (beta * self.optiondf["value"].values[0] + protective_put_price)
            * protective_put_price
        )
        logging.info("month: %s hedged" % self.month)
        return hedge_cost

    def end_month(self):
        if self.optiondf is not None:
            logging.info("month: %s start" % self.month)
            hedge_cost = self.start_month()
            tmp = (
                (self.init - hedge_cost)
                / self.funddf["value"].values[0]
                * self.funddf["value"]
            )
            final = max(tmp.iloc[-1], (self.init - hedge_cost) * self.warning)
            tmp.iloc[-1] = final
            return tmp.map(lambda x: x if x > final else final)
            # return tmp
        else:
            logging.info("month: %s start without hedge" % self.month)
            tmp = self.init / self.funddf["value"].values[0] * self.funddf["value"]
            return tmp


current_with_option = [pd.Series(initial_investment)]
current_without_option = [pd.Series(initial_investment)]
for month in range(1, 5):
    month1 = Month(
        fund, option_etf, month=month, init=current_with_option[-1].values[-1]
    )
    month2 = Month(
        fund, optiondf=None, month=month, init=current_without_option[-1].values[-1]
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
)
result.plot()
plt.savefig("./lib/result/result.png")
# %%
# %%
