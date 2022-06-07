# %% [markdown]
# # 风管模型第四次作业
#
# ## 第一题
#
# ### LN 模型


# %%
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

df = pd.read_excel("./lib/沪深300行情统计.xlsx", index_col=0).sort_index()
df = df.apply(
    lambda x: x.str.replace(",", "").astype(float) if x.dtype == "object" else x
)
df.index = pd.to_datetime(df.index)
result = {}
df["weeknum"] = df.index.map(lambda x: str(x.year) + "-" + str(x.isocalendar().week))
df = df.drop_duplicates(subset="weeknum")
# log = df["收盘价"].apply(np.log)
log = (df["收盘价"] / df["收盘价"].shift(1)).dropna().apply(np.log)
mu = log.mean()
sigma = log.std()
likelyhood_ln = log.apply(
    lambda x: -np.log(sigma) - np.log(2 * np.pi) / 2 - (x - mu) ** 2 / (2 * sigma**2)
).sum()
k, logl = 2, likelyhood_ln
result["LN"] = {
    "log(L)": logl,
    "AIC": logl - k,
    "SBC": logl - k * np.log(log.shape[0]) / 2,
}
pd.DataFrame(result).T

# %% [markdown]
# ### AR(1) 模型

# %%
model = ARIMA(log.values, order=(1, 0, 0))
AR1 = model.fit()
k, logl = 3, AR1.llf
result["AR(1)"] = {
    "log(L)": logl,
    "AIC": logl - k,
    "SBC": logl - k * np.log(log.shape[0]) / 2,
}
print(AR1.summary())
pd.DataFrame(result).T

# %% [markdown]
# ### ARCH 模型

# %%
arch = arch_model(log, p=1, q=0, rescale=False).fit(update_freq=0)
k, logl = 3, arch.loglikelihood
print(arch.summary())
result["ARCH"] = {
    "log(L)": logl,
    "AIC": logl - k,
    "SBC": logl - k * np.log(log.shape[0]) / 2,
}
pd.DataFrame(result).T

# %% [markdown]
# ### GARCH 模型

# %%
garch = arch_model(log, p=1, q=1, rescale=False).fit(update_freq=0)
k, logl = 4, garch.loglikelihood
print(garch.summary())
result["GARCH"] = {
    "log(L)": logl,
    "AIC": logl - k,
    "SBC": logl - k * np.log(log.shape[0]) / 2,
}
pd.DataFrame(result).T

# %% [markdown]
# ### RSLN-2
#
# [reference](https://www.statsmodels.org/dev/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html)

# %%
rsln = MarkovRegression(log.values, 2, switching_variance=True).fit()  # 异方差
k, logl = 6, rsln.llf
print(rsln.summary())
result["RSLN-2"] = {
    "log(L)": logl,
    "AIC": logl - k,
    "SBC": logl - k * np.log(log.shape[0]) / 2,
    "LRT": None,
}
pd.DataFrame(result).T

# %%
for model in result:
    result[model]["LRT"] = 2 * (result[model]["log(L)"] - result[model]["AIC"])

# %%
