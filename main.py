# %% [markdown]
# # 风管模型第三次作业

# %% [markdown]
# ## 单个公司的 KMV 模型
#
# ### 假设
#
# - 无风险利率为 0.03
# - 违约点为 短债+0.5*长债
# - 贷款期限一年
# - 违约回收率 0
# - 贷款利率为 无风险利率+预期违约率*100
#
# ### 计算股票价格波动率

# %%
import glob

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import optimize, stats

companies = glob.glob("lib/trade/*.xlsx")
result = {}
rate = {}
for company in companies:
    df = pd.read_excel(company, index_col=0, skiprows=1).dropna()
    company = company.split("/")[-1].split("[")[0]
    result[company] = {
        "V": df["总市值"].apply(lambda x: float(x.replace(",", ""))) * 10000
    }
    rate[company] = df["涨跌幅"] / 100
rate = pd.DataFrame(rate).sort_index()
rate

# %% [markdown]
# ### 利用 garch 模型计算相关系数与协方差阵

# %%
cov = {}
var = {}
r2 = {}
rate = rate.dropna()
for company in companies:
    company = company.split("/")[-1].split("[")[0]
    r2[company] = {}
    cross = rate[company] * rate[company] * 1000
    arch = arch_model(cross)
    arch_param = arch.fit(update_freq=0)
    var[company] = (arch_param.forecast(reindex=True).variance.values[-1][0]) / 1000
    parameters = arch_param.params[1:]
    cross = rate[[i for i in rate.columns if i != company]].prod(axis=1)
    covlist = [cross[0]]*1000
    print(parameters)
    for j in range(1, len(cross)):
        covlist.append(parameters[0] + parameters[1] * cross[j] + parameters[2] * covlist[-1])
    cov[company] = covlist[-1]/1000
covar = {i: {} for i in cov}
for company in cov:
    r2[company][company] = 1
    covar[company][company] = var[company]
    others = tuple(i for i in cov if i != company)
    r2[others[0]][others[1]] = cov[company] / (
        var[others[0]] ** 0.5 * var[others[1]] ** 0.5
    )
    r2[others[1]][others[0]] = r2[others[0]][others[1]]
    covar[others[0]][others[1]] = cov[company] ** 0.5
    covar[others[1]][others[0]] = cov[company] ** 0.5
corr = pd.DataFrame(r2)
cov = pd.DataFrame(covar)
corr

# %% [markdown]
# ### 计算违约点

# %%
companies = glob.glob("lib/balance/*.xlsx")
for company in companies:
    df = (
        pd.read_excel(company, index_col=0, skipfooter=4)
        .T.rename(columns=lambda x: x.strip())
        .head(1)
    )
    company = company.split("/")[-1].split("[")[0]
    result[company]["F"] = (df["流动负债合计"] + 0.5 * df["非流动负债合计"]).mean()
{company:result[company]["F"] for company in result}

# %% [markdown]
# ### 预期违约概率 EDF

# %%
def kmv(r, sigma_e, t, equity, debt):
    def option(w):
        x, sigma_a = w
        N_d1 = stats.norm.cdf(
            (np.log(abs(x) * equity / debt) + (r + 0.5 * sigma_a**2) * t)
            / (sigma_a * np.sqrt(t))
        )
        N_d2 = stats.norm.cdf(
            (np.log(abs(x) * equity / debt) + (r - 0.5 * sigma_a**2) * t)
            / (sigma_a * np.sqrt(t))
        )
        e1 = equity - (x * equity * N_d1 - debt * N_d2 * np.exp(-r * t))
        e2 = sigma_e - sigma_a * N_d1 * x
        return [e1, e2]

    assets, sigma_a = optimize.fsolve(option, [1, 0.1], maxfev=100000000)
    DD = (assets * equity - debt) / (assets * equity * sigma_a)
    EDF = stats.norm.cdf(-DD)
    return EDF, DD


edf = {}
for company in result:
    kmvmodel = kmv(
        0.03,
        result[company]["V"].pct_change().var() * 252,
        1,
        result[company]["V"].values[0],
        result[company]["F"],
    )
    edf[company] = {"edf": kmvmodel[0], "dd": -kmvmodel[1]}
edf = pd.DataFrame(edf)
edf

# %% [markdown]
# ## 组合的违约概率
#
# 以下 code cell 分别为全部违约、两个公司违约、三个公司违约的概率
#

# %%
cov = rate.cov()
alldefault = stats.multivariate_normal.cdf(x=edf.T["dd"], cov=cov)
alldefault

# %%
twodefault = {}
for company in edf:
    others = [i for i in edf if i != company]
    others_edf = edf[others]
    others_cov = cov[others].loc[others]
    twodefault[tuple(others)] = (
        stats.multivariate_normal.cdf(x=others_edf.T["dd"], cov=others_cov.values)
        - alldefault
    )
twodefault

# %%
onedefault = {}
for company in edf:
    others = [i for i in edf if i != company]
    onedefault[company] = (
        edf.T["edf"].to_dict()[company] - alldefault - twodefault[tuple(others)]
    )
onedefault

# %% [markdown]
# ## 贷款比例与信用风险
#

# %%
partition = [
    {'比亚迪':i / 100, '三一重工':j / 100,'宁德时代':  (100 - i - j) / 100} for i in range(101) for j in range(101 - i)
]
zerodefault = 1 - alldefault - sum(twodefault.values()) - sum(onedefault.values())
return_rate = (1.03+edf.T["edf"] * 10**3).to_dict()
maxfinal = 0
maxGroup = None
for case in partition:
    final = (
        alldefault * 0
        + sum(
            [
                twodefault[i] * return_rate[j]*case[j]
                for i in twodefault
                for j in onedefault
                if j not in i
            ]
        )
        + sum(
            [
                onedefault[i] * return_rate[j]*case[j]
                for i in onedefault
                for j in onedefault
                if j != i
            ]
        )
        + zerodefault * sum([return_rate[i]*case[i] for i in onedefault])
    )
    if final > maxfinal:
        maxfinal = final
        maxGroup = case
case, maxfinal
