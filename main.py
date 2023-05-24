# %% [markdown]
# # 风管模型第三次作业
#
#

# %% [markdown]
# ## 单个公司的 KMV 模型
#
# ### 假设
#
# - 市场的无风险利率为3%
# - 公司的违约实施点为企业1年以下短期债务的价值加上未清偿长期债务账面价值的一半
# - 贷款期限一年
# - 当公司发生违约时，违约回收率为0，即全部贷款均不能回收。
# - 贷款利率为包括流动性溢价和信用溢价，分别用无风险利率和预期违约率$\times$100表示，即贷款利率 = 无风险利率 + 预期违约率 $\times$ 100
#
# ### 计算股票价格波动率
#
# 我们选取了三家上市公司，分别为比亚迪，三一重工，宁德时代，比亚迪为汽车厂商，三一重工为国有大型装备制造厂商，宁德时代为我国新能源电池领头企业，三家厂商都具备较好的现金流以及信用背书，因此是我们选取的贷款对象。
#
# 根据三家公司的历史股价数据，我们计算出三家厂商的股票波动率。

# %%
import glob

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import optimize, stats

companies = glob.glob("data/trade/*.xlsx")
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
#
# 假设三只股票的收益率服从GARCH (1,1)，然后算出得到标的资产组合的协方差矩阵和相关阵。

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
    covlist = [cross[0]]*100000
    print(parameters)
    for j in range(1, len(cross)):
        covlist.append(parameters[0] + parameters[1] * cross[j] + parameters[2] * covlist[-1])
    cov[company] = covlist[-1]/100000
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
companies = glob.glob("data/balance/*.xlsx")
for company in companies:
    df = (
        pd.read_excel(company, index_col=0, skipfooter=4, sheet_name="file")
        .T.rename(columns=lambda x: x.strip())
        .head(1)
    )
    company = company.split("/")[-1].split("(")[0]
    result[company]["F"] = (df["流动负债合计"] + 0.5 * df["非流动负债合计"]).mean()
{company:result[company]["F"] for company in result}

# %% [markdown]
# ### 计算预期违约概率 EDF
#
# 利用KMV模型，我们计算了三家公司各自的预期违约概率，结果如下：

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

    assets, sigma_a = optimize.fsolve(option, [1, 0.1], maxfev=1000000000)
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
# 依据<cite data-cite="任宇航2006信用风险组合管理模型中的相关性问题研究述评">任宇航(2006)</cite>提出的算法，$DD$ 服从联合多元标准正态分布，且相关系数等于资产的相关系数，因而以下 code cell 分别为全部违约、两个公司违约、三个公司违约的概率

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
# 利用以上的结果，我们以贷款给这三家公司的期望收益为标准，选择期望收益最大化的贷款分配比例。由于违约概率都很小，贷款的 $VaR$ 指导意义较弱，因而我们选择最大化期望收益。最终的结果是100%宁德时代，以获得最高的期望收益。

# %%
zerodefault = 1 - alldefault - sum(twodefault.values()) - sum(onedefault.values())
return_rate = (1.03+edf.T["edf"] * 10**3).to_dict()
def expected_return(case):
    case = {'比亚迪': case[0], '三一重工':case[1], '宁德时代': case[2]}
    final = (
        alldefault * 0
        + sum(
            [
                twodefault[i] * return_rate[j] * case[j]
                for i in twodefault
                for j in onedefault
                if j not in i
            ]
        )
        + sum(
            [
                onedefault[i] * return_rate[j] * case[j]
                for i in onedefault
                for j in onedefault
                if j != i
            ]
        )
        + zerodefault * sum([return_rate[i] * case[i] for i in onedefault])
    )
    return -final


case = optimize.minimize(
    expected_return,
    (0,0,1),
    constraints=({"type": "eq", "fun": lambda x: sum(x) - 1}),
    bounds=[(0, 1)] * 3,
).x
{'比亚迪': case[0], '三一重工':case[1], '宁德时代': case[2]}

# %% [markdown]
# \nocite{*}
# \bibliographystyle{plain}
# \bibliography{reference}


