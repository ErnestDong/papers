#%%
#!%cd /Users/dcy/Code/papers.worktrees/risk-management
import statsmodels.api as sm
import pandas as pd

st = pd.read_excel("data/实施ST.xlsx", skipfooter=2)
unst = pd.read_excel("data/撤消ST.xlsx", skipfooter=2)
unst = unst[unst["撤销日期"].dt.year < 2008]
st = st[st["实施日期"].dt.year < 2008]
# %%
st["type"] = "st"
unst["type"] = "unst"
st["date"] = st["实施日期"]
unst["date"] = unst["撤销日期"]
st["name_before"] = st["实施前简称"]
st["name_after"] = st["实施后简称"]
unst["name_before"] = unst["撤销前简称"]
unst["name_after"] = unst["撤销后简称"]
df = pd.concat([st, unst])[["代码", "date", "name_before", "name_after", "type"]]
df.sort_values("date", inplace=True)

# %%
result = []
for stock, stockdf in df.groupby("代码"):
    result.append(
        {
            "stock": stock,
            "name": stockdf["name_after"].values[-1],
            "type": stockdf["type"].values[-1],
            "date": stockdf["date"].values[-1],
        }
    )
result = pd.DataFrame(result).dropna()
# %%
result[result["name"].str.contains("ST")].to_excel("data/sts.xlsx")
# %%
