#%%
# 应该是流动资产<流动负债
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

project_path = Path(__file__).parent.parent

df = pd.read_excel(project_path / "data/sts.xlsx")
st = df[["总资产", "流动负债", "非流动负债"]]
ols = sm.OLS(st["总资产"], st[["流动负债", "非流动负债"]])
res = ols.fit()
summary = res.summary()
with open("results/table1.tex", "w", encoding="utf-8") as f:
    f.write(summary.as_latex())
print(summary)
# %%
