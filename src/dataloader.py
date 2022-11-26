# %%
#!%cd /Users/dcy/Code/papers.worktrees/risk-management

import pandas as pd


class Dataloader:
    mappings = {
        "v_a": "资产价值",
        "v_e": "股权价值",
        "sigma_a": "资产价值波动率",
        "sigma_e": "股权价值波动率",
        "d": "债务价值",
        # "r": "隔夜拆借利率",
        "r": "ROA",
        "assets": "资产规模",
    }

    def __init__(self, variable: str, file: pd.ExcelFile):
        variable = variable.lower()
        if variable not in self.mappings:
            raise KeyError()
        df = file.parse(
            sheet_name=self.mappings[variable],
            skiprows=1,
            index_col=0,
        )
        self.label = {i: df[i].values[0] for i in df.columns}
        df = df.drop(df.index[0]).astype(float)
        df.index = pd.to_datetime(df.index)
        self.data = df


if __name__ == "__main__":
    with pd.ExcelFile("data/KMV模型已知量汇总.xlsx") as f:
        data = Dataloader("Sigma_e", f)
# %%
