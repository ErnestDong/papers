#%%
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from dataloader import Dataloader

project_root = Path(__file__).parent.parent
sns.set(
    rc={"font.family": "SimHei", "figure.figsize": (6, 3), "axes.unicode_minus": False},
    style="white",
)


#!%cd /Users/dcy/Code/papers.worktrees/risk-management
class KMV:
    """利用KMV模型计算违约距离"""

    def __init__(
        self, filename: str = Path(__file__).parent.parent / "data/KMV模型已知量汇总.xlsx"
    ) -> None:
        with pd.ExcelFile(filename) as f:
            self.sigma_e = Dataloader("sigma_e", f).data
            self.sigma_a = Dataloader("sigma_a", f).data
            self.v_e = Dataloader("v_e", f).data
            self.v_a = Dataloader("v_a", f).data
            self.r = Dataloader("r", f).data / 100
            self.d, self.label = (tmp := Dataloader("d", f)).data, tmp.label

    def distance_to_default(self) -> pd.DataFrame:
        """计算违约距离

        Returns:
            pd.DataFrame: 公司x日期的违约距离
        """
        correct = self._filter()
        sigma_a = self.sigma_a[correct]
        v_a = self.v_a[correct]
        d = self.d[correct]
        r = self.r[correct]
        t = 252
        return (np.log(v_a / d) + (r - 0.5 * sigma_a**2 * t)) / (sigma_a * t**0.5)

    def _filter(self):
        """删除股权波动率为0的股票

        Returns:
            list[str]: 股权价值波动率为正的股票
        """
        correct_stock = [i for i in self.sigma_e.columns if self.sigma_e[i].prod() != 0]
        return correct_stock


if __name__ == "__main__":
    model = KMV()
    dd = model.distance_to_default()
    res = dd.stack().reset_index()
    res.columns = ["Date", "Stock", "DD"]
    st = (
        pd.read_excel(
            project_root / "data/sts.xlsx", usecols=["stock", "name", "date", "enddate"]
        )
        .set_index("stock")
        .to_dict()
    )
    res["ST"] = res.apply(
        lambda x: "ST" if model.label[x["Stock"]] in st["name"] else "非ST",
        axis=1,
    )
    res["Date"] = res["Date"].dt.strftime("%y-%m")
    fig = sns.scatterplot(data=res, x="Date", y="DD", hue="ST").get_figure()
    fig.savefig(project_root / "img/dd.png")
# %%
