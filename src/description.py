#%%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dataloader import Dataloader

project_root = Path(__file__).parent.parent

sns.set(
    rc={
        "font.family": "SimHei",
        # "figure.figsize": (6, 3),
        "axes.unicode_minus": False,
    },
    style="white",
)

excel = pd.ExcelFile(project_root / "data/KMV模型已知量汇总.xlsx")
st = pd.read_excel(
    project_root / "data/sts.xlsx", usecols=["stock", "name", "date", "enddate"]
)
st_stocks = st.set_index("stock").to_dict()


# %%
final = False
for item in Dataloader.mappings:
    data, label = (temp := Dataloader(item, excel)).data, temp.label
    result = pd.DataFrame()
    if final is not False:
        data.index = final
    else:
        final = data.index
    for column in data:

        tmp = data[column]
        tmp.name = item
        tmp = tmp.reset_index()
        tmp["stock"] = column
        result = pd.concat([result, tmp], axis=0)
    result["ST"] = result.apply(
        lambda x: "ST" if label[x["stock"]] in st_stocks["name"] else "非ST", axis=1
    )
    result["日期"] = result["日期"].apply(lambda x: x.strftime("%y-%m"))
    plt.cla()
    sns.boxplot(data=result, x="日期", y=item, hue="ST")
    plt.savefig(project_root / f"img/{item}.png")


# %%
excel.close()
df = pd.read_excel(project_root / "data/发行人首次债券违约.xlsx", skiprows=1)

# %%
df["违约时间"] = df["首次违约日期"].dt.year
sns.histplot(data=df, x="违约时间", hue="是否上市公司", multiple="dodge")
plt.savefig(project_root / "img/发行人首次债券违约.png")
# %%
