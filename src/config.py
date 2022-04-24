#%%
"""这段代码解决读取数据问题"""
import glob

import pandas as pd

# NOTE: _use_cols order matters!
_use_cols = {"code": "代码", "name": "名称", "date": "日期", "price": "收盘价(元)"}
_file_path = "./lib/*.xlsx"
expected_return = 0.05
initial_investment = 10_000_000

expected_return = expected_return/365
def prepare_data():
    """
    读取数据，生产收益率数据和股票代码：股票简称映射表
    """

    xlsx_list = glob.glob(_file_path)
    xlsx_list = [i for i in xlsx_list if "~$" not in i]
    assert xlsx_list
    result = pd.DataFrame()
    code_name_map = {}
    for xlsx in xlsx_list:
        df = pd.read_excel(xlsx, usecols=_use_cols.values())
        df.rename(columns={i: j for j, i in _use_cols.items()}, inplace=True)
        df = df[df["code"] != "数据来源：Wind"]
        df = df.dropna()
        code_name_map[df["code"].values[0]] = (
            df["name"].values[0],
            df["price"].values[-1],
        )
        df["price"] = df["price"].pct_change()
        result = pd.concat([result, df])

    result = result.pivot(index="date", columns="code", values="price")
    result = result.dropna()
    if len(result) < 500:
        raise ValueError("数据量不足 500 天")
    return code_name_map, result


if __name__ == "__main__":
    print(prepare_data())
