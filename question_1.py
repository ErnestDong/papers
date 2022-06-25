#%%
import numpy as np
import pandas as pd


class BTCDepth:
    def __init__(
        self, filepath: str = "./data/btc_usdt-2022060621.log", debug: bool = False
    ) -> None:
        """初始化。由于文件较小，我采用了全部读入内存的方式方便多次查询。

        Args:
            filepath (str, optional): logfile 的路径. Defaults to "./data/btc_usdt-2022060621.log".
            debug (bool, optional): 输出人类可以看的时间. Defaults False.
        """
        # 解析文件放到 dataframe 中
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.readlines()
        content = [eval(i) for i in content]  # pylint: disable=eval-used
        df = pd.DataFrame(content)
        # 把时间戳换算成人类可识别的时间
        self.debug = debug
        if self.debug:
            df["tp"] = pd.to_datetime(df["tp"], unit="ms")
            # 看上去是按时间的，可以不要
            df.sort_values(by="tp", inplace=True)
        # init
        self.df = df
        self._timestamp = df[~df["_"].isna()]["tp"].unique()

    def parse(self, until: np.datetime64 | int) -> pd.DataFrame:
        """解析 until 之前的挂单数据

        Args:
            until (np.datetime64|int): 截止某时刻的挂单数据

        Returns:
            pd.DataFrame: 挂单结果，负数为 sell 正数为 buy
        """
        # 数据中的确有足够的数据，即 until 在最小的全量数据和最大的时间戳之间
        assert self._timestamp.min() <= until <= self.df["tp"].max()
        # 找到 until 之前最晚的全量数据
        prev_all = self._timestamp.searchsorted(until, side="right")
        df = self.df[self.df["tp"].between(self._timestamp[prev_all - 1], until)]
        df = df.copy()
        # buy 为 1，sell 为 -1
        df["buy"] = df["s"] * df["t"].apply(lambda x: 1 if x == "buy" else -1)
        # result 为结果，index 为所有的价格
        result = pd.Series(0, index=df["p"].unique())
        for _, tp_df in df.groupby("tp"):
            # 思路是 nan 加任何数都是 nan，
            # 扩展 index 之后，把没有收到增量数据的位置补 0，
            # 把收到的 0 替换成 nan，
            # 就能实现收到 0 在 result 中清零，过程见以下步骤

            # 4. 上一步被清零成 nan 的替换成 0
            result = result.replace(np.nan, 0)
            # 1. 收到 0 替换成 inf，没有收到的数据增量为 nan，否则是增量数据
            # nan 运算优先级大于 inf
            tp_df = tp_df.set_index("p")["buy"].replace(0, np.inf) + pd.Series(
                0, index=df["p"].unique()
            )
            # 2. 替换回来，没有收到的数据增量为 0，收到 0 替换成 nan
            # tp_df = tp_df.replace({np.nan: 0, np.inf: np.nan})
            tp_df = tp_df.replace(np.nan, 0).replace(np.inf, np.nan)
            # 3. 增量数据和 result 相加，收到 0 的结果是 nan 得以清零
            result = result + tp_df
        if self.debug:
            return result.replace(0, np.nan).dropna().sort_index(ascending=False)
        return result.replace(0, np.nan).dropna()

    def dump(self, until: int):
        result = (
            self.parse(until)
            .reset_index()
            .rename(columns={0: "size", "index": "price"})
        )
        buy = result[result["size"] > 0]
        sell = result[result["size"] < 0].abs()
        return {
            "time": "until",
            "buy": buy.to_dict("records"),
            "sell": sell.to_dict("records"),
        }


# %%
if __name__ == "__main__":
    depth = BTCDepth()
    # depth = BTCDepth(debug=True)
    # timestamp = np.datetime64("2022-06-06T21:11:34")
    # print(depth.parse(until=timestamp))
    print(depth.dump(1654552771134))

# %%
