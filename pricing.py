from functools import partial
from typing import Callable, Optional

import numpy as np


class BinTreeModel:
    """`BinTreeModel` 类定义了期权定价的二叉树模型。

    `backward` 方法通过递归地从树的底部移动到顶部来计算树的每个节点的期权价值。

    `_build` 方法通过创建左右子节点递归地构建树，直到达到所需的级别数。
    """
    def __init__(
        self,
        stock:float,
        up: float,
        r: float,
        p: float,
        strike: float,
        option_type:Callable,
        level: int,
        path: Callable,
        hist_stock=None,
    ):
        """
        :param stock: 初始股票价格
        :param up: 上涨幅度
        :param r: 无风险利率
        :param p: 上涨概率
        :param strike: 期权敲出价格
        :param option_type: 期权类型，看涨、看跌期权，分别对应为 `max` 、`min`
        :param level: 树的层数
        :param path: 期权的路径依赖函数，非奇异期权为 lambda x: x[-1]，回望期权为 max/min，亚式期权为 np.mean.
        :param hist_stock: 历史股票的轨迹, by default None
        """
        # 二叉树模型的参数
        self.left, self.right = None, None
        self.up, self.r, self.p = up, r, p
        self.strike = strike
        self.stock = stock
        self.option_type = option_type
        self.path = path
        if hist_stock is None:
            self.hist_stock = []
        else:
            self.hist_stock = list(hist_stock)
        self.hist_stock.append(stock)
        # 递归构建二叉树
        self._build(level, strike, path)
        self.optionvalue: Optional[float] = None

    def backward(self):
        """`backward` 方法通过递归地从树的底部移动到顶部来计算树的每个节点的期权价值。"""
        # 如果是叶子节点，计算期权价值
        if self.left is None or self.right is None:
            stockvalue = self.path(self.hist_stock)
            self.optionvalue = np.abs(self.option_type(stockvalue - self.strike, 0))
        # 如果不是叶子节点，递归地计算左右子树的期权价值
        else:
            self.left.backward()
            self.right.backward()
            optionvalue = self.left.optionvalue * self.p + self.right.optionvalue * (
                1 - self.p
            )
            self.optionvalue = np.exp(-self.r) * optionvalue
        return self.optionvalue

    def _build(self, level, strike, path):
        # 如果层数为 1，停止递归
        if level == 1:
            return
        # 递归构建左右子树
        hist_stock = self.hist_stock
        self.hist_stock = None
        self.left = BinTreeModel(
            stock=self.stock * self.up,
            up=self.up,
            r=self.r,
            p=self.p,
            strike=strike,
            path=path,
            hist_stock=hist_stock,
            option_type=self.option_type,
            level=level - 1,
        )
        self.right = BinTreeModel(
            stock=self.stock / self.up,
            up=self.up,
            r=self.r,
            p=self.p,
            strike=strike,
            path=path,
            hist_stock=hist_stock,
            option_type=self.option_type,
            level=level - 1,
        )


def model(
    s0,
    strike,
    sigma,
    t,
    rfrate,
    ngrid=None,
    method=max,
    path: Callable = lambda x: x[-1],
):
    """
    该函数使用二叉树模型对期权定价进行建模。

    :param s0: 初始股票价格。
    :param strike: 被定价的期权的行使价。
    :param sigma: 标的资产的波动率
    :param t: t 代表期权到期的时间
    :param rfrate: 无风险利率
    :param ngrid: 二叉树模型中使用的时间步数
    :param method: 如何在二叉树的每个节点计算期权值。看涨、看跌期权，分别对应为 `max` 、`min`
    :param path:  确定是否是奇异期权
    """
    if ngrid is None:
        ngrid = t + 1
    deltaT = t / (ngrid - 1)
    u = np.exp(sigma * np.sqrt(deltaT))
    d = 1 / u
    p = (np.exp(rfrate * deltaT) - d) / (u - d)
    if path is max:
        assert method is max
    elif path is min:
        assert method is min
    bintree = BinTreeModel(
        stock=s0,
        up=u,
        r=rfrate * deltaT,
        p=p,
        strike=strike,
        option_type=method,
        level=ngrid,
        path=path,
        hist_stock=None,
    )
    return bintree.backward()

euro_put = partial(
    model, rfrate=0.05 / 252, ngrid=15, method=min, path=lambda x: x[-1]
)
euro_call = partial(
    model, rfrate=0.05 / 252, ngrid=15, method=max, path=lambda x: x[-1]
)
lookback_call = partial(model, rfrate=0.05 / 252, ngrid=15, method=max, path=max)
asian_call = partial(model, rfrate=0.05 / 252, ngrid=15, method=max, path=np.mean)

if __name__ == "__main__":
    print(euro_put(s0=100, strike=90, sigma=10, t=31))
