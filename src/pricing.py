import numpy as np
from functools import partial
from typing import Callable


class BinTreeModel:
    def __init__(
        self,
        stock,
        up,
        r,
        p,
        strike,
        option_type,
        level,
        path,
        hist_stock=None,
    ):
        """二叉树模型

        Parameters
        ----------
        stock : number like
            股票价格
        up : number like
            上涨幅度
        r : number like
            该时间段内的利率
        p : number like
            上涨概率
        strike : number like
            期权翘出价格
        option_type : function
            看涨/看跌期权，分别对应为 max 或 min
        level : number like
            二叉树的层树
        path : function
            是否为奇异期权。非奇异期权为 lambda x: x[-1]，
            回望期权为 max/min，亚式期权为 np.mean etc.
        hist_stock : list, optional
            历史股票的轨迹, by default None
        """
        self.left, self.right = None, None
        self.up, self.r, self.p = up, r, p
        self.strike = strike
        self.stock = stock
        self.option_type = option_type
        self.path = path
        # 考虑奇异期权情况，需要把历史股票轨迹传入
        if hist_stock is None:
            self.hist_stock = []
        else:
            self.hist_stock = list(hist_stock)
        self.hist_stock.append(stock)
        self._build(level, strike, path)
        self.optionvalue: None | float = None

    def backward(self):
        """
        后向求解
        """
        # 如果是叶子节点，则直接返回该节点处的价格
        if self.left is None:
            stockvalue = self.path(self.hist_stock)
            self.optionvalue = np.abs(self.option_type(stockvalue - self.strike, 0))
        # 如果是非叶子节点，则递归调用左右子树的后向求解
        else:
            self.left.backward()
            self.right.backward()
            optionvalue = self.left.optionvalue * self.p + self.right.optionvalue * (
                1 - self.p
            )
            self.optionvalue = np.exp(-self.r) * optionvalue
        return self.optionvalue

    def _build(self, level, strike, path):
        # 如果是一层，则不需要继续创建二叉树
        if level == 1:
            return
        # 节省空间把非叶子节点的历史股票设置为 None
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
    # 构建价格二叉树图参数
    if ngrid == None:
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


if __name__ == "__main__":
    euro_put = partial(
        model, rfrate=0.05 / 252, ngrid=15, method=min, path=lambda x: x[-1]
    )
    euro_call = partial(
        model, rfrate=0.05 / 252, ngrid=15, method=max, path=lambda x: x[-1]
    )
    lookback_call = partial(model, rfrate=0.05 / 252, ngrid=15, method=max, path=max)
    asian_call = partial(model, rfrate=0.05 / 252, ngrid=15, method=max, path=np.mean)
    print(euro_put(s0=100, strike=90, sigma=10, t=31))
