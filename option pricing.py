import math
import numpy as np

# 欧式看涨定价,步数是ngrid-1
def myshout(s0,strike,sigma,t,rfrate,ngrid):
    # 构建价格二叉树图参数
    deltaT = t / (ngrid - 1)
    a = math.exp(rfrate * deltaT)
    u = math.exp(sigma * math.pow(deltaT, 0.5))
    d = 1 / u
    p = (a - d) / (u - d)
    q = 1 - p
    s = np.zeros((ngrid, ngrid))
    opt = np.zeros((ngrid, ngrid))
# 构建价格矩阵,这里的i是行数，j是列数，j代表ud的指数之和，i代表d的指数
    for j in range(ngrid):
        for i in range(j+1):
            s[i][j] = s0 * math.pow(u, j-i) * math.pow(d, i)
    for i in range(ngrid):
        opt[i][-1] = max(s[i][-1]-strike, 0)
    for j in range(ngrid-2, -1, -1):
        for i in range(j, -1, -1):
            opt[i][j] = math.exp(-rfrate * deltaT) * (p * opt[i][j+1] + q * opt[i+1][j+1])
    return opt[0][0]

# 呼叫期权定价
def shout(s0,strike,sigma,t,rfrate,ngrid):
    # 构建价格二叉树图参数
    deltaT = t / (ngrid - 1)
    a = math.exp(rfrate * deltaT)
    u = math.exp(sigma * math.pow(deltaT, 0.5))
    d = 1 / u
    p = (a - d) / (u - d)
    q = 1 - p
    s = np.zeros((ngrid, ngrid))
    opt = np.zeros((ngrid, ngrid))
# 构建价格矩阵,这里的i是行数，j是列数，j代表ud的指数之和，i代表d的指数
    for j in range(ngrid):
        for i in range(j+1):
            s[i][j] = s0 * math.pow(u, j-i) * math.pow(d, i)
    for i in range(ngrid):
        opt[i][-1] = max(s[i][-1]-strike, 0)
    for j in range(ngrid-2, -1, -1):
        for i in range(j, -1, -1):
            # 没有呼叫的情况
            s1 = opt[i][j] = math.exp(-rfrate * deltaT) * (p * opt[i][j+1] + q * opt[i+1][j+1])
            # 有呼叫的情况
            if (s[i][j]>strike):
                call = myshout(s[i][j], s[i][j], sigma, deltaT * (ngrid-j-1), rfrate, ngrid-j)
                s2 = call + (s[i][j] - strike) * math.exp(-rfrate * deltaT * (ngrid-j-1))
            else:
                s2 = 0
            opt[i][j] = max(s1, s2)
    return opt[0][0]

# 回望期权定价
def lookback(s0,strike,sigma,t,rfrate,ngrid,times):
    # 构建价格二叉树图参数
    deltaT = t / (ngrid - 1)
    a = math.exp(rfrate * deltaT)
    u = math.exp(sigma * math.pow(deltaT, 0.5))
    d = 1 / u
    p = (a - d) / (u - d)
    # 使用monte Carlo模拟来做定价
    op = []
    for j in range(times):
        sp = [s0]
        for i in range(ngrid - 1):
            stick = np.random.binomial(1, p)
            if stick == 1:
                sp.append(sp[-1] * u)
            else:
                sp.append(sp[-1] * d)
        op.append(max((math.exp(-rfrate * t) * (max(sp) - strike), 0)))
    return np.mean(op)

# 亚式期权定价
def asian(s0,strike,sigma,t,rfrate,ngrid,times):
    # 构建价格二叉树图参数
    deltaT = t / (ngrid - 1)
    a = math.exp(rfrate * deltaT)
    u = math.exp(sigma * math.pow(deltaT, 0.5))
    d = 1 / u
    p = (a - d) / (u - d)
    # 使用Monte Carlo模拟来做定价
    op = []
    for j in range(times):
        sp = [s0]
        for i in range(ngrid - 1):
            stick = np.random.binomial(1, p)
            if stick == 1:
                sp.append(sp[-1] * u)
            else:
                sp.append(sp[-1] * d)
        op.append(max((math.exp(-rfrate * t) * (np.mean(sp) - strike), 0)))
    return np.mean(op)

price = asian(
    s0=100,
    strike=100,
    sigma=np.log(5),
    t=4,
    rfrate=1,
    ngrid=5,
    times=100000
)
print(price)
