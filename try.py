import math
import numpy as np

s0=1
strike=0.1
sigma=0.5
t=10
rfrate=0.05
ngrid=21

deltaT = t /(ngrid-1)
a = math.exp(rfrate * deltaT)
u = math.exp(sigma * math.pow(deltaT, 0.5))
d = 1 / u
p = (a - d) / (u - d)
q = 1 - p

op = []
sp = [s0]
for i in range(ngrid-1):
    stick = np.random.binomial(1,p)
    if stick == 1:
        sp.append(sp[-1] * u)
    else:
        sp.append(sp[-1] * d)
op.append(max((math.exp(-rfrate * t) * (max(sp)-strike), 0)))
np.mean(op)