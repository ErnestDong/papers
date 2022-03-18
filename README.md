# 风管模型第一次作业
## 假设
无风险利率
收益率符合正态分布
## requirements
recommend to use `virtual environments`.
```shell
$ pip install -r requirements.txt # or poetry install if you have installed poetry
```
## usage
### install as a package (what poetry does)
```python
from src import historical # or monte_carlo, var_cov etc.
print(historical.var(10))
```
### run my unittests (recommend)
``` shell
python tests/test.py # if fails it's maybe python environment problems (x
```
It will print the result of the homework.

Also you can run each part of the package to see detailed result.
## the functionality of each module

### tests
[test_src](./tests/test_src.py)
Just unittest for migrating from different environment. 
Besides it is also designed as solutions of this homework

the result will be shown in [result.log](./result.log)
### src
source code of the homework
#### [config](./src/config.py)
general configuration, like reading from excel files, setting some constants
#### [markowitz](./src/markowitz.py)
solution to the second question

#### [historical](./src/historical.py), [var_cov](./src/var_cov.py), [monte_carlo](./src/monte_carlo.py)
calculate VaR through Historical Simulation Method, Variance-Covariance Method and Monte Carlo Simulation Method. 

each of them provides a `var()` to calc the solution

### lib
trade data of stocks from wind

### doc
maybe [readme](./README.md) is enough?
