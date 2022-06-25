#%%
import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

commit_info = glob.glob("./data/commit_info*.txt")
# 文件名特征：时区的字典
time_zone = {"1": 0, "2": 8}
result = {}
for commit_info_ins in commit_info:
    result[commit_info_ins] = {}
    cursor = open(commit_info_ins, "r", encoding="utf-8")
    # 去掉第一条多余的 commit_done 的信息
    cursor.readline()
    timezone = time_zone[[i for i in time_zone if i in commit_info_ins][0]]
    while (
        (start := cursor.readline())
        and (file_num := cursor.tell())
        and (end := cursor.readline())
    ):
        # 每次读两行，遇到不匹配的退回到两行中间开始下一次循环
        if not ("start" in start and "done" in end):
            print(file_num, "in", commit_info_ins.split("/")[-1], "couldn't match")
            cursor.seek(file_num)
            continue
        # start_num 和开始/结束时间
        start_num = int(start.split("=")[-1].replace(",", ""))
        start_time = datetime.datetime.strptime(
            "2022" + start.split(" ")[1], "%Y[%m-%d|%H:%M:%S.%f]"
        ) - datetime.timedelta(hours=timezone)
        done_time = datetime.datetime.strptime(
            "2022" + end.split(" ")[1], "%Y[%m-%d|%H:%M:%S.%f]"
        ) - datetime.timedelta(hours=timezone)
        result[commit_info_ins][start_num] = done_time - start_time
    cursor.close()
# %%
df = pd.DataFrame(result).dropna()
# df.plot()
diff = df[df.columns[0]] - df[df.columns[1]]
quantile = [0.1, 0.25, 0.5, 0.75, 0.9]
print(f"{df.columns[0]} - {df.columns[1]}的 {quantile} 分位数(单位毫秒)为")
print(diff.quantile(quantile).apply(lambda x:x.total_seconds()*1000))
# %%
