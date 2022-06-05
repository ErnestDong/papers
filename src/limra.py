#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams['font.sans-serif']=['LXGW WenKai Mono']
plt.rcParams['axes.unicode_minus']=False
sns.set(style="white")
male = [0.63,0.53,0.54,0.53,0.54,0.47]
female = [0.65,0.58,0.55,0.56,0.55,0.47]
total = [0.64,0.56,0.55,.55,0.55,.47]
x = ["It’s fast and easy","It avoids the need for medical exam","It provides transparent explanations of risk classiffcation and product pricing", "It avoids the need to see a doctor","It is unbiased and objective","It avoids the need for face-to-face conversation"]
df = pd.DataFrame({"Total":total, "Men":male,"Women":female}, index=x)
df.plot.barh()
plt.title("Appealing features of Simplified Underwriting (by gender)")
plt.plot()
# %%
plt.savefig("./lib/limra.png")
