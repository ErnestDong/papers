#%%
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent


class Volatility:
    def __init__(self, filename=project_root / "data/KMV模型已知量汇总.xlsx"):
        self.filename = filename
        self.data = self._get_volatility()

    def _get_price(self, sheetname="股价", start_date="2008-01-01"):
        df = pd.read_excel(
            self.filename, sheet_name=sheetname, index_col=0, skiprows=[1]
        ).pct_change()
        return df[df.index >= start_date]

    def _get_volatility(
        self, time_series=pd.date_range("2007-12-31", "2011-06-30", freq="6M")
    ):
        price_change = self._get_price()
        result = {}
        for start, end in zip(time_series[:-1], time_series[1:]):
            result[end] = price_change[
                (price_change.index > start) & (price_change.index <= end)
            ].std()
        return pd.DataFrame(result).T


if __name__ == "__main__":
    vol = Volatility()
    print(vol.data)

# %%
