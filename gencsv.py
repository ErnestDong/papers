import pandas as pd

excel = pd.ExcelFile('data/data.xlsx')
for sheet in excel.sheet_names:
    if "raw" in sheet:
        continue
    df = excel.parse(sheet)
    df.to_csv('data/%s.csv' % sheet, index=False, float_format='%.3f')
