import pandas as pd
import numpy as np
import datetime
dfs = pd.read_excel('Book1.xlsx', sheet_name=None)
mdf = None
for df in dfs.values():
    if mdf is None:
        mdf = df[2:146]
    else:
        mdf = mdf.append(df[2:146])
mdf = mdf.reset_index(drop=True)
mdf.columns = ("DT","SP","SLP","prep","temp","RH","AWS","AWD","MWS","MWD","SD")
dt = datetime.datetime(year=2021, month=8, day=1, hour=0, minute=10, second=0)
tdelta = datetime.timedelta(minutes=10)
mdf["date"] = pd.Series([dt + n * tdelta for n in range(144*5)])
keys = np.array(mdf.columns.values)
data = mdf.to_numpy()
np.savez("JMA_AMeDASdata_Tokyo_20210801-001000_20210806-000000.npz",keys,data)

