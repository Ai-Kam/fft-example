import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import drange, DateFormatter
import datetime
keys, data = np.load("JMA_AMeDASdata_Tokyo_20210801-001000_20210806-000000.npz", allow_pickle=True).values()
print(keys, data.shape) #keyの確認

def get_column_by_key(data, key):
    return data[:, np.where(keys == key)].flatten()

dt = get_column_by_key(data, "DT") #Date&Time
temp = get_column_by_key(data, "temp") #Temperature
print(dt[:10])
print(temp[:10])

def fft_and_plot(col, plot=True): #一次元配列をFFT, plot=Trueなら結果をplot
    ffted = np.fft.fft(col)/len(col) #normalize
    if plot == True:
        plt.plot(ffted.real)
        plt.plot(ffted.imag)
        plt.grid()
        plt.show()
    return ffted

ffted_temp = fft_and_plot(temp)

def plot_waves(axes, X, lenx, ffted_data, wavenum, when_plot, additional_label_exp="" ):
    theta = np.linspace(0, 2*np.pi, lenx+1)[:-1]
    expressed_wave = np.zeros_like(theta)
    amps = np.array(ffted_data[0])
    amps = np.append(amps, (ffted_data[1:wavenum+1]+ffted_data[-1:-(wavenum+1):-1].conj()))
    for n in range(wavenum+1):
        expressed_wave += amps[n].real * np.cos(n*theta) - amps[n].imag*np.sin(n*theta)
        if n in when_plot:
            axes.plot(X, expressed_wave, label='%s n=%d' % (additional_label_exp, n))

def date_formatter():
    plt.close(1)
    figure_ = plt.figure(1, figsize=(8,4))
    axes = figure_.add_subplot(111)
    date1 = datetime.datetime(2021, 8, 1, 0, 10)
    date2 = datetime.datetime(2021, 8, 6, 0,   1)
    delta = datetime.timedelta(minutes=10)
    dt = drange(date1, date2, delta)
    xaxis = axes.xaxis
    xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.grid()
    return dt, axes

lenx =144*5
wavenum = 10
"""
dt, axes = date_formatter()
axes.plot(dt, temp, label ="real temp")
plot_waves(axes, dt, lenx, ffted_temp, wavenum, (0,4,5,10))
plt.legend(loc='upper right', framealpha=0.5)
plt.show()

aws = get_column_by_key(data, "AWS")
ffted_aws = fft_and_plot(aws, plot= False)
dt, axes = date_formatter()
axes.plot(dt, aws, label ="real AWS")
plot_waves(axes, dt, lenx, ffted_aws, wavenum, (0,4,5,10))
plt.legend(loc='upper right', framealpha=0.5)
plt.show()

dt, axes = date_formatter()
plot_waves(axes, dt, lenx, ffted_temp, wavenum, (10,), additional_label_exp="temp")
plot_waves(axes, dt, lenx, ffted_aws*5, wavenum, (10,), additional_label_exp="AWS*5")
plt.legend(loc='upper right', framealpha=0.5)
plt.show()
"""

aws = get_column_by_key(data, "AWS")
awd = get_column_by_key(data, "AWD")

def ws_wd_to_u_v_translation(ws,wd):
    wd_dict = {"北":0, "北北西":1, "北西":2, "西北西":3,
               "西":4, "西南西":5, "南西":6, "南南西":7,
               "南":8, "南南東":9, "南東":10, "東南東":11,
               "東":12, "東北東":13, "北東":14, "北北東":15}
    for item in wd_dict.items(): #場合によっては欠損値などが入る恐れあり
        wd = np.where(wd == item[0], item[1], wd)
    wd = wd.astype("float64")
    print(wd, wd.dtype) #確認
    wd *= np.pi/8
    u = -ws*np.sin(wd)
    v = -ws*np.cos(wd)
    return u, v
 
au, av = ws_wd_to_u_v_translation(aws,awd)
ffted_au = fft_and_plot(au, plot= False)
ffted_av = fft_and_plot(av, plot= False)
dt, axes = date_formatter()
axes.plot(dt, au, label ="real AU")
axes.plot(dt, av, label ="real AV")
plot_waves(axes, dt, lenx, ffted_au, wavenum, (10,), additional_label_exp="AU")
plot_waves(axes, dt, lenx, ffted_av, wavenum, (0,10,), additional_label_exp="AV")
plt.legend(loc='upper right', framealpha=0.5)
plt.show()
    

