import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import drange, DateFormatter
import datetime
keys, data = np.load("JMA_AMeDASdata_Tokyo_20210801-001000_20210806-000000.npz", allow_pickle=True).values()
print(keys, data.shape) #keyの確認

def get_column_by_key(data, keys, key):
    return data[:, np.where(keys == key)].flatten()

dt = get_column_by_key(data, keys, "DT") #Date&Time
temp = get_column_by_key(data, keys, "temp") #Temperature
print(dt[:10])
print(temp[:10])

def fft_and_plot(col, plot=True): #一次元配列をFFT, plot=Trueなら結果をplot
    ffted = np.fft.fft(col)/len(col) #normalize
    if plot == True:
        plt.plot(ffted.real, label ="real")
        plt.plot(ffted.imag, label ="imag")
        plt.legend(loc='upper right', framealpha=0.5)
        plt.grid()
        plt.show()
    return ffted

ffted_temp = fft_and_plot(temp)

def calc_amps(ffted_data):
    n = len(ffted_data)
    amps = np.array(ffted_data[0])
    if (n % 2): # odd [0,1,2,3,-3,-2,-1]
        amps = np.append((np.atleast_1d(amps), ffted_data[1:int(n/2)+1]+ffted_data[-1:-int(n/2)-1:-1].conj()))
    else: #even [0,1,2,3,-2,-1]
        amps = np.concatenate((np.atleast_1d(amps), ffted_data[1:int(n/2)]+ffted_data[-1:-int(n/2):-1].conj(), np.atleast_1d(ffted_data[-int(n/2)])))
    return amps

temp_amps = calc_amps(ffted_temp)
print(abs(temp_amps[:21])) #波数0は平均値を表す
plt.plot(np.arange(1,21), abs(temp_amps[1:21]), label ="TEMP amps") #波数5と10にシグナルあり
plt.legend(loc='upper right', framealpha=0.5)
plt.grid()
plt.show()

#グラフを描いてくれるおまじない
def plot_waves(axes, X, amps, when_plot, additional_label_exp="" ):
    theta = np.linspace(0, 2*np.pi, len(X)+1)[:-1]
    expressed_wave = np.zeros_like(theta)
    for n in range(max(when_plot)+1):
        expressed_wave += amps[n].real * np.cos(n*theta) - amps[n].imag*np.sin(n*theta)
        if n in when_plot:
            axes.plot(X, expressed_wave, label='%s n=%d' % (additional_label_exp, n))

#日付軸にしてくれるおまじない
def fig_axisformatter_bydate():
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


dt, axes = fig_axisformatter_bydate()
axes.plot(dt, temp, label ="real temp")
amps = calc_amps(ffted_temp)
plot_waves(axes, dt, amps, (0,4,5,10))
plt.legend(loc='upper right', framealpha=0.5)
plt.show() #気温のplot

aws = get_column_by_key(data, keys, "AWS")
ffted_aws = fft_and_plot(aws, plot= False)

dt, axes = fig_axisformatter_bydate()
axes.plot(dt, aws, label ="real AWS")
amps = calc_amps(ffted_aws)
plot_waves(axes, dt, amps, (0,4,5,10))
plt.legend(loc='upper right', framealpha=0.5)
plt.show() #AWS（平均風速）のplot

dt, axes = fig_axisformatter_bydate()
temp_amps = calc_amps(ffted_temp)
aws_amps = calc_amps(ffted_aws)
plot_waves(axes, dt, temp_amps, (1,10), additional_label_exp="temp")
plot_waves(axes, dt, aws_amps*5, (1,10), additional_label_exp="AWS*5")
plt.legend(loc='upper right', framealpha=0.5)
plt.show() #tempとAWS（平均風速）のplot

def corrcoef_plot(X, amps1, amps2, plot_range, additional_label_exp=""):
    theta = np.linspace(0, 2*np.pi, len(X)+1)[:-1]
    expressed_wave1 = np.zeros_like(theta)
    expressed_wave2 = expressed_wave1.copy()
    n_min = min(plot_range)
    x = np.array(range(len(plot_range)+1))+n_min
    res = np.zeros_like(x).astype("float64")
    for n_k in range(len(plot_range)+1):
        n = n_k + n_min
        expressed_wave1 += amps1[n].real * np.cos(n*theta) - amps1[n].imag*np.sin(n*theta)
        expressed_wave2 += amps2[n].real * np.cos(n*theta) - amps2[n].imag*np.sin(n*theta)
        if (n >= n_min):
            res[n_k] = np.corrcoef(expressed_wave1 ,expressed_wave2)[1,0]
    plt.plot(x, res, label='%s' % (additional_label_exp))
    plt.legend(loc='upper right', framealpha=0.5)
    plt.grid()
    plt.show()

corrcoef_plot(dt, temp_amps, aws_amps, range(1,20), additional_label_exp="temp&AWS")

#WindSpeed, WindDirection -> u, v components
def ws_wd_to_u_v_translation(ws,wd):
    wd_dict = {"北":0, "北北西":1, "北西":2, "西北西":3,
               "西":4, "西南西":5, "南西":6, "南南西":7,
               "南":8, "南南東":9, "南東":10, "東南東":11,
               "東":12, "東北東":13, "北東":14, "北北東":15}
    for item in wd_dict.items(): #場合によっては欠損値などが入る恐れあり
        wd = np.where(wd == item[0], item[1], wd)
    wd = wd.astype("float64")
    print(wd, wd.dtype) #全て変換できているか確認
    wd *= np.pi/8
    u = -ws*np.sin(wd)
    v = -ws*np.cos(wd)
    return u, v

aws = get_column_by_key(data, keys, "AWS")
awd = get_column_by_key(data, keys, "AWD")
au, av = ws_wd_to_u_v_translation(aws,awd)
ffted_au = fft_and_plot(au, plot= False)
ffted_av = fft_and_plot(av, plot= False)

au_amps = calc_amps(ffted_au)
av_amps = calc_amps(ffted_av)
print(abs(au_amps[:21])) #波数0は平均値を表す
print(abs(av_amps[:21])) #波数0は平均値を表す
plt.plot(np.arange(1,21), abs(au_amps[1:21]), label ="AU amps") #波数5と10にシグナルあり
plt.plot(np.arange(1,21), abs(av_amps[1:21]), label ="AV amps") #波数5と10にシグナルあり
plt.legend(loc='upper right', framealpha=0.5)
plt.grid()
plt.show()

dt, axes = fig_axisformatter_bydate()
axes.plot(dt, au, label ="real AU")
axes.plot(dt, av, label ="real AV")
au_amps = calc_amps(ffted_au)
av_amps = calc_amps(ffted_av)
plot_waves(axes, dt, au_amps, (10,), additional_label_exp="AU")
plot_waves(axes, dt, av_amps, (0,5,10,), additional_label_exp="AV") #0成分は背景場の南北風速成分を示す
plt.legend(loc='upper right', framealpha=0.5)
plt.show()
    

