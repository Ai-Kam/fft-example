import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import drange, DateFormatter
import datetime
keys, data = np.load("JMA_AMeDASdata_Tokyo_20210801-001000_20210806-000000.npz", allow_pickle=True).values()
print(keys, data.shape)

def get_column_by_key(data, key):
    return data[:, np.where(keys == key)].flatten()

dt = get_column_by_key(data, "DT")
temp = get_column_by_key(data, "temp")

def fft_and_plot(col, plot=True):
    ffted = np.fft.fft(col)/len(col) #normalize
    if plot == True:
        plt.plot(ffted.real)
        plt.plot(ffted.imag)
        plt.grid()
        plt.show()
    return ffted

ffted_temp = fft_and_plot(temp, plot= False)

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
