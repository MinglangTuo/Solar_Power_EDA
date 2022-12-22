# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from scipy.signal import periodogram
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import normaltest
import holoviews as hv
from holoviews import opts
import cufflinks as cf
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def readDataset():
    #read the dataset
    for dirname, _,filenames in os.walk('./dataset/Solar_Power_Machine_Learning'):
        for filename in filenames:
            print(os.path.join(dirname,filename))

def pre():
    #pre-build
    hv.extension('bokeh')
    cf.set_config_file(offline=True)
    sns.set(style="whitegrid")

def handle_date(path):
    #EDA
    file = path
    plant1_data = pd.read_csv(file)
    plant1_data.tail()
    return plant1_data
    #print(plant1_data.tail())

def inverter_number(plant1_data,time):
    #计算特定时间逆变器的数量
    print('The number of inverter for data_time {} is {}'.format(time, plant1_data[plant1_data.DATE_TIME == time]['SOURCE_KEY'].nunique()))

def check_data(plant1_data):
    #检查数据
    plant1_data.info()

def filiter_date(plant1_data):
    plant1_data = plant1_data.groupby('DATE_TIME')[['DC_POWER','AC_POWER', 'DAILY_YIELD','TOTAL_YIELD']].agg('sum')

    plant1_data = plant1_data.reset_index()

    return plant1_data
def clean_weather_sensor_data(plant1_data):
    plant1_data['DATE_TIME'] = pd.to_datetime(plant1_data['DATE_TIME'], errors='coerce')
    plant1_data['date'] = plant1_data['DATE_TIME'].dt.date
    plant1_data['time'] = plant1_data['DATE_TIME'].dt.time
    del plant1_data['PLANT_ID']
    del plant1_data['SOURCE_KEY']
    return plant1_data

def clean_data(plant1_data):
    plant1_data['DATE_TIME'] = pd.to_datetime(plant1_data['DATE_TIME'], errors='coerce')
    plant1_data['time'] = plant1_data['DATE_TIME'].dt.time
    plant1_data['date'] = plant1_data['DATE_TIME'].dt.date
    #print(plant1_data)
    return plant1_data

def DC_Power_plot(plant1_data):
    plant1_data.plot(x='time', y='DC_POWER', style='.', figsize=(15, 8))
    plant1_data.groupby('time')['DC_POWER'].agg('mean').plot(legend=True, colormap='Reds_r')
    plt.ylabel('DC Power')
    plt.title('DC POWER plot')
    plt.show()

def DC_Power_calendar(plant1_data):
    calendar_dc = plant1_data.pivot_table(values='DC_POWER', index='time', columns='date')
    return calendar_dc

def Daily_yield_calendar(plant1_data):
    daily_yield = plant1_data.pivot_table(values='DAILY_YIELD', index='time', columns='date')
    return daily_yield

def ambient_temperature_calendar(plant1_data):
    ambient = plant1_data.pivot_table(values='AMBIENT_TEMPERATURE', index='time', columns='date')
    return ambient

def multi_plot(data=None, row=None,col=None,title='DC Power'):
    cols = data.columns  # take all column
    gp = plt.figure(figsize=(30, 30))

    gp.subplots_adjust(wspace=0.3, hspace=5)
    for i in range(1, len(cols) + 1):
        ax = gp.add_subplot(row, col, i)
        data[cols[i - 1]].plot(ax=ax, style='k.')
        ax.set_title('{} {}'.format(title, cols[i - 1]))
    plt.show()


def daily_power_bar(plant1_data):
    daily_dc = plant1_data.groupby('date')['DC_POWER'].agg('sum')
    daily_dc.plot.bar(figsize=(15, 5), legend=True)
    plt.title('Daily DC Power')
    plt.show()


def daily_yield(plant1_data):
    plant1_data.plot(x='time',y='DAILY_YIELD',style='b.',figsize=(15,5))
    plant1_data.groupby('time')['DAILY_YIELD'].agg('mean').plot(legend=True, colormap='Reds_r')
    plt.title('DAILY YIELD')
    plt.ylabel('Yield')
    plt.show()

def daily_yield_bar(plant1_data):
    dyield = plant1_data.groupby('date')['DAILY_YIELD'].agg('sum')
    dyield.plot.bar(figsize=(15, 5), legend=True)
    plt.title('Daily YIELD')
    plt.show()

def ambient_temperature(plant1_data):
    plant1_data.plot(x='time', y='AMBIENT_TEMPERATURE', style='b.', figsize=(15, 5))
    plant1_data.groupby('time')['AMBIENT_TEMPERATURE'].agg('mean').plot(legend=True, colormap='Reds_r')
    plt.title('Daily AMBIENT TEMPERATURE MEAN (RED)')
    plt.ylabel('Temperature (°C)')
    plt.show()

def box_temperature(temperature_calendar):
    temperature_calendar.boxplot(figsize=(15, 5), grid=False, rot=90)
    plt.title('AMBIENT TEMPERATURE BOXES')
    plt.ylabel('Temperature (°C)')
    plt.show()

def line_chart(plant1_data):
    am_temp = plant1_data.groupby('date')['AMBIENT_TEMPERATURE'].agg('mean')
    am_temp.plot(grid=True, figsize=(15, 5), legend=True, colormap='Oranges_r')
    plt.title('AMBIENT TEMPERATURE 15 MAY- 17 JUNE')
    plt.ylabel('Temperature (°C)')
    plt.show()

def line_chart_rate(plant1_data):
    am_temp = plant1_data.groupby('date')['AMBIENT_TEMPERATURE'].agg('mean')
    am_change_temp = (am_temp.diff()/am_temp)*100
    am_change_temp.plot(figsize=(15, 5), grid=True, legend=True)
    plt.ylabel('%change')
    plt.title('AMBIENT TEMPERATURE %change')
    plt.show()

def time_series(plant1_data):
    plant1_data = plant1_data.groupby('date')['AMBIENT_TEMPERATURE'].agg('mean')

    decomp = sm.tsa.seasonal_decompose(plant1_data,model='additive',period=5)
    #period不同导致相关的，trend，seasonal，resid的差距太大？？？？

    cols = ['trend', 'seasonal', 'resid']  # take all column
    data = [decomp.trend, decomp.seasonal, decomp.resid]
    gp = plt.figure(figsize=(15, 15))

    gp.subplots_adjust(hspace=0.5)
    for i in range(1, len(cols) + 1):
        ax = gp.add_subplot(3, 1, i)
        data[i - 1].plot(ax=ax)
        ax.set_title('{}'.format(cols[i - 1]))
    plt.show()

def module_temperature(data_result_1):
    data_result_1.plot(x='time', y='MODULE_TEMPERATURE', figsize=(15, 8), style='b.')
    data_result_1.groupby('time')['MODULE_TEMPERATURE'].agg('mean').plot(colormap='Reds_r', legend=True)
    plt.title('DAILY MODULE TEMPERATURE & MEAN(red)')
    plt.ylabel('Temperature(°C)')
    plt.show()

def module_temperature_calendar(data_result_1):
    module_temp = data_result_1.pivot_table(values='MODULE_TEMPERATURE', index='time', columns='date')
    return data_result_1

def orrelation_(data_result_1,plant1_data):
    data_result_1 = data_result_1.merge(plant1_data, left_on='DATE_TIME', right_on='DATE_TIME')
    del data_result_1['date_x']
    del data_result_1['date_y']
    del data_result_1['time_x']
    del data_result_1['time_y']
    corr = data_result_1.drop(columns=['DAILY_YIELD', 'TOTAL_YIELD']).corr(method='spearman')
    return corr

def heat_map(corr):
    plt.figure(dpi=100)
    sns.heatmap(corr, robust=True, annot=True, fmt='0.3f', linewidths=.5, square=True)
    plt.show()

def pair_plot(data_result_1):
    sns.pairplot(data_result_1)
    plt.show()

def compare_dc_power(plant1_data,plant2_data):
    ax = plant1_data.plot(x='time', y='DC_POWER', figsize=(15, 5), legend=True, style='b.')
    plant2_data.plot(x='time', y='DC_POWER', figsize=(15, 5),legend=True, style='r.',ax = ax)
    plt.title('Plant1(blue) vs Plant2(red)')
    plt.ylabel('Power (KW)')
    plt.show()

def compare_ac_power(plant1_data,plant2_data):
    ax1 = plant1_data.plot(x='time', y='AC_POWER', figsize=(15, 5), legend=True, style='b.', )
    plant2_data.plot(x='time', y='AC_POWER', legend=True, style='r.', ax=ax1)
    plt.title('Plant1(blue) vs Plant2(red)')
    plt.ylabel('Power (KW)')

def daily_dc_power(plant1_data,plant2_data):
    p2_daily_dc = plant2_data.groupby('date')['DC_POWER'].agg('sum')
    p1_daily_dc = plant1_data.groupby('date')['DC_POWER'].agg('sum')
    axh = p1_daily_dc.plot.bar(legend=True, figsize=(15, 5), color='Blue', label='DC_POWER Plant I')
    p2_daily_dc.plot.bar(legend=True, color='Red', label='DC_POWER Plant II', stacked=False)
    plt.title('DC POWER COMPARISON')
    plt.ylabel('Power (KW)')
    plt.show()

def create_new_value(data_result):
    del data_result['date_x']
    del data_result['date_y']
    del data_result['time_x']
    del data_result['time_y']
    data_result = data_result.assign(
        DELTA_TEMPERATURE=abs(data_result.MODULE_TEMPERATURE - data_result.AMBIENT_TEMPERATURE),
        NEW_DAILY_YIELD=data_result.DAILY_YIELD.diff(),
        NEW_TOTAL_YIELD=data_result.TOTAL_YIELD.diff(),
        NEW_AMBIENT_TEMPERATURE=data_result.AMBIENT_TEMPERATURE.diff(),
        NEW_MODULE_TEMPERATURE=data_result.MODULE_TEMPERATURE.diff(),
        NEW_AC_POWER=data_result.AC_POWER.diff())

    data_result=data_result.where(data_result['NEW_DAILY_YIELD'].notnull(),0)
    #data_result =data_result.where(data_result['NEW_TOTAL_YIELD'].notnull(), 0)
    return data_result



def heat_map2(data_result):
    plt.figure(dpi=100, figsize=(15, 10))
    sns.heatmap(data_result.corr(method='spearman'), robust=True, annot=True, fmt='0.2f', linewidths=.5, square=False)
    plt.show()

def ac_dc_plot(data_result):
    sns.lmplot(x='DC_POWER', y='AC_POWER', data=data_result)
    plt.title('ac_dc_relationship plot')
    plt.show()

def daily_yield_ac_plot(data_result):
    plt.figure(dpi=(100), figsize=(15, 5))
    sns.regplot(x='AC_POWER', y='NEW_DAILY_YIELD', data=data_result)
    plt.title(' daily_yield_ac_relationship_plot')
    plt.show()

def daily_yield_irradiation(data_result):
    plt.figure(dpi=(100), figsize=(15, 5))
    sns.regplot(x='IRRADIATION', y='NEW_DAILY_YIELD', data=data_result)
    plt.title('daily_yield_irradiation_relationship plot')
    plt.show()

def daily_yield_ac_power(data_result):
    plt.figure(dpi=(100), figsize=(15, 5))
    sns.regplot(x='MODULE_TEMPERATURE', y='NEW_DAILY_YIELD', data=data_result)
    plt.title('daily_yield_ac_power_relationship plot')

def daily_yield_delta_temperature(data_result):
    plt.figure(dpi=(100), figsize=(15, 5))
    sns.regplot(x='DELTA_TEMPERATURE', y='NEW_DAILY_YIELD', data=data_result)
    plt.title('daily_yield_delta_temperature_relationship plot')
    plt.show()

def new_ac_new_module_temperature(data_result):
    plt.figure(dpi=(100), figsize=(15, 5))
    sns.regplot(y='NEW_AC_POWER', x='NEW_MODULE_TEMPERATURE', data=data_result)
    plt.title('Regression plot')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

#单一结果分析！！！！

    #对数据进行检查和处理-Plant_1_Generation_Data
    #readDataset()
    #pre()
    data_result = handle_date("./dataset/Solar_Power_Machine_Learning/Plant_1_Generation_Data.csv")
    data_result_3 = handle_date("./dataset/Solar_Power_Machine_Learning/Plant_2_Generation_Data.csv")

    data_result = filiter_date(data_result)
    data_result_3 = filiter_date(data_result_3)

    plant1_data = clean_data(data_result)
    data_result_3 = clean_data(data_result_3)

    #计算逆变器的数量
    # inverter_number(data_result,'15-05-2020 23:00')
    # check_data(data_result)

    #当天直流电产生的趋势图
    #DC_Power_plot(plant1_data)

    #绘制每天的直流电产生数据趋势图
    #calendar_dc = DC_Power_calendar(plant1_data)

    #multi_plot(data=calendar_dc,row=9,col=4)

    #绘制每天直流电的直方图
    #daily_power_bar(plant1_data)

    #绘制当天光伏产量的趋势
    #daily_yield(plant1_data)

    #绘制每天的光伏产量趋势
    #calendar_yield = Daily_yield_calendar(plant1_data)

    #绘制每天光伏产量直方图
    #daily_yield_bar(plant1_data)

    # 对数据进行检查和处理-Plant_1_Generation_Data
    data_result_1 = handle_date("./dataset/Solar_Power_Machine_Learning/Plant_1_Weather_Sensor_Data.csv")
    data_result_1 = clean_weather_sensor_data(data_result_1)

    data_result_4 = handle_date("./dataset/Solar_Power_Machine_Learning/Plant_2_Weather_Sensor_Data.csv")
    data_result_4 = clean_weather_sensor_data(data_result_4)


    #绘制当天环境温度的散点图
    #ambient_temperature(data_result_1)

    #绘制每天环境温度的箱线图
    #temperature_calendar = ambient_temperature_calendar(data_result_1)
    #box_temperature(temperature_calendar)

    #绘制每天的环境温度折线图
    #line_chart(data_result_1)

    #绘制每天的环境温度增长率折线图
    #line_chart_rate(data_result_1)

    #时间序列分析对于之后趋势和季节性分析
    #time_series(data_result_1)

    #module_temperature(data_result_1)
    #之后的module_temperature，IRRADIATION和前面的结果类似
    #......................................................
    #........................................................
    #.......................................................

#相关性分析！！！！
    #两个数据合并在一起
    #corr = orrelation_(data_result_1,plant1_data)

    #热力图显示
    #heat_map(corr)

    #显示出不同关系的线性表示
    #pair_plot(data_result_1)
    #pair_plot(plant1_data)


    #显示出两个光伏面板的产量和时间比值
    #compare_dc_power(data_result,data_result_3)
    #compare_ac_power(data_result, data_result_3)

    #显示出两个光伏面板dc的产量和每天的比值
    #daily_dc_power(data_result,data_result_3)
    #显示出两个光伏面板ac的产量和每天的比值
    #显示出两个光伏面板yield的产量和每天的比值


    #对于相同站点的传感器和光伏接受装置，也使用相同的方法来使用
    data_result_1 = data_result_1.merge(plant1_data, left_on='DATE_TIME', right_on='DATE_TIME')

    #建立新的指标来处理数据
    data_result_1 = create_new_value(data_result_1);

    #热力图来构建
    #heat_map2(data_result_1)

    #ac和dc之间关系
    #ac_dc_plot(data_result_1)

    #光照日产能与ac之间关系
    #daily_yield_ac_plot(data_result_1)

    #光照日产能与光照量之间关系
    #daily_yield_irradiation(data_result_1)

    #光照日产能与ac之间关系
    #daily_yield_ac_power(data_result_1)

    #光照日产能与delta温度之间关系
    #daily_yield_delta_temperature(data_result_1)

    #新ac与新模块温度之间关系
    #new_ac_new_module_temperature()


    '''conclusion:
plant I produces 6 times more DC power than plant II. And loses 90% of it when converting to AC power.

While Plant II loses nothing when converting DC power to AC power.

AC power output is almost the same for both plants.

The daily yield is almost the same for the two plants.

The gap between The average total yield for plant I and plant II is very large.

Daily yield decrease if delta temperature is less than 5°C.

Daily yield decrease for some value of AC power.


ref:https://www.kaggle.com/code/lumierebatalong/solar-power-machine-learning-i/notebook
'''

