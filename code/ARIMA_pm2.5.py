# -*- coding: utf-8 -*-


'''
#请求mixiot平台数据，返回dataframe格式
from tool import getMixiotData 
#Dataframe
ttt = getMixiotData(dataType='S08',num=1000)
'''



from __future__ import print_function
#表格型矩阵数据构造库
#Series是引用值，改变后所有引用都会变
import pandas as pd 
#矩阵库
import numpy as np
#SciPy是python的一个著名的开源科学库
#SciPy一般都是操纵NumPy数组来进行科学计算
#统计分析，可以说是基于NumPy之上
#SciPy提供了许多科学计算的库函数
#如线性代数，微分方程，信号处理，图像处理，系数矩阵计算等，
from scipy import  stats
#数据可视化
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


import statsmodels.api as sm

from statsmodels.stats.stattools import durbin_watson #DW检验
#from statsmodels.tsa.arima_model import ARIMA #模型
from statsmodels.tsa.arima.model import ARIMA

#qq图
from statsmodels.graphics.api import qqplot
#处理时间格式
from datetime import datetime 

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import threading
import time

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error,mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

import matplotlib.dates as mdate
from matplotlib.pyplot import MultipleLocator

import seaborn as sns


from scipy.optimize import minimize              # 优化函数






def ADF_ACF_PACF_PLOT(data,legend,key):
    pValue = sm.tsa.stattools.adfuller(data)[1]
    fig = plt.figure(figsize=(12,8))
    
    ax1 = fig.add_subplot(1,1,1)
    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M'))
    x_major_locator=MultipleLocator(0.003)
    ax1.xaxis.set_major_locator(x_major_locator)
    
    if(key):
        fig.add_subplot(2,1,1)
    plt.title('Time Series Analysis Plots\nDickey-Fuller: p={0:.5f}'.format(pValue))
    line1, = plt.plot(data,label = legend,marker='o',color="#000")
    plt.legend(loc="upper left")
    plt.grid(False)
    plt.legend(handles=[line1,])
    plt.xlabel('time')
    plt.ylabel('$\mu$g/m3')
    if(key):
        ax2 = fig.add_subplot(2,2,3)
        plt.title('Autocorrelation Of Training Data')
        fig = plot_acf(data,ax=ax2)
        ax3 = fig.add_subplot(2,2,4)
        plt.title('Partial Autocorrelation Of Training Data')
        fig = plot_pacf(data, ax=ax3)
#    ax = plt.gca()
#    ax.spines['right'].set_color('none')
#    ax.spines['top'].set_color('none')
    #plt.savefig(r'../data/diff-adfacfpacf.svg',dpi=600)








    
def split2trainAndTest(series,interval):
    if(interval > 59):
        raise Exception('Interval must be less than 60.\n This ERROR IN Function split2trainAndTest')
    lastTime = series.index[-1]
    startTimeStr = "1924-2-05 00:00"
    if interval < 10:
        interval = '0{i}'.format(i = interval)
    endTimeStr = "1924-2-05 00:{T}".format(T = interval)
    startTime = datetime.strptime(startTimeStr,'%Y-%m-%d %H:%M')
    endTime = datetime.strptime(endTimeStr,'%Y-%m-%d %H:%M')
    intervalTime = endTime - startTime
    split_point = lastTime - intervalTime
    trainData, testData = series[:split_point], series[split_point:]
    #print(testData)
    #print('trainData %d, testData %d' % (len(trainData), len(testData)))
    return trainData, testData




#discfile = r'../data/arima_pm2.5_data.xls'

#工具函数：删除字符串左边的空白字符
def delSpace(serie):
    noSpace = serie.lstrip()
    return noSpace


# 读取数据，指定日期列为索引，nrows=666仅仅读取666行的数据，返回为Datetime格式
dis = r'../data/2020-10-18原始监测数据.xls'
data =pd.read_excel(dis,sheet_name='equipment',\
                    usecols=[0,2],index_col=0,\
                    dtype={"采集时间":"str","PM2.5浓度":"int"},)#nrows=666

#删除时间行标签"  09:45:29"前面的空格
data.rename(delSpace, axis='index',inplace=True)


#改变时间索引格式
def index2datetimeFF(serie):
    #print(serie,type(serie))
    timeIndex = datetime.strptime(serie,'%H:%M:%S')
    return timeIndex

'''
#改变行名的第一种方法
#dfData.index = dfData.index.map(index2datetime)
#print(dfData.index[0].month,type(dfData.index[0]))


#改变行名的第二种方法
# 是否替换原时间序列 inplace=True
#dateIndexdfData = dfData.rename(index2datetime, axis='0')
#print(type(dateIndexdfData.index[0]),dateIndexdfData.index[0].month)

'''
#把DataFrame数据的行索引转成datetime格式
originData = data.rename(index = index2datetimeFF)
originData = originData.copy()
dataStart = 600#450
data = originData[originData.iloc[dataStart].name:originData.iloc[dataStart+660].name]


def creatNewDicFF(dic,key,value):
    #根据采集时间重新创建字符串
    #再重新生成datetime格式的索引
    hour = key.hour
    minute = key.minute
    Y = '2020'
    m = '10'
    d = '18'
    timeStr = '{Y}-{m}-{d} {H}:{M}'.format(Y = Y ,m = m,d = d,H = hour,M = minute)
    dataTimeStr = datetime.strptime(timeStr,'%Y-%m-%d %H:%M')
    #只取到datetime的同一分钟内创建相同的key
    #同一分钟内测得的值成为value，列表形式存放
    if not(dataTimeStr in dic):
        dic[dataTimeStr] = []
        dic[dataTimeStr].append(value)
    else:
        dic[dataTimeStr].append(value)


def createNewdfDataFF(dic):
    #遍历字典一分钟内采集的所有值
    #取平均数为该分钟内的代表值
    for key, value in dic.items():
        #取平均并且保留二位小数
        dic[key] = np.round(np.mean(value),2)
 
    # 第一种
    timeSeriesdfData = pd.DataFrame(dic,index=['PM2.5浓度']).T
    # 第二种
    #timeSeriesdfData = pd.DataFrame.from_dict(dic,orient='index')
    return timeSeriesdfData


def handleDataTimeIndexFF(dateIndexData):
    dataIndexList = dateIndexData.index
    newDic = {}
    for i in dataIndexList:
        #拿到行的时间索引
        timeKey = i
        #拿到对应时间测得的值
        timeValue = dateIndexData.loc[str(i)].values[0]
        #把同一分钟内采集到的值取平均，形成每分钟内具有唯一平均值的字典
        creatNewDicFF(newDic,timeKey,timeValue)
        #print(timeKey,timeValue)
    #根据字典创建新的DataFrame格式时间序列数据
    newdfData = createNewdfDataFF(newDic)
    return newdfData


fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(1,1,1)
ax1.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M'))
x_major_locator=MultipleLocator(0.003)
ax1.xaxis.set_major_locator(x_major_locator)

plt.title('Original PM2.5 Concentrations Data Collected By Sensor')
line1, = plt.plot(data,label = 'PM2.5 Concentrations',\
                  linestyle="--",\
                  color="#000",\
                  marker='s')
plt.legend(loc="upper left")
plt.grid(False)
plt.legend(handles=[line1,])
plt.xlabel('time')
plt.ylabel('$\mu$g/m3')
#ax = plt.gca()
#ax.spines['right'].set_color('none')
#ax.spines['top'].set_color('none')
#plt.savefig(r'../data/Original PM2.5 Concentrations Data Collected By Sensor.svg',dpi=600)


timedata = handleDataTimeIndexFF(data)







fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(1,1,1)
ax1.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M'))
x_major_locator=MultipleLocator(0.003)
ax1.xaxis.set_major_locator(x_major_locator)

plt.title('Time Series Of PM2.5 Concentrations After Pre-processing')
line1, = plt.plot(timedata,label = 'Time Series',\
                  linestyle="--",\
                  color="#000",\
                  marker='o')
plt.legend(loc="upper left")
plt.grid(False)
plt.legend(handles=[line1,])
plt.xlabel('time')
plt.ylabel('$\mu$g/m3')
#plt.gcf().autofmt_xdate()

#ax = plt.gca()
#ax.spines['right'].set_color('none')
#ax.spines['top'].set_color('none')

#plt.savefig(r'../data/Time Series Of PM2.5 Concentrations After Pre-processing.svg',dpi=600)




#Series
seData=timedata['PM2.5浓度']
trainSeData, testSeData = split2trainAndTest(seData,8)





# Encoding: utf-8

'''
author: yhwu
version: 2021-04-19
function: numpy array write in the excel file
'''



## define a as the numpy array
#a = np.array(trainSeData)
## transform a to pandas DataFrame
#a_pd = pd.DataFrame(a)
## create writer to write an excel file
#writer = pd.ExcelWriter('a.xlsx')
## write in ro file, 'sheet1' is the page title, float_format is the accuracy of data
#a_pd.to_excel(writer, 'sheet1', float_format='%.6f')
## save file
#writer.save()
## close writer
#writer.close()
#

















#生成一个datetime格式的一分钟间隔，用于计算
startTimeStr = "1924-2-05 00:00:00"
endTimeStr = "1924-2-05 00:01:00"
startTime = datetime.strptime(startTimeStr,'%Y-%m-%d %H:%M:%S')
endTime = datetime.strptime(endTimeStr,'%Y-%m-%d %H:%M:%S')
intervalTime = endTime - startTime

#双指数平滑
def double_exponential_smoothing(series, alpha, beta):
    """
        series - dataset with timeseries
        alpha - float [0.0, 1.0], smoothing parameter for level
        beta - float [0.0, 1.0], smoothing parameter for trend
    """
    # first value is same as series
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result
#双指数平滑法获得一期的预测值，从哪里开始取决于输入的第一个Series类型的训练集参数
resoult1 = double_exponential_smoothing(trainSeData, 0.9, 0.02)
#生成预测值的所有时间戳索引
time1 = list(trainSeData.index)
time2 = [trainSeData.index[-1] + intervalTime]
resoult2Index = time1 + time2
#print(type(resres1),type(resres2))
#重新组合成Series类型的拟合和预测结果
resoult2 = pd.Series(resoult1,index=resoult2Index)
#分成拟合集和预测集
DES_train = resoult2[:-1]
DES_forecast = resoult2[-2:]



#导入scipy.optimize
#from scipy.optimize import minimize
#导入双指数平滑目标函数
#from '../desTargetFunction.py' import minimize



#预测值  MAPE\RMSE 
rmse2 = round(np.sqrt(sum((DES_forecast-testSeData[0:2])**2)/testSeData[0:2].size),2)
mape2 = round(mean_absolute_percentage_error(DES_forecast,testSeData[0:2]),5)*100
mape2 = ("%.2f" % mape2)
print(rmse2,mape2)





plt.figure(figsize=(20, 8))
plt.xlabel('time')
plt.ylabel('$\mu$g/m3')
#未拟合序列
plt.plot(trainSeData, label = "Fitting",\
         linestyle="--",\
         color="#000",\
         marker='o')
#测试序列
plt.plot(testSeData,color='grey',label='Actual',marker='o')
#拟合序列
plt.plot(DES_train,label="Alpha {}, beta {}".format(0.9, 0.02),color='orange')
#预测序列
plt.plot(DES_forecast,color='#03FE09',label='Predict',marker='s')
#分割线
plt.vlines(DES_train.index[-1], 20, 120, colors = "grey", linestyles = "dashed")
plt.legend(loc="best")
plt.grid(False)
plt.legend(loc='upper left')
plt.title('RMSE(MAPE) Of Double Exponential Smoothing Predict Resoult: {r}({m}%)'.format(r = rmse2,m = mape2))


#plt.savefig(r'../data/des-forcast-1.svg',dpi=600)
#plt.savefig(r'../data/0.02des-forcast-9-j.svg',dpi=600)



def plotDoubleExponentialSmoothing(series, alphas, betas):
    """
        Plots double exponential smoothing with different alphas and betas
        
        series - dataset with timestamps
        alphas - list of floats, smoothing parameters for level
        betas - list of floats, smoothing parameters for trend
    """
    
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(20, 8))
        for alpha in alphas:
            for beta in betas:
                
                resoult1 = double_exponential_smoothing(series, alpha, beta)
                #生成预测值的所有时间戳索引
                time1 = list(trainSeData.index)
                time2 = [trainSeData.index[-1] + intervalTime]
                resoult2Index = time1 + time2
                #print(type(resres1),type(resres2))
                #重新组合成Series类型的拟合和预测结果
                resoult2 = pd.Series(resoult1,index=resoult2Index)
                #分成拟合集和预测集
                DES_train = resoult2[:-1]
                rmse2_fitting = round(np.sqrt(sum((DES_train-series)**2)/series.size),2)
                mape2_fitting = round(mean_absolute_percentage_error(DES_train,series),5)*100
                mape2_fitting = ("%.2f" % mape2_fitting)
#                print(rmse2_fitting,mape2_fitting)

                plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}, RMSE {}, MAPE {}%".format(alpha, beta, rmse2_fitting, mape2_fitting))



        plt.plot(series.values, label = "Actual",color='black',marker='o',linestyle="--")
        plt.legend(loc="best")

        plt.title("Double Exponential Smoothing")
        plt.grid(False)#网格线
#        plt.savefig(r'../data/desFitting.svg',dpi=600)

#plotDoubleExponentialSmoothing(currency.GEMS_GEMS_SPENT, alphas=[0.9, 0.02], betas=[0.9, 0.02])
plotDoubleExponentialSmoothing(trainSeData, alphas=[0.9, 0.02], betas=[0.9, 0.02])
































'''

#三指数平滑模型   Holt-Winters模型
class HoltWinters:
    
    
    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
        #传入的是pandas的Series类型
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        
        
    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i+self.slen] - self.series[i]) / self.slen
        return sum / self.slen  
    
    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series)/self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals   

          
    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []
        
        seasonals = self.initial_seasonal_components()
        
        for i in range(len(self.series)+self.n_preds):
            if i == 0: # components initialization
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i%self.slen])
                
                self.PredictedDeviation.append(0)
                
                self.UpperBond.append(self.result[0] + 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                
                self.LowerBond.append(self.result[0] - 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                continue
                
            if i >= len(self.series): # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m*trend) + seasonals[i%self.slen])
                
                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01) 
                
            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]
                self.result.append(smooth+trend+seasonals[i%self.slen])
                
                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) 
                                               + (1-self.gamma)*self.PredictedDeviation[-1])
                     
            self.UpperBond.append(self.result[-1] + 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.LowerBond.append(self.result[-1] - 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i%self.slen])


'''
#现在，咱们知道了时间序列的数据，
#交叉验证集应该怎么划分。
#接下来开始动手找出 Holt-Winters 模型在玩家每小时的广告浏览量数据集中的最佳参数，
#我们根据常识可知，这个数据集中，
#存在一个明显的季节性变化，变化周期为24小时，
#因此我们设置 slen = 24 :
'''
from sklearn.model_selection import TimeSeriesSplit

def timeseriesCVscore(params, series, loss_function=mean_squared_error, slen=12):

    for train, test in tscv.split(values):

        model = HoltWinters(series=values[train], slen=slen, 
                            alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()
        
        predictions = model.result[-len(test):]
        actual = values[test]
        error = loss_function(predictions, actual)
        errors.append(error)
        
    return np.mean(np.array(errors))


'''
#在 Holt-Winters 模型以及其他指数平滑模型中
#对平滑参数的大小有一个限制，每个参数都在0到1之间。
#因此我们必须选择支持模型参数约束的最优化算法，
#在这里，我们使用 Truncated Newton conjugate gradient (截断牛顿共轭梯度法)
'''

#6过拟合
Data = timedata['PM2.5浓度'][:-20] 
slen = 22 # 30-day seasonality

x = [0, 0, 0] 

opt = minimize(timeseriesCVscore, x0=x, 
               args=(Data, mean_absolute_percentage_error, slen), 
               method="TNC", bounds = ((0, 1), (0, 1), (0, 1))
              )

alpha_final, beta_final, gamma_final = opt.x
print(alpha_final, beta_final, gamma_final)
model = HoltWinters(Data, slen = slen, 
                    alpha = alpha_final, 
                    beta = beta_final, 
                    gamma = gamma_final, 
                    n_preds = 2, scaling_factor = 3)
model.triple_exponential_smoothing()




#将上面训练后得到的最优参数组合（三个平滑系数），绘制图形：

def plotHoltWinters(series, plot_intervals=False, plot_anomalies=False):

    plt.figure(figsize=(20, 10))
    plt.plot(model.result, label = "Model")
    plt.plot(series.values, label = "Actual")
    error = mean_absolute_percentage_error(series.values, model.result[:len(series)])
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    
    if plot_anomalies:
        anomalies = np.array([np.NaN]*len(series))
        anomalies[series.values<model.LowerBond[:len(series)]] = \
            series.values[series.values<model.LowerBond[:len(series)]]
        anomalies[series.values>model.UpperBond[:len(series)]] = \
            series.values[series.values>model.UpperBond[:len(series)]]
        plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    if plot_intervals:
        plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
        plt.plot(model.LowerBond, "r--", alpha=0.5)
        plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond, 
                         y2=model.LowerBond, alpha=0.2, color = "grey")    
        
    plt.vlines(len(series), ymin=min(model.LowerBond), ymax=max(model.UpperBond), linestyles='dashed')
    plt.axvspan(len(series)-20, len(model.result), alpha=0.3, color='lightgrey')

    plt.legend(loc="best", fontsize=13);
    plt.grid(False)#网格线
    
plotHoltWinters(data['PM2.5浓度'], plot_intervals=True, plot_anomalies=False)



'''



















#分解趋势季节和残余
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(trainSeData.values,freq=12,model="additive")

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.figure(figsize=(12,8))
fig = decomposition.plot()
#plt.savefig(r'../data/fenjie.svg',dpi=600)







#用自相关图判断合理差分阶数
df = trainSeData

fig = plt.figure(figsize=(12,8))


ax1 = fig.add_subplot(3,2,1)
plt.title('Original Series')
plt.xticks(rotation=30)
ax1 = plt.plot(df.dropna())

ax2 = fig.add_subplot(3,2,2)
fig = plot_acf(df,ax=ax2)


ax3 = fig.add_subplot(3,2,3)
plt.title('1st Order Differencing')
plt.xticks(rotation=30)
ax3 = plt.plot(df.diff().dropna())

ax4 = fig.add_subplot(3,2,4)
fig = plot_acf(df.diff().dropna(),ax=ax4)


ax5 = fig.add_subplot(3,2,5)
plt.title('2st Order Differencing')
plt.xticks(rotation=30)
ax5 = plt.plot(df.diff().diff().dropna().diff())

ax6 = fig.add_subplot(3,2,6)
fig = plot_acf(df.diff().diff().dropna(),ax=ax6)

plt.tight_layout()

#plt.savefig(r'../data/用自相关图判断合理差分阶数.svg',dpi=600)

























fig = plt.figure(figsize=(12,8))
fig.add_subplot(2,1,1)

ax2 = fig.add_subplot(2,1,1)
plt.title('Autocorrelation Of Training Data')
fig = plot_acf(seData,ax=ax2)

ax3 = fig.add_subplot(2,1,2)
plt.title('Partial Autocorrelation Of Training Data')
fig = plot_pacf(seData, ax=ax3)












#ADF_ACF_PACF_PLOT(trainSeData,'Training Data',True)



'''


#改变时间索引格式
def index2datetime(serie):
    #print(serie,type(serie))
    timeIndex = datetime.strptime(serie,'%Y-%m-%d %H:%M:%S')
    return timeIndex

'''
#改变行名的第一种方法
#dfData.index = dfData.index.map(index2datetime)
#print(dfData.index[0].month,type(dfData.index[0]))


#改变行名的第二种方法
# 是否替换原时间序列 inplace=True
#dateIndexdfData = dfData.rename(index2datetime, axis='0')
#print(type(dateIndexdfData.index[0]),dateIndexdfData.index[0].month)

'''
def creatNewDic(dic,key,value):
    #根据采集时间重新创建字符串
    #再重新生成datetime格式的索引
    year = key.year
    month = key.month
    day = key.day
    hour = key.hour
    minute = key.minute
    timeStr = '{T}-{m}-{d} {H}:{M}'.format(T = year,m = month,d = day,H = hour,M = minute)
    dataTimeStr = datetime.strptime(timeStr,'%Y-%m-%d %H:%M')
    #只取到datetime的同一分钟内创建相同的key
    #同一分钟内测得的值成为value，列表形式存放
    if not(dataTimeStr in dic):
        dic[dataTimeStr] = []
        dic[dataTimeStr].append(value)
    else:
        dic[dataTimeStr].append(value)

def createNewdfData(dic):
    #遍历字典一分钟内采集的所有值
    #取平均数为该分钟内的代表值
    for key, value in dic.items():
        #取平均并且保留二位小数
        dic[key] = round(np.mean(value),2)
 
    # 第一种
    timeSeriesdfData = pd.DataFrame(dic,index=['PM2.5浓度']).T
    # 第二种
    #timeSeriesdfData = pd.DataFrame.from_dict(dic,orient='index')
    return timeSeriesdfData

def handleDataTimeIndex(dateIndexData):
    dataIndexList = dateIndexData.index
    newDic = {}
    for i in dataIndexList:
        #拿到行的时间索引
        timeKey = i
        #拿到对应时间测得的值
        timeValue = dateIndexData.loc[str(i)].values[0][0]
        #把同一分钟内采集到的值取平均，形成每分钟内具有唯一平均值的字典
        creatNewDic(newDic,timeKey,timeValue)
    #根据字典创建新的DataFrame格式时间序列数据
    newdfData = createNewdfData(newDic)
    return newdfData





#请求mixiot平台数据，返回dataframe格式
from tool import getMixiotData 
#Dataframe
dfData = getMixiotData(dataType='S08',num=1000)




#把DataFrame数据的行索引转成datetime格式
dateIndexdfData = dfData.rename(index = index2datetime)

#再转换成df格式
timeSeriesdfData = handleDataTimeIndex(dateIndexdfData)
#Series
seData=timeSeriesdfData['PM2.5浓度']




trainSeData, testSeData = split2trainAndTest(seData,20)

ADF_ACF_PACF_PLOT(trainSeData,'Training data')
'''




# 平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF

#ADF检验：ADF检验的 H0 假设就是存在单位根（即不平稳），
#如果得到的pvalue小于三个置信度(10%，5%，1%)，
#则对应有(90%，95，99%)的把握来拒绝原假设。
#返回值意义为：
#    adf （float）ADF Test result 测试统计
#    pvalue （float）MacKinnon基于MacKinnon的近似p值（1994年，2010年）
#    usedlag （int）使用的滞后数量
#    nobs（ int）用于ADF回归的观察数和临界值的计算
#    critical values（dict）测试统计数据的临界值为1％，5％和10％。基于MacKinnon（2010）
#    icbest（float）如果autolag不是None，则最大化信息标准。
#    resstore （ResultStore，可选）一个虚拟类，其结果作为属性附加

#————————————————
#1%、%5、%10不同程度拒绝原假设的统计值和ADF Test result的比较
#ADF Test result同时小于1%、5%、10%即说明非常好地拒绝该假设
#本数据中，adf结果如果大于三个level的统计值。
#则是不平稳的，需要进行一阶差分后，再进行检验。


#print(u'原始序列的ADF检验结果为：', ADF(seData))
print(u'The ADF test result of training data is:', ADF(trainSeData))


def getADF(data):
    #dfdata
    #adfSeries = list(ADF(data[u'PM2.5浓度']))
    #sedata
    adfSeries = list(ADF(data))
    return adfSeries
# 得到时序平稳之后所经过的差分次数
def getStationarityTime(data):
    #使用前定义dfdata
    dfdata = ''
    index = 0
    Data = data
    #拿到ADF检验总结果：列表
    adf = getADF(Data)
    #拿到时序的ADF假设检验结果
    adfResoult = adf[0]
    #拿到结果判定的标准
    comparDic = adf[4]['5%']
    print('first',adfResoult,comparDic ,adfResoult > comparDic)
    for i in range(10):
        #先判断是否是平稳序列再进行差分
        if (adfResoult > comparDic):
            #大于5%对应的值则说明序列平稳（拒绝原假设）的概率小于95%
            index = index + 1
            Data = Data.diff(index).dropna()
            adf = getADF(Data)
            adfResoult = adf[0]
            comparDic = adf[4]['5%']
            #print('er',index,adfResoult,comparDic)
        else:
            #print('else',index,adfResoult,comparDic)
            break
    if index >= 1:
        dfdata = data.diff(index).dropna()
    return dfdata , index



#difData , difTime = getStationarityTime(timedata)
difData , difTime = getStationarityTime(trainSeData)



print('时间序列需要差分{t}次后才是平稳序列'.format(t = difTime))
#print(difData)

#difTime = 1
#D_data = timedata.diff(difTime).dropna()

difTime = 1
D_data = trainSeData.diff(difTime).dropna()
#D_data = trainSeData


print(u'The ADF test result of D_ata is:', ADF(D_data))







#ADF_ACF_PACF_PLOT(D_data,'1st diff',False)



fig = plt.figure(figsize=(12,8))
fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,1)
plt.title('Autocorrelation Of Training Data')
fig = plot_acf(D_data,ax=ax2)
ax3 = fig.add_subplot(2,1,2)
plt.title('Partial Autocorrelation Of Training Data')
fig = plot_pacf(D_data, ax=ax3)
#plt.savefig(r'../data/trainSeData-acfpacf.svg',dpi=600)




#ADF_ACF_PACF_PLOT(D_data.diff().dropna(),'Training data after 2nd difference transformation')





'''
pValue = sm.tsa.stattools.adfuller(D_data)[1]
fig = plt.figure(figsize=(12,8))
fig.add_subplot(2,1,1)
plt.title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(pValue))
line1, = plt.plot(D_data,label = 'Training data after 1st difference transformation')#color = 'blue'
plt.legend(loc="upper left")
plt.grid(False)
plt.legend(handles=[line1,])
plt.xlabel('time')
plt.ylabel('mg/m^-3')
ax2 = fig.add_subplot(2,2,3)
plt.title('Autocorrelation Of Training Data')
fig = plot_acf(D_data,ax=ax2)
ax3 = fig.add_subplot(2,2,4)
plt.title('Partial Autocorrelation Of Training Data')
fig = plot_pacf(D_data, ax=ax3)
plt.savefig(r'../data/diff1adfacfpacf.svg',dpi=600)
'''


















# 白噪声检验

'''
#首先假设序列为白噪声，根据假设求得的P值如果小于阈值（一般为5%）
#那么假设不成立；反之，假设成立。
#检验结果的第二项为P值，这里为95.57%，远高于阈值5%，
#因此假设是成立的，我们的序列为白噪声序列（实际是随机序列）。


from statsmodels.stats.diagnostic import acorr_ljungbox
LjungBoxTestResoult = list(acorr_ljungbox(D_data,lags=10,return_df = False))[1][0]
#返回统计量和p值
print(u'序列的白噪声检验结果为：',\
      acorr_ljungbox(D_data, lags=10,return_df = False))
print('时间序列的白噪声检验结果中基于卡方分布的p统计量:\
      {resoult} 是否小于0.05（即是否拒绝原假设H0，序列相关，非白噪声）：{key}'\
      .format(resoult = LjungBoxTestResoult ,key = LjungBoxTestResoult < 0.05))


'''





















'''
##定阶数
#符合0，1的AIC和BIC计算函数
train_results = sm.tsa.arma_order_select_ic(D_data, ic=['aic', 'bic'], trend='nc', max_ar=4, max_ma=4)
 
print('the AIC is', train_results.aic_min_order)
print('the BIC is', train_results.bic_min_order)


#信息准则定阶：AIC、BIC、HQIC
#AIC
AIC = sm.tsa.arma_order_select_ic(D_data,\
    max_ar=6,max_ma=4,ic='aic')['aic_min_order']
#BIC
BIC = sm.tsa.arma_order_select_ic(D_data,max_ar=6,\
    max_ma=4,ic='bic')['bic_min_order']
#HQIC
HQIC = sm.tsa.arma_order_select_ic(D_data,max_ar=6,\
    max_ma=4,ic='hqic')['hqic_min_order']
print('the AIC is{},\nthe BIC is{}\n the HQIC is{}'.format(AIC,BIC,HQIC))


#arimaMod4.summary()
arimaMod1 =ARIMA(D_data,order=(4,0,0)).fit()
print(111,arimaMod1.aic,arimaMod1.bic,arimaMod1.hqic)
arimaMod2 =ARIMA(D_data,order =(0,0,1)).fit()
print(222,arimaMod2.aic,arimaMod2.bic,arimaMod2.hqic)
'''




'''
#AIC
#4.建立模型——参数选择
#遍历，寻找适宜的参数
import itertools
 
p_min = 0
d_min = 0
q_min = 0
p_max = 8
d_max = 0
q_max = 8
 
# Initialize a DataFrame to store the results,，以AIC准则
results_aic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                           columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])
 
for p,d,q in itertools.product(range(p_min,p_max+1),
                               range(d_min,d_max+1),
                               range(q_min,q_max+1)):
    if p==0 and d==0 and q==0:
        results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
 
    try:
        model = ARIMA(D_data, order=(p, d, q),
                               #enforce_stationarity=False,
                               #enforce_invertibility=False,
                              )
        results = model.fit()
        results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.aic
    except:
        continue
results_aic = results_aic[results_aic.columns].astype(float)

fig, ax = plt.subplots(figsize=(12, 8))
ax = sns.heatmap(results_aic,
                 mask=results_aic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 )
ax.set_title('AIC')
#plt.savefig(r'../data/AIC.svg',dpi=600)
plt.show()
'''




'''

#BIC

#4.建立模型——参数选择
#遍历，寻找适宜的参数
import itertools
 
p_min = 0
d_min = 0
q_min = 0
p_max = 8
d_max = 0
q_max = 8
 
# Initialize a DataFrame to store the results,，以BIC准则
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                           columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])
 
for p,d,q in itertools.product(range(p_min,p_max+1),
                               range(d_min,d_max+1),
                               range(q_min,q_max+1)):
    if p==0 and d==0 and q==0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
 
    try:
        model = ARIMA(D_data, order=(p, d, q),
                               #enforce_stationarity=False,
                               #enforce_invertibility=False,
                              )
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)

fig, ax = plt.subplots(figsize=(12, 8))
ax = sns.heatmap(results_bic,
                 mask=results_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 )
ax.set_title('BIC')
plt.savefig(r'../data/BIC.svg',dpi=600)
plt.show()
'''








































#主要针对残差进行正态性检验和自相关性检验。
#残差满足正态性，主要是为了残差集中于某一个数值，如果该值与0很接近，则它实际服从均值为0的正态分布，即它是一个白噪声。白噪声是指功率谱密度在整个频域内均匀分布的噪声。白噪声或白杂讯，是一种功率频谱密度为常数的随机信号或随机过程。换句话说，此信号在各个频段上的功率是一样的，由于白光是由各种频率（颜色）的单色光混合而成，因而此信号的这种具有平坦功率谱的性质被称作是“白色的”，此信号也因此被称作白噪声。
#残差满足非自相关性，主要是为了在残差中不再包括AR或者MA过程产生的序列。
#正态性检验可以使用shapiro.test函数来检查，当p-value>0.05时表明满足正态分布，该值越大越好，直到接近于1.
#残差的自相关性可以用函数tsdiag(model)来迅速检验。该函数会列出残差的散点图，自相关性acf检验和Box.test的检验值（pvalue大于0.05即满足非自相关性）。
#为什么残差要是白噪声？
#得到白噪声序列，就说明时间序列中有用的信息已经被提取完毕了，剩下的全是随机扰动，是无法预测和使用的，残差序列如果通过了白噪声检验，则建模就可以终止了，因为没有信息可以继续提取。如果残差不是白噪声，就说明残差中还有有用的信息，需要修改模型或者进一步提取。


#检验残差自相关图偏自相关图
arma_mod = ARIMA(D_data,order=(4,1,0)).fit()
resid = arma_mod.resid 
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(),lags=10,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid,lags=10,ax=ax2)
#plt.savefig(r'../data/410resid-acfpacf.svg',dpi=600)
arma_mod2 = ARIMA(D_data,order=(0,1,1)).fit()
resid2 = arma_mod2.resid
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid2.values.squeeze(),lags=10,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid2,lags=10,ax=ax2)
#plt.savefig(r'../data/011resid-acfpacf.svg',dpi=600)

'''
#（主要看文字）
#这里的模型检验主要有两个：
#1）检验参数估计的显著性（t检验）
#2）检验残差序列的随机性，即残差之间是独立的
#残差序列的随机性可以通过自相关函数法来检验，即做残差的自相关函数图：
model = sm.tsa.ARIMA(D_data, order=(0,1,1))
results = model.fit()
resid = results.resid #赋值
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=10)
plt.show()
'''



#进一步进行D-W检验，德宾-沃森（Durbin-Watson）检验。德宾-沃森检验,简称D-W检验，是目前检验自相关性最常用的方法，但它只使用于检验一阶自相关性。因为自相关系数ρ的值介于-1和1之间，所以 0≤DW≤４。并且DW＝O＝＞ρ＝１　　 即存在正自相关性 
#DW＝４＜＝＞ρ＝－１　即存在负自相关性 
#DW＝２＜＝＞ρ＝０　　即不存在（一阶）自相关性 
#因此，当DW值显著的接近于O或４时，则存在自相关性，而接近于２时，则不存在（一阶）自相关性。这样只要知道ＤＷ统计量的概率分布，在给定的显著水平下，根据临界值的位置就可以对原假设Ｈ０进行检验。

print(sm.stats.durbin_watson(arma_mod.resid.values))





#qq图
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)
#plt.savefig(r'../data/qq.svg',dpi=600)




'''

#3.5 残差序列Ljung-Box检验，也叫Q检验
r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

'''




'''
#残差
#利用QQ图检验残差是否满足正态分布
plt.figure(figsize=(12,8))
qqplot(resid,line='q',fit=True)
#利用D-W检验,检验残差的自相关性
print('D-W检验值为{}'.format(durbin_watson(resid.values)))
'''



'''
我们再次通过计算MSE量化我们预测的预测性能：
# Extract the predicted and true values of our time-series
y_forecasted = pred_dynamic.predicted_mean
y_truth = y['1998-01-01':]
 
# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
'''







from statsmodels.stats.diagnostic import acorr_ljungbox
LjungBoxTestResoult = list(acorr_ljungbox(resid,lags=10,return_df = False))[1][0]
#返回统计量和p值
print(u'序列的白噪声检验结果为：',\
      acorr_ljungbox(resid, lags=10,return_df = False))
print('410参数的残差，时间序列的残差白噪声检验结果中基于卡方分布的p统计量:\
      {resoult} 是否大于0.05（小于0.05就拒绝白噪声的假设，大于0.05就接受白噪声假设）：{key}'\
      .format(resoult = LjungBoxTestResoult ,key = LjungBoxTestResoult > 0.05))


arma_mod2 = ARIMA(D_data,order=(0,1,1)).fit()
resid2 = arma_mod2.resid 
LjungBoxTestResoult = list(acorr_ljungbox(resid2,lags=10,return_df = False))[1][0]
#返回统计量和p值
print(u'序列的白噪声检验结果为：',\
      acorr_ljungbox(resid2, lags=10,return_df = False))
print('011参数的残差，时间序列残差的白噪声检验结果中基于卡方分布的p统计量:\
      {resoult} 是否大于0.05（小于0.05就拒绝白噪声的假设，大于0.05就接受白噪声假设）：{key}'\
      .format(resoult = LjungBoxTestResoult ,key = LjungBoxTestResoult > 0.05))





#r,q,p = sm.tsa.acf(resid.values.squeeze(),qstat=True)
#data2 = np.c_[range(1,41), r[1:], q, p]
#table= pd.DataFrame(data2, columns=[ 'lag','AC','Q','Prob(>Q)'])
#print(table.set_index('lag'))








#残差综合评价

fig = arma_mod.plot_diagnostics(figsize=(12,8))
#plt.savefig(r'../data/zonghepingjia.svg',dpi=600)










'''
#%matplotlib qt5
model = sm.tsa.ARIMA(timedata, order=(1, 0, 0))
results = model.fit()
predict_sunspots = results.predict(start=str('2020-05-22 10:46:00'),end=str('2020-05-22 10:59:00'),dynamic=False)
print(predict_sunspots)
fig, ax = plt.subplots(figsize=(12, 8))
ax = timedata.plot(ax=ax)
predict_sunspots.plot(ax=ax)
plt.show()

'''





















def string2Datetime(string):
    return  datetime.strptime(string, "%Y-%m-%d %H:%M:%S")

#Series类型 pred[4:5],左闭右开[4,5),1个数据

def minNums(startTime, endTime):
    total_seconds = (endTime - startTime).total_seconds()
    # 来获取准确的时间差，并将时间差转换为秒
    print(total_seconds)
    mins = total_seconds / 60
    return int(mins+1)


def forecastData(trainData,time,orde):
    preIndex = []
    preData = []
    temSerise = trainData.copy()
    for i in range(time) :
        d_data = temSerise.diff().dropna()

        model = ARIMA(d_data, order = orde).fit()
        modelResoult = model.predict(start = len(d_data),\
                                     end = len(d_data)+1,\
                                     dynamic=False,\
                                     typ="linear")[0]
        recoverValue = temSerise.shift().dropna()[temSerise.index[-3]]
        actualValue = recoverValue + modelResoult
        
        #生成一个datetime格式的一分钟间隔，用于计算
        startTimeStr = "1924-2-05 00:00:00"
        endTimeStr = "1924-2-05 00:01:00"
        startTime = datetime.strptime(startTimeStr,'%Y-%m-%d %H:%M:%S')
        endTime = datetime.strptime(endTimeStr,'%Y-%m-%d %H:%M:%S')
        intervalTime = endTime - startTime
        
        preTimeIndex = temSerise.index[-5] + intervalTime
         
        preIndex.append(preTimeIndex)
        preData.append(actualValue)
        temSerise[temSerise.index[-1] + intervalTime] = actualValue
        preResoult = pd.Series(preData,index = preIndex)[3:6]

    return preResoult





def getRecover(data,orde):
    
    D_data = data.diff().dropna()
    
    arima_model = ARIMA(D_data,order = orde) #ARIMA模型
    result = arima_model.fit()
    pred = result.predict(dynamic=False,typ="linear")#levels  linear
    idx = pd.date_range(start = D_data.index[1],end = D_data.index[-1],freq='T')
    predList= []
    intervalTime = minNums(D_data.index[1],D_data.index[-1])
    for i in range(intervalTime):
        predList.append(np.array(pred)[i])
    predSeries = pd.Series(np.array(predList),index=idx)
    

    diff_shift = trainSeData.shift(1).dropna()
    diffShiftValues = diff_shift.values
    diffShiftIndex = diff_shift.index
    diff_shift_ts = pd.Series(diffShiftValues,index = diffShiftIndex)
    diff_recover_1 = predSeries.add(diff_shift_ts).dropna()
    recoverModel = diff_recover_1.shift(-1).dropna()
    
    trainTimeSeries = trainSeData[D_data.index[1]:D_data.index[-1]]


    RMSE = np.sqrt(sum((recoverModel-trainTimeSeries[:-1])**2)/trainTimeSeries[:-1].size)
    
    MAPE = mean_absolute_percentage_error(trainTimeSeries[:-1],recoverModel)
    
    R2 = r2_score(trainTimeSeries[:-1],recoverModel)
    print('RMSE,MAERMSE,MAERMSE,MAERMSE,MAERMSE,MAE',RMSE,MAPE,R2)


    return recoverModel,RMSE,MAPE





#getRecover(trainSeData,orde = (0,1,1))
#getRecover(trainSeData,orde = (4,1,0))
#getRecover(trainSeData,orde = (0,1,1))[0].plot()
#getRecover(trainSeData,orde = (4,1,0))[0].plot()
#








#ARIMA model simulations (blue line) and observations (red line)
#拟合有置信区间
def plotModel(data,orde,key):
    
    fit4 , Rmse , MAPE = getRecover(data,orde = (0,1,1))
    
    predValue = forecastData(data,time = 10,orde = orde)
    D_data = data.diff().dropna()
    
    arima_model = ARIMA(D_data,order = orde) #ARIMA模型
    result = arima_model.fit()
    pred = result.predict(dynamic=False,typ="linear")#levels  linear
    idx = pd.date_range(start = D_data.index[1],end = D_data.index[-1],freq='T')
    predList= []
    intervalTime = minNums(D_data.index[1],D_data.index[-1])
    for i in range(intervalTime):
        predList.append(np.array(pred)[i])
    predSeries = pd.Series(np.array(predList),index=idx)
    

    diff_shift = trainSeData.shift(1).dropna()
    diffShiftValues = diff_shift.values
    diffShiftIndex = diff_shift.index
    diff_shift_ts = pd.Series(diffShiftValues,index = diffShiftIndex)
    diff_recover_1 = predSeries.add(diff_shift_ts).dropna()
    recoverModel = diff_recover_1.shift(-1).dropna()
    if(key):
        recoverModel = diff_recover_1.shift(-1).dropna().append(predValue[:2])
        
    trainTimeSeries = trainSeData[D_data.index[1]:D_data.index[-1]]  # 过滤没有预测的记录
    fig=plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(111)
    
    #坐标轴刻度修改
    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M'))
    x_major_locator=MultipleLocator(0.003)
    ax1.xaxis.set_major_locator(x_major_locator)
    
    plt.plot(trainTimeSeries,color='#000',label='Training',marker='o')
    
    #rmse1 = '%.3f' %float(np.sqrt(sum((recoverModel-trainTimeSeries[:-1])**2)/trainTimeSeries[:-1].size))
    #rmse2 = '%.3f' %float(Rmse)
    #print(rmse1,Rmse)
    
    
    rmse2 = round(np.sqrt(sum((predValue[-2:]-testSeData[1:3])**2)/testSeData[1:3].size),3)

    
    mape2 = round(mean_absolute_percentage_error(predValue[-2:],testSeData[1:3]),5)*100
    mape2 = ("%.2f" % mape2)

    #mape2 = int(mape2*1000+0.5)/1000
    print("++++++++++++++++++++++++++++++++++",rmse2,mape2,type(mape2))

    
    
    if(key):
        plt.plot(recoverModel[:-1],color='#FE0100',marker='D',label='ARIMA{o}'.format(o = orde))
        plt.plot(predValue,color='#03FE09',label='Predict',marker='s')
        plt.plot(testSeData,color='grey',label='Actual',marker='o')
        plt.vlines(recoverModel.index[-2], 20, 120, colors = "grey", linestyles = "dashed")
    else:
        
        plt.plot(fit4,color='#0304FD',marker='^',label='ARIMA{o}'.format(o = (0,1,1)))
        plt.plot(recoverModel,color='#FE0100',marker='D',label='ARIMA{o}'.format(o = orde))
    
    if(key):
        plt.title('RMSE(MAPE) Of Predict Resoult: {r}({m}%)'.format(r = rmse2,m = mape2))
    else:
        plt.title('Fitting Model')
    plt.grid(False)#网格线
    
#    window = 2
#    scale = 1.96
#    rolling_mean = recoverModel[:-1].rolling(window).mean()
#    mae = mean_absolute_error(recoverModel[:-1][window:], rolling_mean[window:])
#    deviation = np.std(recoverModel[:-1][window:] - rolling_mean[window:])
#    lower_bond = rolling_mean - (mae + scale * deviation)
#    upper_bond = rolling_mean + (mae + scale * deviation)
#    plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
#    plt.plot(lower_bond, "r--")
    
    #mape2 = mean_absolute_percentage_error(predValue[-2:],testSeData[1:3])
    #rmse2 = np.sqrt(sum((predValue[-2:]-testSeData[1:3])**2)/testSeData[1:3].size)
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$',mape2,rmse2)
    

    confidient = arma_mod.get_prediction(start = len(D_data),\
                     end = len(D_data)+1,\
                     dynamic=False,\
                     typ="linear").conf_int()
    
    lower_bond = []
    upper_bond = []
    
    preIdx = list(predValue.index)
    
    preUpConf1 = confidient.loc[preIdx[1]]['upper PM2.5浓度']
    preLowConf1 = confidient.loc[preIdx[1]]['lower PM2.5浓度']
    preUpConf2 = confidient.loc[preIdx[2]]['upper PM2.5浓度']
    preLowConf2 = confidient.loc[preIdx[2]]['lower PM2.5浓度']
    
    
    
    preUp1 = predValue.loc[preIdx[1]] + preUpConf1
    preLow1 = predValue.loc[preIdx[1]] + preLowConf1
    preUp2 = predValue.loc[preIdx[2]] + preUpConf2
    preLow2 = predValue.loc[preIdx[2]] + preLowConf2
    
    
    
    upper_bond.append(preUp1)
    upper_bond.append(preUp2)
    lower_bond.append(preLow1)
    lower_bond.append(preLow2)
    
    confiIndex = [preIdx[1],preIdx[2]]
    upSe = pd.Series(upper_bond,index = confiIndex)
    lowSe = pd.Series(lower_bond,index = confiIndex)

    if(key):
#        window = 2
#        scale = 1.96
#        rolling_mean = predValue.rolling(window).mean()
#        mae = mean_absolute_error(predValue[1:], rolling_mean[1:])
#        deviation = np.std(predValue[1:] - rolling_mean[1:])
#        lower_bond = rolling_mean - (mae + scale * deviation)
#        upper_bond = rolling_mean + (mae + scale * deviation)
#        plt.plot(upper_bond,color='#621716',linestyle="--", label="Upper Bond / Lower Bond")
#        plt.plot(lower_bond,color='#621716',linestyle="--")
        plt.plot(upSe,color='#621716',linestyle="--", label="Upper Bond / Lower Bond")
        plt.plot(lowSe,color='#621716',linestyle="--")



#    ax = plt.gca()
#    ax.spines['right'].set_color('none')
#    ax.spines['top'].set_color('none')
    
    
    plt.legend(loc='upper left')
    

    from statsmodels.stats.diagnostic import acorr_ljungbox
    arma_mod1 = ARIMA(D_data,order=(4,1,0)).fit()
    resid1 = arma_mod1.resid 
    LjungBoxTestResoult = list(acorr_ljungbox(resid1,lags=10,return_df = False))[1][0]
    #返回统计量和p值
    print(u'序列的白噪声检验结果为：',\
          acorr_ljungbox(resid1, lags=10,return_df = False))
    print('410参数的残差，时间序列的残差白噪声检验结果中基于卡方分布的p统计量:\
          {resoult} 是否大于0.05（小于0.05就拒绝白噪声的假设，大于0.05就接受白噪声假设）：{key}'\
          .format(resoult = LjungBoxTestResoult ,key = LjungBoxTestResoult > 0.05))
    
    
    arma_mod2 = ARIMA(D_data,order=(0,1,1)).fit()
    resid2 = arma_mod2.resid 
    LjungBoxTestResoult = list(acorr_ljungbox(resid2,lags=10,return_df = False))[1][0]
    #返回统计量和p值
    print(u'序列的白噪声检验结果为：',\
          acorr_ljungbox(resid2, lags=10,return_df = False))
    print('011参数的残差，时间序列残差的白噪声检验结果中基于卡方分布的p统计量:\
          {resoult} 是否大于0.05（小于0.05就拒绝白噪声的假设，大于0.05就接受白噪声假设）：{key}'\
          .format(resoult = LjungBoxTestResoult ,key = LjungBoxTestResoult > 0.05))


    print('//////////////////////////////////////',predValue[-2:].index[-1])

    
#    plt.savefig(r'../data/410-forcast-14.svg',dpi=600)
    
    return



#plotModel(trainSeData,orde = (4,1,0),key = True)





#plotModel(trainSeData,orde = (0,1,5),key = False)
#plotModel(trainSeData,orde = (2,1,5),key = False)
#plotModel(trainSeData,orde = (1,1,4),key = False)
#plotModel(trainSeData,orde = (4,1,0),key = False)
#
#plotModel(trainSeData,orde = (0,1,1),key = False)
#plotModel(trainSeData,orde = (1,1,1),key = False)
#plotModel(trainSeData,orde = (0,1,2),key = False)
#
#
#plotModel(trainSeData,orde = (4,1,1),key = True)
#plotModel(trainSeData,orde = (0,1,1),key = True)


##450起取数
#plotFitModel(D_data,orde = (2,1,1),key = False)
#plotFitModel(D_data,orde = (2,1,3),key = False)
#plotFitModel(D_data,orde = (3,1,1),key = False)
#plotFitModel(D_data,orde = (0,1,1),key = False)
#plotFitModel(D_data,orde = (0,1,1),key = True)







'''
#拟合的置信区间??
confidient = arma_mod.get_prediction(start = len(D_data),\
                 end = len(D_data)+1,\
                 dynamic=False,\
                 typ="linear").conf_int()
preResoult = forecastData(trainSeData,time = 10,orde = (4,1,0))

lower_bond = []
upper_bond = []

preIdx = list(preResoult.index)

preUpConf1 = confidient.loc[preIdx[1]]['upper PM2.5浓度']
preLowConf1 = confidient.loc[preIdx[1]]['lower PM2.5浓度']
preUpConf2 = confidient.loc[preIdx[2]]['upper PM2.5浓度']
preLowConf2 = confidient.loc[preIdx[2]]['lower PM2.5浓度']

preUp1 = preResoult.loc[preIdx[1]] + preUpConf1
preLow1 = preResoult.loc[preIdx[1]] + preLowConf1
preUp2 = preResoult.loc[preIdx[2]] + preUpConf2
preLow2 = preResoult.loc[preIdx[2]] + preLowConf2



upper_bond.append(preUp1)
upper_bond.append(preUp2)
lower_bond.append(preLow1)
lower_bond.append(preLow2)

confiIndex = [preIdx[1],preIdx[2]]
upSe = pd.Series(upper_bond,index = confiIndex)
lowSe = pd.Series(lower_bond,index = confiIndex)

print(upSe,lowSe)

'''












#arma_mod.conf_int()


#ccccc = arma_mod.predict(start = len(D_data),\
#                 end = len(D_data)+1,\
#                 dynamic=False,\
#                 typ="linear")


#
#arma_mod.summary()
#
print(arma_mod.summary(),arma_mod.summary().tables[1])




#
##分解趋势季节和残余
#from statsmodels.tsa.seasonal import seasonal_decompose
#decomposition = seasonal_decompose(trainSeData, freq=10, two_sided=False)
#residual = decomposition.resid
#decomposition.plot()
#
#
#






def autoFit():

    
    
    def fun_timer():
        print('Hello Timer!')
        global timer
        timer = threading.Timer(5.5, fun_timer)
        timer.start()
    
    timer = threading.Timer(1, fun_timer)
    timer.start()
    
    time.sleep(15) # 15秒后停止定时器
    timer.cancel()







'''
arima_model = ARIMA(D_data,order =(0,1,1)) #ARIMA模型
result = arima_model.fit()
pred = result.predict(dynamic=False,typ="linear")#levels  linear
idx = pd.date_range(start = D_data.index[1],end = D_data.index[-1],freq='T')
predList= []
intervalTime = minNums(D_data.index[1],D_data.index[-1])
for i in range(intervalTime):
    predList.append(np.array(pred)[i])
predSeries = pd.Series(np.array(predList),index=idx)


diff_shift = trainSeData.shift(1).dropna()
diffShiftValues = diff_shift.values
diffShiftIndex = diff_shift.index
diff_shift_ts = pd.Series(diffShiftValues,index = diffShiftIndex)
diff_recover_1 = predSeries.add(diff_shift_ts).dropna()
recoverModel = diff_recover_1.shift(-1).dropna().append(predValue[:2])

trainTimeSeries = trainSeData[D_data.index[1]:D_data.index[-1]]  # 过滤没有预测的记录
fig=plt.figure(figsize=(12,8))
fig.add_subplot(111)
plt.plot(trainTimeSeries,color='#1F77B4',label='Training Data')
plt.plot(recoverModel,color='#E8BF52',label='Model')
plt.plot(predValue,color='red',label='Predict')
#ax.plot(kkk,color='grey',label='Predicted number')
plt.plot(testSeData,color='#53A15C',label='Actual')
plt.title('RMSE: %.4f'% np.sqrt(sum((recoverModel[:-1]-trainTimeSeries)**2)/trainTimeSeries.size))
plt.legend(loc='best')
plt.grid(True)#网格线
'''













'''
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

plot_intervals = True
if plot_intervals:
    window = 2
    scale = 1.96
    rolling_mean = trainSeData.rolling(window).mean()
    mae = mean_absolute_error(trainSeData[window:], rolling_mean[window:])
    deviation = np.std(trainSeData[window:] - rolling_mean[window:])
    lower_bond = rolling_mean - (mae + scale * deviation)
    upper_bond = rolling_mean + (mae + scale * deviation)
    plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
    plt.plot(lower_bond, "r--",marker='^')
'''


'''


##测试
ARMAModel = ARIMA(D_data, order=(0,1,1)).fit()  # order=(p,d,q)
# fittedvalues和diff对比
plt.figure(figsize=(12, 8))
plt.plot(D_data, 'r', label='Orig')
plt.plot(ARMAModel.fittedvalues, 'g',label='ARMA Model')
plt.legend()
# 样本内预测
predicts = ARMAModel.predict(dynamic=False,typ="linear")#typ="linear",typ="levels"

# 因为预测数据是根据差分值算的，所以要对它一阶差分还原
train_shift = trainSeData.shift(1).dropna()
pred_recover = predicts.add(train_shift).dropna()

## 模型评价指标 1：计算 score
#delta = ARMAModel.fittedvalues - D_data
#score = 1 - delta.var()/trainSeData.var()
#print('score:\n', score)

# 模型评价指标 2：使用均方根误差（RMSE）来评估模型样本内拟合的好坏。
#利用该准则进行判别时，需要剔除“非预测”数据的影响。
train_vs = trainSeData[pred_recover.index]  # 过滤没有预测的记录
plt.figure(figsize=(12, 8))
train_vs.plot(label='Original')
pred_recover.shift(-1).plot(label='Predict')
plt.legend(loc='best')
plt.title('RMSE: %.4f'% np.sqrt(sum((pred_recover-train_vs)**2)/train_vs.size))
plt.show()












def forecastData(trainData,time):
    preIndex = []
    preData = []
    temSerise = trainData.copy()
    for i in range(time) :
        d_data = temSerise.diff().dropna()
        

        
        model = ARIMA(d_data,order =(0,1,1)).fit()
        modelResoult = model.predict(start = len(d_data),end = len(d_data)+1,dynamic=False)[0]
        recoverValue = temSerise.shift().dropna()[temSerise.index[-2]]
        actualValue = recoverValue + modelResoult
        
        
        
        startTimeStr = "1924-2-05 00:00:00"
        endTimeStr = "1924-2-05 00:01:00"
        startTime = datetime.strptime(startTimeStr,'%Y-%m-%d %H:%M:%S')
        endTime = datetime.strptime(endTimeStr,'%Y-%m-%d %H:%M:%S')
        intervalTime = endTime - startTime
        
        
        preTimeIndex = temSerise.index[-1] + intervalTime
        
        
        print(i,actualValue,temSerise[-1])
        
        
        
        preIndex.append(preTimeIndex)
        preData.append(actualValue)
        
        temSerise[preTimeIndex] = actualValue
        
        preResoult = pd.Series(preData,index = preIndex)
        

    return preResoult
ggg = forecastData(trainSeData,10)



    error = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])
    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True);



from statsmodels.tsa.arima_model import ARMA #模型
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample
def forcastDataByParameter(trainData,time):
    preIndex = []
    preData = []
    temSerise = trainData.copy()
    for i in range(time) :
        d_data = temSerise.diff().dropna()
        
        
        res = sm.tsa.ARMA(d_data, (0,1)).fit(trend="nc")
        params = res.params
        residuals = res.resid
        p = res.k_ar
        q = res.k_ma
        k_exog = res.k_exog
        k_trend = res.k_trend
        steps = 1
        
        
        modelResoult = _arma_predict_out_of_sample(params, steps, residuals, p, q, k_trend, k_exog, endog=D_data, exog=None, start=len(D_data))
        actualValue = temSerise[-1] + modelResoult
        
        
        startTimeStr = "1924-2-05 00:00:00"
        endTimeStr = "1924-2-05 00:01:00"
        startTime = datetime.strptime(startTimeStr,'%Y-%m-%d %H:%M:%S')
        endTime = datetime.strptime(endTimeStr,'%Y-%m-%d %H:%M:%S')
        intervalTime = endTime - startTime
        
        
        preTimeIndex = temSerise.index[-1] + intervalTime
        
        
        print(i,actualValue,temSerise[-1])
        
        
        
        preIndex.append(preTimeIndex)
        preData.append(actualValue)
        
        temSerise[preTimeIndex] = actualValue
    return temSerise
kkk = forcastDataByParameter(trainSeData,2)


'''















'''
v1 = [2,4,6]
v2 = [9,1,3,5]
s1 = pd.Series(v1,index = ['9-47','9-48','9-49'])
s2 = pd.Series(v2,index = ['9-46','9-47','9-48','9-49'])
'''









#
#
## 下面深入分解：长期趋势Trend、季节性seasonality和随机残差residuals。
#
## 强行补充小知识：平稳性处理之“分解”
## 所谓分解就是将时序数据分离成不同的成分。statsmodels使用的X-11分解过程，它主要将时序数据分离成长期趋势、季节趋势和随机成分。
## 与其它统计软件一样，statsmodels也支持两类分解模型，加法模型和乘法模型，model的参数设置为"additive"（加法模型）和"multiplicative"（乘法模型）。
#
#
## multiplicative
#res = sm.tsa.seasonal_decompose(trainSeData.values,freq=12,model="additive") 
## 这里用到的.tsa.seasonal_decompose()函数，经尝试：参数trainSeData.values时，横坐标是Time；参数为trainSeData时，横坐标是date_block_num。其他不变。
## freg这个参数容后研究，这里暂且猜测是周期12个月。
#
#plt.figure(figsize=(12,8))
#fig = res.plot()
##plt.savefig(r'../data/fenjie.svg',dpi=600)
## fig.show()  # 此句，可加可不加。
#
## 得到不同的分解成分，接下来可以使用时间序列模型对各个成分进行拟合。







'''

#导入损失度量函数
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
#作用和MAE一样，只不过是以百分比的形式，用来解释模型的质量
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - 时间序列
        window - 移动窗口大小
        plot_intervals - 是否展示置信区间
        plot_anomalies - 显示异常值
    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15,5))
    plt.title("Moving average 移动平均 \n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
        
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")#图例
    plt.grid(True)#网格线
#plotMovingAverage(currency, 7, plot_intervals=True, plot_anomalies=True)
#plotMovingAverage(trainSeData, 10)


'''









'''
import statsmodels.tsa.api as smt
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()

ads_diff = seData - seData.shift(7)
tsplot(ads_diff[7:], lags=10)
'''














