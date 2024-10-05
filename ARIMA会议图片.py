import time
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller, kpss ,acf, pacf, grangercausalitytests
from arch.unitroot import PhillipsPerron
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

#mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

l = ['000001.SS', 'AAPL', 'BTC-USD' , 'DJI', 'Gold_daily','GSPC','IXIC']

for i in l:

    file_path = 'C:/lyx/learning/期刊论文/程序结果/ARIMA/' + i#要保存图片和excel的那个文件夹,每次运行需要修改!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if(os.path.exists(file_path)):
        print('文件夹已存在')
    else:
        os.makedirs(file_path)

    original_acf_pacf_path = file_path + '/original_acf_pacf_result.svg'
    first_order_difference_acf_pacf_path = file_path + '/first_order_difference_acf_pacf_result.svg'#图像保存路径,最后一项不用改
    plot_path = file_path + '/plot_result.svg'
    pre_plot_path = file_path + '/pre_plot_result.svg'
    accu_path = file_path + '/accu.xls'

    division_rate1 = 0.9#分割训练集占比,剩下的是测试集，原来是0.7
    division_rate2 = 1.0

    filename = 'C:/lyx/learning/会议论文/三支同时期数据/' + i + '.csv'
    data_name = i

    df = pd.read_csv(filename, usecols=['Date', 'Adj Close'], parse_dates=['Date'], index_col='Date')
    print(df.head())

    data = df.values
    row1 = round(division_rate1 * data.shape[0])  # 70% split可改动!!!!!!!#round是四舍五入,0.9可能乘出来小数  #shape[0]是result列表中子列表的个数
    row2 = round(division_rate2 * data.shape[0])
    #训练集和测试集划分
    train = data[:int(row1), :]
    test = data[int(row1):int(row2), :]
    df_train = df.iloc[:int(row1), :]
    df_test = df.iloc[int(row1):int(row2), :]


    def plot_df(title='', xlabel='Date', ylabel='Adj Close', dpi=2000):
        plt.ion()
        plt.figure(figsize=(13, 6), dpi=dpi)
        x_train = df_train.index
        y_train = df_train['Adj Close']
        x_test = df_test.index
        y_test = df_test['Adj Close']
        plt.plot(x_train, y_train, color='blue', label='Train')
        plt.plot(x_test, y_test, color='orange', label='Test')
        plt.rcParams.update({'font.size': 20})
        plt.legend(loc='best')
        plt.xticks(fontsize= 20)
        plt.yticks(fontsize= 20)
        plt.title(title, fontsize = 20)
        plt.xlabel(xlabel, fontsize = 20)
        plt.ylabel(ylabel, fontsize= 20)
        values = np.array(data)
        min1 = min(values)
        max1 = max(values) + 1
        plt.ylim([min1, max1])
        #plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel, fontsize = 20)
        plt.savefig(plot_path,dpi = 2000)
        plt.show()
        time.sleep(3)
        plt.close()
    plot_df(title = data_name)

    #单一检验方式不能保证百分百正确,只有干扰项同方差且没有序列相关才可以用ADF，当存在异方差或序列相关用PP比较准确
    #ADF单位根检验序列平稳性,PP用来擦ADF的PP,起辅助检验作用
    def ADF_Test(list):
        #print(list)
        result = adfuller(list, autolag='AIC')
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        for key, value in result[4].items():
            print('Critial Values:')
            print(f'   {key}, {value}')

    #PP单位根检验给ADF擦PP
    def PP_Test(list):
        #不指定trend为默认检验是否为带截距项的平稳过程
        pp = PhillipsPerron(list)
        print(pp.summary().as_text())


    #KPSS单位根检验序列平稳性,也需要和其他检验方法联合起来使用,并不是百分百正确
    def KPSS_Test(list):
        result = kpss(list, regression='c')
        print('\nKPSS Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        for key, value in result[3].items():
            print('Critial Values:')
            print(f'   {key}, {value}')

    #指定trend=‘ct’为检验是否为带截距项和时间趋势项的平稳过程,两个结合在一起看
    ''' pp1 = PhillipsPerron(list)
    pp1.trend = 'ct'
    p0rint(pp1.summary().as_text())'''


    list = df['Adj Close']
    list1 = list.diff(1).dropna()  # list1为list的1阶差分序列,序列的序号从1开始,所以要tolist,这样序号才从0开始. 但是列表不能调用diff
    #或者list1 = np.diff(list)[1:]
    list = list.tolist()
    list1 = list1.tolist()


    #ADF检验,验证1阶差分稳定
    ADF_Test(list)

    ADF_Test(list1)

    #PP检验补充ADF检验
    #PP_Test(list)

    PP_Test(list1)

    #KPSS_Test(list)
    KPSS_Test(list)
    KPSS_Test(list1)

    #检验原序列和1阶差分序列是否为白噪声,白噪声预测无意义
    def White_noise_test(list):
        ljungbox_result = acorr_ljungbox(list, lags=20)
        print(ljungbox_result)



    White_noise_test(list)
    White_noise_test(list1)
    #结果证明list的p值均为0,list1的p值均非常接近0


    #自相关函数(ACF)图检验季节性,当季节模式明显时，ACF 图中季节窗口的整数倍处会反复出现特定的尖峰。
    def ACF_plot(list):
        # Draw Plot
        plt.rcParams.update({'figure.figsize': (9, 5), 'figure.dpi': 120})
        autocorrelation_plot(list.tolist())
        plt.show()
        time.sleep(3)
        plt.close()
    '''ACF_plot(list)'''#经验证,adj close没有季节性

    #自相关和偏自相关图用来确定p,q.  pacf用来看p,acf用来看q.ARIMA(p,n,q),n是几阶差分
    def acf_pacf_plot(list,x):#x是差分阶数,0,1,2...0代表original,1代表first_order_difference!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
        if x == 0:
            fig.suptitle('The original data of {}'.format(data_name))
            plot_acf(list, lags=50, ax=axes[0])
            plot_pacf(list, lags=50, ax=axes[1])
            plt.savefig(original_acf_pacf_path)
            plt.show()
            plt.close()
        if x == 1:
            fig.suptitle('First order difference data of {}'.format(data_name))
            plot_acf(list, lags=50, ax=axes[0])
            plot_pacf(list, lags=50, ax=axes[1])
            plt.savefig(first_order_difference_acf_pacf_path)
            plt.show()
            time.sleep(3)
            plt.close()

    df_train = df_train['Adj Close']
    df_train1 = df_train.diff(1).dropna()
    #df_train和df_train是dataframe格式

    acf_pacf_plot(df_train.tolist(),0)
    acf_pacf_plot(df_train1.tolist(),1)

    '这里p,q选1'

    #自动按照最小AIC或BIC来寻找p,q
    def auto_pq(list):
        trend_evaluate = sm.tsa.arma_order_select_ic(list, ic=['aic', 'bic'], trend='n', max_ar=10,max_ma=10)
        #这里需要设定自动取阶的 p和q 的最大值，即函数里面的max_ar,和max_ma。ic 参数表示选用的选取标准.

        print('train AIC', trend_evaluate.aic_min_order)
        print('train BIC', trend_evaluate.bic_min_order)
        #结果是(5,5)

    #auto_pq(train)  train是二维数组


    '''print('df_train',df_train)
    print('df_train1',df_train1)
    print('train',train)'''

    '''评估模型MSE'''
    # evaluate an ARIMA model for a given order (p,d,q)
    def evaluate_arima_model(train,test,arima_order):
        # prepare training dataset
        train = train.tolist()

        test = test.tolist()
        train = [x for x in train]
        # make predictions
        predictions = []
        model = ARIMA(train, order=arima_order)
        model_fit = model.fit()
        print('the arima_order is :',arima_order)
        print(model_fit.summary())

        start_time = time.time()


        for t in range(len(test)):
            model = ARIMA(train, order=arima_order)
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            train.append(test[t])
        # calculate out of sample error

        end_time = time.time()  # 程序结束时间
        run_time = end_time - start_time  # 程序的运行时间，单位为秒
        print('时间为', run_time)


        test = test[15:]
        test100 = test[0:101]

        predictions = predictions[15:]
        predictions100 = predictions[0:101]
        plt.figure(figsize=(6.4, 4.8), dpi=2000)
        plt.plot(predictions100, color='red', label='Prediction', linewidth=0.6)
        plt.plot(test100, color='blue', label='Actual',linewidth=0.6)
        plt.legend(loc='best')
        plt.title('The test result for {}'.format(data_name))
        plt.xlabel('Days')
        plt.ylabel('Adjusted Closing Price')
        result = np.array(test100)
        min2 = min(result) - 5
        max2 = max(result) + 5
        plt.xlim([0, 101])
        plt.ylim([min2, max2])
        plt.savefig(pre_plot_path,dpi = 2000)  # 保存拟合曲线
        plt.show()
        time.sleep(3)
        plt.close()

        MAPE = sklearn.metrics.mean_absolute_percentage_error(test, predictions)
        # MAPE = calculate_MAPE(pre,real)
        test = np.array(test)

        predictions = np.array(predictions)
        predictions = predictions.reshape(-1,1)



        print('test.shape', test.shape)
        print('test', test)
        print('predictions.shape', predictions.shape)
        print('predictions', predictions)

        MSE = mean_squared_error(test, predictions)
        RMSE = np.sqrt(np.mean(np.square(predictions - test)))
        MAE = np.mean(np.abs(predictions - test))
        R2 = r2_score(predictions, test)
        dict = {'MAPE': MAPE, 'MSE': MSE,'RMSE': RMSE, 'MAE': MAE, 'R2': R2}
        accu = pd.DataFrame([dict])
        accu.to_excel(accu_path)  # 保存网络参数

        predictions = predictions[15:]
        stock = i
        model2 = 'ARIMA'
        csv_path = 'C:/lyx/learning/期刊论文/程序结果/对比图表/' + stock + '/' + model2 + '.xls'
        df = pd.DataFrame(predictions)
        df.columns.name = None
        df.to_excel(csv_path, index=False, header=None)
        print('最终的准确率和指标如下\n', dict)


    # function()   运行的程序


    arima_order = (1,1,1)#经验判断
    evaluate_arima_model(train,test,arima_order)


    '''arima_order = (5,1,5)#自动判断
    evaluate_arima_model(train,test,arima_order)
    '''
    '''寻找最优参'''
    # evaluate combinations of p, d and q values for an ARIMA model
    def evaluate_models(dataset, p_values, d_values, q_values):
        dataset = dataset.astype('float32')
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)
                    try:
                        mse = evaluate_arima_model(dataset, order)
                        if mse < best_score:
                            best_score, best_cfg = mse, order
                        print('ARIMA%s MSE=%.3f' % (order, mse))
                    except:
                        continue
        print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
        return best_cfg



    '''通过AIC准则寻找最优参'''
    def findC(series):
        temp = 1000000
        ansp = 0
        ansq = 0
        ansd = 0
        for p in range(0, 8):
            for q in range(0, 8):
                # if p+q!=0:
                try:
                    testModel = ARIMA(series, order=(p, 0, q))
                    testModel_fit = testModel.fit(disp=0)
                    aic = testModel_fit.aic
                    if aic < temp:
                        temp = aic
                        ansp = p
                        ansq = q
                        ansd = 0
                except:
                    continue
        return ansp,ansd,ansq

    # fit model
    '''寻找最优参'''
    # p_values = [0, 1, 2, 3 , 4, 5,6]
    # d_values = range(0,1)
    # q_values = range(0, 6)
    #

    # X = series.values
    # bestOrder=evaluate_models(X, p_values, d_values, q_values)
    # model = ARIMA(series, order=bestOrder)
    '''
    p,d,q=findC(series.values)
    
    print(p,d,q)
    mse  = evaluate_arima_model(series.values,(p,d,q))
    print("mse = %.3f"%mse)
    model = ARIMA(series, order=(p,d,q))
    # model = ARIMA(series, order=(2,0,0))
    model_fit = model.fit(disp=0)  # disp=0关#闭对训练信息的打印
    
    
    '''
    '''#打印模型信息
    print(model_fit.summary())'''


    # plot residual errors
    # residuals = DataFrame(model_fit.resid)
    # residuals.plot()
    # pyplot.show()
    # residuals.plot(kind='kde')
    # pyplot.show()
    # print(residuals.describe())
