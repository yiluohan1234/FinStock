#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: func.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年5月20日
#    > description: 通用帮助函数
#######################################################################
import akshare as ak
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import datetime
from utils.cons import precision

def get_df_markdown_table(df):
    '''
    获取dataframe数据类型并生成markdown表格
    :param df: 数据
    :type df: pandas.DataFrame
    :return: 无
    :rtype: 无
    '''
    column_types = df.dtypes.to_dict()
    print("| 列名 | 数据类型 |")
    print("| ---------------------------- | ---- |")
    for column, data_type in column_types.items():
        print("|{}|{}|".format(column, data_type))


def get_display_data(df):
    '''
    将数据进行转置
    :param df: 数据
    :type df: pandas.DataFrame
    :return: 可以直接查看的列表数据
    :rtype: pandas.DataFrame
    '''
    ret_columns = df.columns.tolist()
    df_T = df.copy().set_index(ret_columns[0])
    index_row = df_T.index.tolist()
    df_display = pd.DataFrame(df_T.values.T, columns=index_row, index=ret_columns[1:])
    return df_display


def get_report_type(date_str):
    '''
    根据日期获取报告类别
    :param date_str: 报告日期，'20241231'
    :type date_str: str
    :return: 获取报告类别
    :rtype: str
    '''
    if "1231" in date_str:
        return "年报"
    elif "0630" in date_str:
        return "中报"
    elif "0930" in date_str:
        return "三季度报"
    elif "0331" in date_str:
        return "一季度报"


def str2value(valueStr):
    '''
    将带有亿、万和%的字符串转为数字
    :param valueStr: 数字字符串
    :type valueStr: str
    :return: 转换后的数据
    :rtype: float
    '''
    valueStr = str(valueStr)
    idxOfYi = valueStr.find('亿')
    idxOfWan = valueStr.find('万')
    idxOfPercentage = valueStr.find('%')
    if idxOfYi != -1 and idxOfWan != -1:
        return int(float(valueStr[:idxOfYi])*1e8 + float(valueStr[idxOfYi+1:idxOfWan])*1e4)
    elif idxOfYi != -1 and idxOfWan == -1:
        return int(float(valueStr[:idxOfYi])*1e8)
    elif idxOfYi == -1 and idxOfWan != -1:
        return int(float(valueStr[idxOfYi+1:idxOfWan])*1e4)
    elif idxOfYi == -1 and idxOfWan == -1 and idxOfPercentage == -1:
        return float(valueStr)
    elif idxOfYi == -1 and idxOfWan == -1 and idxOfPercentage != -1:
        return float(valueStr[:idxOfPercentage])


def cal_K(df, precision=2):
    '''
    对一段数据进行拟合求斜率
    :param df: 需要设置命名的数据框
    :type df: pandas.DataFrame
    :param precision: 默认保留小数位
    :type precision: int
    :return: 返回一段数据的斜率
    :rtype: float
    '''
    y_arr = np.array(df).ravel()
    x_arr = list(range(1, len(y_arr) + 1))
    fit_K = np.polyfit(x_arr, y_arr, deg=1)
    return round(fit_K[0], precision)


def cal_K1(df, precision=2):
    '''
    对一段数据进行拟合求斜率
    :param df: 需要设置命名的数据框
    :type df: pandas.DataFrame
    :param precision: 默认保留小数位
    :type precision: int
    :return: 返回一段数据的斜率
    :rtype: float
    '''
    from sklearn.linear_model import LinearRegression
    y = np.array(df).ravel()
    x = np.array(range(1, len(y) + 1)).reshape(-1, 1) # 需要将x转换为二维数组
    model = LinearRegression()
    model.fit(x, y)
    return round(model.coef_[0], precision)


def cal_K_predict(df, pencentage=0, precision=2):
    '''
    对一段数据进行拟合求斜率,将下一天收盘数据*（1+pencentage）作为新的数据进行预测
    :param df: 需要设置命名的数据框
    :type df: pandas.DataFrame
    :param precision: 默认保留小数位
    :type precision: int
    :return: 返回一段数据的斜率
    :rtype: float
    '''
    from scipy.stats import linregress
    y = np.array(df).ravel()
    y = np.append(y, y[-1]*(1+pencentage*1.0/100))[1:]
    x = np.array(range(1, len(y) + 1))

    slope, intercept, r_value, p_value, std_err = linregress(x, y) #斜率，截距，相关系数，拟合优度，拟合的均方根误差
    return round(slope, precision)


def MA(df, n):
    """
    计算简单移动平均值
    MA(X,N)，求X的N日移动平均值。算法：(X1+X2+X3+，，，+Xn)/N。例如：MA(CLOSE,10)表示求10日均价。
    :param df: 含有close列的dataframe
    :type df: pandas.DataFrame
    :param n: 移动数
    :type n: int
    :return: 返回简单移动平均值
    :rtype: pandas.DataFrame
    """
    ma_n = df.close.rolling(n).mean()
    return ma_n


def EMA(df_close, n):
    """
    计算指数移动平均值
    EMA(X,N)，求X的N日指数平滑移动平均。算法：若Y=EMA(X,N)则Y=[2*X+(N-1)*Y']/(N+1)，其中Y'表示上一周期Y值。例如：EMA(CLOSE,30)表示求30日指数平滑均价。
    :param df_close: 含有close列的dataframe
    :type df_close: pandas.DataFrame
    :param n: 移动数
    :type n: int
    :return: 返回移动平均值
    :rtype: pandas.DataFrame
    """
    ema_n = df_close.ewm(span=n, min_periods=n, adjust=False).mean()
    return ema_n


def MACD(df, short=12, long=26, mid=9):
    """
    计算MACD指标
    MACD指数平滑异同移动平均线为两条长、短的平滑平均线。参数默认short=12,long=26,M=9
            DIFF : EMA(CLOSE,SHORT) - EMA(CLOSE,LONG);
            DEA  : EMA(DIFF,M);
            MACD : 2*(DIFF-DEA);
    其买卖原则为：
        1.DIFF、DEA均为正，DIFF向上突破DEA，买入信号参考。
        2.DIFF、DEA均为负，DIFF向下跌破DEA，卖出信号参考。
        3.DEA线与K线发生背离，行情可能出现反转信号。
        4.分析MACD柱状线，由红变绿(正变负)，卖出信号参考；由绿变红，买入信号参考。
    :param df: datframe数据
    :type df: pandas.DataFrame
    :param short: short值
    :type short: int
    :param long: long值
    :type long: int
    :param mid: mid值
    :type mid: int
    :return: 返回MACD指标dif、dea和macd
    :rtype: pandas.DataFrame
    """
    dt = {}
    dif = EMA(df.close, short) - EMA(df.close, long)
    dea = EMA(dif, mid)
    macd = (dif - dea) * 2
    dt['DIF'], dt['DEA'], dt['MACD'] = dif, dea, macd
    ret = pd.DataFrame(dt)
    return ret


def KDJ(df, N=9, M1=3, M2=3):
    """
    计算KDJ指标
    返回k、d、j的值，默认N=9,M1=3,M2=3
    RSV=(CLOSE-LLV(LOW,N))/(HHV(HIGH,N)-LLV(LOW,N))*100;
    LLV:求最低值，HHV：求最高值，LOW：当日（周期）最低价，HIGH：当日（周期）最高价
            a=SMA(RSV,M1,1);
            b=SMA(a,M2,1);
            e=3*a-2*b;
            K:a;D:b;J:e;同花顺中默认N=9,M1=3,M2=3；
    KDJ指标指标说明
        KDJ，其综合动量观念、强弱指标及移动平均线的优点，早年应用在期货投资方面，功能颇为显著，目前为股市中最常被使用的指标之一。
    买卖原则
        1 K线由右边向下交叉D值做卖，K线由右边向上交叉D值做买。
        2 高档连续二次向下交叉确认跌势，低挡连续二次向上交叉确认涨势。
        3 D值<20%超卖，D值>80%超买，J>100%超买，J<10%超卖。
        4 KD值于50%左右徘徊或交叉时，无意义。
        5 投机性太强的个股不适用。
        6 可观察KD值同股价的背离，以确认高低点。
    :param df: datframe数据
    :type df: pandas.DataFrame
    :param N: N值
    :type N: int
    :param M1: M1值
    :type M1: int
    :param M2: M2值
    :type M2: int
    :return: 返回KDJ指标k、d和j
    :rtype: pandas.DataFrame
    """
    dt = {}
    llv = df.low.rolling(N).min() # 假设你的low是一个pandas.series的对象
    hhv = df.high.rolling(N).max()
    rsv = (df.close - llv)/(hhv -llv)*100
    k = rsv.ewm(M1-1).mean()
    d = k.ewm(M2-1).mean()
    j = 3*k - 2*d
    dt['K'], dt['D'], dt['J'] = k, d, j
    ret = pd.DataFrame(dt)
    return ret


def RSI(df, N1=6, N2=12, N3=24):
    """
    计算RSI指标
    默认N1=6,N2=12,N3=24，返回6日RSI值、12日RSI值、24日RSI值，RSI一般选用6日、12日、24日作为参考基期
        LC := REF(CLOSE,1);#上一周期的收盘价
        RSI$1:SMA(MAX(CLOSE-LC,0),N1,1)/SMA(ABS(CLOSE-LC),N1,1)*100;
        RSI$2:SMA(MAX(CLOSE-LC,0),N2,1)/SMA(ABS(CLOSE-LC),N2,1)*100;
        RSI$3:SMA(MAX(CLOSE-LC,0),N3,1)/SMA(ABS(CLOSE-LC),N3,1)*100;
        a:20;
        d:80;
    RSI指标：
        RSIS为1978年美国作者Wells WidlerJR。所提出的交易方法之一。所谓RSI英文全名为Relative Strenth Index，中文名称为相对强弱指标．RSI的基本原理是在一个正常的股市中，多
        空买卖双方的力道必须得到均衡，股价才能稳定;而RSI是对于固定期间内，股价上涨总幅度平均值占总幅度平均值的比例。
        1 RSI值于0-100之间呈常态分配，当6日RSI值为80‰以上时，股市呈超买现象，若出现M头，市场风险较大；当6日RSI值在20‰以下时，股市呈超卖现象，若出现W头，市场机会增大。
        2 RSI一般选用6日、12日、24日作为参考基期，基期越长越有趋势性(慢速RSI)，基期越短越有敏感性，(快速RSI)。当快速RSI由下往上突破慢速RSI时，机会增大；当快速RSI由上而下跌破慢速RSI时，风险增大。
    :param df: datframe数据
    :type df: pandas.DataFrame
    :param N1: N1值
    :type N1: int
    :param N2: N2值
    :type N2: int
    :param N3: N3值
    :type N3: int
    :return: 返回KDJ指标k、d和j
    :rtype: pandas.DataFrame
    """
    dt = {}
    lc = df.close.shift(1)
    # 计算前收盘价
    max_diff = (df.close - lc)
    abs_diff = max_diff.copy()

    max_diff[max_diff < 0] = 0  # 实现MAX(CLOSE-LC,0)
    abs_diff = abs_diff.abs()  # 实现ABS(CLOSE-LC)

    RSI1, RSI2, RSI3 = (max_diff.ewm(N-1).mean()/abs_diff.ewm(N-1).mean()*100 for N in [N1, N2, N3])
    dt['RSI1'], dt['RSI2'], dt['RSI3'] = RSI1, RSI2, RSI3
    ret = pd.DataFrame(dt)
    return ret


def BOLL(df, N=20, M=2):
    """
    计算BOLL指标
    :param df: datframe数据
    :type df: pandas.DataFrame
    :param N: N值
    :type N: int
    :param M: M值
    :type M: int
    :return: 返回BOLL指标boll、up、down
    :rtype: pandas.DataFrame
    """
    dt = {}
    boll = df.close.rolling(20).mean()
    delta = df.close - boll
    beta = delta.rolling(20).std()
    up = boll + 2 * beta
    down = boll - 2 * beta
    dt['boll'], dt['up'], dt['down'] = boll, up, down
    ret = pd.DataFrame(dt)
    return ret


def ENE(df, N=10, M=9.0):
    """
    计算包络线ENE(10,9,9)
    ENE代表中轨。MA(CLOSE,N)代表N日均价
    UPPER:(1+M1/100)*MA(CLOSE,N)的意思是，上轨距离N日均价的涨幅为M1%；
    LOWER:(1-M2/100)*MA(CLOSE,N) 的意思是，下轨距离 N 日均价的跌幅为 M2%;
    :param df: datframe数据
    :type df: pandas.DataFrame
    :param N: N值
    :type N: int
    :param M: M值
    :type M: int
    :return: 返回BOLL指标boll、up、down
    :rtype: pandas.DataFrame
    """
    dt = {}
    ene = df.close.rolling(N).mean()
    upper = (1 + M / 100) * ene
    lower = (1 - M / 100) * ene
    dt['ene'], dt['upper'], dt['lower'] = ene, upper, lower
    ret = pd.DataFrame(dt)
    return ret


def ATR(df, N=14):
    """
    求真实波幅的N日移动平均    参数：N 天数，默认取14
    TR:MAX(MAX((HIGH-LOW),ABS(REF(CLOSE,1)-HIGH)),ABS(REF(CLOSE,1)-LOW));
    ATR:MA(TR,N);
    :param df: datframe数据
    :type df: pandas.DataFrame
    :param N: N值
    :type N: int
    :return: 返回ATR
    :rtype: pandas.DataFrame
    """
    dt = {}
    maxx = df['high'] - df['low']
    abs_high = abs(df['close'].shift(1) - df['high'])
    abs_low = abs(df['close'].shift(1) - df['low'])
    a = pd.DataFrame()
    a['maxx'] = maxx.values
    a['abs_high'] = abs_high.values
    a['abs_low'] = abs_low.values
    TR = a.max(axis=1)
    ATR = TR.rolling(N).mean()
    STOP = df['close'].shift(1) - ATR * 3  # 止损价=(上一日收盘价-3*ATR)
    dt['ATR'], dt['stop'] = ATR, STOP
    ret = pd.DataFrame(dt)
    return ret


def get_macd_data(df, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    计算MACD指标
    :param df: datframe数据
    :type df: pandas.DataFrame
    :param fastperiod: 短期天数
    :type fastperiod: int
    :param slowperiod: 长期天数
    :type slowperiod: int
    :param signalperiod: 天数
    :type signalperiod: int
    :return: 返回MACD指标dif、dea和macd
    :rtype: pandas.DataFrame
    # https://cloud.tencent.com/developer/article/1794902
    """
    import talib
    DIF, DEA, MACD = talib.MACDEXT(df['close'], fastperiod=fastperiod, fastmatype=1,
                                   slowperiod=slowperiod, slowmatype=1, signalperiod=signalperiod,
                                   signalmatype=1)
    MACD = MACD * 2
    return DIF, DEA, MACD


def get_name(code, zh_index):
    '''
    获取股票名称
    :param code: 股票代码
    :type code: str
    :param zh_index: 是否位指数
    :type zh_index: bool
    :return: 股票名称
    :rtype: str
    '''
    if not zh_index:
        name_code = ak.stock_zh_a_spot_em()
        name = name_code[name_code['代码'] == code]['名称'].values[0]
        return name
    else:
        code_name = {'sh000001': '上证指数', 'sz399001': '深证成指',
                     'sz399006': '创业板指数', 'sh000688': '科创板指数',
                     # 风格指数
                     'sz399300': '沪深300', 'sh000905': '中证500',
                     'sh000906': '中证800', 'sh000852': '中证1000',
                     'sz399303': '国证2000', 'sh000045': '上证小盘',
                     # 中证行业指数
                     'sh000827': '中证环保', 'sh000928': '中证能源',
                     'sh000933': '中证医药', 'sh000934': '中证金融',
                     'sh000935': '中证信息', 'sz399804': '中证体育',
                     'sz399813': '中证国安', 'sz399971': '中证传媒',
                     'sz399973': '中证国防', 'sz399986': '中证银行',
                     'sz399997': '中证白酒', 'sz399998': '中证煤炭',
                     'sh000932': '中证消费', 'sz399808': '中证新能',
                     'sz399989': '中证医疗', '000001': '上证指数'}
        return code_name[code]


def frb(open_value, close_value):
    '''
    获取volume的标识符
    :param open_value: 开盘数值
    :type open_value: float
    :param close_value: 开盘数值
    :type close_value: float
    :return: 返回标识
    :rtype: int
    '''
    if (close_value - open_value) >= 0:
        return 1
    else:
        return -1


def get_szsh_code(code):
    '''
    获取上证指数的字母前缀
    :param code: 股票代码
    :type code: str
    :return: 返回带前缀的编码
    :rtype: str
    '''
    # https://blog.csdn.net/viki_2/article/details/123775244
    gp_type = ''
    if code.find('60', 0, 3) == 0:
        gp_type = 'SH'+code
    elif code.find('688', 0, 4) == 0:
        gp_type = 'BJ'+code
    elif code.find('900', 0, 4) == 0:
        gp_type = 'SH'+code
    elif code.find('00', 0, 3) == 0:
        gp_type = 'SZ'+code
    elif code.find('300', 0, 4) == 0:
        gp_type = 'SZ'+code
    elif code.find('200', 0, 4) == 0:
        gp_type = 'SZ'+code
    return gp_type


def num2str(num):
    '''
    实现数值转换为万，亿单位，保留2位小数
    :param num: 数字
    :type num: float
    :return: 转换数字位亿万单位的字符串
    :rtype: str
    '''
    if num > 0:
        flag = 1
    else:
        flag = -1
    num = abs(num)
    level = 0
    while num > 10000:
        if level >= 2:
            break
        num /= 10000
        level += 1
    units = ['', '万', '亿']

    return '{}{}'.format(round(flag * num, 3), units[level])


def get_num2str_df(df):
    '''
    实现将dataframe转换为万，亿单位，保留2位小数
    :param df: 数据
    :type df: pandas.DataFrame
    :return: 转换数字位亿万单位的数据
    :rtype: pandas.DataFrame
    '''
    for col in df.columns.tolist():
        if str(df[col].dtype) == 'float64':
            df[col] = df[col].apply(lambda x: num2str(x))
    return df


def get_date_week(current_date):
    '''
    获取当前日期的周数
    :param current_date: 日期字符串，'20240521'
    :type current_date: str
    :return: 返回第n个周
    :rtype: str
    '''
    week = datetime.datetime.strptime(current_date, '%Y%m%d').strftime('%W')
    return week


def get_date_month(current_date):
    '''
    获取当前日期的月份
    :param current_date: 日期字符串，'20240521'
    :type current_date: str
    :return: 返回第n个月
    :rtype: str
    '''
    month = datetime.datetime.strptime(current_date, '%Y%m%d').month
    return str(month)


def transfer_price_freq(df, freq):
    """
    将数据转化为指定周期：开盘价(周期第一天)、收盘价(周期最后一天)、最高价(周期)、最低价(周期)
    :param df: 日数据，包含每天开盘价、收盘价、最高价、最低价
    :type df: pandas.DataFrame
    :param freq: 转换周期，周：'W'，月:'M'，季度:'Q'
    :type freq: str
    :return: 转换后的数据
    :rtype: pandas.DataFrame
    """
    if freq == 'M':
        freq = 'ME'

    df["date"] = pd.to_datetime(df["date"])
    df.set_index('date', inplace=True)

    period_stock_data = round(df.resample(freq).last(), precision)
    period_stock_data['open'] = round(df['open'].resample(freq).first(), precision)
    period_stock_data['close'] = round(df['close'].resample(freq).last(), precision)
    period_stock_data['high'] = round(df['high'].resample(freq).max(), precision)
    period_stock_data['low'] = round(df['low'].resample(freq).min(), precision)
    period_stock_data['volume'] = round(df['volume'].resample(freq).sum(), precision)

    #去除没有交易
    # period_stock_data = df[df['volume'].notnull()]
    period_stock_data.dropna(subset=['close'], how='any', inplace=True)
    if freq == 'W':
        # 周采样默认为周日，改为周五
        period_stock_data.index = period_stock_data.index+datetime.timedelta(days=-2)
    period_stock_data.reset_index(inplace=True)

    return period_stock_data


def k_cross_strategy(df):
    """
    策略：k10、k20和k60为负，k10上穿k20买入；k10、k20和k60为正，k10下穿k20卖出；
    :param df: 数据
    :type df: pandas.DataFrame
    :return: 标记后的数据
    :rtype: pandas.DataFrame
    """
    for i in range(len(df)):
        if i == 0:
            continue
        if (df.loc[i, 'k10'] > df.loc[i, 'k20'] and df.loc[i-1, 'k10'] < df.loc[i-1, 'k20']) and \
            (df.loc[i, 'k10'] < 0 and df.loc[i, 'k20'] < 0 and df.loc[i, 'k60'] < 0) and \
            (df.loc[i, 'k10'] > df.loc[i-1, 'k10'] and df.loc[i, 'k20'] > df.loc[i-1, 'k20'] and df.loc[i, 'k60'] >= df.loc[i-1, 'k60']):
            df.loc[i, 'BUY'] = True
        if (df.loc[i, 'k10'] < df.loc[i, 'k20'] and df.loc[i-1, 'k10'] > df.loc[i-1, 'k20']) and \
            (df.loc[i, 'k10'] > 0 and df.loc[i, 'k20'] > 0 and df.loc[i, 'k60'] > 0) and \
            (df.loc[i, 'k10'] < df.loc[i-1, 'k10'] and df.loc[i, 'k20'] < df.loc[i-1, 'k20'] and df.loc[i, 'k60'] <= df.loc[i-1, 'k60']):
            df.loc[i, 'SELL'] = True
    return df


def max_min_strategy(df, k_name='k20'):
    """
    策略：k_name最小值买入；k_name最大值卖出；
    :param df: 数据
    :type df: pandas.DataFrame
    :param k_name: 斜率名字
    :type k_name: str
    :return: 标记后的数据
    :rtype: pandas.DataFrame
    """
    for i in range(len(df)):
        if i < 2:
            continue
        if df.loc[i, k_name] > df.loc[i - 1, k_name] and df.loc[i - 1, k_name] <= df.loc[i - 2, k_name]:
            df.loc[i, 'BUY'] = True
        if df.loc[i, k_name] < df.loc[i - 1, k_name] and df.loc[i - 1, k_name] >= df.loc[i - 2, k_name]:
            df.loc[i, 'SELL'] = True
    return df


def max_min_low_high_strategy(df, k_name='k20'):
    """
    策略：k10、k20和k60为负，k_name最小值买入；k10、k20和k60为正，k_name最大值卖出；
    :param df: 数据
    :type df: pandas.DataFrame
    :param k_name: 斜率名字
    :type k_name: str
    :return: 标记后的数据
    :rtype: pandas.DataFrame
    """
    for i in range(len(df)):
        if i < 2:
            continue
        if (df.loc[i, k_name] > df.loc[i-1, k_name] and df.loc[i-1, k_name] <= df.loc[i-2, k_name]) and \
            (df.loc[i, 'k10'] < 0 and df.loc[i, 'k20'] < 0 and df.loc[i, 'k60'] < 0) and \
            (df.loc[i, 'k10'] > df.loc[i-1, 'k10'] and df.loc[i, 'k60'] >= df.loc[i-1, 'k60']):
            df.loc[i, 'BUY'] = True
        if (df.loc[i, k_name] < df.loc[i-1, k_name] and df.loc[i-1, k_name] >= df.loc[i-2, k_name]) and \
            (df.loc[i, 'k10'] > 0 and df.loc[i, 'k20'] > 0 and df.loc[i, 'k60'] > 0) and \
            (df.loc[i, 'k10'] < df.loc[i-1, 'k10'] and df.loc[i, 'k60'] <= df.loc[i-1, 'k60']):
            df.loc[i, 'SELL'] = True
    return df


def max_min_low_high_strategy_pre(df, k_name='kp10'):
    """
    策略：kp10、kp20和k60为负，k_name最小值买入；k10、k20和k60为正，k_name最大值卖出；
    :param df: 数据
    :type df: pandas.DataFrame
    :param k_name: 斜率名字
    :type k_name: str
    :return: 标记后的数据
    :rtype: pandas.DataFrame
    """
    for i in range(len(df)):
        if i < 2:
            continue
        if (df.loc[i, k_name] > df.loc[i-1, k_name] and df.loc[i-1, k_name] < df.loc[i-2, k_name]) and \
            df.loc[i, 'kp10'] < 0 and df.loc[i, 'kp20'] < 0 and df.loc[i, 'kp60'] < 0:
            df.loc[i, 'BUY'] = True
        if df.loc[i, k_name] < df.loc[i-1, k_name] and df.loc[i-1, k_name] > df.loc[i-2, k_name]:
            df.loc[i, 'SELL'] = True
    return df


def find_max_min_point(df, k_name='k20'):
    """
    获取数据的局部最大值和最小值的索引
    :param df: 数据
    :type df: pandas.DataFrame
    :param k_name: 斜率名字
    :type k_name: str
    :return: 索引值
    :rtype: numpy.ndarray
    """
    series = np.array(df[k_name])
    peaks, _ = find_peaks(series)  # 纵轴局部最高点
    mins, _ = find_peaks(series*-1)  # 纵轴局部最低点

    for i in range(len(df)):
        if i == 0:
            continue
        if (i in mins.tolist()) and \
            (df.loc[i, 'k10'] < 0 and df.loc[i, 'k20'] < 0 and df.loc[i, 'k60'] < 0):
            df.loc[i, 'BUY'] = True
        if (i in peaks.tolist()) and \
            (df.loc[i, 'k10'] > 0 and df.loc[i, 'k20'] > 0 and df.loc[i, 'k60'] > 0):
            df.loc[i, 'SELL'] = True
    return df


def sma_boll_strategy(df):
    """
    收盘小于ma10买入，高于boll的up卖出
    :param df: 数据
    :type df: pandas.DataFrame
    :return: 索引值
    :rtype: pandas.DataFrame
    """
    for i in range(len(df)):
        if df.loc[i, 'close'] > df.loc[i, 'up']:
            df.loc[i, 'SELL'] = True
        if df.loc[i, 'close'] < df.loc[i, 'ma10']:
            df.loc[i, 'BUY'] = True
    return df


def AMPD(data):
    """
    实现AMPD获取波峰算法
    :param data: 1-D numpy.ndarray
    :return:
    :param data: 数据
    :type data: numpy.ndarray
    :return: 波峰所在索引值的列表
    :rtype: list
    """
    p_data = np.zeros_like(data, dtype=np.int32)
    count = data.shape[0]
    arr_rowsum = []
    for k in range(1, count // 2 + 1):
        row_sum = 0
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                row_sum -= 1
        arr_rowsum.append(row_sum)
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    for k in range(1, max_window_length + 1):
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                p_data[i] += 1
    return np.where(p_data == max_window_length)[0]
    # 这个代码再修改一下，还可以找到第二高的点，第三高的点。在AMPD函数最后一行里面，np.where(p_data == max_window_length-top)[0] 其中top分别为0，1，2就可以代表第一高（波峰），第二高、第三高等。


def transfer_date_format(date_string, formats="%Y%m%d"):
    """
    尝试统一日期字符串格式，返回格式为YYYY-MM-DD的日期格式或无法解析。
    :param date_string:日期字符串
    :param formats: 日期字符串可能的格式列表
    :return: 格式为YYYY-MM-DD的日期格式或
    """
    if '-' in date_string:
        return date_string
    from datetime import datetime
    date_string = date_string.rstrip(', ')
    #  尝试将日期字符串转换为datetime对象
    date_object = datetime.strptime(date_string, formats)
    return date_object.strftime("%Y-%m-%d")


if __name__ == "__main__":
    print(get_date_month("20240521"))
