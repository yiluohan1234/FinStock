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


def cal_K_predict(df, precision=2):
    '''
    对一段数据进行拟合求斜率,将下一天数据作为新的数据进行预测
    :param df: 需要设置命名的数据框
    :type df: pandas.DataFrame
    :param precision: 默认保留小数位
    :type precision: int
    :return: 返回一段数据的斜率
    :rtype: float
    '''
    from scipy.stats import linregress
    y = np.array(df).ravel()
    y = np.append(y, y[-1])[1:]
    x = np.array(range(1, len(y) + 1))

    slope, intercept, r_value, p_value, std_err = linregress(x, y) #斜率，截距，相关系数，拟合优度，拟合的均方根误差
    return round(slope, precision)


def ema(df_close, window):
    """
    计算指数移动平均值
    :param df_close: 收盘dataframe
    :type df_close: pandas.DataFrame
    :param window: 移动数
    :type window: int
    :return: 返回移动平均值
    :rtype: pandas.DataFrame
    """
    return df_close.ewm(span=window, min_periods=window, adjust=False).mean()


def cal_macd(df, short=12, long=26, mid=9):
    """
    计算MACD指标
    :param df: datframe数据
    :type df: pandas.DataFrame
    :param short: 短期天数
    :type short: int
    :param long: 长期天数
    :type long: int
    :param mid: 天数
    :type mid: int
    :return: 返回MACD指标dif、dea和macd
    :rtype: pandas.DataFrame
    """
    dif = ema(df.close, short) - ema(df.close, long)
    dea = ema(dif, mid)
    macd = (dif - dea) * 2
    return dif, dea, macd


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
        code_name = {"000001": "上证指数",
                     "sh880326": "铝"}
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


if __name__ == "__main__":
    print(get_date_month("20240521"))
