#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: data.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年5月20日
#    > description: 数据获取工具
#######################################################################
import time
from utils.func import *
from utils.cons import *
import re
# 设置显示全部行，不省略
import pandas as pd
pd.set_option('display.max_rows', None)
# 设置显示全部列，不省略
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)

def get_data(code, start_date, end_date, freq):
    '''
    获取股票的综合数据
    :param code: 股票代码
    :type code: str
    :param start_date: 开始时间
    :type start_date: str
    :param end_date: 结束时间
    :type end_date: str
    :param freq: 频次，'D': 天, 'min': 分钟
    :type freq: str
    :return: 返回股票综合数据
    :rtype: pandas.DataFrame
    '''
    if freq == 'D' or freq == 'W' or freq == 'M':
        df = ak.stock_zh_a_hist(symbol=code, period=transfer_date_dic[freq], start_date=start_date, end_date=end_date, adjust="qfq").iloc[:, :6]
        # df = ak.stock_zh_a_daily(symbol=self.get_szsh_code(code), start_date=start,end_date=end_date, adjust="qfq")
        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', ]
        df["date"] = pd.to_datetime(df["date"])
    else:
        period = freq[3:]
        # df = ak.stock_zh_a_hist_min_em(symbol=code, start_date=start_date, end_date=end_date, period="60",  adjust="qfq").iloc[:, [0, 1, 3, 4, 2, 7]]
        df = ak.stock_zh_a_minute(symbol=get_szsh_code(code), period=period, adjust="qfq")
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        df["date"] = pd.to_datetime(df["date"])
        df[df.columns.tolist()[1:]] = pd.DataFrame(df[df.columns.tolist()[1:]], dtype=float)

    df['volume'] = round(df['volume'].astype('float') / 10000, 2)

    # 计算主要指标
    df = MAIN_INDICATOR(df)

    # 把date作为日期索引
    df.index = df.date
    return df


def get_index_data(code, start_date, end_date, freq):
    '''
    获取股票的综合数据
    :param code: 股票代码
    :type code: str
    :param start_date: 开始时间
    :type start_date: str
    :param end_date: 结束时间
    :type end_date: str
    :param freq: 频次，'D': 天, 'min': 分钟
    :type freq: str
    :return: 返回股票综合数据
    :rtype: pandas.DataFrame
    '''
    if bool(re.search("[a-zA-Z]", code)):
        df = ak.stock_zh_index_daily_em(symbol=code, start_date=start_date, end_date=end_date).iloc[:, :6]
        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', ]
        df["date"] = pd.to_datetime(df["date"])
    else:
        if freq == 'D' or freq == 'W' or freq == 'M':
            df = ak.index_zh_a_hist(symbol=code, period=transfer_date_dic[freq], start_date=start_date, end_date=end_date).iloc[:, :6]
            df.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
            df["date"] = pd.to_datetime(df["date"])
        else:
            period = freq[3:]
            # df = ak.stock_zh_a_minute(symbol=code, period="60", adjust="qfq")
            # df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', ]
            df = ak.index_zh_a_hist_min_em(symbol=code, period=period, start_date=start_date, end_date=end_date).iloc[:, :6]
            df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', ]
            df["date"] = pd.to_datetime(df["date"])
            df[df.columns.tolist()[1:]] = pd.DataFrame(df[df.columns.tolist()[1:]], dtype=float)
        # else:
        #     df = ak.stock_zh_index_daily(symbol=code).iloc[:, :6]
        #     df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', ]
        #     df = transfer_price_freq(df, freq)

    df['volume'] = round(df['volume'].astype('float') / 100000000, 2)

    # 计算主要指标
    df = MAIN_INDICATOR(df)

    # 把date作为日期索引
    df.index = df.date
    return df


def get_concept_data(symbol, start_date, end_date, freq):
    '''
    获取概念板块的综合数据
    :param symbol: 概念名称
    :type symbol: str
    :param start_date: 开始时间
    :type start_date: str
    :param end_date: 结束时间
    :type end_date: str
    :param freq: 频次，'D': 天, 'min': 分钟
    :type freq: str
    :return: 返回股票综合数据
    :rtype: pandas.DataFrame
    '''
    if freq == 'D' or freq == 'W' or freq == 'M':
        df = ak.stock_board_concept_hist_em(symbol=symbol, start_date=start_date,
                                             end_date=end_date, period=transfer_date_dic[freq],
                                             adjust="qfq").iloc[:, [0, 1, 2, 3, 4, 7]]
        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
        df["date"] = pd.to_datetime(df["date"])
    else:
        period = freq[3:]
        df = ak.stock_board_concept_hist_min_em(symbol=symbol, period=period).iloc[:, [0, 1, 2, 3, 4, 7]]
        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
        df["date"] = pd.to_datetime(df["date"])

    df['volume'] = round(df['volume'].astype('float') / 10000, 2)

    # 计算主要指标
    df = MAIN_INDICATOR(df)

    # 把date作为日期索引
    df.index = df.date
    return df


def get_industry_data(symbol, start_date, end_date, freq):
    '''
    获取行业板块的综合数据
    :param symbol: 行业名称
    :type symbol: str
    :param start_date: 开始时间
    :type start_date: str
    :param end_date: 结束时间
    :type end_date: str
    :param freq: 频次，'D': 天, 'min': 分钟
    :type freq: str
    :return: 返回股票综合数据
    :rtype: pandas.DataFrame
    '''
    if freq == 'D' or freq == 'W' or freq == 'M':
        df = ak.stock_board_industry_hist_em(symbol=symbol, period=transfer_date_dic_zh[freq], start_date=start_date, end_date=end_date,
                                            adjust="qfq").iloc[:, [0, 1, 2, 3, 4, 7]]
        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
        df["date"] = pd.to_datetime(df["date"])
    else:
        period = freq[3:]
        df = ak.stock_board_industry_hist_min_em(symbol=symbol, period=period).iloc[:, [0, 1, 2, 3, 4, 7]]
        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
        df["date"] = pd.to_datetime(df["date"])

    df['volume'] = round(df['volume'].astype('float') / 10000, 2)

    # 计算主要指标
    df = MAIN_INDICATOR(df)

    # 把date作为日期索引
    df.index = df.date
    return df


def get_kline_chart_date(code, start_date, end_date, freq, zh_index):
    '''
    @params:
    - code: str                      #股票代码
    - start_date: str                #开始时间, 如'202000101'
    - end_date: str                  #结束时间, 如'20240202'
    - freq : str                     #默认 'D' :日线数据
    - zh_index :str                  #类型，stock：股票，index：指数，industry：行业，concept：概念
    '''
    date_s = datetime.datetime.strptime(start_date, "%Y%m%d")
    start_end_date = (date_s - datetime.timedelta(days=365)).strftime('%Y%m%d')
    if end_date == '20240202':
        now = datetime.datetime.now()
        if freq[0:3] == 'min':
            end_date = (now + datetime.timedelta(days=1)).strftime('%Y%m%d')
        else:
            if now.hour >= 15:
                end_date = now.strftime('%Y%m%d')
            else:
                yesterday = now - datetime.timedelta(days=1)
                end_date = yesterday.strftime('%Y%m%d')

    if zh_index == 'stock':
        df = get_data(code, start_end_date, end_date, freq)
    elif zh_index == 'index':
        df = get_index_data(code, start_end_date, end_date, freq)
    elif zh_index == 'industry':
        df = get_industry_data(code, start_end_date, end_date, freq)
    else:
        df = get_concept_data(code, start_end_date, end_date, freq)
    df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]
    return df


if __name__ == "__main__":
    time_start = time.time()
    df = get_kline_chart_date(code="000977", start_date='20240101', end_date="20240202", freq='min60', zh_index='stock')
    print(df[(df['BUY'] == True) | (df['SELL'] == True)][['date', 'MACD', 'DIF', 'DEA', 'BUY', 'SELL']])
    # print(df)
    time_end = time.time()
    print("运行耗时{}s".format(round(time_end-time_start, 2)))
