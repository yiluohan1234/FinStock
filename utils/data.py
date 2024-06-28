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
        period = int(freq[3:])
        # df = ak.stock_zh_a_hist_min_em(symbol=code, start_date=start_date, end_date=end_date, period="60",  adjust="qfq").iloc[:, [0, 1, 3, 4, 2, 7]]
        df = ak.stock_zh_a_minute(symbol=get_szsh_code(code), period=period, adjust="qfq")
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        df["date"] = pd.to_datetime(df["date"])
        df[df.columns.tolist()[1:]] = pd.DataFrame(df[df.columns.tolist()[1:]], dtype=float)
    # else:
    #     df = ak.stock_zh_a_hist(symbol=code, start_date=start_end_date, end_date=end_date, adjust="qfq").iloc[:, :6]
    #     df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', ]
    #     df = transfer_price_freq(df, freq)

    df['volume'] = round(df['volume'].astype('float') / 10000, 2)

    # 计算均线、volume均线、抵扣差、乖离率、k率
    for i in ema_list:
        df['ma{}'.format(i)] = round(df.close.rolling(i).mean(), precision)
        df['vma{}'.format(i)] = round(df.volume.rolling(i).mean(), precision)
        df['dkc{}'.format(i)] = round(df["close"] - df["close"].shift(i - 1), precision)
        df['bias{}'.format(i)] = round(
            (df["close"] - df["ma{}".format(i)]) * 100 / df["ma{}".format(i)],
            precision)
        df['k{}'.format(i)] = df.close.rolling(i).apply(cal_K)
        df['kp{}'.format(i)] = df.close.rolling(i).apply(lambda x: cal_K_predict(x))

    for i in fib_list:
        df['ma{}'.format(i)] = round(df.close.rolling(i).mean(), precision)
        df['vma{}'.format(i)] = round(df.volume.rolling(i).mean(), precision)
        df['dkc{}'.format(i)] = round(df["close"] - df["close"].shift(i - 1), precision)
        df['bias{}'.format(i)] = round(
            (df["close"] - df["ma{}".format(i)]) * 100 / df["ma{}".format(i)],
            precision)
        df['k{}'.format(i)] = df.close.rolling(i).apply(cal_K)
        df['kp{}'.format(i)] = df.close.rolling(i).apply(lambda x: cal_K_predict(x))

    # df['ATR'], df['stop'] = ATR(df, 14)
    df = pd.concat([df, ATR(df, 14)], axis=1)

    # BOLL计算 取N=20，M=2
    df = pd.concat([df, BOLL(df, 20, 2)], axis=1)

    # 计算包络线ENE(10,9,9)
    df = pd.concat([df, ENE(df, 10, 9)], axis=1)

    # 计算MACD
    df = pd.concat([df, MACD(df)], axis=1)

    # 计算KDJ
    df = pd.concat([df, KDJ(df)], axis=1)

    # 计算RSI
    df = pd.concat([df, RSI(df)], axis=1)

    # 标记买入和卖出信号
    df = pd.concat([df, find_max_min_point(df, 'kp10')], axis=1)

    # 过滤日期
    # df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # 计算volume的标识
    df['f'] = df.apply(lambda x: frb(x.open, x.close), axis=1)

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
            period = int(freq[3:])
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

    # 计算均线、volume均线、抵扣差、乖离率、k率
    for i in ema_list:
        df['ma{}'.format(i)] = round(df.close.rolling(i).mean(), precision)
        df['vma{}'.format(i)] = round(df.volume.rolling(i).mean(), precision)
        df['dkc{}'.format(i)] = round(df["close"] - df["close"].shift(i - 1), precision)
        df['bias{}'.format(i)] = round(
            (df["close"] - df["ma{}".format(i)]) * 100 / df["ma{}".format(i)],
            precision)
        df['k{}'.format(i)] = df.close.rolling(i).apply(cal_K)
        df['kp{}'.format(i)] = df.close.rolling(i).apply(cal_K_predict)

    for i in fib_list:
        df['ma{}'.format(i)] = round(df.close.rolling(i).mean(), precision)
        df['vma{}'.format(i)] = round(df.volume.rolling(i).mean(), precision)
        df['dkc{}'.format(i)] = round(df["close"] - df["close"].shift(i - 1), precision)
        df['bias{}'.format(i)] = round(
            (df["close"] - df["ma{}".format(i)]) * 100 / df["ma{}".format(i)],
            precision)
        df['k{}'.format(i)] = df.close.rolling(i).apply(cal_K)
        df['kp{}'.format(i)] = df.close.rolling(i).apply(lambda x: cal_K_predict(x))

    # df['ATR'], df['stop'] = ATR(df, 14)
    df = pd.concat([df, ATR(df, 14)], axis=1)

    # BOLL计算 取N=20，M=2
    df = pd.concat([df, BOLL(df, 20, 2)], axis=1)

    # 计算包络线ENE(10,9,9)
    df = pd.concat([df, ENE(df, 10, 9)], axis=1)

    # 计算MACD
    df = pd.concat([df, MACD(df)], axis=1)

    # 计算KDJ
    df = pd.concat([df, KDJ(df)], axis=1)

    # 计算RSI
    df = pd.concat([df, RSI(df)], axis=1)

    # 标记买入和卖出信号
    df = pd.concat([df, find_max_min_point(df, 'kp10')], axis=1)

    # 过滤日期
    # df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # 计算volume的标识
    df['f'] = df.apply(lambda x: frb(x.open, x.close), axis=1)

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
    - zh_index :str                  #是否为指数
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

    if not zh_index:
        df = get_data(code, start_end_date, end_date, freq)
    else:
        df = get_index_data(code, start_end_date, end_date, freq)
    df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]
    return df


# bond_zh_us_rate_df = ak.bond_zh_us_rate(start_date="19901219")
# print(bond_zh_us_rate_df[['日期', '中国国债收益率10年', '美国国债收益率10年']])
# from utils.plot import plot_df_line
# bar = plot_df_line(bond_zh_us_rate_df, '日期' , ['中国国债收益率10年', "美国国债收益率10年"])
# bar.render("./test.html")


if __name__ == "__main__":
    time_start = time.time()
    df = get_kline_chart_date(code="000977", start_date='20240101', end_date="20240202", freq='min60', zh_index=False)
    print(df[(df['BUY'] == True) | (df['SELL'] == True)][['date', 'MACD', 'DIF', 'DEA', 'BUY', 'SELL']])
    # print(df)
    time_end = time.time()
    print("运行耗时{}s".format(round(time_end-time_start, 2)))
