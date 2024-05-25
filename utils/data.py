#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: data.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年5月20日
#    > description: 数据获取工具
#######################################################################
import akshare as ak
import datetime
import pandas as pd
from utils.func import cal_K, cal_macd, frb, get_szsh_code, cal_K_predict
from utils.cons import ema_list, precision, transfer_date_dic


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
    elif freq == 'min':
        # stock_zh_a_hist_min_em
        df = ak.stock_zh_a_minute(symbol=get_szsh_code(code), period="60", adjust="qfq")
        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', ]
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
        df['kp{}'.format(i)] = df.close.rolling(i).apply(cal_K_predict)

    df['ATR1'] = df['high'] - df['low']  # 当日最高价-最低价
    df['ATR2'] = abs(df['close'].shift(1) - df['high'])  # 上一日收盘价-当日最高价
    df['ATR3'] = abs(df['close'].shift(1) - df['low'])  # 上一日收盘价-当日最低价
    df['ATR4'] = df['ATR1']
    for i in range(len(df)):  # 取价格波动的最大值
        if df.loc[i, 'ATR4'] < df.loc[i, 'ATR2']:
            df.loc[i, 'ATR4'] = df.loc[i, 'ATR2']
        if df.loc[i, 'ATR4'] < df.loc[i, 'ATR3']:
            df.loc[i, 'ATR4'] = df.loc[i, 'ATR3']
    df['ATR'] = df.ATR4.rolling(14).mean()  # N=14的ATR值
    df['stop'] = df['close'].shift(1) - df['ATR'] * 3  # 止损价=(上一日收盘价-3*ATR)

    # BOLL计算 取N=20，M=2
    df['boll'] = df.close.rolling(20).mean()
    df['delta'] = df.close - df.boll
    df['beta'] = df.delta.rolling(20).std()
    df['up'] = df['boll'] + 2 * df['beta']
    df['down'] = df['boll'] - 2 * df['beta']

    # 计算包络线ENE(10,9,9)
    # ENE代表中轨。MA(CLOSE,N)代表N日均价
    # UPPER:(1+M1/100)*MA(CLOSE,N)的意思是，上轨距离N日均价的涨幅为M1%；
    # LOWER:(1-M2/100)*MA(CLOSE,N) 的意思是，下轨距离 N 日均价的跌幅为 M2%;
    df['ene'] = df.close.rolling(10).mean()
    df['upper'] = (1 + 9.0 / 100) * df['ene']
    df['lower'] = (1 - 9.0 / 100) * df['ene']

    # 计算MACD
    # df['DIF'], df['DEA'], df['MACD'] = self.get_macd_data(df)
    df['DIF'], df['DEA'], df['MACD'] = cal_macd(df)

    # 标记买入和卖出信号
    # for i in range(len(df)):
    #     if i == 0:
    #         continue
    #     if (df.loc[i, 'k5'] >= df.loc[i, 'k10']) and (df.loc[i-1, 'k5'] < df.loc[i-1, 'k10']) and df.loc[i, 'k10'] > 0 and df.loc[i, 'k20'] > 0:
    #         df.loc[i, 'BUY'] = True
    #     if df.loc[i, 'close'] < df.loc[i, 'ene'] and df.loc[i, 'k20'] > 0:
    #         df.loc[i, 'SELL'] = True
    # 过滤日期
    df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]

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
    if freq == 'D' or freq == 'W' or freq == 'M':
        df = ak.index_zh_a_hist(symbol=code, period=transfer_date_dic[freq], start_date=start_date, end_date=end_date).iloc[:, :6]
        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', ]
        df["date"] = pd.to_datetime(df["date"])
    elif freq == 'min':
        df = ak.stock_zh_a_minute(symbol=code, period="60", adjust="qfq")
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', ]
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

    df['ATR1'] = df['high'] - df['low']  # 当日最高价-最低价
    df['ATR2'] = abs(df['close'].shift(1) - df['high'])  # 上一日收盘价-当日最高价
    df['ATR3'] = abs(df['close'].shift(1) - df['low'])  # 上一日收盘价-当日最低价
    df['ATR4'] = df['ATR1']
    for i in range(len(df)):  # 取价格波动的最大值
        if df.loc[i, 'ATR4'] < df.loc[i, 'ATR2']:
            df.loc[i, 'ATR4'] = df.loc[i, 'ATR2']
        if df.loc[i, 'ATR4'] < df.loc[i, 'ATR3']:
            df.loc[i, 'ATR4'] = df.loc[i, 'ATR3']
    df['ATR'] = df.ATR4.rolling(14).mean()  # N=14的ATR值
    df['stop'] = df['close'].shift(1) - df['ATR'] * 3  # 止损价=(上一日收盘价-3*ATR)

    # BOLL计算 取N=20，M=2
    df['boll'] = df.close.rolling(20).mean()
    df['delta'] = df.close - df.boll
    df['beta'] = df.delta.rolling(20).std()
    df['up'] = df['boll'] + 2 * df['beta']
    df['down'] = df['boll'] - 2 * df['beta']

    # 计算包络线ENE(10,9,9)
    # ENE代表中轨。MA(CLOSE,N)代表N日均价
    # UPPER:(1+M1/100)*MA(CLOSE,N)的意思是，上轨距离N日均价的涨幅为M1%；
    # LOWER:(1-M2/100)*MA(CLOSE,N) 的意思是，下轨距离 N 日均价的跌幅为 M2%;
    df['ene'] = df.close.rolling(10).mean()
    df['upper'] = (1 + 9.0 / 100) * df['ene']
    df['lower'] = (1 - 9.0 / 100) * df['ene']

    # 计算MACD
    # df['DIF'], df['DEA'], df['MACD'] = self.get_macd_data(df)
    df['DIF'], df['DEA'], df['MACD'] = cal_macd(df)

    # 标记买入和卖出信号
    # for i in range(len(df)):
    #     if df.loc[i, 'close'] > df.loc[i, 'up']:
    #         df.loc[i, 'SELL'] = True
    #     if df.loc[i, 'close'] < df.loc[i, 'boll']:
    #         df.loc[i, 'BUY'] = True
    # 过滤日期
    df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]

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
    time_start = datetime.datetime.now()
    # print(get_data("000612", start_date="20240501", end_date="20240521", freq='M'))
    # print(get_index_data("sh000001", start_date="20240110", end_date="20240519", freq='min'))
    # print(k)
    df = get_kline_chart_date(code="300547", start_date='20180301', end_date='20180519', freq='D', zh_index=False)
    print(df)
    time_end = datetime.datetime.now()
    print(f"运行耗时{(time_end - time_start).seconds}s")
