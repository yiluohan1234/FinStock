#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: fundflow.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年5月2日
#    > description: 资金流量
#######################################################################

import akshare as ak
from datetime import datetime
from utils.func import get_num2str_df, get_display_data
# 设置显示全部行，不省略
import pandas as pd
pd.set_option('display.max_rows', None)
# 设置显示全部列，不省略
pd.set_option('display.max_columns', None)
# pd.set_option('display.unicode.ambiguous_as_wide', True)
# pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 180)

def get_individual_fund_flow(code, n, is_display=True):
    '''
    获取个股资金流向
    :param code: 股票代码
    :type code: str
    :param n: 返回数据个数
    :type n: int
    :param is_display: 是否返回展示数据
    :type is_display: bool
    :return: 返回个股资金流动情况数据
    :rtype: pandas.DataFrame
    '''
    if int(code) > 600000:
        market = "sh"
    else:
        market = "sz"

    # market="sh"; 上海证券交易所: sh, 深证证券交易所: sz, 北京证券交易所: bj
    df = ak.stock_individual_fund_flow(stock=code, market=market)
    df = df.sort_values(by='日期', ascending=False)
    df = df.head(n)
    df_display = get_num2str_df(df.copy())
    df_display = get_display_data(df_display)
    for col in df.columns.tolist():
        if str(df[col].dtype) == 'float64':
            df[col] = df[col].apply(lambda x: round(x/100000000, 2))
    if is_display:
        return df_display
    else:
        return df


def get_individual_fund_flow_rank(code, indicator="今日", is_display=True):
    '''
    获取个股资金流动排名情况
    :param code: 股票代码
    :type code: str
    :param indicator: indicator="今日"; choice {"今日", "3日", "5日", "10日"}
    :type indicator: str
    :param is_display: 是否返回展示数据
    :type is_display: bool
    :return: 返回个股资金流动排名数据
    :rtype: pandas.DataFrame
    '''
    df = ak.stock_individual_fund_flow_rank(indicator=indicator)
    df = df[df['代码'] == code]
    for col in ['今日主力净流入-净额', '今日超大单净流入-净额', '今日大单净流入-净额', '今日中单净流入-净额', '今日小单净流入-净额']:
        df[col] = df[col].astype('float64')
    df_display = get_num2str_df(df.copy())
    df_display = get_display_data(df_display)

    if is_display:
        return df_display
    else:
        return df


def get_market_fund_flow(n, is_display=True):
    '''
    获取上证和深证市场资金流向
    :param n: 最近天数
    :type n: str
    :param is_display: 是否返回展示数据
    :type is_display: bool
    :return: 返回个股资金流动排名数据
    :rtype: pandas.DataFrame
    '''
    df = ak.stock_market_fund_flow()
    df = get_num2str_df(df)
    df = df.sort_values(by='日期', ascending=False)
    df = df.head(n)
    df_display = get_display_data(df.head(n))

    if is_display:
        return df_display
    else:
        return df


def get_main_fund_flow(symbol="全部股票"):
    '''主力净流入排名
    @params:
    - indicator: str         symbol="全部股票"；choice of {"全部股票", "沪深A股", "沪市A股", "科创板", "深市A股", "创业板", "沪市B股", "深市B股"}
    '''
    df = ak.stock_main_fund_flow(symbol=symbol)

    return df[df.columns.tolist()[1:]]


def get_sector_fund_flow_hist(symbol="有色金属"):
    '''行业历史资金流
    @params:
    - symbol: str            symbol="有色金属"
    '''
    df = ak.stock_sector_fund_flow_hist(symbol=symbol)
    df = get_num2str_df(df)
    df = df.sort_values(by='日期', ascending=False)

    return df


def get_sector_fund_flow(indicator="今日", sector_type="行业资金流"):
    '''获取行业资金流向
    @params:
    - indicator: str         #indicator="今日"; choice of {"今日", "5日", "10日"}
    - sector_type: str       #sector_type="行业资金流"; choice of {"行业资金流", "概念资金流", "地域资金流"}
    '''
    df = ak.stock_sector_fund_flow_rank(indicator=indicator, sector_type=sector_type)
    df = get_num2str_df(df)
    # df = df.sort_values(by='日期',ascending=False)
    # ret_df = self.get_display_data(df.head(n))

    return df

    #return df[df.columns.tolist()[1:]]


def get_concept_fund_flow_hist(symbol="锂电池"):
    '''概念历史资金流
    @params:
    - symbol: str            symbol="电源设备"
    '''
    df = ak.stock_concept_fund_flow_hist(symbol=symbol)

    return df


def get_cyq_em(symbol, n):
    '''筹码分布
    @params:
    - symbol: str            #股票代码
    - adjust: str            #adjust=""; choice of {"qfq": "前复权", "hfq": "后复权", "": "不复权"}
    '''
    df = ak.stock_cyq_em(symbol=symbol, adjust="qfq")

    return df.tail(n)


def get_north_data(start_date, end_date, symbol="北向资金"):
    '''北向资金历史流入情况
    @params:
    - start_date: int               #开始日期，'20240428'
    - end_date: int                 #结束日期， '20240508'
    - symbol: str            #symbol: choice of {"北向资金", "沪股通", "深股通", "南向资金", "港股通沪", "港股通深"}
    '''
    df = ak.stock_hsgt_hist_em(symbol)
    df = df.loc[(df['日期'].astype(str) >= datetime.strptime(start_date, '%Y%m%d').strftime("%Y-%m-%d")) & (df['日期'].astype(str) <= datetime.strptime(end_date, '%Y%m%d').strftime("%Y-%m-%d"))]
    return df


if __name__ == "__main__":
    # code = "000977"
    # k = get_individual_fund_flow_rank(code, is_display=True)
    # print(k)
    # k = get_individual_fund_flow(code, 10)
    # print(k)
    k = get_north_data(start_date="20240501", end_date="20240520")
    print(k)
