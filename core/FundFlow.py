#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: FundFlow.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年5月2日
#    > description: 资金流量
#######################################################################

import akshare as ak
import pandas as pd
from datetime import datetime


class FundFlow:

    def __init__(self):
        pass

    def num2str(self, num):
        '''实现数值转换为万，亿单位，保留2位小数
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

    def get_num2str_df(self, df):
        for col in df.columns.tolist():
            if str(df[col].dtype) == 'float64':
                df[col] = df[col].apply(lambda x: self.num2str(x))
        return df

    def str2value(self, valueStr):
        valueStr = str(valueStr)
        idxOfYi = valueStr.find('亿')
        idxOfWan = valueStr.find('万')
        if idxOfYi != -1 and idxOfWan != -1:
            return int(float(valueStr[:idxOfYi])*1e8 + float(valueStr[idxOfYi+1:idxOfWan])*1e4)
        elif idxOfYi != -1 and idxOfWan == -1:
            return int(float(valueStr[:idxOfYi])*1e8)
        elif idxOfYi == -1 and idxOfWan != -1:
            return int(float(valueStr[idxOfYi+1:idxOfWan])*1e4)
        elif idxOfYi == -1 and idxOfWan == -1:
            return float(valueStr)

    def get_individual_fund_flow(self, code, n):
        '''获取个股资金流向
        @params:
        - code: str      #股票代码
        - n: int         #最近天数
        '''
        if int(code) > 600000:
            market = "sh"
        else:
            market = "sz"

        # market="sh"; 上海证券交易所: sh, 深证证券交易所: sz, 北京证券交易所: bj
        df = ak.stock_individual_fund_flow(stock=code, market=market)
        df = df.sort_values(by='日期', ascending=False)
        df = df.head(n)
        df_display = self.get_num2str_df(df.copy())
        df_display = self.get_display_data(df_display)
        for col in df.columns.tolist():
            if str(df[col].dtype) == 'float64':
                df[col] = df[col].apply(lambda x: round(x/100000000, 2))

        return df, df_display

    def get_individual_fund_flow_rank(self, code, indicator="今日"):
        '''获取个股资金排名
        @params:
        - n: int              #排名
        - indicator: str      #indicator="今日"; choice {"今日", "3日", "5日", "10日"}
        '''
        df = ak.stock_individual_fund_flow_rank(indicator=indicator)
        df = df[df['代码'] == code]
        for col in ['今日主力净流入-净额', '今日超大单净流入-净额', '今日大单净流入-净额', '今日中单净流入-净额', '今日小单净流入-净额']:
            df[col] = df[col].astype('float64')
        df_display = self.get_num2str_df(df.copy())
        df_display = self.get_display_data(df_display)

        return df, df_display

    def get_market_fund_flow(self, n):
        '''获取市场资金流向
        @params:
        - n: int         #最近天数
        '''
        df = ak.stock_market_fund_flow()
        df = self.get_num2str_df(df)
        df = df.sort_values(by='日期', ascending=False)
        ret_df = self.get_display_data(df.head(n))

        return ret_df

    def get_main_fund_flow(self, symbol="全部股票"):
        '''主力净流入排名
        @params:
        - indicator: str         symbol="全部股票"；choice of {"全部股票", "沪深A股", "沪市A股", "科创板", "深市A股", "创业板", "沪市B股", "深市B股"}
        '''
        df = ak.stock_main_fund_flow(symbol=symbol)

        return df[df.columns.tolist()[1:]]

    def get_sector_fund_flow_summary(self, n, symbol="电源设备"):
        '''行业个股资金流
        @params:
        - symbol: str            symbol="电源设备"
        '''
        df = ak.stock_sector_fund_flow_hist(symbol=symbol)
        df = self.get_num2str_df(df)
        df = df.sort_values(by='日期', ascending=False)
        return df.head(n)

    def get_sector_fund_flow_hist(self, symbol="电源设备"):
        '''行业历史资金流
        @params:
        - symbol: str            symbol="电源设备"
        '''
        df = ak.stock_sector_fund_flow_hist(symbol=symbol)
        df = self.get_num2str_df(df)
        df = df.sort_values(by='日期', ascending=False)

        return df

    def get_sector_fund_flow(self, indicator="今日", sector_type="行业资金流"):
        '''获取行业资金流向
        @params:
        - indicator: str         #indicator="今日"; choice of {"今日", "5日", "10日"}
        - sector_type: str       #sector_type="行业资金流"; choice of {"行业资金流", "概念资金流", "地域资金流"}
        '''
        df = ak.stock_sector_fund_flow_rank(indicator=indicator, sector_type=sector_type)
        df = ak.stock_sector_fund_flow_rank(indicator=indicator, sector_type=sector_type)
        df = self.get_num2str_df(df)
        # df = df.sort_values(by='日期',ascending=False)
        # ret_df = self.get_display_data(df.head(n))

        return df

        #return df[df.columns.tolist()[1:]]

    def get_display_data(self, df):
        '''
        将数据进行转置
        @params:
        - df: dataframe      #数据
        '''
        ret_columns = df.columns.tolist()
        df_T = df.copy().set_index(ret_columns[0])
        index_row = df_T.index.tolist()
        df_display = pd.DataFrame(df_T.values.T, columns=index_row, index=ret_columns[1:])
        return df_display

    def get_concept_fund_flow_hist(self, symbol="锂电池"):
        '''概念历史资金流
        @params:
        - symbol: str            symbol="电源设备"
        '''
        df = ak.stock_concept_fund_flow_hist(symbol=symbol)

        return df

    def get_cyq_em(self, symbol, n):
        '''筹码分布
        @params:
        - symbol: str            #股票代码
        - adjust: str            #adjust=""; choice of {"qfq": "前复权", "hfq": "后复权", "": "不复权"}
        '''
        df = ak.stock_cyq_em(symbol=symbol, adjust="qfq")

        return df.tail(n)

    def get_north_data(self, start_date, end_date, symbol="北向资金"):
        '''北向资金历史流入情况
        @params:
        - start_date: int               #开始日期，'20240428'
        - end_date: int                 #结束日期， '20240508'
        - symbol: str            #symbol: choice of {"北向资金", "沪股通", "深股通", "南向资金", "港股通沪", "港股通深"}
        '''
        df = ak.stock_hsgt_hist_em(symbol)
        df = df.loc[(df['日期'].astype(str) >= datetime.strptime(start_date, '%Y%m%d').strftime("%Y-%m-%d")) & (df['日期'].astype(str) <= datetime.strptime(end_date, '%Y%m%d').strftime("%Y-%m-%d"))]
        return df
