#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: Basic.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年5月3日
#    > description: 基本面分析
#######################################################################

import matplotlib.pyplot as plt
import numpy as np
import akshare as ak
import pandas as pd
from matplotlib import ticker
from matplotlib.font_manager import FontProperties
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Tab, Pie, Page
# 内置主题类型可查看 pyecharts.globals.ThemeType
from pyecharts.globals import ThemeType
import webbrowser
import os


class Basic:

    def __init__(self, code="002230", start_date='20180330', end_date='20221231'):
        self.code = code
        self.start_date = start_date
        self.end_date = end_date

    def get_basic_info(self, code):
        '''同花顺-主营介绍
        @params:
        - code: str      # 股票代码
        '''
#         cninfo_df = ak.stock_profile_cninfo(symbol=code)
#         print("----------------------------- 简介 -----------------------------")
#         print("公司名称:", cninfo_df.iloc[0][0])
#         print("A股简称:", cninfo_df.iloc[0][4])
#         print("成立时间:", cninfo_df.iloc[0][14])
#         print("上市时间:", cninfo_df.iloc[0][15])
        zyjs_ths_df = ak.stock_zyjs_ths(symbol=code)
        print("主营业务:", zyjs_ths_df.iloc[0][1])  # '主营业务'
        print("产品类型:", zyjs_ths_df.iloc[0][2])  # '产品类型'
        print("产品名称:", zyjs_ths_df.iloc[0][3])  # '产品名称'
        print("经营范围:", zyjs_ths_df.iloc[0][4])  # '经营范围'

        # 主营构成-东财
        print("----------------------------- 主营构成 -----------------------------")
        zygc_em_df = ak.stock_zygc_em(symbol=self.get_szsh_code(code))
        zygc_em_df['分类类型'] = zygc_em_df['分类类型'].astype(str).apply(lambda x: x.replace('nan', '其他'))
        zygc_em_df['主营收入'] = round(zygc_em_df['主营收入'] / 100000000, 2)
        zygc_em_df['主营成本'] = round(zygc_em_df['主营成本'] / 100000000, 2)
        zygc_em_df['主营利润'] = round(zygc_em_df['主营利润'] / 100000000, 2)
        zygc_em_df['收入比例'] = round(zygc_em_df['收入比例'] * 100, 2)
        zygc_em_df['成本比例'] = round(zygc_em_df['成本比例'] * 100, 2)
        zygc_em_df['利润比例'] = round(zygc_em_df['利润比例'] * 100, 2)
        zygc_em_df['毛利率'] = round(zygc_em_df['毛利率'] * 100, 2)
        zygc_em_df['毛利率'] = round(zygc_em_df['毛利率'] * 100, 2)
        df = zygc_em_df[zygc_em_df['报告日期'].astype(str) == '2022-12-31']

        df = df.sort_values(by=['分类类型', '收入比例'], ascending=[False, False])
        # df[df.columns.tolist()[2:]]
        print(df[df.columns.tolist()[2:]].to_string(index=False))
        return df

    def str2value(self, valueStr):
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

    def get_basic_import_key(self, code, n, indicator="按年度"):
        '''获取股票代码带字母的
        @params:
        - code: str      # 股票代码
        - n: str         # 返回数据个数
        - indicator: str # 数据类型，indicator="按报告期"; choice of {"按报告期", "按年度", "按单季度"}
        '''
        # indicator="按报告期"; choice of {"按报告期", "按年度", "按单季度"}
        df = ak.stock_financial_abstract_ths(symbol=code, indicator=indicator)
        df = df.head(n)
        for col in df.columns.tolist()[1:]:
            df[col] = df[col].apply(self.str2value)
        # https://blog.csdn.net/a6661314/article/details/133634976
        df['报告期'] = df['报告期'].astype("str")

        df_display = self.get_display_data(df)
        return df, df_display

    def get_szsh_code(self, code):
        '''获取股票代码带字母的
        @params:
        - df: dataframe      #数据
        '''
        if code.find('60', 0, 3) == 0:
            gp_type = 'sh'
        elif code.find('688', 0, 4) == 0:
            gp_type = 'sh'
        elif code.find('900', 0, 4) == 0:
            gp_type = 'sh'
        elif code.find('00', 0, 3) == 0:
            gp_type = 'sz'
        elif code.find('300', 0, 4) == 0:
            gp_type = 'sz'
        elif code.find('200', 0, 4) == 0:
            gp_type = 'sz'
        return gp_type + code

    def get_display_data(self, df):
        '''将数据进行转置
        @params:
        - df: dataframe      #数据
        '''
        ret_columns = df.columns.tolist()
        df_T = df.copy().set_index(ret_columns[0])
        index_row = df_T.index.tolist()
        df_display = pd.DataFrame(df_T.values.T, columns=index_row, index=ret_columns[1:])
        return df_display

    def get_lrb_data(self, code, n, data_type=0):
        '''获取利润表数据
        @params:
        - code: str        #代码
        - n: int           #返回数据个数
        - data_type: int   #返回数据的形式，0为自然，1为季报，2为中报，4为年报
        '''
        ret_columns = ['报告日', '营业总收入', '营业总收入同比',
                       '营业总成本', '营业总成本同比',
                       '营业利润', '营业利润同比',
                       '利润总额', '利润总额同比',
                       '净利润', '净利润同比',
                       '归属于母公司所有者的净利润', '归属于母公司所有者的净利润同比',
                       '销售毛利率', '销售净利率', '销售成本率',
                       '净利润/营业总收入(%)', '营业利润/营业总收入(%)', '息税前利润/营业总收入(%)',
                       '营业总成本/营业总收入(%)', '销售费用/营业总收入(%)',
                       '管理费用/营业总收入(%)', '管理费用(含研发费用)/营业总收入(%)',
                       '财务费用/营业总收入(%)', '研发费用/营业总收入(%)']
        # df_lrb = ak.stock_financial_report_sina(stock=self.code, symbol='利润表')
        df_lrb = ak.stock_profit_sheet_by_report_em(symbol=self.get_szsh_code(code))
        df_lrb = df_lrb.fillna(0)

        # 过滤年报
        df_lrb['报告日'] = df_lrb['REPORT_DATE_NAME']
        # df_lrb = df_lrb.sort_index(ascending=False)
        # 营业收入及同比
        df_lrb['营业总收入'] = round(df_lrb['TOTAL_OPERATE_INCOME'] / 100000000, 2)
        df_lrb['营业总收入同比'] = round(df_lrb['TOTAL_OPERATE_INCOME_YOY'], 2)
        df_lrb['营业总成本'] = round(df_lrb['TOTAL_OPERATE_COST'] / 100000000, 2)
        df_lrb['营业总成本同比'] = round(df_lrb['TOTAL_OPERATE_COST_YOY'], 2)
        df_lrb['营业利润'] = round(df_lrb['OPERATE_PROFIT'] / 100000000, 2)
        df_lrb['营业利润同比'] = round(df_lrb['OPERATE_PROFIT_YOY'], 2)
        df_lrb['利润总额'] = round(df_lrb['TOTAL_PROFIT'] / 100000000, 2)
        df_lrb['利润总额同比'] = round(df_lrb['TOTAL_PROFIT_YOY'], 2)
        df_lrb['净利润'] = round(df_lrb['NETPROFIT'] / 100000000, 2)
        df_lrb['净利润同比'] = round(df_lrb['NETPROFIT_YOY'], 2)
        df_lrb['归属于母公司所有者的净利润'] = round(df_lrb['PARENT_NETPROFIT'] / 100000000, 2)
        df_lrb['归属于母公司所有者的净利润同比'] = round(df_lrb['PARENT_NETPROFIT_YOY'], 2)
        # 净利润/营业总收入(%)
        # 营业利润/营业总收入(%)
        # 息税前利润/营业总收入(%)
        # EBITDA/营业总收入(%)

        df_lrb['净利润/营业总收入(%)'] = round(df_lrb['NETPROFIT'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
        df_lrb['营业利润/营业总收入(%)'] = round(df_lrb['OPERATE_PROFIT'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
        # 息税前利润 （EBTI）= 净利润 + 财务费用 + 所得税费用
        df_lrb['息税前利润/营业总收入(%)'] = round(
            (df_lrb['NETPROFIT'] + df_lrb['FINANCE_EXPENSE'] + df_lrb['INCOME_TAX']) * 100 / df_lrb[
                'TOTAL_OPERATE_INCOME'], 2)
        df_lrb['营业总成本/营业总收入(%)'] = round(df_lrb['TOTAL_OPERATE_COST'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
        df_lrb['销售费用/营业总收入(%)'] = round(df_lrb['SALE_EXPENSE'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
        df_lrb['管理费用/营业总收入(%)'] = round(df_lrb['MANAGE_EXPENSE'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
        df_lrb['管理费用(含研发费用)/营业总收入(%)'] = round(
            (df_lrb['MANAGE_EXPENSE'] + df_lrb['RESEARCH_EXPENSE']) * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
        df_lrb['财务费用/营业总收入(%)'] = round(df_lrb['FINANCE_EXPENSE'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
        df_lrb['研发费用/营业总收入(%)'] = round(df_lrb['RESEARCH_EXPENSE'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
        # 销售毛利率=销售毛利/销售收入×100%=（销售收入-销售成本）/销售收入×100%= (营业收入 - 营业成本 / 营业收入) * 100%
        df_lrb['销售毛利率'] = round((df_lrb['OPERATE_INCOME'] - df_lrb['OPERATE_COST']) * 100 / df_lrb['OPERATE_INCOME'], 2)
        # 净利率=净利润/营业收入
        df_lrb['销售净利率'] = round(df_lrb['NETPROFIT'] * 100 / df_lrb['OPERATE_INCOME'], 2)
        # 销售成本率=销售成本/销售收入净额×100%
        df_lrb['销售成本率'] = round(df_lrb['OPERATE_COST'] * 100 / df_lrb['OPERATE_INCOME'], 2)

        if data_type == 1:
            df_lrb = df_lrb[df_lrb['REPORT_TYPE'] == '一季报']
        elif data_type == 2:
            df_lrb = df_lrb[df_lrb['REPORT_TYPE'] == '中报']
        elif data_type == 4:
            df_lrb = df_lrb[df_lrb['REPORT_TYPE'] == '年报']

        # 返回最近n个数据
        df_lrb = df_lrb[ret_columns].head(n)
        df_lrb_display = self.get_display_data(df_lrb)

        return df_lrb, df_lrb_display

    def get_zcfz_data(self, n, data_type=0):
        '''获取资产负债表数据
        @params:
        - n: int           #返回数据个数
        - data_type: int   #返回数据的形式，0为自然，1为季报，2为中报，4为年报
        '''
        ret_columns = ['报告日', '总资产', '总负债',
                       '资产负债率']
        df_zcfz = ak.stock_balance_sheet_by_report_em(symbol=self.get_szsh_code(self.code))
        df_zcfz = df_zcfz.fillna(0)
        # https://blog.csdn.net/a389085918/article/details/80284812

        df_zcfz['报告日'] = df_zcfz['REPORT_DATE_NAME']
        # df_zcfz = df_zcfz.sort_index(ascending=False)
        # 资产负债率
        df_zcfz['总资产'] = round(df_zcfz['TOTAL_ASSETS'] / 100000000, 2)
        df_zcfz['总负债'] = round(df_zcfz['TOTAL_LIABILITIES'] / 100000000, 2)
        df_zcfz['资产负债率'] = round(df_zcfz['TOTAL_LIABILITIES'] * 100 / df_zcfz['TOTAL_ASSETS'], 2)
        # 应收账款周转率=营业收入/（（期初应收账款+期末应收账款）/2）
        # 应收账款周转天数=365/应收账款周转率

        if data_type == 1:
            df_zcfz = df_zcfz[df_zcfz['REPORT_TYPE'] == '一季报']
        elif data_type == 2:
            df_zcfz = df_zcfz[df_zcfz['REPORT_TYPE'] == '中报']
        elif data_type == 4:
            df_zcfz = df_zcfz[df_zcfz['REPORT_TYPE'] == '年报']
        # 返回最近n个数据
        df_zcfz = df_zcfz[ret_columns].head(n)

        df_zcfz_display = self.get_display_data(df_zcfz)
        return df_zcfz, df_zcfz_display

    def get_xjll_data(self, n, data_type=0):
        '''获取现金流量表数据
        @params:
        - n: int           #返回数据个数
        - data_type: int   #返回数据的形式，0为自然，1为季报，2为中报，4为年报
        '''
        ret_columns = ['报告日', '总资产', '总负债',
                       '资产负债率']

        df_xjll = ak.stock_financial_report_sina(stock=self.code, symbol='现金流量表')
        df_xjll = df_xjll.fillna(0)

        if data_type == 1:
            df_xjll = df_xjll[df_xjll['REPORT_TYPE'] == '一季报']
        elif data_type == 2:
            df_xjll = df_xjll[df_xjll['REPORT_TYPE'] == '中报']
        elif data_type == 4:
            df_xjll = df_xjll[df_xjll['REPORT_TYPE'] == '年报']
        # 返回最近n个数据
        df_xjll = df_xjll[ret_columns].head(n)

        df_xjll_display = self.get_display_data(df_xjll)
        return df_xjll, df_xjll_display

    def get_zygc_data(self):
        '''获取主营构成数据
        @params:
        '''
        # 主营构成-东财
        zygc_em_df = ak.stock_zygc_em(symbol=self.get_szsh_code(self.code))
        zygc_em_df['主营收入'] = round(zygc_em_df['主营收入'] / 100000000, 2)
        zygc_em_df['主营成本'] = round(zygc_em_df['主营成本'] / 100000000, 2)
        zygc_em_df['主营利润'] = round(zygc_em_df['主营利润'] / 100000000, 2)
        zygc_em_df['收入比例'] = round(zygc_em_df['收入比例'] * 100, 2)
        zygc_em_df['成本比例'] = round(zygc_em_df['成本比例'] * 100, 2)
        zygc_em_df['利润比例'] = round(zygc_em_df['利润比例'] * 100, 2)
        zygc_em_df['毛利率'] = round(zygc_em_df['毛利率'] * 100, 2)
        df = zygc_em_df[zygc_em_df['报告日期'].astype(str) == '2023-12-31']
        df = df.sort_values(by=['分类类型', '收入比例'], ascending=[True, False])

        return df[df.columns.tolist()[2:]]

    def plot_amount_ratio(self, df, label, amount, ratio):
        '''绘制一个柱状图和一个折线图
        @params:
        - df: dataframe      #数据
        - label: str         #绘图x轴列名
        - amount: str        #绘图柱状图列名
        - ratio : str        #绘图折线图列名
        '''
        # https://blog.csdn.net/weixin_43364551/article/details/129590524
        # https://baijiahao.baidu.com/s?id=1762978115865709758&wfr=spider&for=pc
        # x坐标相同
        labels = [date[0:4] for date in df[label].values.tolist()]
        x = np.arange(len(labels))
        y1 = df[amount].tolist()
        y2 = df[ratio].tolist()
        # 引入系统中的字体（黑体）
        font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=10)

        # 解决导出图为空图层的问题
        # matplotlib.use('TkAgg')
        # 解决中文乱码问题，并设置字体
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = ['SimHei']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rc('font', family='SimHei', size=10)
        fontsize = 12

        # 调整画布大小 宽8高5
        plt.rcParams['figure.figsize'] = (8, 5)

        # 柱状图柱子宽度
        bar_width = 0.4

        # x轴刻度位置 | 折线图x位置
        x_ticks = range(len(x))
        # 柱状图 - x位置
        bar_x = [ii for ii in x_ticks]

        # 绘制双Y轴
        fig, ax1 = plt.subplots()
        # 设置x轴和y轴刻度字体
        labels1 = ax1.get_xticklabels() + ax1.get_yticklabels()
        [label.set_fontname('SimHei') for label in labels1]

        # 绘制柱状图1
        ax1.bar(bar_x, y1, lw=0.4, color="#1296db", edgecolor="k", label=amount, width=bar_width)
        for a, b in zip(bar_x, y1):
            ax1.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=fontsize, font=font)
        # 设置y轴的刻度范围
        bar_ylim_min = 0 if min(df[amount].values.tolist()) >= 0 else min(df[amount].values.tolist()) * 0.8
        bar_ylim_max = max(df[amount].values.tolist()) * 1.2
        ax1.set_ylim(bar_ylim_min, bar_ylim_max)
        # 设置y轴label
        ax1.set_ylabel(amount, fontsize=fontsize, font=font)
        # 设置图表 loc/bbox_to_anchor-位置参数, borderaxespad-间隙, prop-设置字体
        ax1.legend(loc=3, bbox_to_anchor=(0, 1), borderaxespad=0.2, prop=font)

        # 绘制双Y轴
        ax2 = ax1.twinx()
        # 设置x轴和y轴刻度字体
        labels2 = ax2.get_xticklabels() + ax2.get_yticklabels()
        [label.set_fontname('SimHei') for label in labels2]

        # 绘制折线图
        ax2.plot(x, y2, 'o-', color="#d81e06", label=ratio)
        for a, b in zip(x, y2):
            ax2.text(a + bar_width / 2, b, '{:.1f}%'.format(b), ha='center', va='bottom', fontsize=fontsize, font=font)
        # 设置y轴显示的数值区间
        line_ylim_min = 0 if min(df[ratio].values.tolist()) >= 0 else min(df[ratio].values.tolist()) * 0.8
        line_ylim_max = max(df[ratio].values.tolist()) * 1.2
        ax2.set_ylim(line_ylim_min, line_ylim_max)
        # y轴为百分比形式
        fmt = '%.0f%%'
        yticks = ticker.FormatStrFormatter(fmt)
        # ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))
        ax2.set_ylabel(ratio, fontsize=fontsize)
        ax2.legend(loc=3, bbox_to_anchor=(0.6, 1), borderaxespad=0.2, prop=font)

        plt.xticks(x_ticks, labels=labels)
        plt.tight_layout()
        plt.show()

    def get_df_markdown_table(self, df):
        '''获取dataframe数据类型并生成markdown表格
        @params:
        - df: dataframe      #数据
        '''
        column_types = df.dtypes.to_dict()
        print("| 列名 | 数据类型 |")
        print("| ---------------------------- | ---- |")
        for column, data_type in column_types.items():
            print(f"|{column}|{data_type}|")

    def plot_bar_line(self, df, label, amount_1, amount_2, ratio):
        '''绘制两个柱状图和一个折线图
        @params:
        - df: dataframe      #数据
        - label: str         #绘图x轴列名
        - amount_1: str      #绘图柱状图列名1
        - amount_2: str      #绘图柱状图列名2
        - ratio : str        #绘图折线图列名
        '''
        # https://blog.csdn.net/weixin_43364551/article/details/129590524
        # x坐标相同
        labels = [date[0:4] for date in df[label].values.tolist()]
        x = np.arange(len(labels))
        y1 = df[amount_1].tolist()
        y2 = df[amount_2].tolist()
        y3 = df[ratio].tolist()
        labels = [date[0:4] for date in df[label].values.tolist()]
        # 引入系统中的字体（黑体）
        font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=10)

        # 解决导出图为空图层的问题
        # matplotlib.use('TkAgg') # 在一个新窗口打开图形
        # 解决中文乱码问题，并设置字体
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = ['SimHei']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rc('font', family='SimHei', size=10)
        fontsize = 12

        # 调整画布大小 宽4.5 高2.2
        plt.rcParams['figure.figsize'] = (8, 5)

        # 柱状图柱子宽度
        bar_width = 0.3

        # x轴刻度位置 | 折线图x位置
        x_ticks = range(len(x))
        # 柱状图1 - x位置
        bar_1_x = [ii for ii in x_ticks]
        # 柱状图1 - x位置
        bar_2_x = [ww + bar_width for ww in x_ticks]

        # 绘制双Y轴
        fig, ax1 = plt.subplots()
        # 设置x轴和y轴刻度字体
        labels1 = ax1.get_xticklabels() + ax1.get_yticklabels()
        [label.set_fontname('SimHei') for label in labels1]

        # 绘制柱状图1
        ax1.bar(bar_1_x, y1, lw=0.4, color="#1296db", edgecolor="k", label=amount_1, width=bar_width)
        for a, b in zip(bar_1_x, y1):
            ax1.text(a - 0.05, b + 0.05, '%.2f' % b, ha='center', va='bottom', fontsize=fontsize, font=font)

        # 绘制柱状图2
        # hatch设置填充内容，*3用于设置密度
        # ax1.bar(bar_2_x, y2, lw=0.4, color="#d81e06", edgecolor="k", label=amount_2, width=bar_width, hatch='/'*3)
        ax1.bar(bar_2_x, y2, lw=0.4, color="#d81e06", edgecolor="k", label=amount_2, width=bar_width)
        for a, b in zip(bar_2_x, y2):
            ax1.text(a + 0.05, b + 0.05, '%.2f' % b, ha='center', va='bottom', fontsize=fontsize, font=font)
        # 设置y轴的刻度范围
        min_amount = min(df[amount_2].values.tolist()) if min(df[amount_1].values.tolist()) >= min(
            df[amount_2].values.tolist()) else min(df[amount_1].values.tolist())
        max_amount = max(df[amount_1].values.tolist()) if max(df[amount_1].values.tolist()) >= max(
            df[amount_2].values.tolist()) else max(df[amount_2].values.tolist())
        bar_ylim_min = 0 if min_amount >= 0 else min_amount * 0.8
        bar_ylim_max = max_amount * 1.2
        ax1.set_ylim(bar_ylim_min, bar_ylim_max)
        # 设置y轴label
        ax1.set_ylabel(amount_1 + "和" + amount_2, fontsize=fontsize, font=font)
        # 设置图表 loc/bbox_to_anchor-位置参数, borderaxespad-间隙, prop-设置字体
        ax1.legend(loc=3, bbox_to_anchor=(0, 1), borderaxespad=0.2, prop=font)

        # 绘制双Y轴
        ax2 = ax1.twinx()
        # 设置x轴和y轴刻度字体
        labels2 = ax2.get_xticklabels() + ax2.get_yticklabels()
        [label.set_fontname('SimHei') for label in labels2]

        # 绘制折线图
        ax2.plot(x, y3, 'o-', color="#d4237a", label=ratio)
        for a, b in zip(x, y3):
            ax2.text(a, b - 0.004, '{:.1f}%'.format(b), ha='center', va='bottom', fontsize=fontsize, font=font)
        # 设置y轴显示的数值区间
        line_ylim_min = 0 if min(df[ratio].values.tolist()) >= 0 else min(df[ratio].values.tolist()) * 0.8
        line_ylim_max = max(df[ratio].values.tolist()) * 1.2
        ax2.set_ylim(line_ylim_min, line_ylim_max)
        # y轴为百分比形式
        fmt = '%.0f%%'
        yticks = ticker.FormatStrFormatter(fmt)
        # ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))
        ax2.set_ylabel(ratio, fontsize=fontsize)
        ax2.legend(loc=3, bbox_to_anchor=(0.6, 1), borderaxespad=0.2, prop=font)

        plt.xticks(x_ticks, labels=labels)
        plt.tight_layout()
        plt.show()
        # stock_profile_cninfo_df = ak.stock_profile_cninfo(symbol="600030")
        # b = Basic("600519")
        # data, data_display = b.get_lrb_data(4)
        # b.plot_bar_line(data, '报告日', '营业总收入', '营业总成本', '营业总收入同比')
        # b.plot_amount_ratio(data.tail(5), '报告日', '净利润', '净利润同比')

    def plot_bar(self, data, x, y, title):
        bar = (Bar(init_opts=opts.InitOpts(width="600px", height="400px",theme=ThemeType.DARK))
                .add_xaxis(xaxis_data=data[x].tolist())
                .add_yaxis(
                    series_name=y,
                    y_axis=data[y].values.tolist()
                )
                #.reversal_axis() # 旋转柱形图方向
                .set_series_opts(label_opts=opts.LabelOpts(position="right")) # 设置数字标签位置
                .set_global_opts(
                    title_opts=opts.TitleOpts(is_show=True, title=title, pos_left="center"),
                    yaxis_opts=opts.AxisOpts(
                        name="{}".format(y),
                        type_="value",
                        axislabel_opts=opts.LabelOpts(formatter="{value}"),  # 设置刻度标签的单位
                        axistick_opts=opts.AxisTickOpts(is_show=True),           # 显示坐标轴刻度
                        splitline_opts=opts.SplitLineOpts(is_show=True),         # 显示分割线
                    ),
                    visualmap_opts=opts.VisualMapOpts(
                         max_= max(data[y].values.tolist()),
                          min_= min(data[y].values.tolist()),
                          range_color = ['#ffe100','#e82727'],
                          pos_right='10%',
                          pos_top='60%',
                          is_show=False
                    ),
                )
        )
        return bar

    def plot_line(self, data, x, y, title):
        print(data[x].tolist())
        print(data[y].tolist())
        line = (Line(init_opts=opts.InitOpts(width="600px", height="400px"))
                .add_xaxis(xaxis_data=data[x].tolist())
                .add_yaxis(
                    series_name=y,
                    y_axis=data[y].values.tolist(),
                    label_opts=opts.LabelOpts(is_show=False),
                    symbol="triangle",
                    symbol_size=20,
                )
                .set_series_opts(
                    linestyle_opts=opts.LineStyleOpts(width= 2)
                )   # 设置线条宽度为4
                .set_global_opts(
                    title_opts=opts.TitleOpts(title=title),
                    yaxis_opts=opts.AxisOpts(
                        name="{}".format(y),
                        type_="value",
                        axislabel_opts=opts.LabelOpts(formatter="{value}"),  # 设置刻度标签的单位
                        axistick_opts=opts.AxisTickOpts(is_show=True),           # 显示坐标轴刻度
                        splitline_opts=opts.SplitLineOpts(is_show=True),         # 显示分割线
                    ),
                )
        )
        return line

    def bar_over_line(self, df, x, y_bar, y_line):
        x_data = df[x].tolist()
        y_bar_data = df[y_bar].values.tolist()
        y_line_data = df[y_line].values.tolist()
        bar = (Bar(init_opts=opts.InitOpts(width="600px", height="400px"))
                .add_xaxis(xaxis_data=x_data)
                .add_yaxis(
                    series_name=y_bar,    # 此处为设置图例配置项的名称
                    y_axis=y_bar_data,
                    label_opts=opts.LabelOpts(is_show=False),   # 此处将标签配置项隐藏
                    z=0     # 因为折线图会被柱状图遮挡，所以此处把柱状图置底
                )
                .extend_axis(
                    yaxis=opts.AxisOpts(
                        name="同比增速（%）",
                        type_="value",
                        #axislabel_opts=opts.LabelOpts(formatter="{value} %"),  # 设置刻度标签的单位
                    )
                )
                .set_global_opts(
                    # 设置提示框配置项，触发类型为坐标轴类型，指示器类型为"cross"
                    tooltip_opts=opts.TooltipOpts(
                        is_show=True, trigger="axis", axis_pointer_type="cross"
                    ),
                    # 设置x轴配置项为类目轴，适用于离散的类目数据
                    xaxis_opts=opts.AxisOpts(
                        type_="category",
                        axispointer_opts=opts.AxisPointerOpts(is_show=True, type_="shadow"),
                    ),
                    yaxis_opts=opts.AxisOpts(
                        name="{}（亿元）".format(y_bar),
                        type_="value",
                        #axislabel_opts=opts.LabelOpts(formatter="{value} 亿元"),  # 设置刻度标签的单位
                        axistick_opts=opts.AxisTickOpts(is_show=True),           # 显示坐标轴刻度
                        splitline_opts=opts.SplitLineOpts(is_show=True),         # 显示分割线
                    ),
                    # 设置标题并将其居中
                    title_opts=opts.TitleOpts(
                        is_show=True, title="{}及同比增速".format(y_bar), pos_left="center"
                    ),
                    # 设置图例配置项，并将其放在右下角
                    legend_opts=opts.LegendOpts(
                        pos_right="right",
                        pos_bottom="bottom",
                        is_show=False
                    ),
                )
        )

        line = (Line()
                .add_xaxis(xaxis_data=x_data)
                .add_yaxis(
                    series_name="同比增速",
                    yaxis_index=1,
                    y_axis=y_line_data,
                    label_opts=opts.LabelOpts(is_show=False),
                    symbol="triangle",
                    symbol_size=20,
                )
                .set_series_opts(
                    linestyle_opts=opts.LineStyleOpts(width= 4)
                )   # 设置线条宽度为4
        )

        bar.overlap(line)
        return bar

    def plot_tab(self, df):
        tab = (
            Tab() # 创建Tab类对象
            .add(
                self.bar_over_line(df, '报告日', '营业总收入', '营业总收入同比'), # 图表类型
                "营业总收入" # 选项卡的标签名称
            )
            .add(
                self.bar_over_line(df, '报告日', '净利润', '净利润同比'),
                "净利润"
            )
        )
        return tab



    def title(self, title):
        from datetime import datetime
        now_time = datetime.now().strftime('%Y-%m-%d') # 获取当前时间
        pie = (
            Pie(init_opts=opts.InitOpts(width="600px", height="100px"#,theme=ThemeType.DARK
                                        )) # 不画图，只显示一个标题，用来构成大屏的标题
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title=title,
                        subtitle = f'截至：{now_time}',
                        title_textstyle_opts=opts.TextStyleOpts(
                            font_size=30,
                            #color='#FFFFFF',
                        ),
                        #pos_top=10,
                        pos_left="center"
                    ),
                    legend_opts=opts.LegendOpts(
                        is_show=False
                    )
                )
        )
        return pie
        #https://blog.csdn.net/Student_201/article/details/131189638
    def plot_pie(self, data, x, y, title, classify_type):
        # bar = plot_pie(data, '主营构成', '主营收入', '按产品分类主营构成', '按产品分类')
        data = data[data['分类类型']==classify_type]
        data = data[[x, y]]
        pie = (
            Pie(init_opts=opts.InitOpts(width="600px", height="400px")) # 设置背景的大小
            .add(
                series_name = "按产品分类", # 必须项
                data_pair = data.values.tolist(),
                radius=["20%", "50%"], # 设置环的大小
                rosetype="radius", # 设置玫瑰图类型
                label_opts=opts.LabelOpts(formatter="{b}：{c}\n占比：{d}%"), # 设置标签内容格式
            )
            .set_global_opts(title_opts=opts.TitleOpts(title=title))
        )
        return pie

    def plot_multi_bar(self, x, y, df_list, names_list):
        '''绘制多个柱状图对比图
        @params:
        - x: str             #x轴的名称
        - y: str             #y轴的名称
        - df_list: list      #dataframe列表
        - names_list: list   #公司名称列表
        '''
        x_data = df_list[0][x].tolist()


        if len(df_list) != 0:
            _bar = Bar(init_opts=opts.InitOpts(width='600px',height='400px')).add_xaxis(x_data)
            for i, df in enumerate(df_list):

                _bar.add_yaxis(series_name=names_list[i],
                               y_axis=df[y].values.tolist(),
                               label_opts=opts.LabelOpts(is_show=False),
                               )
                _bar.set_global_opts(
                    tooltip_opts=opts.TooltipOpts(is_show=True, trigger="axis", axis_pointer_type="cross"),
                    xaxis_opts=opts.AxisOpts(type_="category", is_show=True),
                    yaxis_opts=opts.AxisOpts(
                        type_="value",
                        axistick_opts=opts.AxisTickOpts(is_show=True),
                        splitline_opts=opts.SplitLineOpts(is_show=True),
                        is_show=True
                    ),
                )
        # bar = plot_multi_bar('报告日', '营业总收入', [df, df1], ['612', '977'])
        return _bar

    def plot_multi_line(self, x, y, df_list, names_list):
        '''绘制多个折线图对比图
        @params:
        - x: str             #x轴的名称
        - y: str             #y轴的名称
        - df_list: list      #dataframe列表
        - names_list: list   #公司名称列表
        # line = plot_multi_line('报告日', '营业总收入', [df, df1], ['612', '977'])
        '''
        x_data = df_list[0][x].tolist()


        if len(df_list) != 0:
            _line = Line(init_opts=opts.InitOpts(width='600px',height='400px')).add_xaxis(x_data)
            for i, df in enumerate(df_list):

                _line.add_yaxis(series_name=names_list[i],
                               y_axis=df[y].values.tolist(),
                               label_opts=opts.LabelOpts(is_show=False),
                               )
                _line.set_global_opts(
                    tooltip_opts=opts.TooltipOpts(is_show=True, trigger="axis", axis_pointer_type="cross"),
                    xaxis_opts=opts.AxisOpts(type_="category", is_show=True),
                    yaxis_opts=opts.AxisOpts(
                        type_="value",
                        axistick_opts=opts.AxisTickOpts(is_show=True),
                        splitline_opts=opts.SplitLineOpts(is_show=True),
                        is_show=True
                    ),
                )
        # line = plot_multi_line('报告日', '营业总收入', [df, df1], ['612', '977'])
        return _line

    def plot_page(self):
        from core.FundFlow import FundFlow
        # df_lrb, df_lrb_display = self.get_lrb_data("000977", 5, 4)
        # df_lrb1, df_lrb_display1 = self.get_lrb_data("000612", 5, 4)
        f = FundFlow()
        df_fund, df_fund_display = f.get_individual_fund_flow("000977", 5)

        #df_zygc = self.get_basic_info("000977")

        b = Basic()
        df_import, df_import_display= b.get_basic_import_key("000977", 5)
        for col in ['营业总收入', '净利润', '扣非净利润']:
            df_import[col] = round(df_import[col]/100000000, 2)



        page = Page(layout=Page.DraggablePageLayout, page_title="")

        page.add(
            self.title("test"),
            #self.bar_over_line(df_lrb, '报告日', '营业总收入', '营业总收入同比'),
            # self.plot_line(df_fund, '日期', '主力净流入-净额', '资金流量'),
            # self.plot_multi_bar('报告日', '营业总收入', [df_lrb, df_lrb1], ['612', '977'])
            #self.plot_pie(df_zygc, '主营构成', '主营收入', '按产品分类主营构成', '按产品分类')
            self.title("盈利能力"),
            self.bar_over_line(df_import, '报告期', '营业总收入', '营业总收入同比增长率'),
            # self.bar_over_line(df_import, '报告期', '净利润', '净利润同比增长率'),
            # self.bar_over_line(df_import, '报告期', '扣非净利润', '扣非净利润同比增长率'),
            # self.title("财务风险"),
            # self.title("运营能力"),

        )
        page.render('visual.html')
#         # 用于 DraggablePageLayout 布局重新渲染图表
#         page.save_resize_html(
#             # Page 第一次渲染后的 html 文件
#             source="visual.html",
#             # 布局配置文件
#             cfg_file="visual.json",
#             # 重新生成的 .html 存放路径
#             dest="visual_new.html"
#         )

        webbrowser.open_new_tab('file://' + os.path.realpath('visual.html'))
