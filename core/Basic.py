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
from pyecharts.components import Table
# 内置主题类型可查看 pyecharts.globals.ThemeType
from pyecharts.globals import ThemeType
import webbrowser
import os


class Basic:

    def __init__(self, code="002230", start_date='20180330', end_date='20221231'):
        self.code = code
        self.start_date = start_date
        self.end_date = end_date

    def get_basic_info(self, code, indicator='2023-12-31'):
        '''同花顺-主营介绍
        @params:
        - code: str      # 股票代码
        '''
        cninfo_df = ak.stock_profile_cninfo(symbol=code)
        print("----------------------------- 简介 -----------------------------")
        print("公司名称:", cninfo_df.iloc[0][0])
        print("A股简称:", cninfo_df.iloc[0][4])
        print("成立时间:", cninfo_df.iloc[0][14])
        print("上市时间:", cninfo_df.iloc[0][15])
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
        df = zygc_em_df[zygc_em_df['报告日期'].astype(str) == indicator]

        df = df.sort_values(by=['分类类型', '收入比例'], ascending=[False, False])
        # df[df.columns.tolist()[2:]]
        # print(df[df.columns.tolist()[2:]].to_string(index=False))
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

    def get_main_indicators_ths(self, code, n, indicator="按年度"):
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
        for col in ['营业总收入', '归母净利润', '扣非净利润']:
            df[col] = round(df[col]/100000000, 2)
        # https://blog.csdn.net/a6661314/article/details/133634976
        df['报告期'] = df['报告期'].astype("str")

        df_display = self.get_display_data(df)
        return df, df_display

    def get_report_type(self, date_str):
        '''根据日期获取报告类别
        @params:
        - date_str: str      # 报告日期，'20241231'
        '''
        if "1231" in date_str:
            return "年报"
        elif "0630" in date_str:
            return "中报"
        elif "0930" in date_str:
            return "三季度报"
        elif "0331" in date_str:
            return "一季度报"

    def get_main_indicators_sina(self, code, n, indicator="按年度", ret_columns=[]):
        '''获取股票代码带字母的
        @params:
        - code: str      # 股票代码
        - n: str         # 返回数据个数
        - indicator: str # 数据类型，indicator="按报告期"; choice of {"按报告期", "按年度", "按单季度"}
        '''
        df = ak.stock_financial_abstract(code)
        df.drop(['选项'] ,axis=1, inplace=True)
        dT = self.get_display_data(df)
        dT.index.name = '报告期'
        df = dT.reset_index()
        df['报告类型'] = df['报告期'].apply(self.get_report_type)

        for col in ['归母净利润', '营业总收入', '营业成本', '净利润', '扣非净利润', '股东权益合计(净资产)', '经营现金流量净额']:
            df[col] = df[col]/100000000

        for col in df.columns.tolist()[1:]:
            df[col] = round(df[col], 2)

        if indicator == "按年度":
            df = df[df['报告类型'] == '年报']
            df['报告期'] = df['报告期'].apply(lambda x: str(x)[0:4])

        df = df.head(n)
        if len(ret_columns) != 0:
            df = df[ret_columns]

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

    def get_xjll_data(self, code, n, data_type=0):
        '''获取现金流量表数据
        @params:
        - n: int           #返回数据个数
        - data_type: int   #返回数据的形式，0为自然，1为季报，2为中报，4为年报
        '''
        ret_columns = ['报告日', '总资产', '总负债',
                       '资产负债率']
        # 人力投入回报率=企业净利润/员工薪酬福利总额×100%，这是衡量人力资本有效性的核心指标，表明公司在人力资源上的投入和净利润的比值，回报率越高，说明人力资源的效率和效能越高。
        df_xjll = ak.stock_cash_flow_sheet_by_report_em(symbol=self.get_szsh_code(code))
        df_xjll['员工薪酬福利总额'] = round(df_xjll['PAY_STAFF_CASH']/100000000, 2)
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

    def get_df_markdown_table(self, df):
        '''获取dataframe数据类型并生成markdown表格
        @params:
        - df: dataframe      #数据
        '''
        column_types = df.dtypes.to_dict()
        print("| 列名 | 数据类型 |")
        print("| ---------------------------- | ---- |")
        for column, data_type in column_types.items():
            print("|{}|{}|".format(column, data_type))

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
        line = (Line(init_opts=opts.InitOpts(width="600px", height="400px"))
                .add_xaxis(xaxis_data=data[x].tolist())
                .add_yaxis(
                    series_name=y,
                    y_axis=data[y].values.tolist(),
                    label_opts=opts.LabelOpts(is_show=False),
                    # symbol="triangle",
                    # symbol_size=20,
                )
                .set_series_opts(
                    linestyle_opts=opts.LineStyleOpts(width=1)
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

    def plot_bar_line(self, df, x, y_bar, y_line):
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
                        # subtitle = f'截至：{now_time}',
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
        data = data[data['分类类型'] == classify_type]
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
            _bar = Bar().add_xaxis(x_data) #init_opts=opts.InitOpts(width='600px',height='400px')
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

    def plot_table(self, data, headers, title):
        table = Table()

        rows = data[headers].values().tolist()
        table.add(headers, rows).set_global_opts(
            title_opts=opts.ComponentTitleOpts(title=title)
        )
        return table

    def plot_page(self):
        from core.FundFlow import FundFlow
        df_lrb, df_lrb_display = self.get_lrb_data("000977", 5, 4)
        # df_lrb1, df_lrb_display1 = self.get_lrb_data("000612", 5, 4)
        f = FundFlow()
        df_fund, df_fund_display = f.get_individual_fund_flow("000977", 5)

        #df_zygc = self.get_basic_info("000977")

        b = Basic()
        df_import, df_import_display = b.get_main_indicators_ths("000977", 5)
        df_main, df_main_display = b.get_main_indicators_sina("000977", 5)



        # df_north = f.get_north_data(start_date='20240202', end_date='20240511')
        # df_sh = f.get_north_data(start_date='20240202', end_date='20240511', symbol="沪股通")
        # df_sz = f.get_north_data(start_date='20240202', end_date='20240511', symbol="深股通")

        page = Page(layout=Page.DraggablePageLayout, page_title="")

        page.add(
            self.title("test"),
            #self.plot_bar_line(df_lrb, '报告日', '营业总收入', '营业总收入同比'),
            # self.plot_line(df_fund, '日期', '主力净流入-净额', '资金流量'),
            # self.plot_multi_bar('报告日', '营业总收入', [df_lrb, df_lrb1], ['612', '977'])
            #self.plot_pie(df_zygc, '主营构成', '主营收入', '按产品分类主营构成', '按产品分类')
            # self.title("关键指标"),
            # self.plot_bar_line(df_import, '报告期', '营业总收入', '营业总收入同比增长率'),
            # self.plot_bar_line(df_lrb, '报告日', '净利润', '净利润'),
            # self.plot_bar_line(df_import, '报告期', '扣非净利润', '扣非净利润同比增长率'),
            # self.title("每股指标"),
            # self.plot_line(df_import, '报告期', '基本每股收益', '基本每股收益'),
            # self.plot_line(df_import, '报告期', '每股净资产', '每股净资产'),
            # self.plot_line(df_import, '报告期', '每股资本公积金', '每股资本公积金'),
            # self.plot_line(df_import, '报告期', '每股未分配利润', '每股未分配利润'),
            # self.plot_line(df_import, '报告期', '每股经营现金流', '每股经营现金流'),
            # self.title("盈利能力"),
            # self.plot_line(df_import, '报告期', '净资产收益率', '净资产收益率'),
            # self.plot_line(df_import, '报告期', '净资产收益率-摊薄', '净资产收益率-摊薄'),
            # self.plot_line(df_main, '报告期', '总资产报酬率', '总资产报酬率'),
            # self.plot_line(df_import, '报告期', '销售净利率', '销售净利率'),
            # self.plot_line(df_import, '报告期', '销售毛利率', '销售毛利率'),
            # self.title("财务风险"),
            # self.plot_line(df_import, '报告期', '资产负债率', '资产负债率'),
            # self.plot_line(df_import, '报告期', '流动比率', '流动比率'),
            # self.plot_line(df_import, '报告期', '速动比率', '速动比率'),
            # self.plot_line(df_main, '报告期', '权益乘数', '权益乘数'),
            # self.plot_line(df_import, '报告期', '产权比率', '产权比率'),
            # self.plot_line(df_main, '报告期', '现金比率', '现金比率'),
            # self.plot_line(df_import, '报告期', '产权比率', '产权比率'),
            # self.title("运营能力"),
            # self.plot_line(df_import, '报告期', '存货周转天数', '存货周转天数'),
            # self.plot_line(df_import, '报告期', '应收账款周转天数', '应收账款周转天数'),
            # self.plot_line(df_import, '报告期', '应收账款周转天数', '应收账款周转天数'),
            # self.plot_line(df_import, '报告期', '营业周期', '营业周期'),
            # self.plot_line(df_main, '报告期', '总资产周转率', '总资产周转率'),
            # self.plot_line(df_main, '报告期', '存货周转率', '存货周转率'),
            # self.plot_line(df_main, '报告期', '应收账款周转率', '应收账款周转率'),
            # self.plot_line(df_main, '报告期', '应付账款周转率', '应付账款周转率'),
            # self.plot_line(df_main, '报告期', '流动资产周转率', '流动资产周转率'),
            # self.plot_multi_line('日期', '当日资金流入', [df_north, df_sh, df_sz], ['北向资金', "沪股通", "深股通"])


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
