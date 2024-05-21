#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: main.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年4月26日
#    > description:
#######################################################################
from core.KLineChart import KLineChart
from core.FundFlow import FundFlow
from core.Basic import Basic
from core.AIndex import AIndex
import warnings
warnings.filterwarnings("ignore")
#设置显示全部行，不省略
import pandas as pd
pd.set_option('display.max_rows', None)
#设置显示全部列，不省略
pd.set_option('display.max_columns', None)


def main(code):
    k = KLineChart(code)

    k.plot(area=['V', 'KL'], width=1200, height=600,
           klines=['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250'],
           KLlines=['k5', 'k10'],
           is_notebook=False
           )
    # k.web()
    print(k.get_data_k(20))

    f = FundFlow()
    df, df_display = f.get_individual_fund_flow_rank(code)
    print(df_display)
    print("----------------------------- 近5日资金流动 -----------------------------")
    df_fund_111, df_fund_111_display = f.get_individual_fund_flow(code, 10)
    print(df_fund_111_display)
    # b = Basic()
    # df_zygc = b.get_basic_info(code)
    # print(df_zygc)
    # df_lrb_year, df_lrb_display_year = b.get_lrb_data(code, 8, 4)
    # print("----------------------------- 利润表（年度） -----------------------------")
    # print(df_lrb_display_year)
    # df_lrb_111, df_lrb_111_display = b.get_lrb_data(code, 8)
    # print("----------------------------- 利润表（自然季度） -----------------------------")
    # print(df_lrb_111_display)
    # print("----------------------------- 关键指标 -----------------------------")
    # df_main, df_main_display = b.get_main_indicators_sina(
    #     code=code,
    #     n=5,
    #     ret_columns=['报告期', '营业总收入', '营业总收入增长率', '净利润', '扣非净利润'])
    # print(df_main_display)
    # print("----------------------------- 盈利能力 -----------------------------")
    # df_main_yingli, df_main_yingli_display = b.get_main_indicators_sina(
    #     code=code,
    #     n=5,
    #     ret_columns=['报告期', '净资产收益率(ROE)', '摊薄净资产收益率', '总资产报酬率', '销售净利率', '毛利率'])
    # print(df_main_yingli_display)
    # print("----------------------------- 财务风险 -----------------------------")
    # df_main_fengxian, df_main_fengxian_display = b.get_main_indicators_sina(
    #     code=code,
    #     n=5,
    #     ret_columns=['报告期', '资产负债率', '流动比率', '速动比率', '权益乘数', '产权比率', '现金比率', '产权比率'])
    # print(df_main_fengxian_display)
    # print("----------------------------- 运营能力 -----------------------------")
    # df_main_yunying, df_main_yunying_display = b.get_main_indicators_sina(
    #     code=code,
    #     n=5,
    #     ret_columns=['报告期', '存货周转天数', '应收账款周转天数', '总资产周转率', '存货周转率', '应收账款周转率', '应付账款周转率', '流动资产周转率'])
    # print(df_main_yunying_display)


def AI():
    a = AIndex()
    a.plot(n=20, area=['V', 'BIAS'], width=1200,height=600,
           klines=['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250'],
           KLlines=['k60', 'k120'],
           is_notebook=False
    )
    a.web()


if __name__ == "__main__":
    main("000977")

