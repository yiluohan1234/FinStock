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
import warnings
from utils.fundflow import get_individual_fund_flow_rank, get_individual_fund_flow
from utils.basic import get_basic_info, get_lrb_data, get_main_indicators_sina
warnings.filterwarnings("ignore")
# 设置显示全部行，不省略
import pandas as pd
pd.set_option('display.max_rows', None)
# 设置显示全部列，不省略
pd.set_option('display.max_columns', None)


def main(code):
    k = KLineChart(code)
    k.plot(n=20, area=['V', 'KL'], width=1600, height=900,
           klines=['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250'],
           vlines=['vma10', 'vma20'],
           KLlines=['k10', 'k60'],
           KPLlines=['kp10', 'kp60'],
           is_notebook=False
           )
    k.web()

    df_display = get_individual_fund_flow_rank(code)
    print(df_display)
    print("----------------------------- 近5日资金流动 -----------------------------")
    df_fund_111_display = get_individual_fund_flow(code, 10)
    print(df_fund_111_display)

    # df_zygc = get_basic_info(code)
    # print(df_zygc)
    # df_lrb_display_year = get_lrb_data(code, 8, 4)
    # print("----------------------------- 利润表（年度） -----------------------------")
    # print(df_lrb_display_year)
    # df_lrb_111_display = get_lrb_data(code, 8)
    # print("----------------------------- 利润表（自然季度） -----------------------------")
    # print(df_lrb_111_display)
    # print("----------------------------- 关键指标 -----------------------------")
    # df_main_display = get_main_indicators_sina(
    #     code=code,
    #     n=5,
    #     ret_columns=['报告期', '营业总收入', '营业总收入增长率', '净利润', '扣非净利润'])
    # print(df_main_display)
    # print("----------------------------- 盈利能力 -----------------------------")
    # df_main_yingli_display = get_main_indicators_sina(
    #     code=code,
    #     n=5,
    #     ret_columns=['报告期', '净资产收益率(ROE)', '摊薄净资产收益率', '总资产报酬率', '销售净利率', '毛利率'])
    # print(df_main_yingli_display)
    # print("----------------------------- 财务风险 -----------------------------")
    # df_main_fengxian_display = get_main_indicators_sina(
    #     code=code,
    #     n=5,
    #     ret_columns=['报告期', '资产负债率', '流动比率', '速动比率', '权益乘数', '产权比率', '现金比率', '产权比率'])
    # print(df_main_fengxian_display)
    # print("----------------------------- 运营能力 -----------------------------")
    # df_main_yunying_display = get_main_indicators_sina(
    #     code=code,
    #     n=5,
    #     ret_columns=['报告期', '存货周转天数', '应收账款周转天数', '总资产周转率', '存货周转率', '应收账款周转率', '应付账款周转率', '流动资产周转率'])
    # print(df_main_yunying_display)

def main_a(code):
    k = KLineChart(code, zh_index=True)
    k.plot(n=20, area=['V', 'KL'], width=1600, height=900,
           klines=['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250'],
           vlines=['vma10', 'vma60'],
           KLlines=['k10', 'k60'],
           KPLlines=['kp10', 'kp60'],
           is_notebook=False
           )
    k.web()


if __name__ == "__main__":
    main("000737")
    # main_a("000001")
