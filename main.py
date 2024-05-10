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


def main():
    b = Basic()
    df, df_display= b.get_basic_import_key("000977", 5)
    # 盈利能力['净资产收益率', '总资产回报率', '毛利率']
    # 财务风险['资产负债率', '流动比率', '速动比率']
    # 运营能力['存货周转天数', '应收账款周转天数']
    print(df)


def line():
    k=KLineChart("000612")
    # k=KLineChart("000960")
    # k=KLineChart("000612")
    # k=KLineChart("600595")
    # k=KLineChart("001872")

    k.plot(area=['V','KL'], width=1200, height=600,
           # klines=['ma5','ma10','ma20','ma60', 'ma120', 'ma250'],
           klines=['ma5', 'ma20', 'ene', 'upper', 'lower'],
           # jxPoints=[[("2024-02-21",18.2), ("2024-04-12",18.2)]],
           # jxLines=[18.2, 16.5, "2024-04-12", "2024-04-30"]
           )
    k.web()


def AI():
    a = AIndex()
    a.plot(n=60, area=['V', 'BIAS'], width=1200,height=600,
           klines=['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250'],
           # jxPoints=[[("2024-02-21",1800.2), ("2024-04-12",1800.2)]],
           # jxLines=[1800.2, 1700.5, "2024-04-12", "2024-04-30"]
           )
    a.web()


def fund():
    import akshare as ak
    f = FundFlow()
    df, df_display = f.get_individual_fund_flow_rank("000612")
    print(df_display)
    # print(ak.stock_hsgt_fund_min_em("北向资金"))
    # print(ak.stock_hsgt_fund_flow_summary_em())
    # print(ak.stock_hsgt_hist_em("北向资金").dtypes)


if __name__ == "__main__":
    # main()
    # AI()
    # b = Basic()
    # b.plot_page()
    fund()
    # import akshare as ak

