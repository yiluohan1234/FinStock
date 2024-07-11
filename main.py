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


def main(code):
    k = KLineChart(code, start_date="20240101", freq='min5')
    k.plot(n=20, width=1600, height=800, area=['V', 'M', 'KPL', 'MUL'], multiLines=['bias10'],
           klines=['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250'],
           is_notebook=False
           )
    k.web()


def main_concept(symbol):
    k = KLineChart(symbol, start_date="20230601", zh_index='concept')
    k.plot(n=20, width=1600, height=800,
           klines=['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250'],
           is_notebook=False
           )
    k.web()


def main_industry(symbol):
    k = KLineChart(symbol, start_date="20230601", zh_index='industry')
    k.plot(n=20, width=1600, height=800,
           klines=['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250'],
           is_notebook=False
           )
    k.web()


def main_a(code):
    k = KLineChart(code, zh_index='index')
    k.plot(n=20, width=1600, height=800,
           klines=['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250'],
           is_notebook=False
           )
    k.web()


if __name__ == "__main__":
    main("600160")
    # main_concept("液冷服务器") # 人工智能
    # main_industry("房地产开发") # 有色金属、计算机设备
    # main_a("000001")
    # code_list = ["000737", "000977", "002948", "601877", "600595", "000612"]
    # for code in code_list:
    #     main(code)
