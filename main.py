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
    k = KLineChart(code, start_date="20230601")
    k.plot(n=20, area=['V', 'M', 'KPL'], width=1600, height=900,
           klines=['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250'],
           vlines=['vma10', 'vma20'], dmalines=['ma10', 'ma60'],
           KLlines=['k10', 'k60'], KPLlines=['kp10', 'kp60'],
           is_notebook=False
           )
    k.web()


def main_a(code):
    k = KLineChart(code, zh_index=True)
    k.plot(n=20, area=['V', 'M', 'KPL'], width=1600, height=900,
           klines=['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250'],
           vlines=['vma10', 'vma60'],  dmalines=['ma10', 'ma60'],
           KLlines=['k10', 'k60'], KPLlines=['kp10', 'kp60'],
           is_notebook=False
           )
    k.web()


if __name__ == "__main__":
    main("000977")
    # main_a("000001")
