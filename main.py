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
    k=KLineChart("000612")

    k.plot(area=['V','KL'], width=1200, height=600,
           klines=['ma5', 'ma20', 'ene', 'upper', 'lower'],
           # jxPoints=[[("2024-02-21",18.2), ("2024-04-12",18.2)]],
           # jxLines=[18.2, 16.5, "2024-04-12", "2024-04-30"],
           is_notebook=False
           )
    # k.web()
    print(k.get_data_k(20))

    f = FundFlow()
    df, df_display = f.get_individual_fund_flow_rank("000612")
    print(df_display)

def AI():
    a = AIndex(start_date="20200101", end_date="20211231", freq='W')
    a.plot(n=60, area=['V', 'BIAS'], width=1200,height=600,
           klines=['ma5','ma10','ma20','ma60', 'ma120', 'ma250'],
        #    jxPoints=[[("2024-02-21",1800.2), ("2024-04-12",1800.2)]],
        #    jxLines=[1800.2, 1700.5, "2024-04-12", "2024-04-30"],
            is_notebook=False
           )
    a.web()


if __name__ == "__main__":
    main()

