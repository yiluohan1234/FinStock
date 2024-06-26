#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: k_cross_strategy.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年5月27日
#    > description: 斜率策略
#######################################################################


import backtrader as bt
import backtrader.indicators as btind
import backtrader.feeds as btfeeds
import backtrader.analyzers as btanalyzers
from utils.data import get_kline_chart_date
import datetime

class KCross(bt.Strategy):
    params = (
        ("printlog", False),
    )

    def __init__(self):
        self.data_close = self.datas[0].close  # 指定价格序列
        # 初始化交易指令、买卖价格和手续费
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        self.k10 = self.datas[0].k10
        self.k10_pre = self.datas[-1].k10
        self.k20 = self.datas[0].k20
        self.k20_pre = self.datas[-1].k20
        self.k60 = self.datas[0].k60
        self.k60_pre = self.datas[-1].k60
        self.crossover = btind.CrossOver(self.k10, self.k20)

    def next(self):
        if self.order:  # 检查是否有指令等待执行
            return
        # 检查是否持仓
        if self.position.size == 0:
            if self.crossover > 0 and self.k10 < 0 and self.k20 < 0 and self.k60 < 0:
                amount_to_invest = (self.broker.cash * 0.95)
                self.size = int(amount_to_invest / self.data.close)
                self.log("BUY CREATE, %.2f" % self.data_close[0])
                self.order = self.buy()
        elif self.position.size > 0:
            if self.crossover < 0 and self.k10 > 0 and self.k20 > 0 and self.k60 > 0:
                self.log("SELL CREATE, %.2f" % self.data_close[0])
                self.order = self.sell()

    def log(self, txt, dt=None, do_print=False):  # 日志函数
        if self.params.printlog or do_print:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def notify_order(self, order):  # 记录交易执行情况
        # 如果order为submitted/accepted,返回空
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 指令为buy/sell,报告价格结果
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"买入:\n价格:{order.executed.price},\
                成本:{order.executed.value},\
                手续费:{order.executed.comm}"
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(
                    f"卖出:\n价格：{order.executed.price},\
                成本: {order.executed.value},\
                手续费{order.executed.comm}"
                )
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("交易失败")  # 指令取消/交易失败, 报告结果
        self.order = None

    def notify_trade(self, trade):  # 记录交易收益情况
        if not trade.isclosed:
            return
        self.log(f"策略收益：\n毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}")

    def stop(self):  # 回测结束后输出结果
        self.log(
            "期末总资金 %.2f" % (self.broker.getvalue()),
            do_print=True,
            )


class PandasDataPlus(bt.feeds.PandasData):
    lines = ('k10', 'k20', 'k60')  # 要添加的列名
    # 设置 line 在数据源上新增的位置
    params = (
        ('k10', -1),  # turnover对应传入数据的列名，这个-1会自动匹配backtrader的数据类与原有pandas文件的列名
        ('k20', -1),
        ('k60', -1),
        # 如果是个大于等于0的数，比如8，那么backtrader会将原始数据下标8(第9列，下标从0开始)的列认为是turnover这一列
    )


if __name__ == "__main__":
    sdate = "20240101"  # 回测时间段
    now = datetime.datetime.now()
    if now.hour >= 15:
        edate = now.strftime('%Y%m%d')
    else:
        yesterday = now - datetime.timedelta(days=1)
        edate = yesterday.strftime('%Y%m%d')
    cerebro = bt.Cerebro()  # 创建回测系统实例
    # 利用AKShare获取股票的前复权数据的前6列
    df = get_kline_chart_date(code="000977", start_date=sdate, end_date=edate, freq='D', zh_index=False)

    start_date = datetime.datetime.strptime(sdate, "%Y%m%d")  # 转换日期格式
    end_date = datetime.datetime.strptime(edate, "%Y%m%d")

    # data = bt.feeds.PandasData(dataname=df, fromdate=start_date, todate=end_date)
    data = PandasDataPlus(dataname=df, fromdate=start_date, todate=end_date)

    # 将数据添加到引擎中
    cerebro.adddata(data)
    cerebro.addstrategy(KCross, printlog=True)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="PyFolio")
    cerebro.broker.setcash(10000.0)  # broker设置资金
    cerebro.broker.setcommission(commission=0.0005)  # broker手续费
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)  # 设置买入数量
    print("期初总资金: %.2f" % 10000.0)
    back = cerebro.run()  # 运行回测
    end_value = cerebro.broker.getvalue()  # 获取回测结束后的总资金
    print("期末总资金: %.2f" % end_value)
    # cerebro.plotinfo.plotname = "BOLL线 回测结果"
    #cerebro.plot()
