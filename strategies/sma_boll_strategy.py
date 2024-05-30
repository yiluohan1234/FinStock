#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: boll_strategy.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年5月27日
#    > description: boll策略
#######################################################################


import datetime
import backtrader as bt
from utils.data import get_kline_chart_date


class BollStrategy(bt.Strategy):  # BOLL策略程序
    params = (
        ("nk", 13),  # 求均值的天数
        ("maperiod", 10),
        ("printlog", False),
    )  # 打印log

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.data_close = self.datas[0].close
        # To keep track of pending orders
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod
        )
        # Boll指标计算
        self.top = bt.indicators.BollingerBands(
            self.datas[0], period=self.params.nk
        ).top
        self.bot = bt.indicators.BollingerBands(
            self.datas[0], period=self.params.nk
        ).bot
        # 添加移动均线指标
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.nk
        )

    def next(self):  # 买卖策略
        if self.order:  # 检查是否有指令等待执行
            return
        # 检查是否持仓
        """
        if not self.position:  # 没有持仓
            if self.data_close[0] > self.sma[0]:  # 执行买入条件判断：收盘价格上涨突破20日均线
                self.order = self.buy(size=100)   # 执行买入
        else:
            if self.data_close[0] < self.sma[0]:  # 执行卖出条件判断：收盘价格跌破20日均线
                self.order = self.sell(size=100)  # 执行卖出
        """
        if not self.position:  # 没有持仓
            if self.data_close[0] < self.sma[0]:  # 收盘价格跌破下轨
                self.log("BUY CREATE, %.2f" % self.data_close[0])
                self.order = self.buy()  # 执行买入
        else:
            if self.data_close[0] > self.top[0]:  # 收盘价格上涨突破上轨
                self.log("SELL CREATE, %.2f" % self.data_close[0])
                self.order = self.sell()  # 执行卖出

    def log(self, txt, dt=None, do_print=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog or do_print:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def notify_order(self, order):
        # Buy/Sell order submitted/accepted to/by broker - Nothing to do
        if order.status in [order.Submitted, order.Accepted]:
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    '买入, 价格: %.2f, 成本: %.2f, 手续费: %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log('卖出, 价格: %.2f, 成本: %.2f, 手续费: %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

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
            "(BOLL线： %2d日) 期末总资金 %.2f" % (self.params.nk, self.broker.getvalue()),
            do_print=True,
            )


if __name__ == "__main__":
    sdate = "20240101"  # 回测时间段
    now = datetime.datetime.now()
    if now.hour >= 15:
        edate = now.strftime('%Y%m%d')
    else:
        yesterday = now - datetime.timedelta(days=1)
        edate = yesterday.strftime('%Y%m%d')
    # Get data from AkShare
    df = get_kline_chart_date(code="000977", start_date=sdate, end_date=edate, freq='D', zh_index=False)
    start_date = datetime.datetime.strptime(sdate, "%Y%m%d")
    end_date = datetime.datetime.strptime(edate, "%Y%m%d")

    # Create a cerebro entity
    cerebro = bt.Cerebro()
    # Create a Data Feed
    data = bt.feeds.PandasData(dataname=df, fromdate=start_date, todate=end_date)
    # Add the Data Feed to Cerebro
    cerebro.adddata(data)
    cerebro.addstrategy(BollStrategy, nk=13, maperiod=20, printlog=True)  # 加载交易策略
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="PyFolio")
    # Set our desired cash start
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0005)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)
    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    back = cerebro.run()  # 运行回测
    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    # cerebro.plotinfo.plotname = "BOLL线 回测结果"
    # cerebro.plot()
