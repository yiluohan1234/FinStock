'''
Author: Charmve yidazhang1@gmail.com
Date: 2023-03-12 18:56:23
LastEditors: Charmve yidazhang1@gmail.com
LastEditTime: 2023-03-12 19:36:14
FilePath: /Qbot/qbot/strategies/multi_strategy_bt.py
Version: 1.0.1
Blogs: charmve.blog.csdn.net
GitHub: https://github.com/Charmve
Description: 在这个示例中，我们使用了backtrader的内置数据源GenericCSVData，读取了名为mydata.csv的数据文件。
        在初始化策略时，我们使用了SimpleMovingAverage和RelativeStrengthIndex等指标，用于判断买卖点。
        在每个交易点，我们使用log函数来输出交易信息。在运行引擎后，我们使用ShapreRatio性能分析器来输出交易性能指标。

Copyright (c) 2023 by Charmve, All Rights Reserved.
Licensed under the MIT License.
'''
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind
import backtrader.analyzers as btanalyzers
from utils.data import get_kline_chart_date

from datetime import date, datetime

class MultiStrategy(bt.Strategy):
    params = (
        ('exitbars', 5),
        ('maperiod', 15),
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.sma = btind.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod
        )
        self.rsi = btind.RelativeStrengthIndex()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm: %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm: %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        self.log('Close, %.2f' % self.dataclose[0])
        if self.order:
            return
        if not self.position:
            if (self.rsi[0] < 50 and
                self.rsi[-1] >= 50 and
                self.dataclose[0] > self.sma[0]):
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.order = self.buy()
        else:
            if (self.rsi[0] > 50 and
                self.rsi[-1] <= 50 and
                self.dataclose[0] < self.sma[0]):
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                self.order = self.sell()

    def stop(self):
        self.log('(MA Period %2d) Ending Value %.2f' %
                 (self.params.maperiod, self.broker.getvalue()), dt=None)


if __name__ == "__main__":

    # 创建Cerebro引擎
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MultiStrategy)

    code = "000977"  # 股票代码
    start_cash = 10000  # 初始自己为10000
    stake = 100  # 单次交易数量为1手
    commfee = 0.0005  # 佣金为万5
    sdate = "20240101"  # 回测时间段
    edate = "20240526"
    df = get_kline_chart_date(code=code, start_date=sdate, end_date=edate, freq='D', zh_index=False)
    start_date = datetime.strptime(sdate, "%Y%m%d")  # 转换日期格式
    end_date = datetime.strptime(edate, "%Y%m%d")

    data = bt.feeds.PandasData(dataname=df, fromdate=start_date, todate=end_date)
    print(data)

    # 将数据添加到引擎中
    cerebro.adddata(data)

    # 设置初始资金
    cerebro.broker.setcash(10000.0)

    # 设置交易手续费
    cerebro.broker.setcommission(commission=0.001)

    # 添加性能分析器
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='mysharpe')

    # 运行引擎
    # cerebro.run()
    thestrats = cerebro.run()

    # 输出性能指标
    thestrat = thestrats[0]
    print('Sharpe Ratio:', thestrat.analyzers.mysharpe.get_analysis()['sharperatio'])

    # Plot the result
    # cerebro.plot()
