#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: sma_cross_strategy.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年5月27日
#    > description: 在这个例子中，我们使用了Backtrader框架和Alpaca API来实现自动化交易。我们首先定义了策略`SmaCross`，该策略基于两个简单移动平均线的交叉来进行买卖点的判断。在`next`方法中，我们检查当前是否持有头寸，如果没有，我们计算可用资金并计算购买股票的数量，然后买入该股票。如果当前持有头寸，则判断是否需要卖出。
#
# 在数据源方面，我们使用了Alpaca API作为数据源，并设置了证券交易所、证券代码、时间间隔、历史数据范围等参数。在引擎方面，我们使用Cerebro引擎，并将数据源和策略添加到引擎中。我们还设置了初始资金、交易费用等参数，并添加了性能分析器来输出交易性能指标。
#
# 请注意，这只是一个简单的示例，实际交易需要更多的参数和逻辑处理。在实际交易前，请务必进行充分的测试和评估。
#######################################################################

import backtrader as bt
import backtrader.indicators as btind
import backtrader.feeds as btfeeds
import backtrader.analyzers as btanalyzers
from utils.data import get_kline_chart_date
from datetime import date, datetime

class SmaCross(bt.Strategy):
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
        self.k60 = self.datas[0].k20
        self.k60_pre = self.datas[-1].k60
        self.crossover = btind.CrossUp(self.k10, self.k20)

    def next(self):
        if self.order:  # 检查是否有指令等待执行
            return
        # 检查是否持仓
        if self.position.size == 0:
            if self.crossover > 0:# and self.k10 < 0 and self.k20 < 0 and self.k60 < 0 and self.k10 > self.k10_pre and self.k20 > self.k20_pre and self.k60 >= self.k60_pre:
                amount_to_invest = (self.broker.cash * 0.95)
                self.size = int(amount_to_invest / self.data.close)
                self.log("BUY CREATE, %.2f" % self.data_close[0])
                self.order = self.buy()
        elif self.position.size > 0:
            if self.crossover < 0:# and self.k10 > 0 and self.k20 > 0 and self.k60 > 0 and self.k10 < self.k10_pre and self.k20 < self.k20_pre and self.k60 <= self.k60_pre:
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
    code = "000737"  # 股票代码
    start_cash = 10000  # 初始自己为10000
    stake = 100  # 单次交易数量为1手
    commfee = 0.0005  # 佣金为万5
    sdate = "20240101"  # 回测时间段
    edate = "20240526"
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross)

    # 利用AKShare获取股票的前复权数据的前6列
    df = get_kline_chart_date(code=code, start_date=sdate, end_date=edate, freq='D', zh_index=False)

    start_date = datetime.strptime(sdate, "%Y%m%d")  # 转换日期格式
    end_date = datetime.strptime(edate, "%Y%m%d")

    # data = bt.feeds.PandasData(dataname=df, fromdate=start_date, todate=end_date)
    data = PandasDataPlus(dataname=df, fromdate=start_date, todate=end_date)

    # 将数据添加到引擎中
    cerebro.adddata(data)
    cerebro.addstrategy(SmaCross, printlog=True)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="PyFolio")
    cerebro.broker.setcash(start_cash)  # broker设置资金
    cerebro.broker.setcommission(commission=commfee)  # broker手续费
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)  # 设置买入数量
    print("期初总资金: %.2f" % start_cash)
    back = cerebro.run()  # 运行回测
    end_value = cerebro.broker.getvalue()  # 获取回测结束后的总资金
    print("期末总资金: %.2f" % end_value)
    # cerebro.plotinfo.plotname = "BOLL线 回测结果"
    cerebro.plot()

