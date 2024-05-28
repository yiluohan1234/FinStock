#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: bigger_than_ema.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年5月27日
#    > description: 收盘价大于简单移动平均价。
#######################################################################

from datetime import date, datetime  # For datetime objects

# Import the backtrader platform
import backtrader as bt
import pandas as pd
from utils.data import get_kline_chart_date


# 创建策略继承bt.Strategy
class BiggerThanEmaStrategy(bt.Strategy):
    params = (
        # 均线参数设置15天，15日均线
        ("maperiod", 15),
    )

    def log(self, txt, dt=None):
        # 记录策略的执行日志
        dt = dt or self.datas[0].datetime.date(0)
        print("%s, %s" % (dt.isoformat(), txt))

    def __init__(self):
        # 保存收盘价的引用
        self.dataclose = self.datas[0].close
        # 跟踪挂单
        self.order = None
        # 买入价格和手续费
        self.buyprice = None
        self.buycomm = None
        # 加入指标
        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod
        )
        # Indicators for the plotting show
        bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
        bt.indicators.WeightedMovingAverage(self.datas[0], period=25, subplot=True)
        bt.indicators.StochasticSlow(self.datas[0])
        bt.indicators.MACDHisto(self.datas[0])
        rsi = bt.indicators.RSI(self.datas[0])
        bt.indicators.SmoothedMovingAverage(rsi, period=10)
        bt.indicators.ATR(self.datas[0], plot=False)

    # 订单状态通知，买入卖出都是下单
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # broker 提交/接受了，买/卖订单则什么都不做
            return
        # 检查一个订单是否完成
        # 注意: 当资金不足时，broker会拒绝订单
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    "已买入, 价格: %.2f, 费用: %.2f, 佣金 %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log(
                    "已卖出, 价格: %.2f, 费用: %.2f, 佣金 %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )
            # 记录当前交易数量
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("订单取消/保证金不足/拒绝")
        # 其他状态记录为：无挂起订单
        self.order = None

    # 交易状态通知，一买一卖算交易
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log("交易利润, 毛利润 %.2f, 净利润 %.2f" % (trade.pnl, trade.pnlcomm))

    def next(self):
        # 记录收盘价
        self.log("Close, %.2f" % self.dataclose[0])
        # 如果有订单正在挂起，不操作
        if self.order:
            return
        # 如果没有持仓则买入
        if not self.position:
            # 今天的收盘价在均线价格之上
            if self.dataclose[0] > self.sma[0]:
                # 买入
                self.log("买入单, %.2f" % self.dataclose[0])
                # 跟踪订单避免重复
                self.order = self.buy()
        else:
            # 如果已经持仓，收盘价在均线价格之下
            if self.dataclose[0] < self.sma[0]:
                # 全部卖出
                self.log("卖出单, %.2f" % self.dataclose[0])
                # 跟踪订单避免重复
                self.order = self.sell()


if __name__ == "__main__":
    code = "000977"  # 股票代码
    start_cash = 10000  # 初始自己为10000
    stake = 100  # 单次交易数量为1手
    commfee = 0.0005  # 佣金为万5
    sdate = "20240101"  # 回测时间段
    edate = "20240526"
    df = get_kline_chart_date(code=code, start_date=sdate, end_date=edate, freq='D', zh_index=False)

    start_date = datetime.strptime(sdate, "%Y%m%d")  # 转换日期格式
    end_date = datetime.strptime(edate, "%Y%m%d")
    df["openinterest"] = 0
    # 初始化cerebro回测系统设置
    cerebro = bt.Cerebro()
    # 取得股票历史数据
    data = bt.feeds.PandasData(dataname=df, fromdate=start_date, todate=end_date)
    print(data)
    # 为Cerebro引擎添加策略
    cerebro.addstrategy(BiggerThanEmaStrategy)
    # 加载交易数据
    cerebro.adddata(data)
    # 设置投资金额
    cerebro.broker.setcash(start_cash)
    # 每笔交易使用固定交易量
    cerebro.addsizer(bt.sizers.FixedSize, stake=stake)
    # 设置佣金为0.001, 除以100去掉%号
    cerebro.broker.setcommission(commission=commfee)
    # 获取回测开始时的总资金
    print("期初资金: %.2f" % cerebro.broker.getvalue())
    # 运行回测系统
    cerebro.run()
    # 获取回测结束后的总资金
    print("期末资金: %.2f" % cerebro.broker.getvalue())
    # cerebro.plotinfo.plotname = "收盘价大于简单移动平均价"
    # Plot the result
    cerebro.plot()
