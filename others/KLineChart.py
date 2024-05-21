#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: KLineChart.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年4月26日
#    > description:
#######################################################################
from pyecharts.charts import Kline, Scatter, Line, Grid, Bar, EffectScatter
from pyecharts import options as opts
from pyecharts.render import make_snapshot
from pyecharts.globals import SymbolType  # ,ThemeType
from pyecharts.commons.utils import JsCode
# https://blog.csdn.net/qq_42571592/article/details/122826752
from pyecharts.globals import CurrentConfig, NotebookType

# CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_NOTEBOOK
CurrentConfig.ONLINE_HOST = 'https://assets.pyecharts.org/assets/'

from snapshot_pyppeteer import snapshot
from pyecharts.render import make_snapshot
import webbrowser
import os
import akshare as ak
import pandas as pd
import numpy as np
import talib
import datetime

import warnings

warnings.filterwarnings("ignore")


class KLineChart:

    def __init__(self, code, start_date='20200101', end_date='20240202', freq='D', precision=2):
        '''
        @params:
        - code: str                      #股票代码
        - start_date: str                #开始时间
        - end_date: str                  #结束时间
        - freq : str                     #默认 D 日线数据
        - precision :str                 #数据精度,默认2
        '''
        if end_date == '20240202':
            now = datetime.datetime.now()
            if now.hour >= 15:
                end_date = now.strftime('%Y%m%d')
            else:
                yesterday = now - datetime.timedelta(days=1)
                end_date = yesterday.strftime('%Y%m%d')
        name_code = ak.stock_zh_a_spot_em()
        name = name_code[name_code['代码'] == code]['名称'].values[0]
        # self.title = "{}-K线及均线".format(name)
        self.title = name
        self.precision = precision
        if freq == 'D':
            df = self.get_data(code, start_date, end_date)
            self.data = df.copy()
            self.dateindex = df.index.strftime("%Y-%m-%d").tolist()
        else:
            now = datetime.datetime.now()
            yesterday = now - datetime.timedelta(days=1)
            current_date = yesterday.strftime('%Y%m%d')
            df = self.get_data_min(code, current_date)
            self.data = df.copy()
            self.dateindex = df.index.strftime('%H:%M').tolist()

        self.data['f'] = self.data.apply(lambda x: self.frb(x.open, x.close), axis=1)

        self.prices_cols = ['open', 'close', 'low', 'high']

    def frb(self, open_value, close_value):
        if (close_value - open_value) >= 0:
            return 1
        else:
            return -1

    def get_szsh_code(self, code):
        if code.find('60', 0, 3) == 0:
            gp_type = 'sh'
        elif code.find('688', 0, 4) == 0:
            gp_type = 'sh'
        elif code.find('900', 0, 4) == 0:
            gp_type = 'sh'
        elif code.find('00', 0, 3) == 0:
            gp_type = 'sz'
        elif code.find('300', 0, 4) == 0:
            gp_type = 'sz'
        elif code.find('200', 0, 4) == 0:
            gp_type = 'sz'
        return gp_type + code

    def get_data(self, code, start_date, end_date):
        date_s = datetime.datetime.strptime(start_date, "%Y%m%d")
        start = (date_s - datetime.timedelta(days=365)).strftime('%Y%m%d')

        df = ak.stock_zh_a_hist(symbol=code, start_date=start, end_date=end_date, adjust="qfq").iloc[:, :6]
        # df = ak.stock_zh_a_daily(symbol=self.get_szsh_code(code), start_date=start,end_date=end_date, adjust="qfq")

        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', ]
        df['volume'] = round(df['volume'].astype('float') / 10000, 2)

        # 计算均线
        df['ma5'] = df.close.rolling(5).mean()
        df['ma10'] = df.close.rolling(10).mean()
        df['ma20'] = df.close.rolling(20).mean()
        df['ma30'] = df.close.rolling(30).mean()
        df['ma60'] = df.close.rolling(60).mean()
        df['ma120'] = df.close.rolling(120).mean()
        df['ma250'] = df.close.rolling(250).mean()

        # 计算抵扣差
        for i in [5, 10, 20, 30, 60, 120, 250]:
            df['dkc{}'.format(i)] = round(df["close"] - df["close"].shift(i - 1), self.precision)

        # 计算乖离率
        for i in [5, 10, 20, 30, 60, 120, 250]:
            df['bias{}'.format(i)] = round(
                (df["close"] - df["ma{}".format(i)]) * 100 / df["ma{}".format(i)],
                self.precision)

        # 计算k率
        for i in [5, 10, 20, 30, 60, 120, 250]:
            df['k{}'.format(i)] = df.close.rolling(i).apply(self.cal_K)


        df.index = range(len(df))  # 修改索引为数字序号
        df['ATR1'] = df['high'] - df['low']  # 当日最高价-最低价
        df['ATR2'] = abs(df['close'].shift(1) - df['high'])  # 上一日收盘价-当日最高价
        df['ATR3'] = abs(df['close'].shift(1) - df['low'])  # 上一日收盘价-当日最低价
        df['ATR4'] = df['ATR1']
        for i in range(len(df)):  # 取价格波动的最大值
            if df.loc[i, 'ATR4'] < df.loc[i, 'ATR2']:
                df.loc[i, 'ATR4'] = df.loc[i, 'ATR2']
            if df.loc[i, 'ATR4'] < df.loc[i, 'ATR3']:
                df.loc[i, 'ATR4'] = df.loc[i, 'ATR3']
        df['ATR'] = df.ATR4.rolling(14).mean()  # N=14的ATR值
        df['stop'] = df['close'].shift(1) - df['ATR'] * 3  # 止损价=(上一日收盘价-3*ATR)

        # BOLL计算 取N=20，M=2
        df['boll'] = df.close.rolling(20).mean()
        df['delta'] = df.close - df.boll
        df['beta'] = df.delta.rolling(20).std()
        df['up'] = df['boll'] + 2 * df['beta']
        df['down'] = df['boll'] - 2 * df['beta']

        # 计算MACD
        df = self.get_data_macd(df)

        # 标记买入和卖出信号
        # for i in range(len(df)):
        #     if df.loc[i, 'close'] > df.loc[i, 'up']:
        #         df.loc[i, 'SELL'] = True
        #     if df.loc[i, 'close'] < df.loc[i, 'boll']:
        #         df.loc[i, 'BUY'] = True

        start_date = datetime.datetime.strptime(start_date, '%Y%m%d').date()
        end_date = datetime.datetime.strptime(end_date, '%Y%m%d').date()
        df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]
        # 把date作为日期索引
        df.index = pd.to_datetime(df.date)
        return df

    def get_data_min(self, code, current_date):
        if int(code) > 600000:
            symbol = "sh" + str(code)
        else:
            symbol = "sz" + str(code)
        df = ak.stock_zh_a_minute(symbol=symbol, period="1", adjust="qfq")
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', ]
        df['volume'] = round(df['volume'].astype('float') / 10000, 2)
        df = df[pd.to_datetime(df['date']).dt.date.astype('str') == '2024-03-18']

        # 计算均线
        df['ma5'] = df.close.rolling(5).mean()
        df['ma10'] = df.close.rolling(10).mean()
        df['ma20'] = df.close.rolling(20).mean()
        df['ma30'] = df.close.rolling(30).mean()
        df['ma60'] = df.close.rolling(60).mean()
        df['ma120'] = df.close.rolling(120).mean()
        df['ma250'] = df.close.rolling(250).mean()

        # 计算乖离率
        df['bias5'] = round((df["close"] - df["ma5"]) * 100 / df["ma5"], self.precision)
        df['bias10'] = round((df["close"] - df["ma10"]) * 100 / df["ma10"], self.precision)
        df['bias20'] = round((df["close"] - df["ma20"]) * 100 / df["ma20"], self.precision)
        df['bias30'] = round((df["close"] - df["ma30"]) * 100 / df["ma30"], self.precision)
        df['bias60'] = round((df["close"] - df["ma60"]) * 100 / df["ma60"], self.precision)
        df['bias120'] = round((df["close"] - df["ma120"]) * 100 / df["ma120"], self.precision)
        df['bias250'] = round((df["close"] - df["ma250"]) * 100 / df["ma250"], self.precision)

        df.index = range(len(df))  # 修改索引为数字序号
        df['ATR1'] = df['high'] - df['low']  # 当日最高价-最低价
        df['ATR2'] = abs(df['close'].shift(1) - df['high'])  # 上一日收盘价-当日最高价
        df['ATR3'] = abs(df['close'].shift(1) - df['low'])  # 上一日收盘价-当日最低价
        df['ATR4'] = df['ATR1']
        for i in range(len(df)):  # 取价格波动的最大值
            if df.loc[i, 'ATR4'] < df.loc[i, 'ATR2']:
                df.loc[i, 'ATR4'] = df.loc[i, 'ATR2']
            if df.loc[i, 'ATR4'] < df.loc[i, 'ATR3']:
                df.loc[i, 'ATR4'] = df.loc[i, 'ATR3']
        df['ATR'] = df.ATR4.rolling(14).mean()  # N=14的ATR值
        df['stop'] = df['close'].shift(1) - df['ATR'] * 3  # 止损价=(上一日收盘价-3*ATR)

        # BOLL计算 取N=20，M=2
        df['boll'] = df.close.rolling(20).mean()
        df['delta'] = df.close - df.boll
        df['beta'] = df.delta.rolling(20).std()
        df['up'] = df['boll'] + 2 * df['beta']
        df['down'] = df['boll'] - 2 * df['beta']

        # 计算k率
        df['k5'] = df.close.rolling(5).apply(self.cal_K)
        df['k10'] = df.close.rolling(10).apply(self.cal_K)
        df['k20'] = df.close.rolling(20).apply(self.cal_K)
        df['k30'] = df.close.rolling(30).apply(self.cal_K)
        df['k60'] = df.close.rolling(60).apply(self.cal_K)
        df['k120'] = df.close.rolling(120).apply(self.cal_K)
        df['k250'] = df.close.rolling(250).apply(self.cal_K)

        # 计算抵扣差
        df['dkc5'] = round(df["close"] - df["close"].shift(4), 2)
        df['dkc10'] = round(df["close"] - df["close"].shift(9), 2)
        df['dkc20'] = round(df["close"] - df["close"].shift(19), 2)
        df['dkc30'] = round(df["close"] - df["close"].shift(29), 2)
        df['dkc60'] = round(df["close"] - df["close"].shift(59), 2)
        df['dkc120'] = round(df["close"] - df["close"].shift(119), 2)
        df['dkc250'] = round(df["close"] - df["close"].shift(249), 2)

        # 标记买入和卖出信号
        for i in range(len(df)):
            if df.loc[i, 'close'] > df.loc[i, 'up']:
                df.loc[i, 'SELL'] = True
            if df.loc[i, 'close'] < df.loc[i, 'down']:
                df.loc[i, 'BUY'] = True

        # 计算MACD
        df = self.get_data_macd(df)

        # 把date作为日期索引
        df.index = pd.to_datetime(df.date)
        return df

    def cal_K(self, arr, precision=2):
        y_arr = np.array(arr).ravel()
        x_arr = list(range(1, len(y_arr) + 1))
        fit_K = np.polyfit(x_arr, y_arr, deg=1)
        return round(fit_K[0], precision)

    def get_data_macd(self, df, fastperiod=12, slowperiod=26, signalperiod=9):
        df['DIF'], df['DEA'], df['MACD'] = talib.MACDEXT(df['close'], fastperiod=fastperiod, fastmatype=1,
                                                         slowperiod=slowperiod, slowmatype=1, signalperiod=signalperiod,
                                                         signalmatype=1)
        df['MACD'] = df['MACD'] * 2
        return df

    def K(self) -> Kline:
        data = self.data[self.prices_cols].values.tolist()
        c = (
            Kline()
                .add_xaxis(self.dateindex)
                .add_yaxis(
                series_name="k线",  # 序列名称
                y_axis=data,  # Y轴数据
                itemstyle_opts=opts.ItemStyleOpts(color="#ec0000", color0="#00da3c"),
                markpoint_opts=opts.MarkPointOpts(
                    data=[  # 添加标记符
                        opts.MarkPointItem(type_='max', name='最大值', value_dim='highest'),
                        opts.MarkPointItem(type_='min', name='最小值', value_dim='lowest'), ],
                ),
            )
                .set_global_opts(
                title_opts=opts.TitleOpts(title=self.title, pos_left='10%'),
                legend_opts=opts.LegendOpts(
                    is_show=True, pos_top=10, pos_left="center"
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(
                        is_show=False,
                        type_="inside",  # 内部缩放
                        xaxis_index=[0, 1],  # 可缩放的x轴坐标编号
                        range_start=80,
                        range_end=100,  # 初始显示范围
                    ),
                    opts.DataZoomOpts(
                        is_show=True,
                        xaxis_index=[0, 1],
                        type_="slider",
                        pos_top="90%",
                        range_start=80,
                        range_end=100,
                    ),
                ],
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=False),
                                         axistick_opts=opts.AxisTickOpts(is_show=False), ),
                yaxis_opts=opts.AxisOpts(
                    is_scale=True,
                    splitarea_opts=opts.SplitAreaOpts(
                        is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                    ),
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis",
                    axis_pointer_type="cross",
                    background_color="rgba(245, 245, 245, 0.8)",
                    border_width=2,
                    border_color="#ccc",
                    textstyle_opts=opts.TextStyleOpts(color="#000"),
                ),
                visualmap_opts=opts.VisualMapOpts(
                    is_show=False, dimension=2,
                    series_index=5, is_piecewise=True,
                    pieces=[
                        {"value": 1, "color": "#00da3c"},
                        {"value": -1, "color": "#ec0000"},
                    ],
                ),
                axispointer_opts=opts.AxisPointerOpts(
                    is_show=True,
                    link=[{"xAxisIndex": "all"}],
                    label=opts.LabelOpts(background_color="#777"),
                ),
                brush_opts=opts.BrushOpts(
                    x_axis_index="all",
                    brush_link="all",
                    out_of_brush={"colorAlpha": 0.1},
                    brush_type="lineX",
                ),
            )
        )
        if len(self.klines) != 0:
            kLine = Line().add_xaxis(self.dateindex)
            for i in self.klines:
                kLine.add_yaxis(i, round(self.data[i], self.precision).values.tolist(),
                                is_smooth=True,
                                is_symbol_show=False,
                                is_hover_animation=False,
                                label_opts=opts.LabelOpts(is_show=False),
                                linestyle_opts=opts.LineStyleOpts(type_='solid', width=2),
                                )
            kLine.set_global_opts(xaxis_opts=opts.AxisOpts(type_="category", is_show=False))
            c.overlap(kLine)

        if 'BUY' in self.data.columns:
            v1 = self.data[self.data['BUY'] == True].index.strftime("%Y-%m-%d").tolist()
            v2 = self.data[self.data['BUY'] == True]['close'].values.tolist()
            es_buy = (
                Scatter()
                    .add_xaxis(v1)
                    .add_yaxis(series_name='',
                               y_axis=v2,
                               xaxis_index=0,
                               symbol='triangle',
                               symbol_size=10,  # 设置散点的大小
                               )
                    .set_series_opts(label_opts=opts.LabelOpts(is_show=False),
                                     itemstyle_opts=opts.ItemStyleOpts(color="red"))
                    .set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
                    .set_global_opts(visualmap_opts=opts.VisualMapOpts(is_show=False))
            )
            c.overlap(es_buy)

        if 'SELL' in self.data.columns:
            v1 = self.data[self.data['SELL'] == True].index.strftime("%Y-%m-%d").tolist()
            v2 = self.data[self.data['SELL'] == True]['close'].values.tolist()
            es_sell = (
                Scatter()
                    .add_xaxis(v1)
                    .add_yaxis(series_name='',
                               y_axis=v2,
                               xaxis_index=0,
                               symbol='triangle',
                               symbol_size=10,  # 设置散点的大小
                               symbol_rotate=180,
                               )
                    .set_series_opts(label_opts=opts.LabelOpts(is_show=False),
                                     itemstyle_opts=opts.ItemStyleOpts(color="green"))
                    .set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
                    .set_global_opts(visualmap_opts=opts.VisualMapOpts(is_show=False))
            )
            c.overlap(es_sell)

        return c

    def V(self) -> Bar:
        '''绘制成交量图
        '''
        db = self.data[['volume', 'f']].reset_index()
        db['i'] = db.index
        v = (Bar()
            .add_xaxis(self.dateindex)
            .add_yaxis(
            series_name="成交量",
            y_axis=db[['i', 'volume', 'f']].values.tolist(),
            xaxis_index=0,
            yaxis_index=1,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(
                color=JsCode(
                    """
                    function(params) {
                        var colorList;
                        if (params.data[2] >= 0) {
                            colorList = '#ef232a';
                        } else {
                            colorList = '#14b143';
                        }
                        return colorList;
                    }
                    """
                )
            )
        )
            .set_series_opts(
            label_opts=opts.LabelOpts(is_show=False),
        )
            .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                is_scale=True,
                grid_index=1,
                # boundary_gap=False,
                # axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                # axistick_opts=opts.AxisTickOpts(is_show=False),
                # splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=1,
                is_scale=True,
                split_number=2,
                axislabel_opts=opts.LabelOpts(is_show=False),
                axisline_opts=opts.AxisLineOpts(is_show=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
            ),
            axispointer_opts=opts.AxisPointerOpts(
                is_show=False,
                link=[{"xAxisIndex": "all"}],
                label=opts.LabelOpts(background_color="#777"),
            ),
            legend_opts=opts.LegendOpts(orient='vertical', pos_left="right", pos_top="70%", is_show=False)
            # legend_opts=opts.LegendOpts(is_show=False),
        )
        )

        # 叠加成交量线
        if len(self.vlines) != 0:
            vLine = Line().add_xaxis(self.dateindex)
            for i in self.vlines:
                vLine.add_yaxis(
                    series_name=i,
                    y_axis=round(self.data[i], self.precision).values.tolist(),
                    is_smooth=True,
                    is_symbol_show=False,
                    is_hover_animation=False,
                    label_opts=opts.LabelOpts(is_show=False),
                    linestyle_opts=opts.LineStyleOpts(type_='solid', width=2)
                )
            vLine.set_global_opts(xaxis_opts=opts.AxisOpts(type_="category"))
            v.overlap(vLine)
        return v

    def MACD(self) -> Bar:
        c = (
            Bar()
                .add_xaxis(self.dateindex)
                .add_yaxis("macd", round(self.data.MACD, self.precision).values.tolist(), stack="v",
                           category_gap=2,
                           itemstyle_opts=opts.ItemStyleOpts(
                               color=JsCode(
                                   """
                                       function(params) {
                                           var colorList;
                                           if (params.data >= 0) {
                                               colorList = '#ef232a';
                                           } else {
                                               colorList = '#14b143';
                                           }
                                       return colorList;
                                       }
                                   """
                               )
                           ),
                           )
                .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
                legend_opts=opts.LegendOpts(orient='vertical', pos_left="top", pos_top="70%",
                                            is_show=False),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    is_scale=True,
                    grid_index=1,
                    boundary_gap=False,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    axislabel_opts=opts.LabelOpts(is_show=False),
                )
            )
        )
        dea = round(self.data.DEA, self.precision).values.tolist()
        dif = round(self.data.DIF, self.precision).values.tolist()
        macd_line = (
            Line()
                .add_xaxis(self.dateindex)
                .add_yaxis("DIF", dif,
                           is_symbol_show=False,
                           label_opts=opts.LabelOpts(is_show=False),
                           linestyle_opts=opts.LineStyleOpts(type_='solid', width=2),

                           )
                .add_yaxis("DEA", dea,
                           is_symbol_show=False,
                           label_opts=opts.LabelOpts(is_show=False)
                           )
                .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(datazoom_opts=[opts.DataZoomOpts()],
                                 )

        )
        c.overlap(macd_line)
        return c

    def DKC(self) -> Line:
        dkc_name = "dkc{}".format(self.n)

        c = (
            Line()
                .add_xaxis(self.dateindex)  # X轴数据
                .add_yaxis(
                series_name="抵扣差{}".format(self.n),
                y_axis=self.data[dkc_name].values.tolist(),  # Y轴数据
                xaxis_index=1,
                yaxis_index=1,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    color='#ef232a'  # '#14b143'
                ),
                markline_opts=opts.MarkLineOpts(
                    data=[opts.MarkLineItem(name='0值', y=0, symbol='none', )],
                    linestyle_opts=opts.LineStyleOpts(width=0.1, color='#301934', ),
                )
            )
                .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",  # 坐标轴类型-离散数据
                    grid_index=1,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )
        return c

    def BIAS(self) -> Line:
        bias_name = "bias{}".format(self.n)

        c = (
            Line()
                .add_xaxis(self.dateindex)  # X轴数据
                .add_yaxis(
                series_name="乖离率{}".format(self.n),
                y_axis=self.data[bias_name].values.tolist(),  # Y轴数据
                xaxis_index=1,
                yaxis_index=1,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    color='#ef232a'  # '#14b143'
                ),
                markline_opts=opts.MarkLineOpts(
                    data=[opts.MarkLineItem(name='0值', y=0, symbol='none', )],
                    linestyle_opts=opts.LineStyleOpts(width=0.1, color='#301934', ),
                )
            )
                .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",  # 坐标轴类型-离散数据
                    grid_index=1,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )
        return c

    def KL(self) -> Line:
        k_name = "k{}".format(self.n)

        c = (
            Line()
                .add_xaxis(self.dateindex)  # X轴数据
                .add_yaxis(
                series_name="k率{}".format(self.n),
                y_axis=self.data[k_name].values.tolist(),  # Y轴数据
                xaxis_index=1,
                yaxis_index=1,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    color='#ef232a'  # '#14b143'
                ),
                markline_opts=opts.MarkLineOpts(
                    data=[opts.MarkLineItem(name='0值', y=0, symbol='none', )],
                    linestyle_opts=opts.LineStyleOpts(width=0.1, color='#301934', ),
                )
            )
                .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",  # 坐标轴类型-离散数据
                    grid_index=1,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )
        return c

    def DMA(self):
        # 计算均线差
        self.data['DMA'] = round(self.data[self.dmalines[0]] - self.data[self.dmalines[1]], self.precision)
        c = (
            Line()
                .add_xaxis(self.dateindex)  # X轴数据
                .add_yaxis(
                series_name="dma-{}-{}".format(self.dmalines[0], self.dmalines[1]),
                y_axis=self.data['DMA'].values.tolist(),  # Y轴数据
                xaxis_index=1,
                yaxis_index=1,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    color='#ef232a'  # '#14b143'
                ),
                markline_opts=opts.MarkLineOpts(
                    data=[opts.MarkLineItem(name='0值', y=0, symbol='none', )],
                    linestyle_opts=opts.LineStyleOpts(width=1, color='#301934', ),
                )
            )
                .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",  # 坐标轴类型-离散数据
                    grid_index=1,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )
        return c

    def plot(self, n=20, area=['V', 'DKC'], width=1000, height=600, klines=[], vlines=[], dmalines=[]) -> Grid:
        '''
        @params:
        - area : list   #显示区域
                       'V'      交易量
                       'M'      k线+MACD
                       'DKC'    抵扣差
                       'BIAS'   乖离率
                       'KL'     K率
                       'DMA'    K率
                       FieldName: string   Dataframe中的字段名
                       [Field1,Field2,...] Dataframe中的字段名列表，将显示在一个区域
          width: int   #图表宽度 px
          height:int   #图表高度 px
          klines:str   #K线区域显示的数据，Dataframe中的字段名，如['ma5','ma10','ma20','ma60', 'ma120', 'ma250', 'boll', 'up', 'down', 'stop']
          vline: str   #Volume区域显示的数据，Dataframe中的字段名，如MA...
        - sample:
           chart=data.plot(area=[['V','DKC'],'V'],vlines=['vMA5','vMA10'],klines=['ma5','ma10'])
        '''
        self.n = n
        self.klines = klines
        self.vlines = vlines
        self.dmalines = dmalines
        grid = (
            Grid(init_opts=opts.InitOpts(
                width=str(width) + "px",
                height=str(height) + "px",
                animation_opts=opts.AnimationOpts(animation=False),
            )
            )
        )
        c = self.K()
        iTop = 10
        iButton = 10
        iWindows = len(area)
        iStep = 0
        if iWindows == 0:
            grid.add(c, grid_opts=opts.GridOpts(pos_top="8%", pos_bottom="10%"))
        elif iWindows > 1:
            grid.add(c, grid_opts=opts.GridOpts(pos_top="8%", pos_bottom="50%"))
            iStep = int(30 / iWindows)
            iButton = 50
        else:
            grid.add(c, grid_opts=opts.GridOpts(pos_top="8%", pos_bottom="30%"))
            iStep = 15
            iButton = 70
        icount = 0

        for w in area:
            # print(iStep)
            if type(w) == list:
                window = Line().add_xaxis(self.dateindex)
                for l in w:
                    window.add_yaxis(series_name=l,
                                     y_axis=round(self.data[l], self.precision).values.tolist(),
                                     is_smooth=True,
                                     is_symbol_show=False,
                                     is_hover_animation=False,
                                     label_opts=opts.LabelOpts(is_show=False),
                                     linestyle_opts=opts.LineStyleOpts(type_='solid', width=2)
                                     )
                window.axislabel_opts = opts.LabelOpts(is_show=False),
                window.set_global_opts(datazoom_opts=[opts.DataZoomOpts()],
                                       xaxis_opts=opts.AxisOpts(
                                           type_="category",
                                           axislabel_opts=opts.LabelOpts(is_show=False),
                                       ),
                                       legend_opts=opts.LegendOpts(orient='vertical', pos_left="top",
                                                                   pos_top=str(iButton) + "%"),
                                       )


            elif w == 'V':
                window = self.V()
            elif w == 'M':
                window = self.MACD()
            elif w == 'DKC':
                window = self.DKC()
            elif w == 'BIAS':
                window = self.BIAS()
            elif w == 'KL':
                window = self.KL()
            elif w == 'DMA':
                window = self.DMA()
            else:
                window = Line().add_xaxis(self.dateindex)
                if isinstance(w, list):
                    ws = w
                else:
                    ws = [w]
                for wi in ws:
                    window.add_yaxis(series_name=wi,
                                     y_axis=round(self.data[w], self.precision).values.tolist(),
                                     is_smooth=True,
                                     is_symbol_show=False,
                                     is_hover_animation=False,
                                     label_opts=opts.LabelOpts(is_show=False),
                                     linestyle_opts=opts.LineStyleOpts(type_='solid', width=2)
                                     )
                window.axislabel_opts = opts.LabelOpts(is_show=True),
                window.set_global_opts(datazoom_opts=[opts.DataZoomOpts()],
                                       xaxis_opts=opts.AxisOpts(
                                           type_="category",
                                           axislabel_opts=opts.LabelOpts(is_show=False),

                                       ),
                                       legend_opts=opts.LegendOpts(orient='horizontal',
                                                                   pos_left=str(icount + 20) + "%"),

                                       )
                if '_' + w + '_flag' in self.data.columns:
                    # print("find_flag")
                    v1 = self.data[self.data['_' + w + '_flag'] == True].index.strftime("%Y-%m-%d").tolist()
                    v2 = self.data[self.data['_' + w + '_flag'] == True][w]
                    c_flag = (
                        EffectScatter()
                            .add_xaxis(v1)
                            .add_yaxis("", v2)
                    )
                    window.overlap(c_flag)
                # grid.add(vLine,grid_opts=opts.GridOpts(pos_top= str(iButton)+'%',height=str(iStep)+'%'))
            icount += 1
            # 横坐标最后一行加上x刻度
            if icount == iWindows:
                window.options['xAxis'][0]['axisLabel'].opts['show'] = True
            grid.add(window, grid_opts=opts.GridOpts(pos_top=str(iButton) + '%', height=str(iStep) + '%'))
            iButton = iButton + iStep
        # grid.grid_opts=opts.GridOpts(pos_left="8%", pos_right="8%", height="50%"),
        grid.options['dataZoom'][0].opts['xAxisIndex'] = list(range(0, iWindows + 1))
        grid.render("kline.html")
        # return grid.render_notebook()
        return grid

    def save_png(self, charts, filename):
        make_snapshot(snapshot, charts.render(), filename)

    def web(self):
        webbrowser.open_new_tab('file://' + os.path.realpath('kline.html'))

    def get_date_k(self, n=10):
        now = datetime.datetime.now()
        yesterday = now - datetime.timedelta(days=n)
        current_date = yesterday.strftime('%Y-%m-%d')
        data_n = self.data[self.data['date'].astype(str) > current_date][['close', 'k10', 'k20', 'k60']]
        return data_n

    def get_three_stage(self, min_y, max_y, jx):
        if max_y > jx:
            min_y = 29.38
            max_y = 29.44
            jx = 24.6
            print("止损位：{}x( 1 + 3% )={}".format(jx, round(jx * 1.03, 2)))
            print("顶点到颈线的距离：{} - {} = {} 元".format(max_y, jx, round(max_y - jx, 2)))
            print("第一跌幅满足位：{} - {} = {} 元".format(jx, round(max_y - jx, 2), round(jx - (max_y - jx), 2)))
            print("第二跌幅满足位：{} - {} = {} 元".format(round(jx - (max_y - jx), 2), round(max_y - jx, 2),
                                                  round(jx - 2 * (max_y - jx), 2)))
            print("第三跌幅满足位：{} - {} = {} 元".format(round(jx - 2 * (max_y - jx), 2), round(max_y - jx, 2),
                                                  round(jx - 3 * (max_y - jx), 2)))
        else:
            print("止损位：{}x( 1 - 3% )={}".format(jx, round(jx * 0.97, 2)))
            print("顶点到颈线的距离：{} - {} = {} 元".format(jx, min_y, round(jx - min_y, 2)))
            print("第一涨幅满足位：{} + {} = {} 元".format(jx, round(jx - min_y, 2), round(jx + (jx - min_y), 2)))
            print("第二涨幅满足位：{} + {} = {} 元".format(round(jx + (jx - min_y), 2), round(jx - min_y, 2),
                                                  round(jx + 2 * (jx - min_y), 2)))
            print("第三涨幅满足位：{} + {} = {} 元".format(round(jx + 2 * (jx - min_y), 2), round(jx - min_y, 2),
                                                  round(jx + 3 * (jx - min_y), 2)))
if __name__ == "__main__":
    k = KLineChart("000612", start_date="20240101", end_date="20240516")

    k.plot(area=['V', 'KL'], width=1200, height=600, klines=['ma5', 'ma10'])
    # k.web()
    print(k.data)
