#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: AIndex.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年4月27日
#    > description: A股K线行情
#######################################################################
from pyecharts.charts import Kline, Scatter, Line, Grid, Bar, EffectScatter
from pyecharts import options as opts
from pyecharts.globals import SymbolType, ThemeType
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
import datetime


class AIndex:

    def __init__(self, code='sh000001', start_date='20200101', end_date='20240202', freq='D', precision=2):
        '''
        @params:
        - code: str                      #股票代码
        - start_date: str                #开始时间
        - end_date: str                  #结束时间
        - freq : str                     #默认 D 日线数据
        - precision :str                 #数据精度,默认2
        '''
        self.title = self.code2name(code)
        self.precision = precision
        self.ema_list = [5, 10, 20, 30, 60, 120, 250]

        # 如果默认日期为'20240202'，则end_date转为最新的日期
        if end_date == '20240202':
            now = datetime.datetime.now()
            if now.hour >= 15:
                end_date = now.strftime('%Y%m%d')
            else:
                yesterday = now - datetime.timedelta(days=1)
                end_date = yesterday.strftime('%Y%m%d')

        if freq == 'D':
            df = self.get_data(code, start_date, end_date)
            self.data = df.copy()
            self.dateindex = df.index.strftime("%Y-%m-%d").tolist()

        # 获取volume的标识符
        self.data['f'] = self.data.apply(lambda x: self.frb(x.open, x.close), axis=1)
        self.prices_cols = ['open', 'close', 'low', 'high']

    def code2name(self, code):
        '''获取股票代码名称
        @params:
        - code: str                      #股票代码
        '''
        code_name = {"sh000001": "上证指数",
                     "sh880326": "铝"}

        return code_name[code]

    def frb(self, open_value, close_value):
        '''获取volume的标识
        @params:
        - code: str                      #股票代码
        '''
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
        '''获取股票的综合数据
        @params:
        - code: str                      #股票代码
        - start_date: str                #开始时间
        - end_date: str                  #结束时间
        '''
        df = ak.stock_zh_index_daily(symbol=code).iloc[:, :6]

        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', ]
        df['volume'] = round(df['volume'].astype('float') / 10000, 2)

        # 计算均线
        for i in self.ema_list:
            df['ma{}'.format(i)] = round(df.close.rolling(i).mean(), self.precision)

        # 计算抵扣差
        for i in self.ema_list:
            df['dkc{}'.format(i)] = round(df["close"] - df["close"].shift(i - 1), self.precision)

        # 计算乖离率
        for i in self.ema_list:
            df['bias{}'.format(i)] = round(
                (df["close"] - df["ma{}".format(i)]) * 100 / df["ma{}".format(i)],
                self.precision)

        # 计算k率
        for i in self.ema_list:
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

        # 计算包络线ENE(10,9,9)
        # ENE代表中轨。MA(CLOSE,N)代表N日均价
        # UPPER:(1+M1/100)*MA(CLOSE,N)的意思是，上轨距离N日均价的涨幅为M1%；
        # LOWER:(1-M2/100)*MA(CLOSE,N) 的意思是，下轨距离 N 日均价的跌幅为 M2%;
        df['ene'] = df.close.rolling(10).mean()
        df['upper'] = (1 + 9.0 / 100) * df['ene']
        df['lower'] = (1 - 9.0 / 100) * df['ene']

        # 计算MACD
        # df['DIF'], df['DEA'], df['MACD'] = self.get_macd_data(df)
        df['DIF'], df['DEA'], df['MACD'] = self.cal_macd(df)

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
        '''获取股票分钟的综合数据
        @params:
        - code: str                      #股票代码
        - current_date: str              #日期，如'20240202'
        '''
        if int(code) > 600000:
            symbol = "sh" + str(code)
        else:
            symbol = "sz" + str(code)

        df = ak.stock_zh_a_minute(symbol=symbol, period="30", adjust="qfq")
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', ]
        df['volume'] = round(df['volume'].astype('float') / 10000, 2)
        df = df[pd.to_datetime(df['date']).dt.date.astype('str') == current_date]

       # 计算均线
        for i in self.ema_list:
            df['ma{}'.format(i)] = round(df.close.rolling(i).mean(), self.precision)

        # 计算抵扣差
        for i in self.ema_list:
            df['dkc{}'.format(i)] = round(df["close"] - df["close"].shift(i - 1), self.precision)

        # 计算乖离率
        for i in self.ema_list:
            df['bias{}'.format(i)] = round(
                (df["close"] - df["ma{}".format(i)]) * 100 / df["ma{}".format(i)],
                self.precision)

        # 计算k率
        for i in self.ema_list:
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

        # 计算包络线ENE(10,9,9)
        # ENE代表中轨。MA(CLOSE,N)代表N日均价
        # UPPER:(1+M1/100)*MA(CLOSE,N)的意思是，上轨距离N日均价的涨幅为M1%；
        # LOWER:(1-M2/100)*MA(CLOSE,N) 的意思是，下轨距离 N 日均价的跌幅为 M2%;
        df['ene'] = df.close.rolling(10).mean()
        df['upper'] = (1 + 9.0 / 100) * df['ene']
        df['lower'] = (1 - 9.0 / 100) * df['ene']

        # 计算MACD
        # df['DIF'], df['DEA'], df['MACD'] = self.get_macd_data(df)
        df['DIF'], df['DEA'], df['MACD'] = self.cal_macd(df)

        # 标记买入和卖出信号
        for i in range(len(df)):
            if df.loc[i, 'close'] > df.loc[i, 'up']:
                df.loc[i, 'SELL'] = True
            if df.loc[i, 'close'] < df.loc[i, 'down']:
                df.loc[i, 'BUY'] = True

        # 把date作为日期索引
        df.index = pd.to_datetime(df.date)
        return df

    def cal_K(self, df, precision=2):
        '''获取股票斜率
        @params:
        - df: dataframe               #数据
        - precision: int              #保留小数位
        '''
        y_arr = np.array(df).ravel()
        x_arr = list(range(1, len(y_arr) + 1))
        fit_K = np.polyfit(x_arr, y_arr, deg=1)
        return round(fit_K[0], precision)

    def ema(self, df_close, window):
        """计算指数移动平均值
        @params:
        - df_close: dataframe      #收盘dataframe
        - window: int              #移动数
        """
        return df_close.ewm(span=window, min_periods=window, adjust=False).mean()

    def cal_macd(self, df, short=12, long=26, mid=9):
        """计算MACD指标
        @params:
        - df: dataframe           #datframe数据
        - short: int              #短期
        - long: int               #长期
        - mid: int                #天数
        """
        dif = self.ema(df.close, short) - self.ema(df.close, long)
        dea = self.ema(dif, mid)
        macd = (dif - dea) * 2
        return dif, dea, macd

    def get_macd_data(self, df, fastperiod=12, slowperiod=26, signalperiod=9):
        '''获取macd的指标数据
        @params:
        - df: dataframe                  #数据
        - fastperiod: str                #长期
        - slowperiod: str                #短期
        - signalperiod: str              #天数
        # https://cloud.tencent.com/developer/article/1794902
        '''
        import talib
        DIF, DEA, MACD = talib.MACDEXT(df['close'], fastperiod=fastperiod, fastmatype=1,
                                                         slowperiod=slowperiod, slowmatype=1, signalperiod=signalperiod,
                                                         signalmatype=1)
        MACD = MACD * 2
        return DIF, DEA, MACD

    def K(self):
        '''绘制k线图
        '''
        data = self.data[self.prices_cols].values.tolist()
        c = (Kline()
            .add_xaxis(self.dateindex)
            .add_yaxis(
                series_name="k线",  # 序列名称
                y_axis=data,        # Y轴数据
                itemstyle_opts=opts.ItemStyleOpts(color="#ec0000", color0="#00da3c"), # 红色，绿色
                markpoint_opts=opts.MarkPointOpts(
                    data=[          # 添加标记符
                        opts.MarkPointItem(type_='max', name='最大值', value_dim='highest'),
                        opts.MarkPointItem(type_='min', name='最小值', value_dim='lowest'), ],
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=self.title, pos_left='10%'), # 标题设置
                legend_opts=opts.LegendOpts(                                 # 图例配置项
                    is_show=True, pos_top=10, pos_left="center"
                ),
                datazoom_opts=[                                              # 区域缩放
                    opts.DataZoomOpts(
                        is_show=False,
                        type_="inside",      # 内部缩放
                        xaxis_index=[0, 1],  # 可缩放的x轴坐标编号
                        range_start=80,      # 初始显示范围
                        range_end=100,       # 初始显示范围
                    ),
                    opts.DataZoomOpts(
                        is_show=True,
                        xaxis_index=[0, 1],
                        type_="slider",       # 外部滑动缩放
                        pos_top="90%",        # 放置位置
                        range_start=80,       # 初始显示范围
                        range_end=100,        # 初始显示范围
                    ),
                ],
                xaxis_opts=opts.AxisOpts(     # x坐标轴配置项
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                ),
                yaxis_opts=opts.AxisOpts(     # y坐标轴配置项
                    is_scale=True,
                    splitarea_opts=opts.SplitAreaOpts( # 直角坐标系分割区域配置
                        is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                    ),
                ),
                tooltip_opts=opts.TooltipOpts(  # 提示框配置顶
                    trigger="axis",
                    axis_pointer_type="cross",
                    background_color="rgba(245, 245, 245, 0.8)",
                    border_width=2,
                    border_color="#ccc",
                    textstyle_opts=opts.TextStyleOpts(color="#000"),
                ),
                visualmap_opts=opts.VisualMapOpts( # 视觉映射配置项
                    is_show=False, dimension=2,
                    series_index=5, is_piecewise=True,
                    pieces=[
                        {"value": 1, "color": "#00da3c"},
                        {"value": -1, "color": "#ec0000"},
                    ],
                ),
                axispointer_opts=opts.AxisPointerOpts( # 坐标轴指示器配置项
                    is_show=True,
                    link=[{"xAxisIndex": "all"}],
                    label=opts.LabelOpts(background_color="#777"),
                ),
                brush_opts=opts.BrushOpts(  # 区域选择组件配置项
                    x_axis_index="all",
                    brush_link="all",
                    out_of_brush={"colorAlpha": 0.1},
                    brush_type="lineX",
                ),
            )
        )

        # 叠加均线
        if len(self.klines) != 0:
            kLine = Line().add_xaxis(self.dateindex)
            for i in self.klines:
                kLine.add_yaxis(
                    series_name=i,
                    y_axis=round(self.data[i], self.precision).values.tolist(),
                    is_smooth=True,
                    is_symbol_show=False,
                    is_hover_animation=False,
                    label_opts=opts.LabelOpts(is_show=False),
                    linestyle_opts=opts.LineStyleOpts(type_='solid', width=2),
                )
            kLine.set_global_opts(xaxis_opts=opts.AxisOpts(type_="category", is_show=False))
            c.overlap(kLine)

        # 叠加自定义直线
        if len(self.jxPoints) != 0:
            jxs = self.plot_multi_jx(self.jxPoints)
            c.overlap(jxs)

        # 叠加三个涨跌幅满足位直线
        if len(self.jxLines) != 0:
            lines = self.plot_three_stage(self.jxLines)
            c.overlap(lines)

        # 叠加买卖标记
        if 'BUY' in self.data.columns:
            v1 = self.data[self.data['BUY'] == True].index.strftime("%Y-%m-%d").tolist()
            v2 = self.data[self.data['BUY'] == True]['close'].values.tolist()
            es_buy = (
                Scatter()
                    .add_xaxis(v1)
                    .add_yaxis(
                        series_name='',
                        y_axis=v2,
                        xaxis_index=0,
                        symbol='triangle',
                        symbol_size=10,  # 设置散点的大小
                    )
                    .set_series_opts(
                        label_opts=opts.LabelOpts(is_show=False),
                        itemstyle_opts=opts.ItemStyleOpts(color="red")
                    )
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
                    .add_yaxis(
                        series_name='',
                        y_axis=v2,
                        xaxis_index=0,
                        symbol='triangle',
                        symbol_size=10,  # 设置散点的大小
                        symbol_rotate=180,
                    )
                    .set_series_opts(
                        label_opts=opts.LabelOpts(is_show=False),
                        itemstyle_opts=opts.ItemStyleOpts(color="green")
                    )
                    .set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
                    .set_global_opts(visualmap_opts=opts.VisualMapOpts(is_show=False))
            )
            c.overlap(es_sell)

        return c

    def V(self):
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
                    axisline_opts=opts.AxisLineOpts(is_show=True),
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

    def MACD(self):
        '''绘制MACD图
        '''
        c = (Bar()
            .add_xaxis(self.dateindex)
            .add_yaxis(
                series_name="macd",
                y_axis=round(self.data.MACD, self.precision).values.tolist(), stack="v",
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
                    legend_opts=opts.LegendOpts(
                        orient='vertical',
                        pos_left="top",
                        pos_top="70%",
                        is_show=False
                    ),
                    xaxis_opts=opts.AxisOpts(
                        type_="category",
                        is_scale=True,
                        grid_index=1,
                        boundary_gap=False,
                        axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                        axistick_opts=opts.AxisTickOpts(is_show=False),
                        splitline_opts=opts.SplitLineOpts(is_show=False),
                        axislabel_opts=opts.LabelOpts(is_show=False),
                    ),
                    yaxis_opts=opts.AxisOpts(
                        axislabel_opts=opts.LabelOpts(is_show=False),
                        axisline_opts=opts.AxisLineOpts(is_show=True),
                        axistick_opts=opts.AxisTickOpts(is_show=False), # 不显示刻度线
                        # splitline_opts=opts.SplitLineOpts(is_show=False),
                    ),
            )
        )
        dea = round(self.data.DEA, self.precision).values.tolist()
        dif = round(self.data.DIF, self.precision).values.tolist()
        macd_line = (Line()
                .add_xaxis(self.dateindex)
                .add_yaxis(
                    series_name="DIF",
                    y_axis=dif,
                    is_symbol_show=False,
                    label_opts=opts.LabelOpts(is_show=False),
                    linestyle_opts=opts.LineStyleOpts(type_='solid', width=2),
                )
                .add_yaxis(
                    series_name="DEA",
                    y_axis=dea,
                    is_symbol_show=False,
                    label_opts=opts.LabelOpts(is_show=False)
                )
                .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(datazoom_opts=[opts.DataZoomOpts()],)
        )
        c.overlap(macd_line)
        return c

    def DKC(self):
        '''绘制抵扣差图
        '''
        dkc_name = "dkc{}".format(self.n)

        c = (Line()
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
                    axisline_opts=opts.AxisLineOpts(is_show=True),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
                yaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    axisline_opts=opts.AxisLineOpts(is_show=True),
                    axistick_opts=opts.AxisTickOpts(is_show=False), # 不显示刻度线
                    # splitline_opts=opts.SplitLineOpts(is_show=False),
                ),
            )
        )
        return c

    def BIAS(self):
        '''绘制乖离率图
        '''
        bias_name = "bias{}".format(self.n)

        c = (Line()
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
                    data=[
                        opts.MarkLineItem(name='0值', y=0, symbol='none', ),
                        opts.MarkLineItem(name='最大值', y=20, symbol='none', ),
                        opts.MarkLineItem(name='最小值', y=-20, symbol='none', )
                    ],
                    linestyle_opts=opts.LineStyleOpts(width=0.1, color='#301934', ),
                )
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",  # 坐标轴类型-离散数据
                    grid_index=1,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    axisline_opts=opts.AxisLineOpts(is_show=True),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
                yaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    axisline_opts=opts.AxisLineOpts(is_show=True),
                    axistick_opts=opts.AxisTickOpts(is_show=False), # 不显示刻度线
                    # splitline_opts=opts.SplitLineOpts(is_show=False),
                ),
            )
        )
        return c

    def KL(self):
        '''绘制斜率图
        @params:
        '''
        k_name = "k{}".format(self.n)

        c = (Line()
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
                    axisline_opts=opts.AxisLineOpts(is_show=True),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
                yaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    axisline_opts=opts.AxisLineOpts(is_show=True),
                    axistick_opts=opts.AxisTickOpts(is_show=False), # 不显示刻度线
                    # splitline_opts=opts.SplitLineOpts(is_show=False),
                ),
            )
        )
        return c

    def DMA(self):
        '''绘制均线差图
        '''
        # 计算均线差
        self.data['DMA'] = round(self.data[self.dmalines[0]] - self.data[self.dmalines[1]], self.precision)
        c = (Line()
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
                    axisline_opts=opts.AxisLineOpts(is_show=True),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
                yaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    axisline_opts=opts.AxisLineOpts(is_show=True),
                    axistick_opts=opts.AxisTickOpts(is_show=False), # 不显示刻度线
                    # splitline_opts=opts.SplitLineOpts(is_show=False),
                ),
            )
        )
        return c

    def plot(self, n=20, area=['V', 'DKC'], width=1000, height=600, klines=[], vlines=[], dmalines=[], jxPoints=[], jxLines=[], is_notebook=False):
        '''
        @params:
        - n:int                 #抵扣差、乖离率、斜率的计算天数
        - area : list           #显示区域
                                'V'      交易量
                                'M'      k线+MACD
                                'DKC'    抵扣差
                                'BIAS'   乖离率
                                'KL'     K率
                                'DMA'    均线差
                                FieldName: string   Dataframe中的字段名
                                [Field1,Field2,...] Dataframe中的字段名列表，将显示在一个区域
          width: int            #图表宽度 px
          height: int            #图表高度 px
          klines: list           #K线区域显示的数据，Dataframe中的字段名，如['ma5','ma10','ma20','ma60', 'ma120', 'ma250', 'boll', 'up', 'down', 'stop', 'ene', 'upper', 'lower']
          vline: list           #Volume区域显示的数据，Dataframe中的字段名，如MA...
          dmalines: list        #线误差的两个均线选择，如['ma5', 'ma10']
          jxPoints: list        #绘制多个颈线的坐标，如jxPoints=[[("2024-03-01",38.80), ("2024-04-09",38.80)], [("2024-01-11",18.80), ("2024-01-31",28.80)]])
          jxLines: list        #绘制多个颈线的坐标，如[jx, max_y, start_date, end_date]
          is_notebook: bool    #是否在notebook绘制
        - sample:
           chart=data.plot(area=[['V','DKC'],'V'],vlines=['vMA5','vMA10'],klines=['ma5','ma10'])
        '''
        self.n = n
        self.klines = klines
        self.vlines = vlines
        self.dmalines = dmalines
        self.jxPoints = jxPoints
        self.jxLines = jxLines
        grid = (Grid(init_opts=opts.InitOpts(
                width=str(width) + "px",
                height=str(height) + "px",
                animation_opts=opts.AnimationOpts(animation=False),
                ))
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
                    window.add_yaxis(
                        series_name=l,
                        y_axis=round(self.data[l], self.precision).values.tolist(),
                        is_smooth=True,
                        is_symbol_show=False,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False),
                        linestyle_opts=opts.LineStyleOpts(type_='solid', width=2)
                    )
                window.axislabel_opts = opts.LabelOpts(is_show=False),
                window.set_global_opts(
                    datazoom_opts=[opts.DataZoomOpts()],
                    xaxis_opts=opts.AxisOpts(
                        type_="category",
                        axislabel_opts=opts.LabelOpts(is_show=False),
                    ),
                    legend_opts=opts.LegendOpts(orient='vertical', pos_left="top", pos_top=str(iButton) + "%"),
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
                    window.add_yaxis(
                        series_name=wi,
                        y_axis=round(self.data[w], self.precision).values.tolist(),
                        is_smooth=True,
                        is_symbol_show=False,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False),
                        linestyle_opts=opts.LineStyleOpts(type_='solid', width=2)
                    )
                window.axislabel_opts = opts.LabelOpts(is_show=True),
                window.set_global_opts(
                    datazoom_opts=[opts.DataZoomOpts()],
                    xaxis_opts=opts.AxisOpts(
                        type_="category",
                        axislabel_opts=opts.LabelOpts(is_show=False),

                    ),
                    legend_opts=opts.LegendOpts(orient='horizontal', pos_left=str(icount + 20) + "%"),

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
        if is_notebook:
            return grid.render_notebook()
        else:
            grid.render("./kline.html")
            # return grid.render_notebook()
            return grid

    def save_png(self, charts, filename):
        '''保存pyecharts图片
        @params:
        - charts:int            #pyecharts对象
        - filename:str         #保存文件名称
        '''
        make_snapshot(snapshot, charts.render(), filename)

    def web(self):
        '''通过浏览器打开pyecharts的html文件
        '''
        webbrowser.open_new_tab('file://' + os.path.realpath('./kline.html'))

    def get_date_k(self, n=10):
        '''返回最近n个交易日的斜率数据
        @params:
        - n: int         #最近n天
        '''
        now = datetime.datetime.now()
        yesterday = now - datetime.timedelta(days=n)
        current_date = yesterday.strftime('%Y-%m-%d')
        data_n = self.data[self.data['date'].astype(str) > current_date][['close', 'k10', 'k20', 'k60']]
        return data_n

    def get_three_stage(self, jx, max_y, is_print=False):
        '''获取三个涨跌幅满足位
        @params:
        - jx: float        #颈线数据
        - max_y: float     #颈线下最低值或颈线上最高值
        - is_print: bool   #是否打印结果
        '''
        if max_y > jx:
            stop_line = round(jx * 1.03, 2)
            jx_to_stop = round(max_y - jx, 2)
            one_stage = round(jx - (max_y - jx), 2)
            two_stage = round(jx - 2 * (max_y - jx), 2)
            three_stage = round(jx - 3 * (max_y - jx), 2)
            if is_print:
                print("止损位：{}x( 1 + 3% )={}".format(jx, stop_line))
                print("顶点到颈线的距离：{} - {} = {} 元".format(max_y, jx, jx_to_stop))
                print("第一跌幅满足位：{} - {} = {} 元".format(jx, jx_to_stop, one_stage))
                print("第二跌幅满足位：{} - {} = {} 元".format(one_stage, jx_to_stop, two_stage))
                print("第三跌幅满足位：{} - {} = {} 元".format(two_stage, jx_to_stop, three_stage))
        else:
            stop_line = round(jx * 0.97, 2)
            jx_to_stop = round(jx - max_y, 2)
            one_stage = round(jx + (jx - max_y), 2)
            two_stage = round(jx + 2 * (jx - max_y), 2)
            three_stage = round(jx + 3 * (jx - max_y), 2)
            if is_print:
                print("止损位：{}x( 1 - 3% )={}".format(jx, stop_line))
                print("顶点到颈线的距离：{} - {} = {} 元".format(jx, max_y, jx_to_stop))
                print("第一涨幅满足位：{} + {} = {} 元".format(jx, jx_to_stop, one_stage))
                print("第二涨幅满足位：{} + {} = {} 元".format(one_stage, jx_to_stop, two_stage))
                print("第三涨幅满足位：{} + {} = {} 元".format(two_stage, jx_to_stop, three_stage))
        return [stop_line, one_stage, two_stage, three_stage]

    def plot_jx(self, points_list):
        '''绘制一个颈线
        @params:
        - points_list : list   #颈线的开始和结束节点的元组列表
        '''
        #bar = plot_jx([("Mon",820), ("Tue",932)])
        x_data = []
        y_data = []
        for point in points_list:
            x_data.append(point[0])
            y_data.append(point[1])

        jx = (Line()
                .set_global_opts(
                    tooltip_opts=opts.TooltipOpts(is_show=False),
                    xaxis_opts=opts.AxisOpts(type_="category", is_show=False),
                    yaxis_opts=opts.AxisOpts(
                        type_="value",
                        axistick_opts=opts.AxisTickOpts(is_show=True),
                        splitline_opts=opts.SplitLineOpts(is_show=True),
                        is_show=False
                    ),
                )
                .add_xaxis(xaxis_data=x_data)
                .add_yaxis(
                    series_name="",
                    y_axis=y_data,
                    symbol="emptyCircle",
                    is_symbol_show=True,
                    label_opts=opts.LabelOpts(is_show=False),
                    linestyle_opts=opts.LineStyleOpts(type_='solid', width=2, color='red'),
                )
        )
        return jx

    def plot_multi_jx(self, lines_list):
        '''绘制多个颈线，每个颈线由一个list组成
        @params:
        - lines_list : list   #颈线的开始和结束节点的元组列表
        '''
        x_data = []

        for line in lines_list:
            for point in line:
                x_data.append(point[0])

        if len(lines_list) != 0:
            _line = Line().add_xaxis(x_data)
            for i, line in enumerate(lines_list):
                y_data = [None for a in range(i*2)]
                for point in line:
                    y_data.append(point[1])

                _line.add_yaxis(
                    "", y_data,
                    is_smooth=True,
                    is_symbol_show=False,
                    is_hover_animation=False,
                    label_opts=opts.LabelOpts(is_show=False),
                    linestyle_opts=opts.LineStyleOpts(type_='solid', width=2),
                    #markpoint_opts=opts.MarkPointOpts(
                    #  data=[opts.MarkPointItem(name='',coord=[x_data[i*2],y_data[i*2]],value=y_data[i*2])]
                    #)
                )
                _line.set_global_opts(
                    tooltip_opts=opts.TooltipOpts(is_show=False),
                    xaxis_opts=opts.AxisOpts(type_="category", is_show=False),
                    yaxis_opts=opts.AxisOpts(
                        type_="value",
                        axistick_opts=opts.AxisTickOpts(is_show=True),
                        splitline_opts=opts.SplitLineOpts(is_show=True),
                        is_show=False
                    ),
                )
        return _line

    def plot_three_stage(self, jxLines):
        '''绘制多个颈线，每个颈线由一个list组成
        @params:
        - jxLines : list   #jxLines, [jx, max_y, start_date, end_date]
        '''
        jx = jxLines[0]
        max_y = jxLines[1]
        start_date = jxLines[2]
        end_date = jxLines[3]
        lines_list = []
        ret_list = self.get_three_stage(jx, max_y)
        for re in ret_list:
            lines_list.append([(start_date, re), (end_date, re)])

        x_data = []

        for line in lines_list:
            for point in line:
                x_data.append(point[0])

        if len(lines_list) != 0:
            _line = Line().add_xaxis(x_data)
            for i, line in enumerate(lines_list):
                y_data = [None for a in range(i*2)]
                for point in line:
                    y_data.append(point[1])

                _line.add_yaxis(
                    "",
                    y_data,
                    is_smooth=True,
                    is_symbol_show=False,
                    is_hover_animation=False,
                    label_opts=opts.LabelOpts(is_show=False),
                    linestyle_opts=opts.LineStyleOpts(type_='solid', width=2),
                    markpoint_opts=opts.MarkPointOpts(
                      data=[opts.MarkPointItem(name='',coord=[x_data[i*2],y_data[i*2]],value=y_data[i*2])]
                    )
                )
                _line.set_global_opts(
                    tooltip_opts=opts.TooltipOpts(is_show=False),
                    xaxis_opts=opts.AxisOpts(type_="category", is_show=False),
                    yaxis_opts=opts.AxisOpts(
                        type_="value",
                        axistick_opts=opts.AxisTickOpts(is_show=True),
                        splitline_opts=opts.SplitLineOpts(is_show=True),
                        is_show=False
                    ),
                )
        return _line
