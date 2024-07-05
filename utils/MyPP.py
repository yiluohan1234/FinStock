#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: MyPP.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年7月5日
#    > description: 数据绘图工具
#######################################################################
from pyecharts.charts import Kline, Scatter, Line, Grid, Bar, EffectScatter, Tab, Pie, Page
from pyecharts.components import Table
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode


def PLINE(df, lines=[], precision=2):
    '''
    利用pyecharts绘制折线图
    :param df: 数据
    :param lines: y坐标名称列表，列表最大为7
    :param precision: 数据小数位
    :return: Line对象
    '''
    kind = "%Y-%m-%d" if df.index[0].hour == 0 else "%Y-%m-%d %H:%M"
    x_index = df.index.strftime(kind).tolist()
    color = ['green', 'red', 'blue', 'cyan', 'yellow', 'orange', 'purple']
    _line = Line().add_xaxis(x_index)
    for i, line in enumerate(lines):
        _line.add_yaxis(
            series_name=line,
            y_axis=round(df[line], precision).values.tolist(),
            is_smooth=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(type_='solid', width=1),
            itemstyle_opts=opts.ItemStyleOpts(color=color[i]),
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(name='0', y=0, symbol='none', )],
                linestyle_opts=opts.LineStyleOpts(width=1, color='#301934', ),
            ),
            is_symbol_show=False, symbol_size=1
        )
    _line.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="category",
            # 分割线
            splitline_opts=opts.SplitLineOpts(is_show=False),
            axislabel_opts=opts.LabelOpts(is_show=False),
        ),
        yaxis_opts=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(is_show=False),
            # 坐标轴刻度
            axistick_opts=opts.AxisTickOpts(is_show=False),
            # 分割线
            splitline_opts=opts.SplitLineOpts(is_show=False),
        ),
        legend_opts=opts.LegendOpts(is_show=False),
        # 区域缩放配置
        datazoom_opts=[
            opts.DataZoomOpts(is_show=False, type_="inside", range_start=0, range_end=100),
            opts.DataZoomOpts(is_show=False, type_="slider", range_start=0, range_end=100),
        ],
        # 工具箱配置
        tooltip_opts=opts.TooltipOpts(
            trigger="axis",
            axis_pointer_type="cross",
            background_color="rgba(245, 245, 245, 0.8)",
            border_width=2,
            border_color="#ccc",
            textstyle_opts=opts.TextStyleOpts(color="#000", font_size=10),
        ),
        # 坐标轴指示器配置
        axispointer_opts=opts.AxisPointerOpts(
            is_show=True,
            link=[{"xAxisIndex": "all"}],
            label=opts.LabelOpts(background_color="#777"),
        ),
    )
    return _line


def PBUY_SELL(df, col_name):
    '''
    利用pyecharts绘制买卖散点图，叠加到col_name直线上
    :param df: 数据
    :param col_name: 叠加的曲线列名称
    :return: 返回(buy,sell)对象
    '''
    kind = "%Y-%m-%d" if df.index[0].hour == 0 else "%Y-%m-%d %H:%M"
    x_buy = df[df['BUY'] == True].index.strftime(kind).tolist()
    x_sell = df[df['SELL'] == True].index.strftime(kind).tolist()
    y_buy = df[df['BUY'] == True][col_name].values.tolist()
    y_sell = df[df['SELL'] == True][col_name].values.tolist()
    es_buy = (
        Scatter()
            .add_xaxis(x_buy)
            .add_yaxis(series_name='', y_axis=y_buy, symbol='triangle', symbol_size=5, )
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False), itemstyle_opts=opts.ItemStyleOpts(color="red"))
            # 图例和视觉映射配置
            .set_global_opts(legend_opts=opts.LegendOpts(is_show=False), visualmap_opts=opts.VisualMapOpts(is_show=False))
    )
    es_sell = (
        Scatter()
            .add_xaxis(x_sell)
            .add_yaxis(series_name='', y_axis=y_sell, symbol='triangle', symbol_size=5, symbol_rotate=180, )
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False), itemstyle_opts=opts.ItemStyleOpts(color="green"))
            # 图例和视觉映射配置
            .set_global_opts(legend_opts=opts.LegendOpts(is_show=False), visualmap_opts=opts.VisualMapOpts(is_show=False))
    )
    return es_buy, es_sell


## 应用类
def PMACD(df, col_name=['MACD', 'DIF', 'DEA'], precision=2):
    '''
    利用pyecharts绘制MACD图
    :param df: 数据
    :param col_name: 相关列名的列表
    :param precision: 数据小数位
    :return: Bar对象
    '''
    kind = "%Y-%m-%d" if df.index[0].hour == 0 else "%Y-%m-%d %H:%M"
    x_index = df.index.strftime(kind).tolist()
    bar = (
        Bar()
            .add_xaxis(x_index)
            .add_yaxis(
                series_name=col_name[0],
                y_axis=round(df[col_name[0]], precision).values.tolist(),
                label_opts=opts.LabelOpts(is_show=False),
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
                markline_opts=opts.MarkLineOpts(
                    data=[opts.MarkLineItem(name='0', y=0, symbol='none', ), ],
                    linestyle_opts=opts.LineStyleOpts(width=1, color='#301934', ),
                )
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    # 分割线
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                yaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    # 坐标轴刻度
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    # 分割线
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
                # 区域缩放配置
                datazoom_opts=[
                    opts.DataZoomOpts(is_show=False, type_="inside", range_start=0, range_end=100),
                    opts.DataZoomOpts(is_show=False, type_="slider", range_start=0, range_end=100),
                ],
                # 工具箱配置
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis",
                    axis_pointer_type="cross",
                    background_color="rgba(245, 245, 245, 0.8)",
                    border_width=2,
                    border_color="#ccc",
                    textstyle_opts=opts.TextStyleOpts(color="#000", font_size=10),
                ),
                # 坐标轴指示器配置
                axispointer_opts=opts.AxisPointerOpts(
                    is_show=True,
                    link=[{"xAxisIndex": "all"}],
                    label=opts.LabelOpts(background_color="#777"),
                ),
            )
    )
    line = PLINE(df, col_name[1:])
    macd = bar.overlap(line)

    return macd


def PVOL(df, col_name='volume', precision=2):
    '''
    利用pyecharts绘制VOL图
    :param df: 数据
    :param col_name: VOL列名
    :param precision: 数据小数位
    :return: Bar对象
    '''
    kind = "%Y-%m-%d" if df.index[0].hour == 0 else "%Y-%m-%d %H:%M"
    x_index = df.index.strftime(kind).tolist()
    volume = (
        Bar()
            .add_xaxis(x_index)
            .add_yaxis(
                series_name=col_name,
                y_axis=round(df[col_name], precision).values.tolist(),
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode(
                        """
                        function(params) {
                            var colorList;
                            if (barData[params.dataIndex][1] > barData[params.dataIndex][0]) {
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
            .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                # 分割线
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(is_show=False),
                # 坐标轴刻度
                axistick_opts=opts.AxisTickOpts(is_show=False),
                # 分割线
                splitline_opts=opts.SplitLineOpts(is_show=False),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
            # 区域缩放配置
            datazoom_opts=[
                opts.DataZoomOpts(is_show=False, type_="inside", range_start=0, range_end=100),
                opts.DataZoomOpts(is_show=False, type_="slider", range_start=0, range_end=100),
            ],
            # 工具箱配置
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                background_color="rgba(245, 245, 245, 0.8)",
                border_width=2,
                border_color="#ccc",
                textstyle_opts=opts.TextStyleOpts(color="#000", font_size=10),
            ),
            # 坐标轴指示器配置
            axispointer_opts=opts.AxisPointerOpts(
                is_show=True,
                link=[{"xAxisIndex": "all"}],
                label=opts.LabelOpts(background_color="#777"),
            ),
        )
    )
    # 传递js函数所需的数据
    volume.add_js_funcs("var barData = {}".format(df[['open', 'close', 'low', 'high']].values.tolist()))
    return volume
