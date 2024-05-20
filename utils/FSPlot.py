#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: FSPlot.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年5月20日
#    > description: 数据绘图工具
#######################################################################
from pyecharts.charts import Kline, Scatter, Line, Grid, Bar, EffectScatter, Tab, Pie, Page
from pyecharts.components import Table
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
# https://blog.csdn.net/qq_42571592/article/details/122826752
from pyecharts.globals import CurrentConfig, NotebookType
# CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_NOTEBOOK
CurrentConfig.ONLINE_HOST = 'https://assets.pyecharts.org/assets/'
from utils.FSData import get_data
import os
import webbrowser
from utils.cons import precision, prices_cols


def K(data, title, klines, jxPoints, jxLines) -> Kline:
    '''
    绘制k线图
    :param data: 数据
    :type data: pandas.DataFrame
    :param title: Kline的标题
    :type title: str
    :param klines: 均线列表
    :type klines: list
    :param jxPoints: 颈线端点的列表，每段颈线一个列表
    :type jxPoints: list
    :param jxLines: 颈线上三个涨降幅满足位列表
    :type jxLines: list
    :return: 返回Kline对象
    :rtype: Kline
    '''
    dateindex = data.index.strftime("%Y-%m-%d").tolist()
    data_k = data[prices_cols].values.tolist()
    c = (Kline()
        .add_xaxis(dateindex)
        .add_yaxis(
            series_name="k线",  # 序列名称
            y_axis=data_k,        # Y轴数据
            itemstyle_opts=opts.ItemStyleOpts(color="#ec0000", color0="#00da3c"), # 红色，绿色
            markpoint_opts=opts.MarkPointOpts(
                data=[          # 添加标记符
                    opts.MarkPointItem(type_='max', name='最大值', value_dim='highest'),
                    opts.MarkPointItem(type_='min', name='最小值', value_dim='lowest'), ],
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title, pos_left='10%'), # 标题设置
            legend_opts=opts.LegendOpts(                                 # 图例配置项
                is_show=True, pos_top=10, pos_left="center"
            ),
            datazoom_opts=[                                              # 区域缩放
                opts.DataZoomOpts(
                    is_show=False,
                    type_="inside",      # 内部缩放
                    xaxis_index=[0, 1],  # 可缩放的x轴坐标编号
                    range_start=0,      # 初始显示范围
                    range_end=100,       # 初始显示范围
                ),
                opts.DataZoomOpts(
                    is_show=False,
                    xaxis_index=[0, 1],
                    type_="slider",       # 外部滑动缩放
                    pos_top="90%",        # 放置位置
                    range_start=0,       # 初始显示范围
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
    if len(klines) != 0:
        kLine = Line().add_xaxis(dateindex)
        for i in klines:
            kLine.add_yaxis(
                series_name=i,
                y_axis=round(data[i], precision).values.tolist(),
                is_smooth=True,
                is_symbol_show=False,
                is_hover_animation=False,
                label_opts=opts.LabelOpts(is_show=False),
                linestyle_opts=opts.LineStyleOpts(type_='solid', width=2),
            )
        kLine.set_global_opts(xaxis_opts=opts.AxisOpts(type_="category", is_show=False))
        c.overlap(kLine)

    # 叠加自定义直线
    if len(jxPoints) != 0:
        jxs = plot_multi_jx(jxPoints)
        c.overlap(jxs)

    # 叠加三个涨跌幅满足位直线
    if len(jxLines) != 0:
        lines = plot_three_stage(jxLines)
        c.overlap(lines)

    # 叠加买卖标记
    if 'BUY' in data.columns:
        v1 = data[data['BUY'] == True].index.strftime("%Y-%m-%d").tolist()
        v2 = data[data['BUY'] == True]['close'].values.tolist()
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

    if 'SELL' in data.columns:
        v1 = data[data['SELL'] == True].index.strftime("%Y-%m-%d").tolist()
        v2 = data[data['SELL'] == True]['close'].values.tolist()
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


def V(data, vlines) -> Bar:
    '''
    绘制成交量图
    :param data: 数据
    :type data: pandas.DataFrame
    :param vlines: 成交量均线列表
    :type vlines: list
    :return: 返回成交量的Bar对象
    :rtype: Bar
    '''
    dateindex = data.index.strftime("%Y-%m-%d").tolist()
    db = data[['volume', 'f']].reset_index()
    db['i'] = db.index
    v = (Bar()
        .add_xaxis(dateindex)
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
    if len(vlines) != 0:
        vLine = Line().add_xaxis(dateindex)
        for i in vlines:
            vLine.add_yaxis(
                series_name=i,
                y_axis=round(data[i], precision).values.tolist(),
                is_smooth=True,
                is_symbol_show=False,
                is_hover_animation=False,
                label_opts=opts.LabelOpts(is_show=False),
                linestyle_opts=opts.LineStyleOpts(type_='solid', width=2)
            )
        vLine.set_global_opts(xaxis_opts=opts.AxisOpts(type_="category"))
        v.overlap(vLine)
    return v


def MACD(data) -> Bar:
    '''
    绘制MACD图
    :param data: 数据
    :type data: pandas.DataFrame
    :return: 返回MACD的Bar对象
    :rtype: Bar
    '''
    dateindex = data.index.strftime("%Y-%m-%d").tolist()
    c = (Bar()
        .add_xaxis(dateindex)
        .add_yaxis(
            series_name="macd",
            y_axis=round(data.MACD, precision).values.tolist(), stack="v",
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
    dea = round(data.DEA, precision).values.tolist()
    dif = round(data.DIF, precision).values.tolist()
    macd_line = (Line()
                 .add_xaxis(dateindex)
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


def plot_multi_jx(lines_list) -> Line:
    '''
    绘制多个颈线，每个颈线由一个list组成
    :param lines_list: 颈线的开始和结束节点的元组列表
    :type lines_list: list
    :return: 返回自定义颈线的Line对象
    :rtype: Line
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


def get_three_stage(jx, max_y, min_y, is_up=True, is_print=False):
    '''
    获取三个涨跌幅满足位
    :param jx: 颈线数据值
    :type jx: float
    :param max_y: 颈线下最低值或颈线上最高值
    :type max_y: float
    :param min_y: 颈线下最低值或颈线上最高值
    :type min_y: float
    :param is_up: 是否向上
    :type is_up: bool
    :param is_print: 是否打印结果
    :type is_print: bool
    :return: 返回三个满足位列表
    :rtype: list
    '''
    if not is_up:
        stop_line = round(jx * 1.03, 2)
        h = round(max_y - min_y, 2)
        one_stage = round(jx - h, 2)
        two_stage = round(jx - 2 * h, 2)
        three_stage = round(jx - 3 * h, 2)
        if is_print:
            print("止损位：{}x( 1 + 3% )={}".format(jx, stop_line))
            print("顶点到颈线的距离：{} - {} = {} 元".format(max_y, min_y, h))
            print("第一跌幅满足位：{} - {} = {} 元".format(jx, h, one_stage))
            print("第二跌幅满足位：{} - {} = {} 元".format(one_stage, h, two_stage))
            print("第三跌幅满足位：{} - {} = {} 元".format(two_stage, h, three_stage))
    else:
        stop_line = round(jx * 0.97, 2)
        h = round(max_y - min_y, 2)
        one_stage = round(jx + h, 2)
        two_stage = round(jx + 2 * h, 2)
        three_stage = round(jx + 3 * h, 2)
        if is_print:
            print("止损位：{}x( 1 - 3% )={}".format(jx, stop_line))
            print("顶点到颈线的距离：{} - {} = {} 元".format(jx, max_y, h))
            print("第一涨幅满足位：{} + {} = {} 元".format(jx, h, one_stage))
            print("第二涨幅满足位：{} + {} = {} 元".format(one_stage, h, two_stage))
            print("第三涨幅满足位：{} + {} = {} 元".format(two_stage, h, three_stage))
    return [stop_line, one_stage, two_stage, three_stage]


def plot_three_stage(jxLines) -> Line:
    '''绘制多个颈线，每个颈线由一个list组成
    @params:
    - jxLines : list   #jxLines, [jx, max_y, min_y, is_up, start_date, end_date]
    :param jxLines: jxLines, [jx, max_y, min_y, is_up, start_date, end_date]
    :type jxLines: list
    :return: 返回三个满足位的Line对象
    :rtype: Line
    '''
    jx = jxLines[0]
    max_y = jxLines[1]
    min_y = jxLines[2]
    is_up = jxLines[3]
    start_date = jxLines[4]
    end_date = jxLines[5]
    lines_list = []
    ret_list = get_three_stage(jx, max_y, min_y, is_up)
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


def DKC(data, n) -> Line:
    '''
    绘制抵扣差图
    :param data: 数据
    :type data: pandas.DataFrame
    :param n: 数字级别
    :type n: int
    :return: 返回抵扣差Line对象
    :rtype: Line
    '''
    dateindex = data.index.strftime("%Y-%m-%d").tolist()
    dkc_name = "dkc{}".format(n)

    c = (Line()
        .add_xaxis(dateindex)  # X轴数据
        .add_yaxis(
            series_name="抵扣差{}".format(n),
            y_axis=data[dkc_name].values.tolist(),  # Y轴数据
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


def BIAS(data, n) -> Line:
    '''
    绘制乖离率图
    :param data: 数据
    :type data: pandas.DataFrame
    :param n: 数字级别
    :type n: int
    :return: 返回乖离率Line对象
    :rtype: Line
    '''
    dateindex = data.index.strftime("%Y-%m-%d").tolist()
    bias_name = "bias{}".format(n)

    c = (Line()
        .add_xaxis(dateindex)  # X轴数据
        .add_yaxis(
            series_name="乖离率{}".format(n),
            y_axis=data[bias_name].values.tolist(),  # Y轴数据
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


def KL(data, n, KLlines) -> Line:
    '''
    绘制斜率图
    :param data: 数据
    :type data: pandas.DataFrame
    :param n: 数字级别
    :type n: int
    :param KLlines: 斜率列表，['k5', 'k10']
    :type KLlines: list
    :return: 返回斜率Line对象
    :rtype: Line
    '''
    dateindex = data.index.strftime("%Y-%m-%d").tolist()
    k_name = "k{}".format(n)

    c = (Line()
        .add_xaxis(dateindex)  # X轴数据
        .add_yaxis(
            series_name="k{}".format(n),
            y_axis=data[k_name].values.tolist(),  # Y轴数据
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
    # 叠加多个k线
    if len(KLlines) != 0:
        klLine = Line().add_xaxis(dateindex)
        for i in KLlines:
            klLine.add_yaxis(
                series_name=i,
                y_axis=round(data[i], precision).values.tolist(),
                is_smooth=True,
                is_symbol_show=False,
                is_hover_animation=False,
                label_opts=opts.LabelOpts(is_show=False),
                linestyle_opts=opts.LineStyleOpts(type_='solid', width=2),
            )
        klLine.set_global_opts(xaxis_opts=opts.AxisOpts(type_="category", is_show=False))
        c.overlap(klLine)
    return c


def DMA(data, n, dmalines) -> Line:
    '''
    绘制均线差图
    :param data: 数据
    :type data: pandas.DataFrame
    :param n: 数字级别
    :type n: int
    :param dmalines: 均线列表，['ma5', 'ma10']
    :type dmalines: list
    :return: 返回均线差Line对象
    :rtype: Line
    '''
    dateindex = data.index.strftime("%Y-%m-%d").tolist()
    # 计算均线差
    data['DMA'] = round(data[dmalines[0]] - data[dmalines[1]], precision)
    c = (Line()
        .add_xaxis(dateindex)  # X轴数据
        .add_yaxis(
            series_name="dma-{}-{}".format(dmalines[0], dmalines[1]),
            y_axis=data['DMA'].values.tolist(),  # Y轴数据
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


def plot_jx(points_list) -> Line:
    '''
    绘制一个颈线
    :param points_list: 颈线的开始和结束节点的元组列表
    :type points_list: list
    :return: 返回单条颈线的Line对象
    :rtype: Line
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


def plot_bar(data, x, y, title) -> Bar:
    '''
    绘制直方图
    :param data: 数据
    :type data: list
    :param x: 横坐标轴的列名称
    :type x: str
    :param y: 纵坐标轴列名称
    :type y: str
    :param title: 图的标题
    :type title: str
    :return: 返回单条颈线的Line对象
    :rtype: Line
    '''
    bar = (Bar()
        .add_xaxis(xaxis_data=data[x].tolist())
        .add_yaxis(
            series_name=y,
            y_axis=data[y].values.tolist()
        )
        #.reversal_axis() # 旋转柱形图方向
        .set_series_opts(label_opts=opts.LabelOpts(position="right")) # 设置数字标签位置
        .set_global_opts(
        title_opts=opts.TitleOpts(is_show=True, title=title, pos_left="center"),
        yaxis_opts=opts.AxisOpts(
            name="{}".format(y),
            type_="value",
            axislabel_opts=opts.LabelOpts(formatter="{value}"),  # 设置刻度标签的单位
            axistick_opts=opts.AxisTickOpts(is_show=True),           # 显示坐标轴刻度
            splitline_opts=opts.SplitLineOpts(is_show=True),         # 显示分割线
        ),
        visualmap_opts=opts.VisualMapOpts(
            max_=max(data[y].values.tolist()),
            min_=min(data[y].values.tolist()),
            range_color=['#ffe100','#e82727'],
            pos_right='10%',
            pos_top='60%',
            is_show=False
        ),
    )
    )
    return bar


def plot_line(data, x, y, title) -> Line:
    '''
    绘制折线图
    :param data: 数据
    :type data: list
    :param x: 横坐标轴的列名称
    :type x: str
    :param y: 纵坐标轴列名称
    :type y: str
    :param title: 图的标题
    :type title: str
    :return: 返回折线图的Line对象
    :rtype: Line
    '''
    line = (Line()
        .add_xaxis(xaxis_data=data[x].tolist())
        .add_yaxis(
            series_name=y,
            y_axis=data[y].values.tolist(),
            label_opts=opts.LabelOpts(is_show=False),
            # symbol="triangle",
            # symbol_size=20,
        )
        .set_series_opts(
            linestyle_opts=opts.LineStyleOpts(width=1)
        )   # 设置线条宽度为4
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            yaxis_opts=opts.AxisOpts(
                name="{}".format(y),
                type_="value",
                axislabel_opts=opts.LabelOpts(formatter="{value}"),  # 设置刻度标签的单位
                axistick_opts=opts.AxisTickOpts(is_show=True),           # 显示坐标轴刻度
                splitline_opts=opts.SplitLineOpts(is_show=True),         # 显示分割线
            ),
        )
    )
    return line


def plot_bar_line(data, x, y_bar, y_line) -> Bar:
    '''
    绘制业务同比增长的直方折线图
    :param data: 数据
    :type data: list
    :param x: 横坐标轴的列名称
    :type x: str
    :param y_bar: 纵坐标轴列名称(bar)
    :type y_bar: str
    :param y_line: 纵坐标轴列名称(line)
    :type y_line: str
    :return: 返回直方折线图的Bar对象
    :rtype: Bar
    '''
    x_data = df[x].tolist()
    y_bar_data = df[y_bar].values.tolist()
    y_line_data = df[y_line].values.tolist()
    bar = (Bar()
        .add_xaxis(xaxis_data=x_data)
        .add_yaxis(
            series_name=y_bar,    # 此处为设置图例配置项的名称
            y_axis=y_bar_data,
            label_opts=opts.LabelOpts(is_show=False),   # 此处将标签配置项隐藏
            z=0     # 因为折线图会被柱状图遮挡，所以此处把柱状图置底
        )
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="同比增速（%）",
                type_="value",
                #axislabel_opts=opts.LabelOpts(formatter="{value} %"),  # 设置刻度标签的单位
            )
        )
        .set_global_opts(
            # 设置提示框配置项，触发类型为坐标轴类型，指示器类型为"cross"
            tooltip_opts=opts.TooltipOpts(
                is_show=True, trigger="axis", axis_pointer_type="cross"
            ),
            # 设置x轴配置项为类目轴，适用于离散的类目数据
            xaxis_opts=opts.AxisOpts(
                type_="category",
                axispointer_opts=opts.AxisPointerOpts(is_show=True, type_="shadow"),
            ),
            yaxis_opts=opts.AxisOpts(
                name="{}（亿元）".format(y_bar),
                type_="value",
                #axislabel_opts=opts.LabelOpts(formatter="{value} 亿元"),  # 设置刻度标签的单位
                axistick_opts=opts.AxisTickOpts(is_show=True),           # 显示坐标轴刻度
                splitline_opts=opts.SplitLineOpts(is_show=True),         # 显示分割线
            ),
            # 设置标题并将其居中
            title_opts=opts.TitleOpts(
                is_show=True, title="{}及同比增速".format(y_bar), pos_left="center"
            ),
            # 设置图例配置项，并将其放在右下角
            legend_opts=opts.LegendOpts(
                pos_right="right",
                pos_bottom="bottom",
                is_show=False
            ),
        )
    )

    line = (Line()
        .add_xaxis(xaxis_data=x_data)
        .add_yaxis(
            series_name="同比增速",
            yaxis_index=1,
            y_axis=y_line_data,
            label_opts=opts.LabelOpts(is_show=False),
            symbol="triangle",
            symbol_size=20,
        )
        .set_series_opts(
            linestyle_opts=opts.LineStyleOpts(width= 4)
        )   # 设置线条宽度为4
    )

    bar.overlap(line)
    return bar


def plot_tab(df) ->Tab:
    '''
    绘制tab图
    :return: 返回Tab对象
    :rtype: Tab
    '''
    tab = (
        Tab() # 创建Tab类对象
            .add(
                plot_bar_line(df, '报告日', '营业总收入', '营业总收入同比'), # 图表类型
                "营业总收入" # 选项卡的标签名称
            )
            .add(
                plot_bar_line(df, '报告日', '净利润', '净利润同比'),
                "净利润"
            )
    )
    return tab


def title(title) -> Pie:
    '''
    绘制title图
    :param title: 标题
    :return: 返回Pie对象
    :rtype: Pie
    '''
    from datetime import datetime
    now_time = datetime.now().strftime('%Y-%m-%d') # 获取当前时间
    pie = (
        Pie(init_opts=opts.InitOpts(width="600px", height="100px"#,theme=ThemeType.DARK
                                    )) # 不画图，只显示一个标题，用来构成大屏的标题
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=title,
                    # subtitle = f'截至：{now_time}',
                    title_textstyle_opts=opts.TextStyleOpts(
                        font_size=30,
                        #color='#FFFFFF',
                    ),
                    #pos_top=10,
                    pos_left="center"
                ),
                legend_opts=opts.LegendOpts(
                    is_show=False
                )
            )
    )
    return pie
    #https://blog.csdn.net/Student_201/article/details/131189638


def plot_pie(data, x, y, title, classify_type):
    '''
    绘制pie图
    :param data: 数据
    :type data: list
    :param x: 横坐标轴的列名称
    :type x: str
    :param y: 纵坐标轴列名称
    :type y: str
    :param title: 标题
    :type title: str
    :param classify_type: 分类的字段
    :type classify_type: str
    :return: 返回Pie对象
    :rtype: Pie
    '''
    # bar = plot_pie(data, '主营构成', '主营收入', '按产品分类主营构成', '按产品分类')
    data = data[data['分类类型'] == classify_type]
    data = data[[x, y]]
    pie = (
        Pie() # 设置背景的大小
            .add(
                series_name = "按产品分类", # 必须项
                data_pair = data.values.tolist(),
                radius=["20%", "50%"], # 设置环的大小
                rosetype="radius", # 设置玫瑰图类型
                label_opts=opts.LabelOpts(formatter="{b}：{c}\n占比：{d}%"), # 设置标签内容格式
            )
            .set_global_opts(title_opts=opts.TitleOpts(title=title))
    )
    return pie


def plot_multi_bar(x, y, df_list, names_list) -> Bar:
    '''
    绘制多个柱状图对比图
    :param x: 横坐标轴的列名称
    :type x: str
    :param y: 纵坐标轴列名称
    :type y: str
    :param df_list: dataframe列表
    :type df_list: list
    :param names_list: 公司名称列表
    :type names_list: str
    :return: 返回Bar对象
    :rtype: Bar
    '''
    x_data = df_list[0][x].tolist()

    if len(df_list) != 0:
        _bar = Bar().add_xaxis(x_data)
        for i, df in enumerate(df_list):
            _bar.add_yaxis(series_name=names_list[i],
                           y_axis=df[y].values.tolist(),
                           label_opts=opts.LabelOpts(is_show=False),
                           )
            _bar.set_global_opts(
                tooltip_opts=opts.TooltipOpts(is_show=True, trigger="axis", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(type_="category", is_show=True),
                yaxis_opts=opts.AxisOpts(
                    type_="value",
                    axistick_opts=opts.AxisTickOpts(is_show=True),
                    splitline_opts=opts.SplitLineOpts(is_show=True),
                    is_show=True
                ),
            )
    # bar = plot_multi_bar('报告日', '营业总收入', [df, df1], ['612', '977'])
    return _bar


def plot_multi_line(x, y, df_list, names_list):
    '''
    绘制多个折线图对比图
    :param x: 横坐标轴的列名称
    :type x: str
    :param y: 纵坐标轴列名称
    :type y: str
    :param df_list: dataframe列表
    :type df_list: list
    :param names_list: 公司名称列表
    :type names_list: str
    :return: 返回Line对象
    :rtype: Line
    # line = plot_multi_line('报告日', '营业总收入', [df, df1], ['612', '977'])
    '''
    x_data = df_list[0][x].tolist()

    if len(df_list) != 0:
        _line = Line().add_xaxis(x_data)
        for i, df in enumerate(df_list):
            _line.add_yaxis(series_name=names_list[i],
                            y_axis=df[y].values.tolist(),
                            label_opts=opts.LabelOpts(is_show=False),
                            )
            _line.set_global_opts(
                tooltip_opts=opts.TooltipOpts(is_show=True, trigger="axis", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(type_="category", is_show=True),
                yaxis_opts=opts.AxisOpts(
                    type_="value",
                    axistick_opts=opts.AxisTickOpts(is_show=True),
                    splitline_opts=opts.SplitLineOpts(is_show=True),
                    is_show=True
                ),
            )
    # line = plot_multi_line('报告日', '营业总收入', [df, df1], ['612', '977'])
    return _line


def plot_table(data, headers, title):
    table = Table()

    rows = data[headers].values().tolist()
    table.add(headers, rows).set_global_opts(
        title_opts=opts.ComponentTitleOpts(title=title)
    )
    return table


def plot_page(self):
    from core.FundFlow import FundFlow
    from core.Basic import Basic
    df_lrb, df_lrb_display = self.get_lrb_data("000977", 5, 4)
    # df_lrb1, df_lrb_display1 = self.get_lrb_data("000612", 5, 4)
    f = FundFlow()
    df_fund, df_fund_display = f.get_individual_fund_flow("000977", 5)

    #df_zygc = self.get_basic_info("000977")

    b = Basic()
    df_import, df_import_display = b.get_main_indicators_ths("000977", 5)
    df_main, df_main_display = b.get_main_indicators_sina("000977", 5)

    # df_north = f.get_north_data(start_date='20240202', end_date='20240511')
    # df_sh = f.get_north_data(start_date='20240202', end_date='20240511', symbol="沪股通")
    # df_sz = f.get_north_data(start_date='20240202', end_date='20240511', symbol="深股通")

    page = Page(layout=Page.DraggablePageLayout, page_title="")

    page.add(
        self.title("test"),
        # self.plot_bar_line(df_lrb, '报告日', '营业总收入', '营业总收入同比'),
        # self.plot_line(df_fund, '日期', '主力净流入-净额', '资金流量'),
        # self.plot_multi_bar('报告日', '营业总收入', [df_lrb, df_lrb1], ['612', '977'])
        # self.plot_pie(df_zygc, '主营构成', '主营收入', '按产品分类主营构成', '按产品分类')
        # self.title("关键指标"),
        # self.plot_bar_line(df_import, '报告期', '营业总收入', '营业总收入同比增长率'),
        # self.plot_bar_line(df_lrb, '报告日', '净利润', '净利润'),
        # self.plot_bar_line(df_import, '报告期', '扣非净利润', '扣非净利润同比增长率'),
        # self.title("每股指标"),
        # self.plot_line(df_import, '报告期', '基本每股收益', '基本每股收益'),
        # self.plot_line(df_import, '报告期', '每股净资产', '每股净资产'),
        # self.plot_line(df_import, '报告期', '每股资本公积金', '每股资本公积金'),
        # self.plot_line(df_import, '报告期', '每股未分配利润', '每股未分配利润'),
        # self.plot_line(df_import, '报告期', '每股经营现金流', '每股经营现金流'),
        # self.title("盈利能力"),
        # self.plot_line(df_import, '报告期', '净资产收益率', '净资产收益率'),
        # self.plot_line(df_import, '报告期', '净资产收益率-摊薄', '净资产收益率-摊薄'),
        # self.plot_line(df_main, '报告期', '总资产报酬率', '总资产报酬率'),
        # self.plot_line(df_import, '报告期', '销售净利率', '销售净利率'),
        # self.plot_line(df_import, '报告期', '销售毛利率', '销售毛利率'),
        # self.title("财务风险"),
        # self.plot_line(df_import, '报告期', '资产负债率', '资产负债率'),
        # self.plot_line(df_import, '报告期', '流动比率', '流动比率'),
        # self.plot_line(df_import, '报告期', '速动比率', '速动比率'),
        # self.plot_line(df_main, '报告期', '权益乘数', '权益乘数'),
        # self.plot_line(df_import, '报告期', '产权比率', '产权比率'),
        # self.plot_line(df_main, '报告期', '现金比率', '现金比率'),
        # self.plot_line(df_import, '报告期', '产权比率', '产权比率'),
        # self.title("运营能力"),
        # self.plot_line(df_import, '报告期', '存货周转天数', '存货周转天数'),
        # self.plot_line(df_import, '报告期', '应收账款周转天数', '应收账款周转天数'),
        # self.plot_line(df_import, '报告期', '应收账款周转天数', '应收账款周转天数'),
        # self.plot_line(df_import, '报告期', '营业周期', '营业周期'),
        # self.plot_line(df_main, '报告期', '总资产周转率', '总资产周转率'),
        # self.plot_line(df_main, '报告期', '存货周转率', '存货周转率'),
        # self.plot_line(df_main, '报告期', '应收账款周转率', '应收账款周转率'),
        # self.plot_line(df_main, '报告期', '应付账款周转率', '应付账款周转率'),
        # self.plot_line(df_main, '报告期', '流动资产周转率', '流动资产周转率'),
        # self.plot_multi_line('日期', '当日资金流入', [df_north, df_sh, df_sz], ['北向资金', "沪股通", "深股通"])


    )
    page.render('visual.html')
    #         # 用于 DraggablePageLayout 布局重新渲染图表
    #         page.save_resize_html(
    #             # Page 第一次渲染后的 html 文件
    #             source="visual.html",
    #             # 布局配置文件
    #             cfg_file="visual.json",
    #             # 重新生成的 .html 存放路径
    #             dest="visual_new.html"
    #         )

    webbrowser.open_new_tab('file://' + os.path.realpath('visual.html'))


if __name__ == "__main__":
    df = get_data("000612", start_date="20240101", end_date="20240519")

