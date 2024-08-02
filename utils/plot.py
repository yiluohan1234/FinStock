#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: plot.py
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
from utils.MyPP import PKLINE, PBUY_SELL, PVOL, PLINE, PMACD
from utils.data import *

CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_NOTEBOOK
CurrentConfig.ONLINE_HOST = 'https://assets.pyecharts.org/assets/'
import os
import webbrowser
from utils.basic import *
from utils.fundflow import *
from utils.func import *

def K(dateindex, data, title, klines, jxLines, threeLines) -> Kline:
    '''
    绘制k线图
    :param dateindex: x轴
    :param data: 数据
    :param title: Kline的标题
    :param klines: 均线列表
    :param jxLines: 颈线端点的列表，每段颈线一个列表
    :param threeLines: 颈线上三个涨降幅满足位列表
    :return: 返回Kline对象
    '''
    c = PKLINE(data, title)

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
    if len(jxLines) != 0:
        jxs = plot_multi_jx(jxLines)
        c.overlap(jxs)

    # 叠加三个涨跌幅满足位直线
    if len(threeLines) != 0:
        lines = plot_three_stage(threeLines)
        c.overlap(lines)

    # 叠加买卖标记
    if 'BUY' in data.columns and 'SELL' in data.columns:
        es_buy, es_sell = PBUY_SELL(data, 'close')
        c.overlap(es_buy)
        c.overlap(es_sell)

    return c


def V(dateindex, data, vlines) -> Bar:
    '''
    绘制成交量图
    :param dateindex: x轴
    :param data: 数据
    :param vlines: 成交量均线列表
    :return: 返回成交量的Bar对象
    '''
    v = PVOL(data)

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


def plot_multi_jx(jxLines) -> Line:
    '''
    绘制多个颈线，每个颈线由一个list组成
    :param jxLines: 颈线的开始和结束节点的元组列表
    :return: 返回自定义颈线的Line对象
    '''
    x_data = []

    for line in jxLines:
        for point in line:
            x_data.append(transfer_date_format(point[0]))

    if len(jxLines) != 0:
        _line = Line().add_xaxis(x_data)
        for i, line in enumerate(jxLines):
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


def get_three_stage(jx, max_y, min_y, rate=3, is_up=True, stage=3, is_print=False):
    '''
    获取三个涨跌幅满足位
    :param jx: 颈线数据值
    :param max_y: 颈线下最低值或颈线上最高值
    :param min_y: 颈线下最低值或颈线上最高值
    :param rate: 止损幅度%
    :param is_up: 是否向上
    :param stage: 返回满足位阶数
    :param is_print: 是否打印结果
    :return: 返回三个满足位列表
    '''
    if not is_up:
        stop_line = round(jx * (1 + rate*1.0/100), 2)
        h = round(max_y - min_y, 2)
        one_stage = round(jx - h, 2)
        two_stage = round(jx - 2 * h, 2)
        three_stage = round(jx - 3 * h, 2)
        if is_print:
            print("止损位：{}x(1 + {}%) = {} 元。\n".format(jx, rate, stop_line))
            print("顶点到颈线的距离：{} - {} = {} 元。\n".format(max_y, min_y, h))
            print("第一跌幅满足位：{} - {} = {} 元。\n".format(jx, h, one_stage))
            print("第二跌幅满足位：{} - {} = {} 元。\n".format(one_stage, h, two_stage))
            print("第三跌幅满足位：{} - {} = {} 元。\n".format(two_stage, h, three_stage))
            print("第一跌幅满足位收益：1 - {} ÷ {} = {}%。\n".format(one_stage, jx, round((1-one_stage*1.0/jx)*100, 2)))
            print("第二跌幅满足位收益：1 - {} ÷ {} = {}%。\n".format(two_stage, jx, round((1-two_stage*1.0/jx)*100, 2)))
            print("第三跌幅满足位收益：1 - {} ÷ {} = {}%。\n".format(three_stage, jx, round((1-three_stage*1.0/jx)*100, 2)))
    else:
        stop_line = round(jx * (1 - rate*1.0/100), 2)
        h = round(max_y - min_y, 2)
        one_stage = round(jx + h, 2)
        two_stage = round(jx + 2 * h, 2)
        three_stage = round(jx + 3 * h, 2)
        if is_print:
            print("止损位：{}x(1 - {}%)=  {} 元。\n".format(jx, rate, stop_line))
            print("顶点到颈线的距离：{} - {} = {} 元。\n".format(jx, max_y, h))
            print("第一涨幅满足位：{} + {} = {} 元。\n".format(jx, h, one_stage))
            print("第二涨幅满足位：{} + {} = {} 元。\n".format(one_stage, h, two_stage))
            print("第三涨幅满足位：{} + {} = {} 元。\n".format(two_stage, h, three_stage))
            print("第一涨幅满足位收益：{} ÷ {} - 1 = {}%。\n".format(one_stage, jx, round((one_stage*1.0/jx-1)*100, 2)))
            print("第二涨幅满足位收益：{} ÷ {} - 1 = {}%。\n".format(two_stage, jx, round((one_stage*1.0/jx-1)*100, 2)))
            print("第三涨幅满足位收益：{} ÷ {} - 1 = {}%。\n".format(three_stage, jx, round((one_stage*1.0/jx-1)*100, 2)))
    if stage == 3:
        return [stop_line, one_stage, two_stage, three_stage]
    elif stage == 2:
        return [stop_line, one_stage, two_stage]
    elif stage == 1:
        return [stop_line, one_stage]


def plot_three_stage(jxLines) -> Line:
    '''绘制多个颈线，每个颈线由一个list组成
    @params:
    - jxLines : list   #jxLines, [jx, max_y, min_y, is_up, start_date, end_date]
    :param jxLines: jxLines, [jx, max_y, min_y, is_up, start_date, end_date]
    :return: 返回三个满足位的Line对象
    '''
    jx = jxLines[0]
    max_y = jxLines[1]
    min_y = jxLines[2]
    rate = jxLines[3]
    is_up = jxLines[4]
    stage = jxLines[5]
    start_date = transfer_date_format(jxLines[6])
    end_date = transfer_date_format(jxLines[7])
    lines_list = []
    ret_list = get_three_stage(jx, max_y, min_y, rate, is_up, stage)
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


def plot_bar(data, x, y, title) -> Bar:
    '''
    绘制直方图
    :param data: 数据
    :param x: 横坐标轴的列名称
    :param y: 纵坐标轴列名称
    :param title: 图的标题
    :return: 返回单条颈线的Line对象
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
    :param x: 横坐标轴的列名称
    :param y: 纵坐标轴列名称
    :param title: 图的标题
    :return: 返回折线图的Line对象
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
            datazoom_opts=[                                              # 区域缩放
                opts.DataZoomOpts(
                    is_show=False,
                    type_="inside",      # 内部缩放
                    range_start=0,      # 初始显示范围
                    range_end=100,       # 初始显示范围
                ),
                opts.DataZoomOpts(
                    is_show=True,
                    type_="slider",       # 外部滑动缩放
                    range_start=0,       # 初始显示范围
                    range_end=100,        # 初始显示范围
                ),
            ],
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


def plot_bar_line(df, x, y_bar, y_line) -> Bar:
    '''
    绘制业务同比增长的直方折线图
    :param df: 数据
    :param x: 横坐标轴的列名称
    :param y_bar: 纵坐标轴列名称(bar)
    :param y_line: 纵坐标轴列名称(line)
    :return: 返回直方折线图的Bar对象
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
    :param x: 横坐标轴的列名称
    :param y: 纵坐标轴列名称
    :param title: 标题
    :param classify_type: 分类的字段
    :return: 返回Pie对象
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
    :param y: 纵坐标轴列名称
    :param df_list: dataframe列表
    :param names_list: 公司名称列表
    :return: 返回Bar对象
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
    :param y: 纵坐标轴列名称
    :param df_list: dataframe列表
    :param names_list: 公司名称列表
    :return: 返回Line对象
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


def plot_df_line(df, x, y_list):
    '''
    一个df绘制多个折线图对比图
    :param df: dataframe列表
    :param x: 横坐标轴的列名称
    :param y_list: 纵坐标轴列名称
    :return: 返回Line对象
    # line = plot_multi_line('报告日', '营业总收入', [df, df1], ['612', '977'])
    '''
    x_data = df[x].tolist()

    if len(y_list) != 0:
        _line = Line().add_xaxis(x_data)
        for col in y_list:
            _line.add_yaxis(series_name=col,
                            y_axis=df[col].values.tolist(),
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


def plot_bond_zh_us_rate():
    '''
    10年国债收益率
    '''
    bond_zh_us_rate_df = ak.bond_zh_us_rate(start_date="19901219")
    bar = plot_df_line(bond_zh_us_rate_df, '日期', ['中国国债收益率10年', "美国国债收益率10年"])
    bar.render("./bond_zh_us_rate.html")
    webbrowser.open_new_tab('file://' + os.path.realpath('bond_zh_us_rate.html'))


def plot_table(data, headers, title):
    table = Table()

    rows = data[headers].values().tolist()
    table.add(headers, rows).set_global_opts(
        title_opts=opts.ComponentTitleOpts(title=title)
    )
    return table


def plot_page(self):
    df_lrb, df_lrb_display = self.get_lrb_data("000977", 5, 4)
    # df_lrb1, df_lrb_display1 = self.get_lrb_data("000612", 5, 4)
    df_fund, df_fund_display = get_individual_fund_flow("000977", 5)

    #df_zygc = self.get_basic_info("000977")

    df_import, df_import_display = get_main_indicators_ths("000977", 5)
    df_main, df_main_display = get_main_indicators_sina("000977", 5)

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


def plot_main(symbol, start_date='20240501', end_date="20240202", freq='min60', is_notebook=True):
    df = get_kline_chart_date(code=symbol, start_date=start_date, end_date=end_date, freq=freq, zh_index='stock')
    macd = PMACD(df, ['MACD', 'DIF', 'DEA'])
    kpl = PLINE(df, ['kp10', 'kp20', 'kp60'])
    es_buy, es_sell = PBUY_SELL(df, 'kp10')
    kpl.overlap(es_buy)
    kpl.overlap(es_sell)
    kdj = PLINE(df, ['K', 'D', 'J'])

    grid_chart = Grid(init_opts=opts.InitOpts(width="1000px", height="300px",))
    grid_chart.add(
        macd,
        grid_opts=opts.GridOpts(pos_top="0%", height="30%"),
    )
    grid_chart.add(
        kpl, grid_opts=opts.GridOpts(pos_top="30%", height="30%"),
    )
    grid_chart.add(
        kdj, grid_opts=opts.GridOpts(pos_top="60%", height="30%"),
    )
    # 多个子图同时区域缩放功能
    grid_chart.options['dataZoom'][0].opts['xAxisIndex'] = list(range(0, 3))
    if is_notebook:
        return grid_chart.render_notebook()
    else:
        return grid_chart


def plot_main_tx(df, width=1000, height=300, is_kline=False, is_notebook=True):
    kline = PKLINE(df, 'K线')
    macd = PMACD(df, ['MACD', 'DIF', 'DEA'])
    kpl = PLINE(df, ['kp10', 'kp20', 'kp60'])
    es_buy, es_sell = PBUY_SELL(df, 'kp10')
    kpl.overlap(es_buy)
    kpl.overlap(es_sell)
    bias = PLINE(df, ['bias10', 'bias20', 'bias60'])

    grid = Grid(init_opts=opts.InitOpts(width=str(width) + "px", height=str(height) + "px",))
    area = [kline, macd, kpl, bias] if is_kline else [macd, kpl, bias]
    # area = [kline, macd, kpl, bias]
    iWindows = len(area)
    if kline in area:
        iButton = 45
        iStep = 15
    else:
        iButton = 0
        iStep = 30

    for window in area:
        if window == kline:
            grid.add(window, grid_opts=opts.GridOpts(pos_top='8%', pos_bottom='55%')) # 100 - iButton
        else:
            grid.add(window, grid_opts=opts.GridOpts(pos_top=str(iButton) + '%', height=str(iStep) + '%'))
            iButton = iButton + iStep

    # 多个子图同时区域缩放功能
    grid.options['dataZoom'][0].opts['xAxisIndex'] = list(range(0, iWindows))
    if is_notebook:
        return grid.render_notebook()
    else:
        return grid


if __name__ == "__main__":
    print(get_three_stage(3.12, 3.12, 2.62, 5, True, 3, is_print=True))
