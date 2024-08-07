#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: AIndex.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年4月27日
#    > description: A股K线行情
#######################################################################
from pyecharts.charts import Line, Grid, EffectScatter
from pyecharts import options as opts
import webbrowser
import os
import datetime
from utils.data import get_index_data
from utils.plot import K, V, MACD, DKC, BIAS, KL, DMA
from utils.cons import precision


class AIndex:

    def __init__(self, code='sh000001', start_date='20200101', end_date='20240202', freq='D', precision=1):
        '''
        @params:
        - code: str                      #股票代码
        - start_date: str                #开始时间
        - end_date: str                  #结束时间
        - freq : str                     #默认 D 日线数据
        - precision :str                 #数据精度,默认2
        '''
        self.title = self.code2name(code)
        # 如果默认日期为'20240202'，则end_date转为最新的日期
        if end_date == '20240202':
            now = datetime.datetime.now()
            if now.hour >= 15:
                end_date = now.strftime('%Y%m%d')
            else:
                yesterday = now - datetime.timedelta(days=1)
                end_date = yesterday.strftime('%Y%m%d')
        df = get_index_data(code, start_date, end_date, freq)
        self.data = df.copy()
        if freq == 'min':
            self.dateindex = df.index.strftime('%Y-%m-%d %H').tolist()
        else:
            self.dateindex = df.index.strftime("%Y-%m-%d").tolist()

    def code2name(self, code):
        '''获取股票代码名称
        @params:
        - code: str                      #股票代码
        '''
        code_name = {"sh000001": "上证指数",
                     "sh880326": "铝"}

        return code_name[code]

    def plot(self, n=20, area=['V', 'DKC'], width=1000, height=600, klines=[], vlines=[], dmalines=[], jxPoints=[],
             jxLines=[], KLlines=[], is_notebook=True):
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
          vline: list           #Volume区域显示的数据，Dataframe中的字段名，如['vma5','vma10','vma20','vma60', 'vma120', 'vma250']
          dmalines: list        #线误差的两个均线选择，如['ma5', 'ma10']
          jxPoints: list        #绘制多个颈线的坐标，如jxPoints=[[("2024-03-01",38.80), ("2024-04-09",38.80)], [("2024-01-11",18.80), ("2024-01-31",28.80)]])
          jxLines: list        #绘制多个颈线的坐标，如[jx, max_y, start_date, end_date]
          KLlines: list        #绘制多个K线，如['k60', 'k120']
          is_notebook: bool    #是否在notebook绘制
        - sample:
           chart=data.plot(area=[['V','DKC'],'V'],vlines=['vMA5','vMA10'],klines=['ma5','ma10'])
        '''
        grid = (Grid(init_opts=opts.InitOpts(
            width=str(width) + "px",
            height=str(height) + "px",
            animation_opts=opts.AnimationOpts(animation=False),
        ))
        )
        c = K(self.data, self.title, klines, jxPoints, jxLines)
        iTop = 10
        iButton = 10
        iWindows = len(area)
        iStep = 0
        if iWindows == 0:
            grid.add(c, grid_opts=opts.GridOpts(pos_top="8%", pos_bottom="10%"))
        elif iWindows > 1:
            grid.add(c, grid_opts=opts.GridOpts(pos_top="8%", pos_bottom="50%"))
            iStep = int(40 / iWindows)
            iButton = 50
        else:
            grid.add(c, grid_opts=opts.GridOpts(pos_top="8%", pos_bottom="30%"))
            iStep = 20
            iButton = 70
        icount = 0

        for w in area:
            # print(iStep)
            if type(w) == list:
                window = Line().add_xaxis(self.dateindex)
                for l in w:
                    window.add_yaxis(
                        series_name=l,
                        y_axis=round(self.data[l], precision).values.tolist(),
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
                window = V(self.data, vlines)
            elif w == 'M':
                window = MACD(self.data)
            elif w == 'DKC':
                window = DKC(self.data, n)
            elif w == 'BIAS':
                window = BIAS(self.data, n)
            elif w == 'KL':
                window = KL(self.data, n, KLlines)
            elif w == 'DMA':
                window = DMA(self.data, n, dmalines)
            else:
                window = Line().add_xaxis(self.dateindex)
                if isinstance(w, list):
                    ws = w
                else:
                    ws = [w]
                for wi in ws:
                    window.add_yaxis(
                        series_name=wi,
                        y_axis=round(self.data[w], precision).values.tolist(),
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
            grid.render("./aline.html")
            # return grid.render_notebook()
            return grid

    def web(self):
        '''通过浏览器打开pyecharts的html文件
        '''
        webbrowser.open_new_tab('file://' + os.path.realpath('./aline.html'))


if __name__ == "__main__":
    a = AIndex()
    a.plot(area=['V', 'KL'],
           klines=['ma5', 'ma20', 'ma60', 'ma120', 'ma250'],
           is_notebook=False
           )
    a.web()
