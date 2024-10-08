#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: KLineChart.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年4月26日
#    > description: 绘制股票K线
#######################################################################
from utils.func import *
from utils.data import get_kline_chart_date
from utils.plot import *
from utils.cons import precision


class KLineChart:

    def __init__(self, code, start_date='20200101', end_date='20240202', freq='D', zh_index='stock'):
        '''
        @params:
        - code: str                      #股票代码
        - start_date: str                #开始时间, 如'202000101'
        - end_date: str                  #结束时间, 如'20240202'
        - freq : str                     #默认 'D' :日线数据
        - zh_index :str                  #类型，stock：股票，index：指数，industry：行业，concept：概念
        '''
        self.title = get_name(code, zh_index)
        df = get_kline_chart_date(code, start_date, end_date, freq, zh_index)
        self.data = df.copy()

        if 'm' in freq:
            self.dateindex = df.index.strftime('%Y-%m-%d %H:%M').tolist()
        else:
            self.dateindex = df.index.strftime("%Y-%m-%d").tolist()

    def plot(self, n=20, area=['V', 'M', 'KPL'], width=1000, height=600, klines=[], vlines=['vma10', 'vma60'],
             dmalines=['ma10', 'ma60'], KLlines=['k10', 'k20', 'k60'], KPLlines=['kp10', 'kp20', 'kp60'],
             multiLines=['bias10', 'bias20', 'bias60'], jxLines=[], threeLines=[], is_notebook=True):
        '''
        @params:
        - n:int                 #抵扣差、乖离率、斜率的计算天数
        - area : list           #显示区域
                                'V'      交易量
                                'M'      k线+MACD
                                'DKC'    抵扣差
                                'BIAS'   乖离率
                                'KL/KPL' K率/预测K率
                                'DMA'    均线差
                                FieldName: string   Dataframe中的字段名
                                [Field1,Field2,...] Dataframe中的字段名列表，将显示在一个区域
          width: int            #图表宽度 px
          height: int           #图表高度 px
          klines: list          #K线区域显示的数据，Dataframe中的字段名，如['ma5','ma10','ma20','ma60', 'ma120', 'ma250', 'boll', 'up', 'down', 'stop', 'ene', 'upper', 'lower']
          vline: list           #Volume区域显示的数据，Dataframe中的字段名，如['vma5','vma10','vma20','vma60', 'vma120', 'vma250']
          dmalines: list        #线误差的两个均线选择，如['ma5', 'ma10']
          jxLines: list         #绘制多个颈线的坐标，如jxPoints=[[("2024-03-01",38.80), ("2024-04-09",38.80)], [("2024-01-11",18.80), ("2024-01-31",28.80)]])
          threeLines: list      #绘制三个涨跌幅满足位颈线的坐标，如[jx, max_y, rate, is_up, stage, start_date, end_date]
          KLlines: list         #绘制多个K线，如['k60', 'k120']
          KPLlines: list        #绘制多个预测K线，如['kp60', 'kp120']
          multiLines: list      #绘制多个直线，如['kp60', 'kp120']
          is_notebook: bool     #是否在notebook绘制
        - sample:
           chart=data.plot(n=20, area=['V', 'M', 'KPL'], vlines=['vma5','vma10'], klines=['ma5','ma10'])
        '''
        grid = (Grid(init_opts=opts.InitOpts(
            width=str(width) + "px",
            height=str(height) + "px",
            animation_opts=opts.AnimationOpts(animation=False),))
        )
        c = K(self.dateindex, self.data, self.title, klines, jxLines, threeLines)
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
            if w == 'V':
                window = V(self.dateindex, self.data, vlines)
            elif w == 'M':
                window = PMACD(self.data, x_splitline=True)
            elif w == 'DKC':
                window = PLINE(self.data, lines=["dkc{}".format(n)], x_splitline=True)
            elif w == 'BIAS':
                window = PLINE(self.data, lines=["bias{}".format(n)], x_splitline=True)
            elif w == 'KL':
                window = PLINE(self.data, lines=KLlines, x_splitline=True)
            elif w == 'KPL':
                window = PLINE(self.data, lines=KPLlines, x_splitline=True)
            elif w == 'DMA':
                self.data["DMA-({}-{})".format(dmalines[0], dmalines[1])] = round(self.data[dmalines[0]] - self.data[dmalines[1]], precision)
                window = PLINE(self.data, lines=["DMA-({}-{})".format(dmalines[0], dmalines[1])], x_splitline=True)
            elif w == 'KDJ':
                window = PLINE(self.data, lines=['K', 'D', 'J'], x_splitline=True)
            elif w == 'MUL':
                window = PLINE(self.data, lines=multiLines, x_splitline=True)
            icount += 1
            # 横坐标最后一行加上x刻度
            if icount == iWindows:
                window.options['xAxis'][0]['axisLabel'].opts['show'] = True
            grid.add(window, grid_opts=opts.GridOpts(pos_top=str(iButton) + '%', height=str(iStep) + '%'))
            iButton = iButton + iStep
        grid.options['dataZoom'][0].opts['xAxisIndex'] = list(range(0, iWindows + 1))
        if is_notebook:
            return grid.render_notebook()
        else:
            grid.render("./kline.html")
            return grid

    def web(self):
        '''通过浏览器打开pyecharts的html文件
        '''
        webbrowser.open_new_tab('file://' + os.path.realpath('./kline.html'))


if __name__ == "__main__":
    k = KLineChart("000612", freq='W')
    k.plot(area=['V', 'KL'], width=1200, height=600,
           klines=['ma5', 'ma20', 'ene', 'upper', 'lower'],
           # jxPoints=[[("2024-02-21",18.2), ("2024-04-12",18.2)]],
           # jxLines=[18.2, 16.5, "2024-04-12", "2024-04-30"],
           is_notebook=False
           )
    k.web()
