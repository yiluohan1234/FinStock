#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: Basic.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年5月3日
#    > description: 基本面分析
#######################################################################
import akshare as ak
from utils.func import get_szsh_code, str2value, get_display_data, get_report_type


def get_basic_info(code, indicator='2023-12-31'):
    '''同花顺-主营介绍
    @params:
    - code: str      # 股票代码
    '''
    cninfo_df = ak.stock_profile_cninfo(symbol=code)
    print("----------------------------- 简介 -----------------------------\n")
    print("公司名称:", cninfo_df.iloc[0][0])
    print("A股简称:", cninfo_df.iloc[0][4])
    print("成立时间:", cninfo_df.iloc[0][14])
    print("上市时间:", cninfo_df.iloc[0][15])
    zyjs_ths_df = ak.stock_zyjs_ths(symbol=code)
    print("主营业务:", zyjs_ths_df.iloc[0][1])  # '主营业务'
    print("产品类型:", zyjs_ths_df.iloc[0][2])  # '产品类型'
    print("产品名称:", zyjs_ths_df.iloc[0][3])  # '产品名称'
    print("经营范围:", zyjs_ths_df.iloc[0][4])  # '经营范围'

    # 主营构成-东财
    print("\n----------------------------- 主营构成 -----------------------------\n")
    ret_columns = ['报告日期', '分类类型', '主营构成', '主营收入', '收入比例', '主营成本', '成本比例', '主营利润', '利润比例', '毛利率']
    zygc_em_df = ak.stock_zygc_em(symbol=get_szsh_code(code))
    zygc_em_df['分类类型'] = zygc_em_df['分类类型'].astype(str).apply(lambda x: x.replace('nan', '其他'))
    zygc_em_df['主营收入'] = round(zygc_em_df['主营收入'] / 100000000, 2)
    zygc_em_df['主营成本'] = round(zygc_em_df['主营成本'] / 100000000, 2)
    zygc_em_df['主营利润'] = round(zygc_em_df['主营利润'] / 100000000, 2)
    zygc_em_df['收入比例'] = round(zygc_em_df['收入比例'] * 100, 2)
    zygc_em_df['成本比例'] = round(zygc_em_df['成本比例'] * 100, 2)
    zygc_em_df['利润比例'] = round(zygc_em_df['利润比例'] * 100, 2)
    zygc_em_df['毛利率'] = round(zygc_em_df['毛利率'] * 100, 2)
    zygc_em_df['毛利率'] = round(zygc_em_df['毛利率'] * 100, 2)
    df = zygc_em_df[zygc_em_df['报告日期'].astype(str) == indicator]

    df = df.sort_values(by=['分类类型', '收入比例'], ascending=[False, False])
    # df[df.columns.tolist()[2:]]
    # print(df[df.columns.tolist()[2:]].to_string(index=False))
    return df[ret_columns]


def get_main_indicators_ths(code, n, indicator="按年度"):
    '''获取股票代码带字母的
    @params:
    - code: str      # 股票代码
    - n: str         # 返回数据个数
    - indicator: str # 数据类型，indicator="按报告期"; choice of {"按报告期", "按年度", "按单季度"}
    '''
    # indicator="按报告期"; choice of {"按报告期", "按年度", "按单季度"}
    df = ak.stock_financial_abstract_ths(symbol=code, indicator=indicator)
    df = df.head(n)
    for col in df.columns.tolist()[1:]:
        df[col] = df[col].apply(str2value)
    for col in ['营业总收入', '归母净利润', '扣非净利润']:
        df[col] = round(df[col]/100000000, 2)
    # https://blog.csdn.net/a6661314/article/details/133634976
    df['报告期'] = df['报告期'].astype("str")

    df_display = get_display_data(df)
    return df, df_display


def get_main_indicators_sina(code, n, indicator="按年度", ret_columns=[]):
    '''获取股票代码带字母的
    @params:
    - code: str      # 股票代码
    - n: str         # 返回数据个数
    - indicator: str # 数据类型，indicator="按报告期"; choice of {"按报告期", "按年度", "按单季度"}
    '''
    df = ak.stock_financial_abstract(code)
    df.drop(['选项'] ,axis=1, inplace=True)
    dT = get_display_data(df)
    dT.index.name = '报告期'
    df = dT.reset_index()
    df['报告类型'] = df['报告期'].apply(get_report_type)

    for col in ['归母净利润', '营业总收入', '营业成本', '净利润', '扣非净利润', '股东权益合计(净资产)', '经营现金流量净额']:
        df[col] = df[col]/100000000

    for col in df.columns.tolist()[1:]:
        df[col] = round(df[col], 2)

    if indicator == "按年度":
        df = df[df['报告类型'] == '年报']
        df['报告期'] = df['报告期'].apply(lambda x: str(x)[0:4])

    df = df.head(n)
    if len(ret_columns) != 0:
        df = df[ret_columns]

    df_display = get_display_data(df)
    df_display.drop_duplicates(subset=df_display.columns.tolist(), keep='first', inplace=True)
    return df, df_display


def get_lrb_data(code, n, data_type=0):
    '''获取利润表数据
    @params:
    - code: str        #代码
    - n: int           #返回数据个数
    - data_type: int   #返回数据的形式，0为自然，1为季报，2为中报，4为年报
    '''
    ret_columns = ['报告日', '营业总收入', '营业总收入同比',
                   '营业总成本', '营业总成本同比',
                   '营业利润', '营业利润同比',
                   '利润总额', '利润总额同比',
                   '净利润', '净利润同比',
                   '归属于母公司所有者的净利润', '归属于母公司所有者的净利润同比',
                   '销售毛利率', '销售净利率', '销售成本率',
                   '净利润/营业总收入(%)', '营业利润/营业总收入(%)', '息税前利润/营业总收入(%)',
                   '营业总成本/营业总收入(%)', '销售费用/营业总收入(%)',
                   '管理费用/营业总收入(%)', '管理费用(含研发费用)/营业总收入(%)',
                   '财务费用/营业总收入(%)', '研发费用/营业总收入(%)']
    # df_lrb = ak.stock_financial_report_sina(stock=self.code, symbol='利润表')
    df_lrb = ak.stock_profit_sheet_by_report_em(symbol=get_szsh_code(code))
    df_lrb = df_lrb.fillna(0)

    # 过滤年报
    df_lrb['报告日'] = df_lrb['REPORT_DATE_NAME']
    # df_lrb = df_lrb.sort_index(ascending=False)
    # 营业收入及同比
    df_lrb['营业总收入'] = round(df_lrb['TOTAL_OPERATE_INCOME'] / 100000000, 2)
    df_lrb['营业总收入同比'] = round(df_lrb['TOTAL_OPERATE_INCOME_YOY'], 2)
    df_lrb['营业总成本'] = round(df_lrb['TOTAL_OPERATE_COST'] / 100000000, 2)
    df_lrb['营业总成本同比'] = round(df_lrb['TOTAL_OPERATE_COST_YOY'], 2)
    df_lrb['营业利润'] = round(df_lrb['OPERATE_PROFIT'] / 100000000, 2)
    df_lrb['营业利润同比'] = round(df_lrb['OPERATE_PROFIT_YOY'], 2)
    df_lrb['利润总额'] = round(df_lrb['TOTAL_PROFIT'] / 100000000, 2)
    df_lrb['利润总额同比'] = round(df_lrb['TOTAL_PROFIT_YOY'], 2)
    df_lrb['净利润'] = round(df_lrb['NETPROFIT'] / 100000000, 2)
    df_lrb['净利润同比'] = round(df_lrb['NETPROFIT_YOY'], 2)
    df_lrb['归属于母公司所有者的净利润'] = round(df_lrb['PARENT_NETPROFIT'] / 100000000, 2)
    df_lrb['归属于母公司所有者的净利润同比'] = round(df_lrb['PARENT_NETPROFIT_YOY'], 2)
    # 净利润/营业总收入(%)
    # 营业利润/营业总收入(%)
    # 息税前利润/营业总收入(%)
    # EBITDA/营业总收入(%)

    df_lrb['净利润/营业总收入(%)'] = round(df_lrb['NETPROFIT'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    df_lrb['营业利润/营业总收入(%)'] = round(df_lrb['OPERATE_PROFIT'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    # 息税前利润 （EBTI）= 净利润 + 财务费用 + 所得税费用
    df_lrb['息税前利润/营业总收入(%)'] = round(
        (df_lrb['NETPROFIT'] + df_lrb['FINANCE_EXPENSE'] + df_lrb['INCOME_TAX']) * 100 / df_lrb[
            'TOTAL_OPERATE_INCOME'], 2)
    df_lrb['营业总成本/营业总收入(%)'] = round(df_lrb['TOTAL_OPERATE_COST'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    df_lrb['销售费用/营业总收入(%)'] = round(df_lrb['SALE_EXPENSE'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    df_lrb['管理费用/营业总收入(%)'] = round(df_lrb['MANAGE_EXPENSE'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    df_lrb['管理费用(含研发费用)/营业总收入(%)'] = round(
        (df_lrb['MANAGE_EXPENSE'] + df_lrb['RESEARCH_EXPENSE']) * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    df_lrb['财务费用/营业总收入(%)'] = round(df_lrb['FINANCE_EXPENSE'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    df_lrb['研发费用/营业总收入(%)'] = round(df_lrb['RESEARCH_EXPENSE'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    # 销售毛利率=销售毛利/销售收入×100%=（销售收入-销售成本）/销售收入×100%= (营业收入 - 营业成本 / 营业收入) * 100%
    df_lrb['销售毛利率'] = round((df_lrb['OPERATE_INCOME'] - df_lrb['OPERATE_COST']) * 100 / df_lrb['OPERATE_INCOME'], 2)
    # 净利率=净利润/营业收入
    df_lrb['销售净利率'] = round(df_lrb['NETPROFIT'] * 100 / df_lrb['OPERATE_INCOME'], 2)
    # 销售成本率=销售成本/销售收入净额×100%
    df_lrb['销售成本率'] = round(df_lrb['OPERATE_COST'] * 100 / df_lrb['OPERATE_INCOME'], 2)

    if data_type == 1:
        df_lrb = df_lrb[df_lrb['REPORT_TYPE'] == '一季报']
    elif data_type == 2:
        df_lrb = df_lrb[df_lrb['REPORT_TYPE'] == '中报']
    elif data_type == 4:
        df_lrb = df_lrb[df_lrb['REPORT_TYPE'] == '年报']

    # 返回最近n个数据
    df_lrb = df_lrb[ret_columns].head(n)
    df_lrb_display = get_display_data(df_lrb)

    return df_lrb, df_lrb_display

def get_zcfz_data(code, n, data_type=0):
    '''获取资产负债表数据
    @params:
    - n: int           #返回数据个数
    - data_type: int   #返回数据的形式，0为自然，1为季报，2为中报，4为年报
    '''
    ret_columns = ['报告日', '总资产', '总负债',
                   '资产负债率']
    df_zcfz = ak.stock_balance_sheet_by_report_em(symbol=get_szsh_code(code))
    df_zcfz = df_zcfz.fillna(0)
    # https://blog.csdn.net/a389085918/article/details/80284812

    df_zcfz['报告日'] = df_zcfz['REPORT_DATE_NAME']
    # df_zcfz = df_zcfz.sort_index(ascending=False)
    # 资产负债率
    df_zcfz['总资产'] = round(df_zcfz['TOTAL_ASSETS'] / 100000000, 2)
    df_zcfz['总负债'] = round(df_zcfz['TOTAL_LIABILITIES'] / 100000000, 2)
    df_zcfz['资产负债率'] = round(df_zcfz['TOTAL_LIABILITIES'] * 100 / df_zcfz['TOTAL_ASSETS'], 2)
    # 应收账款周转率=营业收入/（（期初应收账款+期末应收账款）/2）
    # 应收账款周转天数=365/应收账款周转率

    if data_type == 1:
        df_zcfz = df_zcfz[df_zcfz['REPORT_TYPE'] == '一季报']
    elif data_type == 2:
        df_zcfz = df_zcfz[df_zcfz['REPORT_TYPE'] == '中报']
    elif data_type == 4:
        df_zcfz = df_zcfz[df_zcfz['REPORT_TYPE'] == '年报']
    # 返回最近n个数据
    df_zcfz = df_zcfz[ret_columns].head(n)

    df_zcfz_display = get_display_data(df_zcfz)
    return df_zcfz, df_zcfz_display


def get_xjll_data(code, n, data_type=0):
    '''获取现金流量表数据
    @params:
    - n: int           #返回数据个数
    - data_type: int   #返回数据的形式，0为自然，1为季报，2为中报，4为年报
    '''
    ret_columns = ['报告日', '总资产', '总负债',
                   '资产负债率']
    # 人力投入回报率=企业净利润/员工薪酬福利总额×100%，这是衡量人力资本有效性的核心指标，表明公司在人力资源上的投入和净利润的比值，回报率越高，说明人力资源的效率和效能越高。
    df_xjll = ak.stock_cash_flow_sheet_by_report_em(symbol=get_szsh_code(code))
    df_xjll['员工薪酬福利总额'] = round(df_xjll['PAY_STAFF_CASH']/100000000, 2)
    df_xjll = df_xjll.fillna(0)

    if data_type == 1:
        df_xjll = df_xjll[df_xjll['REPORT_TYPE'] == '一季报']
    elif data_type == 2:
        df_xjll = df_xjll[df_xjll['REPORT_TYPE'] == '中报']
    elif data_type == 4:
        df_xjll = df_xjll[df_xjll['REPORT_TYPE'] == '年报']
    # 返回最近n个数据
    df_xjll = df_xjll[ret_columns].head(n)

    df_xjll_display = get_display_data(df_xjll)
    return df_xjll, df_xjll_display

def get_zygc_data(code):
    '''获取主营构成数据
    @params:
    '''
    ret_columns = ['报告日期', '分类类型', '主营构成', '主营收入', '收入比例', '主营成本', '成本比例', '主营利润', '利润比例', '毛利率']
    # 主营构成-东财
    zygc_em_df = ak.stock_zygc_em(symbol=get_szsh_code(code))
    zygc_em_df['主营收入'] = round(zygc_em_df['主营收入'] / 100000000, 2)
    zygc_em_df['主营成本'] = round(zygc_em_df['主营成本'] / 100000000, 2)
    zygc_em_df['主营利润'] = round(zygc_em_df['主营利润'] / 100000000, 2)
    zygc_em_df['收入比例'] = round(zygc_em_df['收入比例'] * 100, 2)
    zygc_em_df['成本比例'] = round(zygc_em_df['成本比例'] * 100, 2)
    zygc_em_df['利润比例'] = round(zygc_em_df['利润比例'] * 100, 2)
    zygc_em_df['毛利率'] = round(zygc_em_df['毛利率'] * 100, 2)
    df = zygc_em_df[zygc_em_df['报告日期'].astype(str) == '2023-12-31']
    df = df.sort_values(by=['分类类型', '收入比例'], ascending=[True, False])

    return df[ret_columns]

if __name__ == "__main__":
    df = get_basic_info("000737")
    print(df)
