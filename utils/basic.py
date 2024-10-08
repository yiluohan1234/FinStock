#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: basic.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2024年5月3日
#    > description: 基本面分析
#######################################################################
import akshare as ak
import pandas as pd
from utils.func import get_szsh_code, str2value, get_display_data, get_report_type
from utils.cons import precision, lrb_ret_columns, xjll_ret_columns, zcfz_ret_columns


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
    # name_code = ak.stock_zh_a_spot_em()
    # name = name_code[name_code['代码'] == code]['名称'].values[0]
    # print("----------------------------- 简介 -----------------------------\n")
    # print("公司名称:", name)
    # stock_ipo_summary_cninfo_df = ak.stock_ipo_summary_cninfo(symbol=code)
    # print("上市时间:", stock_ipo_summary_cninfo_df.iloc[0][9])
    zyjs_ths_df = ak.stock_zyjs_ths(symbol=code)
    print("主营业务:", zyjs_ths_df.iloc[0][1])  # '主营业务'
    print("产品类型:", zyjs_ths_df.iloc[0][2])  # '产品类型'
    print("产品名称:", zyjs_ths_df.iloc[0][3])  # '产品名称'
    print("经营范围:", zyjs_ths_df.iloc[0][4])  # '经营范围'
    print("----------------------------- 股东人数 -----------------------------\n")
    print(get_gd_info(code, 6))
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


def get_free_top_10_em(code, date='20240331', is_display=True):
    '''
    获取10大流通股
    :param code: 股票代码
    :type code: str
    :param date: 日期
    :type date: str
    :param is_display: 是否返回展示数据
    :type is_display: bool
    :return: 返回股东人数
    :rtype: pandas.DataFrame
    :return: 返回10大流通股
    :rtype: pandas.DataFrame
    '''
    df = ak.stock_gdfx_free_top_10_em(symbol=get_szsh_code(code), date=date)
    ret_df_display = get_display_data(df)
    if is_display:
        return ret_df_display
    else:
        return df


def get_gd_info(code, n, is_display=True):
    '''
    获取股东人数
    :param code: 股票代码
    :type code: str
    :param n: 返回数据个数
    :type n: str
    :param is_display: 是否返回展示数据
    :type is_display: bool
    :return: 返回股东人数
    :rtype: pandas.DataFrame
    '''
    df_gd = ak.stock_zh_a_gdhs_detail_em(symbol=code)

    dt = {}
    dt['报告日'] = df_gd['股东户数统计截止日']
    dt['股东户数'] = round(df_gd['股东户数-本次'] / 10000, precision)
    dt['增减比例'] = round(df_gd['股东户数-增减比例'], precision)

    ret_df = pd.DataFrame(dt)
    ret_df = pd.DataFrame(dt)
    ret_df = ret_df.fillna(0)
    ret_df = ret_df.head(n)

    ret_df_display = get_display_data(ret_df)

    if is_display:
        return ret_df_display
    else:
        return ret_df


def get_main_indicators_ths(code, n, indicator="按年度", is_display=True):
    '''
    获取主要指标数据
    :param code: 股票代码
    :type code: str
    :param n: 返回数据个数
    :type n: str
    :param indicator: 数据类型，indicator="按报告期"; choice of {"按报告期", "按年度", "按单季度"}
    :type indicator: str
    :param is_display: 是否返回展示数据
    :type is_display: bool
    :return: 返回主要指标数据
    :rtype: pandas.DataFrame
    '''
    # indicator="按报告期"; choice of {"按报告期", "按年度", "按单季度"}
    df = ak.stock_financial_abstract_ths(symbol=code, indicator=indicator)
    df = df.head(n)
    for col in df.columns.tolist()[1:]:
        df[col] = df[col].apply(str2value)
    for col in ['营业总收入', '净利润', '扣非净利润']:
        df[col] = round(df[col]/100000000, 2)
    # https://blog.csdn.net/a6661314/article/details/133634976
    df['报告期'] = df['报告期'].astype("str")

    df_display = get_display_data(df)

    if is_display:
        return df_display
    else:
        return df


def get_main_indicators_sina(code, n, indicator="按年度", ret_columns=[], is_display=True):
    '''
    获取主要指标数据
    :param code: 股票代码
    :type code: str
    :param n: 返回数据个数
    :type n: str
    :param indicator: 数据类型，indicator="按报告期"; choice of {"按报告期", "按年度", "按单季度"}
    :type indicator: int
    :param ret_columns: 返回数据列名列表
    :type ret_columns: list
    :param is_display: 是否返回展示数据
    :type is_display: bool
    :return: 返回主要指标数据
    :rtype: pandas.DataFrame
    '''
    df = ak.stock_financial_abstract(code)
    df.drop(['选项'], axis=1, inplace=True)
    dT = get_display_data(df)
    dT.index.name = '报告期'
    df = dT.reset_index()
    df['报告类型'] = df['报告期'].apply(get_report_type)

    for col in ['归母净利润', '营业总收入', '营业成本', '净利润', '扣非净利润', '股东权益合计(净资产)', '经营现金流量净额']:
        df[col] = df[col]/100000000

    for col in df.columns.tolist()[1:-1]:
        df[col] = round(df[col], 2)

    if indicator == "按年度":
        df = df[df['报告类型'] == '年报']
        df['报告期'] = df['报告期'].apply(lambda x: str(x)[0:4])

    df = df.head(n)
    if len(ret_columns) != 0:
        df = df[ret_columns]

    df_display = get_display_data(df)
    df_display.drop_duplicates(subset=df_display.columns.tolist(), keep='first', inplace=True)
    if is_display:
        return df_display
    else:
        return df


def get_lrb_data(code, n, data_type=0, is_display=True, ret_columns=lrb_ret_columns):
    '''
    获取利润表数据
    :param code: 股票代码
    :type code: str
    :param n: 返回数据个数
    :type n: int
    :param data_type: 返回数据的形式，0为自然，1为季报，2为中报，4为年报
    :type data_type: int
    :param is_display: 是否返回展示数据
    :type is_display: bool
    :param ret_columns: 返回数据列名列表,必须包含"报告日"字段
    :type ret_columns: list
    :return: 返回利润表数据
    :rtype: pandas.DataFrame
    '''
    # df_lrb = ak.stock_financial_report_sina(stock=self.code, symbol='利润表')
    df_lrb = ak.stock_profit_sheet_by_report_em(symbol=get_szsh_code(code))
    dt = {}
    # 过滤年报
    dt['报告日'] = df_lrb['REPORT_DATE_NAME']
    dt['REPORT_TYPE'] = df_lrb['REPORT_TYPE']
    # df_lrb = df_lrb.sort_index(ascending=False)
    # 营业收入
    dt['营业总收入'] = round(df_lrb['TOTAL_OPERATE_INCOME'] / 100000000, 2)
    dt['营业总收入同比'] = round(df_lrb['TOTAL_OPERATE_INCOME_YOY'], 2)
    dt['营业收入'] = round(df_lrb['OPERATE_INCOME'] / 100000000, 2)
    dt['营业收入同比'] = round(df_lrb['OPERATE_INCOME_YOY'], 2)
    dt['利息收入'] = round(df_lrb['INTEREST_INCOME'] / 100000000, 2)
    dt['利息收入同比'] = round(df_lrb['INTEREST_INCOME_YOY'], 2)
    # 营业成本
    dt['营业总成本'] = round(df_lrb['TOTAL_OPERATE_COST'] / 100000000, 2)
    dt['营业总成本同比'] = round(df_lrb['TOTAL_OPERATE_COST_YOY'], 2)
    dt['利息支出'] = round(df_lrb['INTEREST_EXPENSE'] / 100000000, 2)
    dt['手续费及佣金支出'] = round(df_lrb['FEE_COMMISSION_EXPENSE'] / 100000000, 2)
    dt['税金及附加'] = round(df_lrb['OPERATE_TAX_ADD'] / 100000000, 2)
    dt['销售费用'] = round(df_lrb['SALE_EXPENSE'] / 100000000, 2)
    dt['管理费用'] = round(df_lrb['MANAGE_EXPENSE'] / 100000000, 2)
    dt['研发费用'] = round(df_lrb['RESEARCH_EXPENSE'] / 100000000, 2)
    dt['财务费用'] = round(df_lrb['FINANCE_EXPENSE'] / 100000000, 2)
    dt['利息费用'] = round(df_lrb['FE_INTEREST_EXPENSE'] / 100000000, 2)
    dt['利息收入'] = round(df_lrb['FE_INTEREST_INCOME'] / 100000000, 2)
    # 其他经营收益
    dt['公允价值变动收益'] = round(df_lrb['FAIRVALUE_CHANGE_INCOME'] / 100000000, 2)
    dt['投资收益'] = round(df_lrb['INVEST_INCOME'] / 100000000, 2)
    dt['资产处置收益'] = round(df_lrb['ASSET_DISPOSAL_INCOME'] / 100000000, 2)
    dt['信用减值损失'] = round(df_lrb['CREDIT_IMPAIRMENT_INCOME'] / 100000000, 2)
    dt['其他收益'] = round(df_lrb['OTHER_INCOME'] / 100000000, 2)
    # 营业利润
    dt['营业利润'] = round(df_lrb['OPERATE_PROFIT'] / 100000000, 2)
    dt['营业利润同比'] = round(df_lrb['OPERATE_PROFIT_YOY'], 2)
    dt['营业外收入'] = round(df_lrb['NONBUSINESS_INCOME'] / 100000000, 2)
    dt['营业外支出'] = round(df_lrb['NONBUSINESS_EXPENSE'] / 100000000, 2)
    dt['利润总额'] = round(df_lrb['TOTAL_PROFIT'] / 100000000, 2)
    dt['利润总额同比'] = round(df_lrb['TOTAL_PROFIT_YOY'], 2)
    dt['所得税'] = round(df_lrb['INCOME_TAX'] / 100000000, 2)
    dt['净利润'] = round(df_lrb['NETPROFIT'] / 100000000, 2)
    dt['净利润同比'] = round(df_lrb['NETPROFIT_YOY'], 2)
    dt['持续经营净利润'] = round(df_lrb['CONTINUED_NETPROFIT'] / 100000000, 2)
    dt['归属于母公司股东的净利润'] = round(df_lrb['PARENT_NETPROFIT'] / 100000000, 2)
    dt['归属于母公司股东的净利润同比'] = round(df_lrb['PARENT_NETPROFIT_YOY'], 2)
    dt['少数股东损益'] = round(df_lrb['MINORITY_INTEREST'] / 100000000, 2)
    dt['扣除非经常性损益后的净利润'] = round(df_lrb['DEDUCT_PARENT_NETPROFIT'] / 100000000, 2)
    # 每股收益
    dt['基本每股收益'] = round(df_lrb['BASIC_EPS'], 2)
    dt['稀释每股收益'] = round(df_lrb['DILUTED_EPS'], 2)
    # 其他综合收益
    dt['其他综合收益'] = round(df_lrb['OTHER_COMPRE_INCOME'] / 100000000, 2)
    dt['归属于母公司股东的其他综合收益'] = round(df_lrb['PARENT_OCI'] / 100000000, 2)
    # 综合收益总额
    dt['综合收益总额'] = round(df_lrb['TOTAL_COMPRE_INCOME'] / 100000000, 2)
    dt['归属于母公司股东的综合收益总额'] = round(df_lrb['DEDUCT_PARENT_NETPROFIT'] / 100000000, 2)
    dt['归属于少数股东的综合收益总额'] = round(df_lrb['MINORITY_INTEREST'] / 100000000, 2)


    dt['净利润/营业总收入(%)'] = round(df_lrb['NETPROFIT'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    dt['营业利润/营业总收入(%)'] = round(df_lrb['OPERATE_PROFIT'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    # 息税前利润 （EBTI）= 净利润 + 财务费用 + 所得税费用
    dt['息税前利润/营业总收入(%)'] = round(
        (df_lrb['NETPROFIT'] + df_lrb['FINANCE_EXPENSE'] + df_lrb['INCOME_TAX']) * 100 / df_lrb[
            'TOTAL_OPERATE_INCOME'], 2)
    dt['营业总成本/营业总收入(%)'] = round(df_lrb['TOTAL_OPERATE_COST'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    dt['销售费用/营业总收入(%)'] = round(df_lrb['SALE_EXPENSE'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    dt['管理费用/营业总收入(%)'] = round(df_lrb['MANAGE_EXPENSE'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    dt['管理费用(含研发费用)/营业总收入(%)'] = round(
        (df_lrb['MANAGE_EXPENSE'] + df_lrb['RESEARCH_EXPENSE']) * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    dt['财务费用/营业总收入(%)'] = round(df_lrb['FINANCE_EXPENSE'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    dt['研发费用/营业总收入(%)'] = round(df_lrb['RESEARCH_EXPENSE'] * 100 / df_lrb['TOTAL_OPERATE_INCOME'], 2)
    # 销售毛利率=销售毛利/销售收入×100%=（销售收入-销售成本）/销售收入×100%= (营业收入 - 营业成本 / 营业收入) * 100%
    dt['销售毛利率'] = round((df_lrb['OPERATE_INCOME'] - df_lrb['OPERATE_COST']) * 100 / df_lrb['OPERATE_INCOME'], 2)
    # 净利率=净利润/营业收入
    dt['销售净利率'] = round(df_lrb['NETPROFIT'] * 100 / df_lrb['OPERATE_INCOME'], 2)
    # 销售成本率=销售成本/销售收入净额×100%
    dt['销售成本率'] = round(df_lrb['OPERATE_COST'] * 100 / df_lrb['OPERATE_INCOME'], 2)

    ret_df = pd.DataFrame(dt)
    ret_df = ret_df.fillna(0)

    if data_type == 1:
        ret_df = ret_df[ret_df['REPORT_TYPE'] == '一季报']
    elif data_type == 2:
        ret_df = ret_df[ret_df['REPORT_TYPE'] == '中报']
    elif data_type == 4:
        ret_df = ret_df[ret_df['REPORT_TYPE'] == '年报']

    # 返回最近n个数据
    ret_df.drop('REPORT_TYPE', axis=1, inplace=True)
    if len(ret_columns) != 0:
        ret_df = ret_df[ret_columns]
    ret_df = ret_df.head(n)
    ret_df_display = get_display_data(ret_df)

    if is_display:
        return ret_df_display
    else:
        return ret_df


def get_zcfz_data(code, n, data_type=0, is_display=True, ret_columns=zcfz_ret_columns):
    '''
    获取资产负债表数据
    :param code: 股票代码
    :type code: str
    :param n: 返回数据个数
    :type n: int
    :param data_type: 返回数据的形式，0为自然，1为季报，2为中报，4为年报
    :type data_type: int
    :param is_display: 是否返回展示数据
    :type is_display: bool
    :param ret_columns: 返回数据列名列表,必须包含"报告日"字段
    :type ret_columns: list
    :return: 返回现金流量数据
    :rtype: pandas.DataFrame
    '''
    df_zcfz = ak.stock_balance_sheet_by_report_em(symbol=get_szsh_code(code))
    #print(df_zcfz.columns.tolist())
    # https://blog.csdn.net/a389085918/article/details/80284812
    dt = {}
    dt['报告日'] = df_zcfz['REPORT_DATE_NAME']
    dt['REPORT_TYPE'] = df_zcfz['REPORT_TYPE']
    # df_zcfz = df_zcfz.sort_index(ascending=False)
    # 资产负债率
    dt['总资产'] = round(df_zcfz['TOTAL_ASSETS'] / 100000000, 2)
    dt['总负债'] = round(df_zcfz['TOTAL_LIABILITIES'] / 100000000, 2)
    dt['资产负债率'] = round(df_zcfz['TOTAL_LIABILITIES'] * 100 / df_zcfz['TOTAL_ASSETS'], 2)
    dt['货币资金/总资产(%)'] = round(df_zcfz['MONETARYFUNDS'] * 100 / df_zcfz['TOTAL_ASSETS'], 2)
    # 流动资产
    dt['货币资金'] = round(df_zcfz['MONETARYFUNDS'] / 100000000, 2)
    dt['拆出资金'] = round(df_zcfz['LEND_FUND'] / 100000000, 2)
    dt['交易性金融资产'] = round(df_zcfz['TRADE_FINASSET'] / 100000000, 2)
    dt['衍生金融资产'] = round(df_zcfz['DERIVE_FINASSET'] / 100000000, 2)
    dt['应收票据及应收账款'] = round(df_zcfz['NOTE_ACCOUNTS_RECE'] / 100000000, 2)
    dt['应收票据'] = round(df_zcfz['NOTE_RECE'] / 100000000, 2)
    dt['应收账款'] = round(df_zcfz['ACCOUNTS_RECE'] / 100000000, 2)
    dt['应收款项融资'] = round(df_zcfz['FINANCE_RECE'] / 100000000, 2)
    dt['预付款项'] = round(df_zcfz['PREPAYMENT'] / 100000000, 2)
    dt['其他应收款(合计)'] = round(df_zcfz['TOTAL_OTHER_RECE'] / 100000000, 2)
    dt['买入返售金融资产'] = round(df_zcfz['BUY_RESALE_FINASSET'] / 100000000, 2)
    dt['存货'] = round(df_zcfz['INVENTORY'] / 100000000, 2)
    dt['其他流动资产'] = round(df_zcfz['OTHER_CURRENT_ASSET'] / 100000000, 2)
    dt['流动资产合计'] = round(df_zcfz['TOTAL_CURRENT_ASSETS'] / 100000000, 2)
    # 非流动资产
    dt['发放贷款及垫款'] = round(df_zcfz['LOAN_ADVANCE'] / 100000000, 2)
    dt['债权投资'] = round(df_zcfz['CREDITOR_INVEST'] / 100000000, 2)
    dt['其他非流动金融资产'] = round(df_zcfz['OTHER_NONCURRENT_FINASSET'] / 100000000, 2)
    dt['投资性房地产'] = round(df_zcfz['INVEST_REALESTATE'] / 100000000, 2)
    dt['固定资产'] = round(df_zcfz['FIXED_ASSET'] / 100000000, 2)
    dt['在建工程'] = round(df_zcfz['CIP'] / 100000000, 2)
    dt['使用权资产'] = round(df_zcfz['USERIGHT_ASSET'] / 100000000, 2)
    dt['无形资产'] = round(df_zcfz['INTANGIBLE_ASSET'] / 100000000, 2)
    dt['开发支出'] = round(df_zcfz['DEVELOP_EXPENSE'] / 100000000, 2)
    dt['长期待摊费用'] = round(df_zcfz['LONG_PREPAID_EXPENSE'] / 100000000, 2)
    dt['递延所得税资产'] = round(df_zcfz['DEFER_TAX_ASSET'] / 100000000, 2)
    dt['其他非流动资产'] = round(df_zcfz['OTHER_NONCURRENT_ASSET'] / 100000000, 2)
    dt['非流动资产合计'] = round(df_zcfz['TOTAL_NONCURRENT_ASSETS'] / 100000000, 2)
    dt['资产合计'] = round(df_zcfz['TOTAL_ASSETS'] / 100000000, 2)
    # 流动负债
    dt['应付票据及应付账款'] = round(df_zcfz['NOTE_ACCOUNTS_PAYABLE'] / 100000000, 2)
    dt['应付账款'] = round(df_zcfz['ACCOUNTS_PAYABLE'] / 100000000, 2)
    dt['合同负债'] = round(df_zcfz['CONTRACT_LIAB'] / 100000000, 2)
    dt['应付职工薪酬'] = round(df_zcfz['STAFF_SALARY_PAYABLE'] / 100000000, 2)
    dt['应缴税费'] = round(df_zcfz['TAX_PAYABLE'] / 100000000, 2)
    dt['其他应付款(合计)'] = round(df_zcfz['TOTAL_OTHER_PAYABLE'] / 100000000, 2)
    dt['应付股利'] = round(df_zcfz['DIVIDEND_PAYABLE'] / 100000000, 2)
    dt['一年内到期的非流动负债'] = round(df_zcfz['NONCURRENT_LIAB_1YEAR'] / 100000000, 2)
    dt['其他流动负债'] = round(df_zcfz['OTHER_CURRENT_LIAB'] / 100000000, 2)
    dt['流动负债合计'] = round(df_zcfz['TOTAL_CURRENT_LIAB'] / 100000000, 2)
    # 非流动负债
    dt['租赁负债'] = round(df_zcfz['LEASE_LIAB'] / 100000000, 2)
    dt['递延所得税负债'] = round(df_zcfz['DEFER_TAX_LIAB'] / 100000000, 2)
    dt['非流动负债合计'] = round(df_zcfz['TOTAL_NONCURRENT_LIAB'] / 100000000, 2)

    dt['应付手续费及佣金'] = round(df_zcfz['FEE_COMMISSION_PAYABLE'] / 100000000, 2)
    dt['应付手续费及佣金'] = round(df_zcfz['FEE_COMMISSION_PAYABLE'] / 100000000, 2)
    dt['负债合计'] = round(df_zcfz['TOTAL_LIABILITIES'] / 100000000, 2)
    # 所有者权益
    dt['实收资本'] = round(df_zcfz['SHARE_CAPITAL'] / 100000000, 2)
    dt['资本公积'] = round(df_zcfz['CAPITAL_RESERVE'] / 100000000, 2)
    dt['其他综合收益'] = round(df_zcfz['OTHER_COMPRE_INCOME'] / 100000000, 2)
    dt['盈余公积'] = round(df_zcfz['SURPLUS_RESERVE'] / 100000000, 2)
    dt['一般风险准备'] = round(df_zcfz['GENERAL_RISK_RESERVE'] / 100000000, 2)
    dt['未分配利润'] = round(df_zcfz['UNASSIGN_RPOFIT'] / 100000000, 2)
    dt['归属于母公司股东权益总计'] = round(df_zcfz['TOTAL_PARENT_EQUITY'] / 100000000, 2)
    dt['少数股东权益'] = round(df_zcfz['MINORITY_EQUITY'] / 100000000, 2)
    dt['股东权益合计'] = round(df_zcfz['TOTAL_EQUITY'] / 100000000, 2)
    dt['负债和股东权益总计'] = round(df_zcfz['TOTAL_LIAB_EQUITY'] / 100000000, 2)

    dt['短期借款'] = round(df_zcfz['SHORT_LOAN'] / 100000000, 2)
    dt['长期借款'] = round(df_zcfz['LONG_LOAN'] / 100000000, 2)
    dt['交易性金融负债'] = round(df_zcfz['TRADE_FINLIAB'] / 100000000, 2)
    dt['衍生金融负债'] = round(df_zcfz['DERIVE_FINLIAB'] / 100000000, 2)
    dt['预收款项'] = round(df_zcfz['ADVANCE_RECEIVABLES'] / 100000000, 2)
    dt['划分为持有待售负债'] = round(df_zcfz['ADVANCE_RECEIVABLES'] / 100000000, 2)
    dt['预提费用'] = round(df_zcfz['ACCRUED_EXPENSE'] / 100000000, 2)
    dt['递延收益-流动负债'] = round(df_zcfz['DEFER_TAX_LIAB'] / 100000000, 2)
    dt['应付短期债券'] = round(df_zcfz['SHORT_BOND_PAYABLE'] / 100000000, 2)
    dt['其他金融类流动负债'] = round(df_zcfz['OTHER_NONCURRENT_LIAB'] / 100000000, 2)
    # dt['流动负债差额(特殊报表项目)'] = round(df_zcfz['ADVANCE_RECEIVABLES'] / 100000000, 2)
    # dt['流动负债差额(合计平衡项目)'] = round(df_zcfz['ADVANCE_RECEIVABLES'] / 100000000, 2)
    dt['生产性生物资产'] = round(df_zcfz['PRODUCTIVE_BIOLOGY_ASSET'] / 100000000, 2)
    dt['归属母公司股东权益'] = round(df_zcfz['PARENT_EQUITY_BALANCE'] / 100000000, 2)
    dt['长期应收款'] = round(df_zcfz['LONG_RECE'] / 100000000, 2)
    dt['长期股权投资'] = round(df_zcfz['LONG_EQUITY_INVEST'] / 100000000, 2)
    dt['工程物资'] = round(df_zcfz['PROJECT_MATERIAL'] / 100000000, 2)
    dt['固定资产清理'] = round(df_zcfz['FIXED_ASSET_DISPOSAL'] / 100000000, 2)
    dt['长期应付款'] = round(df_zcfz['LONG_PAYABLE'] / 100000000, 2)

    # 有息负债=短期借款+一年内到期的非流动负债+长期借款+应付债券+长期应付款
    dt['有息负债'] = round((df_zcfz['SHORT_LOAN'] + df_zcfz['NONCURRENT_LIAB_1YEAR'] + df_zcfz['LONG_LOAN'] + df_zcfz['LONG_PAYABLE'] + df_zcfz['BOND_PAYABLE']) / 100000000, 2)
    dt['银行存款'] = round(df_zcfz['ACCEPT_DEPOSIT_INTERBANK'] / 100000000, 2)
    dt['有息负债率'] = round((df_zcfz['SHORT_LOAN'] + df_zcfz['NONCURRENT_LIAB_1YEAR'] + df_zcfz['LONG_LOAN'] + df_zcfz['LONG_PAYABLE'] + df_zcfz['BOND_PAYABLE']) / df_zcfz['TOTAL_LIABILITIES'], 2)

    ret_df = pd.DataFrame(dt)
    ret_df = ret_df.fillna(0)
    # 应收账款周转率=营业收入/（（期初应收账款+期末应收账款）/2）
    # 应收账款周转天数=365/应收账款周转率

    if data_type == 1:
        ret_df = ret_df[ret_df['REPORT_TYPE'] == '一季报']
    elif data_type == 2:
        ret_df = ret_df[ret_df['REPORT_TYPE'] == '中报']
    elif data_type == 4:
        ret_df = ret_df[ret_df['REPORT_TYPE'] == '年报']

    # 返回最近n个数据
    ret_df.drop('REPORT_TYPE', axis=1, inplace=True)
    if len(ret_columns) != 0:
        ret_df = ret_df[ret_columns]
    ret_df = ret_df.head(n)

    ret_df_display = get_display_data(ret_df)

    if is_display:
        return ret_df_display
    else:
        return ret_df


def get_xjll_data(code, n, data_type=0, is_display=True, ret_columns=xjll_ret_columns):
    '''
    获取现金流量表数据
    :param code: 股票代码
    :type code: str
    :param n: 返回数据个数
    :type n: int
    :param data_type: 返回数据的形式，0为自然，1为季报，2为中报，4为年报
    :type data_type: int
    :param is_display: 是否返回展示数据
    :type is_display: bool
    :param ret_columns: 返回数据列名列表,必须包含"报告日"字段
    :type ret_columns: list
    :return: 返回现金流量数据
    :rtype: pandas.DataFrame
    '''
    # 人力投入回报率=企业净利润/员工薪酬福利总额×100%，这是衡量人力资本有效性的核心指标，表明公司在人力资源上的投入和净利润的比值，回报率越高，说明人力资源的效率和效能越高。
    df_xjll = ak.stock_cash_flow_sheet_by_report_em(symbol=get_szsh_code(code))
    dt = {}
    dt['报告日'] = df_xjll['REPORT_DATE_NAME']
    dt['REPORT_TYPE'] = df_xjll['REPORT_TYPE']
    # 经营活动产生的现金流量
    dt['销售商品、提供劳务收到的现金'] = round(df_xjll['SALES_SERVICES'] / 100000000, precision)
    dt['客户存款和同业存放款项净增加额'] = round(df_xjll['DEPOSIT_INTERBANK_ADD'] / 100000000, precision)
    dt['收取利息、手续费及佣金的现金'] = round(df_xjll['RECEIVE_INTEREST_COMMISSION'] / 100000000, precision)
    dt['收到的税收返还'] = round(df_xjll['RECEIVE_TAX_REFUND'] / 100000000, precision)
    dt['收到其他与经营活动有关的现金'] = round(df_xjll['RECEIVE_OTHER_OPERATE'] / 100000000, precision)
    dt['经营活动现金流入小计'] = round(df_xjll['TOTAL_OPERATE_INFLOW'] / 100000000, precision)
    dt['购买商品、接受劳务支付的现金'] = round(df_xjll['BUY_SERVICES'] / 100000000, precision)
    dt['客户贷款及垫款净增加额'] = round(df_xjll['LOAN_ADVANCE_ADD'] / 100000000, precision)
    dt['存放中央银行和同业款项净增加额'] = round(df_xjll['PBC_INTERBANK_ADD'] / 100000000, precision)
    dt['支付利息、手续费及佣金的现金'] = round(df_xjll['PAY_INTEREST_COMMISSION'] / 100000000, precision)
    dt['支付给职工以及为职工支付的现金'] = round(df_xjll['PAY_STAFF_CASH']/100000000, precision)
    dt['支付的各项税费'] = round(df_xjll['PAY_ALL_TAX']/100000000, precision)
    dt['支付其他与经营活动有关的现金'] = round(df_xjll['PAY_OTHER_OPERATE']/100000000, precision)
    dt['经营活动现金流出的其他项目'] = round(df_xjll['OPERATE_OUTFLOW_OTHER']/100000000, precision)
    dt['经营活动现金流出小计'] = round(df_xjll['TOTAL_OPERATE_OUTFLOW']/100000000, precision)
    dt['经营活动产生的现金流量净额'] = round(df_xjll['NETCASH_OPERATE'] / 100000000, precision)

    ## 投资活动产生的现金流量
    dt['收回投资收到的现金'] = round(df_xjll['WITHDRAW_INVEST']/100000000, precision)
    dt['取得投资收益收到的现金'] = round(df_xjll['RECEIVE_INVEST_INCOME'] / 100000000, precision)
    dt['处置固定资产、无形资产和其他长期资产收回的现金净额'] = round(df_xjll['DISPOSAL_LONG_ASSET'] / 100000000, precision)
    dt['收到的其他与投资活动有关的现金'] = round(df_xjll['RECEIVE_OTHER_INVEST'] / 100000000, precision)
    dt['投资活动现金流入小计'] = round(df_xjll['TOTAL_INVEST_INFLOW'] / 100000000, precision)
    dt['购建固定资产、无形资产和其他长期资产支付的现金'] = round(df_xjll['CONSTRUCT_LONG_ASSET'] / 100000000, precision)
    dt['投资支付的现金'] = round(df_xjll['INVEST_PAY_CASH'] / 100000000, precision)
    dt['支付其他与投资活动有关的现金'] = round(df_xjll['PAY_OTHER_INVEST'] / 100000000, precision)
    dt['投资活动现金流出小计'] = round(df_xjll['TOTAL_INVEST_OUTFLOW'] / 100000000, precision)
    dt['投资活动产生的现金流量净额'] = round(df_xjll['NETCASH_INVEST'] / 100000000, precision)
    # 筹资活动产生的现金流量
    dt['分配股利、利润或偿付利息支付的现金'] = round(df_xjll['ASSIGN_DIVIDEND_PORFIT'] / 100000000, precision)
    dt['其中:子公司支付给少数股东的股利、利润'] = round(df_xjll['SUBSIDIARY_PAY_DIVIDEND'] / 100000000, precision)
    dt['支付的其他与筹资活动有关的现金'] = round(df_xjll['PAY_OTHER_FINANCE'] / 100000000, precision)
    dt['筹资活动现金流出小计'] = round(df_xjll['TOTAL_FINANCE_OUTFLOW'] / 100000000, precision)
    dt['筹资活动产生的现金流量净额'] = round(df_xjll['NETCASH_FINANCE'] / 100000000, precision)
    dt['汇率变动对现金及现金等价物的影响'] = round(df_xjll['RATE_CHANGE_EFFECT'] / 100000000, precision)
    dt['现金及现金等价物净增加额'] = round(df_xjll['CCE_ADD'] / 100000000, precision)
    dt['加:期初现金及现金等价物余额'] = round(df_xjll['BEGIN_CCE'] / 100000000, precision)
    dt['期末现金及现金等价物余额'] = round(df_xjll['END_CCE'] / 100000000, precision)
    # 补充资料
    dt['净利润'] = round(df_xjll['NETPROFIT'] / 100000000, precision)
    dt['固定资产和投资性房地产折旧'] = round(df_xjll['FA_IR_DEPR'] / 100000000, precision)
    dt['其中:固定资产折旧、油气资产折耗、生产性生物资产折旧'] = round(df_xjll['OILGAS_BIOLOGY_DEPR'] / 100000000, precision)
    dt['无形资产摊销'] = round(df_xjll['IA_AMORTIZE'] / 100000000, precision)
    dt['长期待摊费用摊销'] = round(df_xjll['LPE_AMORTIZE'] / 100000000, precision)
    dt['处置固定资产、无形资产和其他长期资产的损失'] = round(df_xjll['DISPOSAL_LONGASSET_LOSS'] / 100000000, precision)
    dt['固定资产报废损失'] = round(df_xjll['FA_SCRAP_LOSS'] / 100000000, precision)
    dt['公允价值变动损失'] = round(df_xjll['FAIRVALUE_CHANGE_LOSS'] / 100000000, precision)
    dt['财务费用'] = round(df_xjll['FINANCE_EXPENSE'] / 100000000, precision)
    dt['投资损失'] = round(df_xjll['INVEST_LOSS'] / 100000000, precision)
    dt['递延所得税'] = round(df_xjll['DEFER_TAX'] / 100000000, precision)
    dt['其中:递延所得税资产减少'] = round(df_xjll['DT_ASSET_REDUCE'] / 100000000, precision)
    dt['递延所得税负债增加'] = round(df_xjll['DT_LIAB_ADD'] / 100000000, precision)
    dt['存货的减少'] = round(df_xjll['INVENTORY_REDUCE'] / 100000000, precision)
    dt['经营性应收项目的减少'] = round(df_xjll['OPERATE_RECE_REDUCE'] / 100000000, precision)
    dt['经营性应付项目的增加'] = round(df_xjll['OPERATE_PAYABLE_ADD'] / 100000000, precision)
    dt['经营活动产生的现金流量净额'] = round(df_xjll['NETCASH_OPERATE'] / 100000000, precision)
    dt['现金的期末余额'] = round(df_xjll['END_CASH'] / 100000000, precision)
    dt['减:现金的期初余额'] = round(df_xjll['BEGIN_CASH'] / 100000000, precision)
    dt['加:现金等价物的期末余额'] = round(df_xjll['END_CASH_EQUIVALENTS'] / 100000000, precision)
    dt['现金及现金等价物的净增加额'] = round(df_xjll['CCE_ADDNOTE'] / 100000000, precision)

    ret_df = pd.DataFrame(dt)
    ret_df = ret_df.fillna(0)

    if data_type == 1:
        ret_df = ret_df[ret_df['REPORT_TYPE'] == '一季报']
    elif data_type == 2:
        ret_df = ret_df[ret_df['REPORT_TYPE'] == '中报']
    elif data_type == 4:
        ret_df = ret_df[ret_df['REPORT_TYPE'] == '年报']

    # 返回最近n个数据
    ret_df.drop('REPORT_TYPE', axis=1, inplace=True)
    if len(ret_columns) != 0:
        ret_df = ret_df[ret_columns]
    ret_df = ret_df.head(n)

    ret_df_display = get_display_data(ret_df)

    if is_display:
        return ret_df_display
    else:
        return ret_df


def get_zygc_data(code, data_date, indicator="全部"):
    '''
    获取主营构成数据
    :param code: 股票代码
    :type code: str
    :param data_date: 报告日期，'2023-12-31'
    :type data_date: str
    :param indicator: 分类类型，按产品分类，按地区分类，全部
    :type indicator: str
    :return: 返回主营构成数据
    :rtype: pandas.DataFrame
    '''
    # 主营构成-东财
    zygc_em_df = ak.stock_zygc_em(symbol=get_szsh_code(code))
    zygc_em_df = zygc_em_df[zygc_em_df['报告日期'].astype(str) == data_date]
    dt = {}
    dt['报告日期'] = zygc_em_df['报告日期']
    dt['分类类型'] = zygc_em_df['分类类型']
    dt['主营收入'] = round(zygc_em_df['主营收入'] / 100000000, 2)
    dt['收入比例'] = round(zygc_em_df['收入比例'] * 100, 2)
    dt['主营成本'] = round(zygc_em_df['主营成本'] / 100000000, 2)
    dt['成本比例'] = round(zygc_em_df['成本比例'] * 100, 2)
    dt['主营利润'] = round(zygc_em_df['主营利润'] / 100000000, 2)
    dt['利润比例'] = round(zygc_em_df['利润比例'] * 100, 2)
    dt['毛利率'] = round(zygc_em_df['毛利率'] * 100, 2)
    ret_df = pd.DataFrame(dt)
    ret_df.reset_index()
    ret_df = ret_df.sort_values(by=['分类类型', '收入比例'], ascending=[True, False])
    if indicator != "全部":
        ret_df = ret_df[ret_df['分类类型'] == indicator]

    return ret_df

if __name__ == "__main__":
    code = "000977"
    df_zygc = get_basic_info(code)
    print(df_zygc)
    df_lrb_display_year = get_lrb_data(code, 8, 4)
    print("----------------------------- 利润表（年度） -----------------------------")
    print(df_lrb_display_year)
    df_lrb_111_display = get_lrb_data(code, 8)
    print("----------------------------- 利润表（自然季度） -----------------------------")
    print(df_lrb_111_display)
    print("----------------------------- 关键指标 -----------------------------")
    df_main_display = get_main_indicators_sina(
        code=code,
        n=5,
        ret_columns=['报告期', '营业总收入', '营业总收入增长率', '净利润', '扣非净利润'])
    print(df_main_display)
    print("----------------------------- 盈利能力 -----------------------------")
    df_main_yingli_display = get_main_indicators_sina(
        code=code,
        n=5,
        ret_columns=['报告期', '净资产收益率(ROE)', '摊薄净资产收益率', '总资产报酬率', '销售净利率', '毛利率'])
    print(df_main_yingli_display)
    print("----------------------------- 财务风险 -----------------------------")
    df_main_fengxian_display = get_main_indicators_sina(
        code=code,
        n=5,
        ret_columns=['报告期', '资产负债率', '流动比率', '速动比率', '权益乘数', '产权比率', '现金比率', '产权比率'])
    print(df_main_fengxian_display)
    print("----------------------------- 运营能力 -----------------------------")
    df_main_yunying_display = get_main_indicators_sina(
        code=code,
        n=5,
        ret_columns=['报告期', '存货周转天数', '应收账款周转天数', '总资产周转率', '存货周转率', '应收账款周转率', '应付账款周转率', '流动资产周转率'])
    print(df_main_yunying_display)
