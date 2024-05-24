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
    if is_display:
        return df_display
    else:
        return df


def get_lrb_data(code, n, data_type=0, is_display=True, ret_columns = []):
    '''
    获取利润表数据
    :param code: 股票代码
    :type code: str
    :param n: 返回数据个数
    :type n: str
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
    # 营业收入及同比
    dt['营业总收入'] = round(df_lrb['TOTAL_OPERATE_INCOME'] / 100000000, 2)
    dt['营业总收入同比'] = round(df_lrb['TOTAL_OPERATE_INCOME_YOY'], 2)
    dt['营业总成本'] = round(df_lrb['TOTAL_OPERATE_COST'] / 100000000, 2)
    dt['营业总成本同比'] = round(df_lrb['TOTAL_OPERATE_COST_YOY'], 2)
    dt['营业利润'] = round(df_lrb['OPERATE_PROFIT'] / 100000000, 2)
    dt['营业利润同比'] = round(df_lrb['OPERATE_PROFIT_YOY'], 2)
    dt['利润总额'] = round(df_lrb['TOTAL_PROFIT'] / 100000000, 2)
    dt['利润总额同比'] = round(df_lrb['TOTAL_PROFIT_YOY'], 2)
    dt['净利润'] = round(df_lrb['NETPROFIT'] / 100000000, 2)
    dt['净利润同比'] = round(df_lrb['NETPROFIT_YOY'], 2)
    dt['归属于母公司所有者的净利润'] = round(df_lrb['PARENT_NETPROFIT'] / 100000000, 2)
    dt['归属于母公司所有者的净利润同比'] = round(df_lrb['PARENT_NETPROFIT_YOY'], 2)
    # 净利润/营业总收入(%)
    # 营业利润/营业总收入(%)
    # 息税前利润/营业总收入(%)
    # EBITDA/营业总收入(%)

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


def get_zcfz_data(code, n, data_type=0, is_display=True, ret_columns=[]):
    '''
    获取资产负债表数据
    :param code: 股票代码
    :type code: str
    :param n: 返回数据个数
    :type n: str
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
    # https://blog.csdn.net/a389085918/article/details/80284812
    dt = {}
    dt['报告日'] = df_zcfz['REPORT_DATE_NAME']
    dt['REPORT_TYPE'] = df_zcfz['REPORT_TYPE']
    # df_zcfz = df_zcfz.sort_index(ascending=False)
    # 资产负债率
    # dt['总资产'] = round(df_zcfz['TOTAL_ASSETS'] / 100000000, 2)
    # dt['总负债'] = round(df_zcfz['TOTAL_LIABILITIES'] / 100000000, 2)
    # dt['资产负债率'] = round(df_zcfz['TOTAL_LIABILITIES'] * 100 / df_zcfz['TOTAL_ASSETS'], 2)
    # 流动资产
    dt['货币资金'] = round(df_zcfz['MONETARYFUNDS'] / 100000000, 2)
    dt['拆出资金'] = round(df_zcfz['LEND_FUND'] / 100000000, 2)
    dt['交易性金融资产'] = round(df_zcfz['TRADE_FINASSET'] / 100000000, 2)
    # dt['衍生金融资产'] = round(df_zcfz['DERIVE_FINASSET'] / 100000000, 2)
    dt['应收票据及应收账款'] = round(df_zcfz['NOTE_ACCOUNTS_RECE'] / 100000000, 2)
    dt['应收票据'] = round(df_zcfz['NOTE_RECE'] / 100000000, 2)
    dt['应收账款'] = round(df_zcfz['ACCOUNTS_RECE'] / 100000000, 2)
    # dt['应收款项融资'] = round(df_zcfz['FINANCE_RECE'] / 100000000, 2)
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
    dt['应付帐款'] = round(df_zcfz['ACCOUNTS_PAYABLE'] / 100000000, 2)
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


def get_xjll_data(code, n, data_type=0, is_display=True, ret_columns=[]):
    '''
    获取现金流量表数据
    :param code: 股票代码
    :type code: str
    :param n: 返回数据个数
    :type n: str
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
    # 销售商品、提供劳务收到的现金
    # 客户存款和同业存放款项净增加额
    # 收取利息、手续费及佣金的现金
    # 收到的税收返还
    # 收到其他与经营活动有关的现金
    # 经营活动现金流入小计
    # 购买商品、接受劳务支付的现金
    # 客户贷款及垫款净增加额
    # 存放中央银行和同业款项净增加额
    # 支付利息、手续费及佣金的现金
    # 支付给职工以及为职工支付的现金
    # 支付的各项税费
    # 支付其他与经营活动有关的现金
    # 经营活动现金流出的其他项目
    # 经营活动现金流出小计
    # 经营活动产生的现金流量净额
    ## 投资活动产生的现金流量
    # 收回投资收到的现金
    # 取得投资收益收到的现金
    # 处置固定资产、无形资产和其他长期资产收回的现金净额
    # 收到的其他与投资活动有关的现金
    # 投资活动现金流入小计
    # 购建固定资产、无形资产和其他长期资产支付的现金
    # 投资支付的现金
    # 支付其他与投资活动有关的现金
    # 投资活动现金流出小计
    # 投资活动产生的现金流量净额
    ## 筹资活动产生的现金流量
    # 分配股利、利润或偿付利息支付的现金
    # 其中:子公司支付给少数股东的股利、利润
    # 支付的其他与筹资活动有关的现金
    # 筹资活动现金流出小计
    # 筹资活动产生的现金流量净额
    # 汇率变动对现金及现金等价物的影响
    # 现金及现金等价物净增加额
    # 加:期初现金及现金等价物余额
    # 期末现金及现金等价物余额
    ## 补充资料
    # 净利润
    # 固定资产和投资性房地产折旧
    # 其中:固定资产折旧、油气资产折耗、生产性生物资产折旧
    # 无形资产摊销
    # 长期待摊费用摊销
    # 处置固定资产、无形资产和其他长期资产的损失
    # 固定资产报废损失
    # 公允价值变动损失
    # 财务费用
    # 投资损失
    # 递延所得税
    # 其中:递延所得税资产减少
    # 递延所得税负债增加
    # 存货的减少
    # 经营性应收项目的减少
    # 经营性应付项目的增加
    # 经营活动产生的现金流量净额
    # 现金的期末余额
    # 减:现金的期初余额
    # 加:现金等价物的期末余额
    # 现金及现金等价物的净增加额

    dt['员工薪酬福利总额'] = round(df_xjll['PAY_STAFF_CASH']/100000000, 2)
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
    code = "600519"
    # df = get_basic_info("000737")
    # print(df)
    # zygc_em_df = get_zygc_data("000737", "2023-12-31", indicator="按产品分类")
    # print(zygc_em_df)
    df_xjll = get_xjll_data(code, 6, data_type=0, is_display=True)
    print(df_xjll)
