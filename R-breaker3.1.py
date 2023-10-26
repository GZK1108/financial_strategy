# 导入函数库
from jqdata import *
import numpy as np
import pandas as pd
import datetime
import time

# 初始化函数，设定基准等等
def initialize(context):
    set_parameter(context)
    # 设定基准银华日利，在多品种的回测当中基准没有参考意义
    set_benchmark('511880.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 过滤掉order系列API产生的比error级别低的log
    log.set_level('order', 'error')
    ### 期货相关设定 ###
    # 设定账户为金融账户
    set_subportfolios([SubPortfolioConfig(cash=context.portfolio.starting_cash, type='futures')])
    # 期货类每笔交易时的手续费是：买入时万分之0.23,卖出时万分之0.23,平今仓为万分之23
    set_order_cost(OrderCost(open_commission=0.00005, close_commission=0.00005, close_today_commission=0.00005),
                   type='index_futures')
    # 设定保证金比例，如果不设置，使用默认保证金比例
    set_option('futures_margin_rate', 0.15)
    # 设置滑点（单边万5，双边千1）
    set_slippage(PriceRelatedSlippage(0.001), type='future')
    # 开盘前运行
    run_daily(before_market_open, time='before_open', reference_security=get_future_code('JD'))
    # 开盘时运行，使用鸡蛋JD作为参考，即不交易夜盘
    run_daily(market_open, time='every_bar', reference_security=get_future_code('JD'))
    # 收盘前运行
    run_daily(close_amount, time='14:59:59', reference_security=get_future_code('JD'))
    # 收盘后运行
    run_daily(after_market_close, time='after_close', reference_security=get_future_code('JD'))


# 设置参数函数
def set_parameter(context):
    # 变量设置
    g.future_list = []  # 设置期货品种列表
    g.MappingReal = {}  # 真实合约映射（key为symbol，value为主力合约）
    g.MappingIndex = {}  # 指数合约映射 （key为 symbol，value为指数合约
    g.break_price = {}  # 突破价格映射
    g.maxmin_price = {}  # 存储各个期货品种的当日最高价和最低价
    g.amount = 10  # 设置默认交易的手数  value = 最新价 * 手数 * 保证金率 * 乘数
    g.multiple = 10  # 设置交易的乘数
    g.stop_loss_price = 1  # 设置止损的点数

    # 交易的期货品种信息
    g.instruments = ['RU', 'TA', 'PP', 'CU', 'AG', 'L', 'RB', 'I', 'J', 'JD']

    # 价格列表初始化
    set_future_list(context)


# 合约信息
def set_future_list(context):
    for ins in g.instruments:
        idx = get_future_code(ins)
        dom = get_dominant_future(ins)
        # 填充映射字典
        g.MappingIndex[ins] = idx
        g.MappingReal[ins] = dom
        #设置主力合约已上市的品种基本参数
        if dom == '':
            pass
        else:
            if dom not in g.future_list:
                g.future_list.append(dom)


# 交易函数
def judge(bar_data, context, RealFuture):
    # 输出当前可用保证金
    # log.info('当前可用保证金：' , context.portfolio.available_cash)

    # 设置合约乘数
    symbol = RealFuture[:2]
    multiple = get_lots(symbol)
    break_price = g.break_price

    # 持仓情况
    # print('多仓：'+str(context.portfolio.long_positions))
    # print('空仓：'+str(context.portfolio.short_positions))
    # 突破策略，该策略用于在价格高于（低于）突破买入价（突破卖出价）时，采取趋势策略，即在该点位开仓做多（做空）
    # 首先判断是否有持仓，若持仓，则判断止损，若不持仓，则判断是否满足开仓条件
    if (RealFuture not in context.portfolio.long_positions.keys()) and (RealFuture not in context.portfolio.short_positions.keys()):

        if bar_data['close'][0] > break_price[RealFuture]['Bbreak'].values and context.portfolio.available_cash > 500000:
            # 在空仓的情况下，如果盘中价格超过突破买入价，则采取趋势策略，即在该点位开仓做多
            values = bar_data['close'][0] * g.amount * multiple  # value = 最新价 * 手数 * 保证金率 * 乘数
            order_value(RealFuture, values, style=StopMarketOrderStyle('stop_loss', break_price[RealFuture]['Bbreak'].values), side='long', pindex=0, close_today=False)
            print('开多仓'+str(RealFuture)+str(values))
            # print('当前仓位：'+str(context.portfolio.long_positions.keys()))
        elif bar_data['close'][0] < break_price[RealFuture]['Sbreak'].values and context.portfolio.available_cash > 500000:
            # 在空仓的情况下，如果盘中价格跌破突破卖出价，则采取趋势策略，即在该点位开仓做空
            values = bar_data['close'][0] * g.amount * multiple  # value = 最新价 * 手数 * 保证金率 * 乘数
            order_value(RealFuture, values, style=StopMarketOrderStyle('stop_loss', break_price[RealFuture]['Sbreak'].values), side='short', pindex=0, close_today=False)
            print('开空仓'+str(RealFuture)+str(values))

    else:
        # 反转策略，第一次反转时，相当于止损，之后反转则正常盈利
        if RealFuture in context.portfolio.long_positions.keys():  # 多仓
            if (g.maxmin_price[RealFuture]['max'].values > break_price[RealFuture]['Ssetup'].values) and (bar_data['close'][0] < break_price[RealFuture]['Senter'].values) and (context.portfolio.available_cash > 500000):
                order_target(RealFuture, 0, side='long')  # 平仓
                values = bar_data['close'][0] * g.amount * multiple  # value = 最新价 * 手数 * 保证金率 * 乘数
                order_value(RealFuture, values, style=StopMarketOrderStyle('stop_loss', break_price[RealFuture]['Senter'].values), side='short', pindex=0, close_today=False)
                print('平仓开空仓'+str(RealFuture)+str(values))

        elif RealFuture in context.portfolio.short_positions.keys():  # 空仓
            if (g.maxmin_price[RealFuture]['min'].values < break_price[RealFuture]['Bsetup'].values) and (bar_data['close'][0] > break_price[RealFuture]['Benter'].values) and (context.portfolio.available_cash > 500000):
                order_target(RealFuture, 0, side='short')  # 平仓
                values = bar_data['close'][0] * g.amount * multiple  # value = 最新价 * 手数 * 保证金率 * 乘数
                order_value(RealFuture, values, style=StopMarketOrderStyle('stop_loss', break_price[RealFuture]['Benter'].values), side='long', pindex=0, close_today=False)
                print('平仓开多仓'+str(RealFuture)+str(values))


# 开盘前运行函数
def before_market_open(context):
    # 输出运行时间
    log.info('函数运行时间(before_market_open)：' + str(context.current_dt.time()))

    # 过滤无主力合约的品种，传入并修改期货字典信息
    filt_dominant_future(context)
    # 循环设置对比的参数
    set_break_price(context)


# 开盘前平仓
def close_amount(context):
    for LastFuture in context.portfolio.long_positions.keys():
        order_target(LastFuture,0,side='long')
    print('收盘前平多仓')

    for LastFuture in context.portfolio.short_positions.keys():
        order_target(LastFuture,0,side='short')
    print('收盘前平空仓')

    # 查看仓位
    log.info(str('持仓情况：'+str(context.portfolio.long_positions)+str(context.portfolio.short_positions)))


# 开盘运行函数
def market_open(context):
    # 以下是主循环
    for ins in g.instruments:
        # 过滤空主力合约品种
        if g.MappingReal[ins] != '':
            RealFuture = g.MappingReal[ins]  # 主力合约
            # 获取当月合约交割日期
            end_date = get_CCFX_end_date(RealFuture)
            # 当月合约交割日当天不开仓
            if (context.current_dt.date() == end_date):
                return
            else:
                # 获取当前数据
                current_data = get_bars(RealFuture, count=1, unit='1m', fields=['open', 'close', 'high', 'low'],
                                        include_now=True, end_dt=None, fq_ref_date=None)
                # 如果没有数据，返回
                if current_data.shape[0] < 1:
                    return
                else:
                    # 执行交易
                    # print('交易品种：' + str(RealFuture))
                    get_maxmin_price(current_data, RealFuture)
                    judge(current_data, context, RealFuture)


# 收盘后运行函数
def after_market_close(context):
    log.info(str('函数运行时间(after_market_close):'+str(context.current_dt.time())))
    # 得到当天所有成交记录
    # trades = get_trades()
    # for _trade in trades.values():
    #     log.info('成交记录：'+str(_trade))
    # 把参数置空，用于第二天的数据
    g.break_price = {}  # 突破价格映射
    g.maxmin_price = {}  # 存储各个期货品种的当日最高价和最低价
    log.info('一天结束')
    log.info('##############################################################')


# 获取当日最高价和最低价
def get_maxmin_price(current_data, RealFuture):
    # 获取当日最高价和最低价
    if RealFuture not in g.maxmin_price.keys():
        g.maxmin_price[RealFuture] = pd.DataFrame(columns=['max', 'min'])
        g.maxmin_price[RealFuture]['max'] = current_data['high']
        g.maxmin_price[RealFuture]['min'] = current_data['low']
    else:
        if current_data['high'] > g.maxmin_price[RealFuture]['max'].values:
            g.maxmin_price[RealFuture]['max'] = current_data['high']
        if current_data['low'] < g.maxmin_price[RealFuture]['min'].values:
            g.maxmin_price[RealFuture]['min'] = current_data['low']


# 移仓模块：当主力合约更换时，平当前持仓，更换为最新主力合约
# 补充：R-breaker为日内高频交易策略，理论上不需要更换主力合约，仅仅放这儿表示期货交易基本流程
def replace_old_futures(context,ins,dom):

    LastFuture = g.MappingReal[ins]

    if LastFuture in context.portfolio.long_positions.keys():
        lots_long = context.portfolio.long_positions[LastFuture].total_amount
        order_target(LastFuture,0,side='long')
        order_target(dom,lots_long,side='long')
        print('主力合约更换，平多仓换新仓')

    if LastFuture in context.portfolio.short_positions.keys():
        lots_short = context.portfolio.short_positions[dom].total_amount
        order_target(LastFuture,0,side='short')
        order_target(dom,lots_short,side='short')
        print('主力合约更换，平空仓换新仓')

    g.MappingReal[ins] = dom


# 过滤模块
def filt_dominant_future(context):
    # 过滤无主力合约的品种，传入并修改期货字典信息
    for ins in g.instruments:
        dom = get_dominant_future(ins)
        if dom == '':
            pass
        else:
            # 判断是否执行replace_old_futures
            if dom == g.MappingReal[ins]:
                pass
            else:
                replace_old_futures(context, ins, dom)
                g.future_list.append(dom)


# 计算break_price模块
def set_break_price(context):
    # 循环设置对比的参数
    for ins in g.instruments:
        # 过滤空主力合约品种
        if g.MappingReal[ins] != '':
            RealFuture = g.MappingReal[ins]  # 主力合约
            # 获取当月合约交割日期
            end_date = get_CCFX_end_date(RealFuture)
            # 当月合约交割日当天不开仓
            if (context.current_dt.date() == end_date):
                return
            else:
                # 计算需要的突破数据，context.previous_date：获取前一日数据
                daily_ND = get_price(RealFuture, start_date=None, end_date=context.previous_date, frequency='daily',
                                     fields=['open', 'close', 'high', 'low'], skip_paused=False, fq='pre', count=1,
                                     panel=True)
                # a,b,c,d设置参数，参数可调
                a = 0.35
                b = 1.07
                c = 0.07
                d = 0.25
                break_price = pd.DataFrame(columns=['Ssetup', 'Bsetup', 'Senter', 'Benter', 'Bbreak', 'Sbreak'])
                break_price['Ssetup'] = daily_ND['high'] + a * (daily_ND['close'] - daily_ND['low'])
                break_price['Bsetup'] = daily_ND['low'] - a * (daily_ND['high'] - daily_ND['close'])

                break_price['Senter'] = b / 2 * (daily_ND['high'] + daily_ND['low']) - c * daily_ND['low']
                break_price['Benter'] = b / 2 * (daily_ND['high'] + daily_ND['low']) - c * daily_ND['high']

                break_price['Bbreak'] = break_price['Ssetup'] + d * (break_price['Ssetup'] - break_price['Bsetup'])
                break_price['Sbreak'] = break_price['Bsetup'] - d * (break_price['Ssetup'] - break_price['Bsetup'])
                break_price.index = [context.current_dt]
                g.break_price[RealFuture] = break_price  # 传入全局变量，用于Judge函数


# 获取合约乘数函数
def get_lots(symbol):
    future_coef_list = {'A':10, 'AG':15, 'AL':5, 'AU':1000,
                        'B':10, 'BB':500, 'BU':10, 'C':10,
                        'CF':5, 'CS':10, 'CU':5, 'ER':10,
                        'FB':500, 'FG':20, 'FU':50, 'GN':10,
                        'HC':10, 'I':100, 'IC':200, 'IF':300,
                        'IH':300, 'J':100, 'JD':5, 'JM':60,
                        'JR':20, 'L':5, 'LR':10, 'M':10,
                        'MA':10, 'ME':10, 'NI':1, 'OI':10,
                        'P':10, 'PB':5, 'PM':50, 'PP':5,
                        'RB':10, 'RI':20, 'RM':10, 'RO':10,
                        'RS':10, 'RU':10, 'SF':5, 'SM':5,
                        'SN':1, 'SR':10, 'T':10000, 'TA':5,
                        'TC':100, 'TF':10000, 'V':5, 'WH':20,
                        'WR':10, 'WS':50, 'WT':10, 'Y':10,
                        'ZC':100, 'ZN':5}
    if symbol not in future_coef_list.keys():
        lots = g.multiple
    else:
        lots = future_coef_list.get(symbol)
    return lots


# 获取当天时间正在交易的期货主力合约函数
def get_future_code(symbol):
    future_code_list = {'A':'A8888.XDCE', 'AG':'AG8888.XSGE', 'AL':'AL8888.XSGE', 'AU':'AU8888.XSGE',
                        'B':'B8888.XDCE', 'BB':'BB8888.XDCE', 'BU':'BU8888.XSGE', 'C':'C8888.XDCE',
                        'CF':'CF8888.XZCE', 'CS':'CS8888.XDCE', 'CU':'CU8888.XSGE', 'ER':'ER8888.XZCE',
                        'FB':'FB8888.XDCE', 'FG':'FG8888.XZCE', 'FU':'FU8888.XSGE', 'GN':'GN8888.XZCE',
                        'HC':'HC8888.XSGE', 'I':'I8888.XDCE', 'IC':'IC8888.CCFX', 'IF':'IF8888.CCFX',
                        'IH':'IH8888.CCFX', 'J':'J8888.XDCE', 'JD':'JD8888.XDCE', 'JM':'JM8888.XDCE',
                        'JR':'JR8888.XZCE', 'L':'L8888.XDCE', 'LR':'LR8888.XZCE', 'M':'M8888.XDCE',
                        'MA':'MA8888.XZCE', 'ME':'ME8888.XZCE', 'NI':'NI8888.XSGE', 'OI':'OI8888.XZCE',
                        'P':'P8888.XDCE', 'PB':'PB8888.XSGE', 'PM':'PM8888.XZCE', 'PP':'PP8888.XDCE',
                        'RB':'RB8888.XSGE', 'RI':'RI8888.XZCE', 'RM':'RM8888.XZCE', 'RO':'RO8888.XZCE',
                        'RS':'RS8888.XZCE', 'RU':'RU8888.XSGE', 'SF':'SF8888.XZCE', 'SM':'SM8888.XZCE',
                        'SN':'SN8888.XSGE', 'SR':'SR8888.XZCE', 'T':'T8888.CCFX', 'TA':'TA8888.XZCE',
                        'TC':'TC8888.XZCE', 'TF':'TF8888.CCFX', 'V':'V8888.XDCE', 'WH':'WH8888.XZCE',
                        'WR':'WR8888.XSGE', 'WS':'WS8888.XZCE', 'WT':'WT8888.XZCE', 'Y':'Y8888.XDCE',
                        'ZC':'ZC8888.XZCE', 'ZN':'ZN8888.XSGE'}
    try:
        return future_code_list[symbol]
    except:
        return 'WARNING: 无此合约'


# 获取金融期货合约到期日
def get_CCFX_end_date(fature_code):
    # 获取金融期货合约到期日
    return get_security_info(fature_code).end_date
