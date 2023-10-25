# 克隆自聚宽文章：https://www.joinquant.com/post/15565
# 标题：期货 AdaptiveMA自适应均线
# 作者：cicikml

# 导入函数库
from jqdata import *
import talib
from math import isnan

def initialize(context):
    # 设置参数
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
    set_order_cost(OrderCost(open_commission=0.00005, close_commission=0.00005,close_today_commission=0.00005), type='index_futures')
    # 设定保证金比例
    set_option('futures_margin_rate', 0.15)
    # 设置滑点（单边万5，双边千1）
    set_slippage(PriceRelatedSlippage(0.001),type='future')
      # 开盘前运行
    run_daily( before_market_open, time='before_open', reference_security=get_future_code('RB'))
      # 开盘时运行
    run_daily( market_open, time='open', reference_security=get_future_code('RB'))
      # 收盘后运行
    run_daily( after_market_close, time='after_close', reference_security=get_future_code('RB'))

   # 参数设置函数
def set_parameter(context):

    #######变量设置########
    g.LastRealPrice = {} # 最新真实合约价格字典
    g.HighPrice = {} # 各品种最高价字典（用于吊灯止损）
    g.LowPrice = {} # 各品种最低价字典（用于吊灯止损）
    g.future_list = []  # 设置期货品种列表
    g.TradeLots = {}  # 各品种的交易手数信息
    g.Price_dict = {} # 各品种价格列表字典
    g.Times = {} # 计数器（用于防止止损重入）
    g.Reentry_long = False # 止损后重入标记
    g.Reentry_short = False # 止损后重入标记
    g.ATR = {} # ATR值字典
    g.AMA = {} # AMA值字典
    g.PriceArray = {} # 信号计算价格字典
    g.Filter ={} # 过滤器金额（计算买卖条件）
    g.MappingReal = {} # 真实合约映射（key为symbol，value为主力合约）
    g.MappingIndex = {} # 指数合约映射 （key为 symbol，value为指数合约

    #######参数设置########
    g.Cross = 0 # 均线交叉判定信号
    g.FilterTimes = 0.3 # AMA的过滤乘数
    g.NATRstop = 4 # ATR止损倍数
    g.Window = 15 # AMA等窗口参数（方便起见，系数统一，绩效会有下降，但是安全））

    # 交易的期货品种信息
    g.instruments = ['RU','TA','PP','CU','AG','L','RB','I','J','JD']

    # 价格列表初始化
    set_future_list(context)


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
                g.HighPrice[dom] = False
                g.LowPrice[dom] = False
                g.Times[dom] = 0

## 开盘前运行函数
def before_market_open(context):
    # 输出运行时间
    log.info('函数运行时间(before_market_open)：'+str(context.current_dt.time()))
    send_message('开始交易')

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
                replace_old_futures(context,ins,dom)
                g.future_list.append(dom)
                g.HighPrice[dom] = False
                g.LowPrice[dom] = False
                g.Times[dom] = 0

            g.TradeLots[dom] = get_lots(context.portfolio.starting_cash/len(g.instruments),ins)


## 开盘时运行函数
def market_open(context):

    # 以下是主循环
    for ins in g.instruments:
        # 过滤空主力合约品种
        if g.MappingReal[ins] != '':
            IndexFuture = g.MappingIndex[ins]
            RealFuture = g.MappingReal[ins]
            # 获取当月合约交割日期
            end_date = get_CCFX_end_date(RealFuture)
            # 当月合约交割日当天不开仓
            if (context.current_dt.date() == end_date):
                return
            else:
                g.LastRealPrice[RealFuture] = attribute_history(RealFuture,1,'1d',['close'])['close'][-1]
                # 获取价格list
                g.PriceArray[IndexFuture] = attribute_history(IndexFuture,50,'1d',['close','open','high','low'])
                g.CurrentPrice = g.PriceArray[IndexFuture]['close'][-1]
                g.ClosePrice = g.PriceArray[IndexFuture]['close']
                # 如果没有数据，返回
                if len(g.PriceArray[IndexFuture]) < 50:
                    return
                else:
                    close = np.array(g.PriceArray[IndexFuture]['close'])
                    high = np.array(g.PriceArray[IndexFuture]['high'])
                    low = np.array(g.PriceArray[IndexFuture]['low'])

                    # =========================================================================================
                    # 计算AMA，仅传入一个参数g.Window
                    g.AMA[IndexFuture] = talib.KAMA(close,g.Window)
                    # 计算ATR
                    g.ATR[IndexFuture] = talib.ATR(high,low,close, g.Window)[-1]
                    if not isnan(g.AMA[IndexFuture][-1]) :
                        g.Filter[IndexFuture] = talib.STDDEV(g.AMA[IndexFuture][-g.Window:],g.Window)[-1]

                        # 判断AMA两日差值，是否大于标准差过滤器
                        if g.AMA[IndexFuture][-1]-g.AMA[IndexFuture][-2] > g.Filter[IndexFuture]*g.FilterTimes:
                            g.Cross = 1
                        elif g.AMA[IndexFuture][-2]-g.AMA[IndexFuture][-1] > g.Filter[IndexFuture]*g.FilterTimes:
                            g.Cross = -1
                        else:
                            g.Cross = 0

                        # 判断交易信号：均线交叉+可二次入场条件成立
                        if  g.Cross == 1 and g.Reentry_long == False:
                            g.Signal = 1
                        elif g.Cross == -1 and g.Reentry_short == False:
                            g.Signal = -1
                        else:
                            g.Signal = 0

                    # =========================================================================================

                    # 执行交易
                    Trade(context,RealFuture,IndexFuture)
                    # 运行防止充入模块
                    Re_entry(context,RealFuture)
                    # 计数器+1
                    if RealFuture in g.Times.keys():
                        g.Times[RealFuture] += 1
                    else:
                        g.Times[RealFuture] = 0


## 收盘后运行函数
def after_market_close(context):
    #log.info(str('函数运行时间(after_market_close):'+str(context.current_dt.time())))
    # 得到当天所有成交记录
    trades = get_trades()
    for _trade in trades.values():
        log.info('成交记录：'+str(_trade))
    log.info('一天结束')
    log.info('##############################################################')


## 交易模块
def Trade(context,RealFuture,IndexFuture):

    # 快线高于慢线，且追踪止损失效，则可开多仓
    if g.Signal == 1 and context.portfolio.long_positions[RealFuture].total_amount == 0:
        if context.portfolio.long_positions[RealFuture].total_amount != 0:
            log.info('空头有持仓：%s'%(RealFuture))
        order_target(RealFuture,0,side='short')
        order_target(RealFuture,g.TradeLots[RealFuture],side='long')
        g.HighPrice[RealFuture] = g.LastRealPrice[RealFuture]
        g.LowPrice[RealFuture] = False
        log.info('正常买多合约：%s'%(RealFuture))


    elif g.Signal == -1 and context.portfolio.short_positions[RealFuture].total_amount == 0:
        if context.portfolio.short_positions[RealFuture].total_amount != 0:
            log.info('多头有持仓：%s'%(RealFuture))
        order_target(RealFuture,0,side ='long')
        order_target(RealFuture,g.TradeLots[RealFuture],side='short')
        g.LowPrice[RealFuture] = g.LastRealPrice[RealFuture]
        g.HighPrice[RealFuture] = False
        log.info('正常卖空合约：%s'%(RealFuture))
    else:
        # 追踪止损
        Trailing_Stop(context,RealFuture,IndexFuture)


# 防止止损后立刻重入模块
def Re_entry(context,future):
    # 防重入模块：上一次止损后20根bar内不交易，但如果出现价格突破事件则20根bar的限制失效

    #设置最高价与最低价（注意：需要错一位，不能算入当前价格）
    g.Highest_high_2_20 = g.ClosePrice[-21:-1].max()
    g.Lowest_low_2_20 = g.ClosePrice[-21:-1].min()

    if  g.Reentry_long == True:
        if g.Times[future] > 20 or g.CurrentPrice > g.Highest_high_2_20 :
            g.Reentry_long = False
    if  g.Reentry_short == True:
        if g.Times[future] > 20 or g.CurrentPrice < g.Lowest_low_2_20 :
            g.Reentry_short = False

# 追踪止损模块
def Trailing_Stop(context,RealFuture,IndexFuture):

    long_positions = context.portfolio.long_positions
    short_positions = context.portfolio.short_positions

    if RealFuture in long_positions.keys():
        if long_positions[RealFuture].total_amount > 0:
            if g.HighPrice[RealFuture]:
                g.HighPrice[RealFuture] = max(g.HighPrice[RealFuture], g.LastRealPrice[RealFuture])
                if g.LastRealPrice[RealFuture]  < g.HighPrice[RealFuture]  - g.NATRstop*g.ATR[IndexFuture]:
                    log.info('多头止损:\t' +  RealFuture)
                    order_target(RealFuture,0,side = 'long')
                    g.Reentry_long = True

    if RealFuture in short_positions.keys():
        if short_positions[RealFuture].total_amount > 0:
            if g.LowPrice[RealFuture]:
                g.LowPrice[RealFuture] = min(g.LowPrice[RealFuture], g.LastRealPrice[RealFuture])
                if g.LastRealPrice[RealFuture]  > g.LowPrice[RealFuture] + g.NATRstop*g.ATR[IndexFuture]:
                    log.info('空头止损:\t' + RealFuture)
                    order_target(RealFuture,0,side = 'short')
                    g.Reentry_short = True


# 移仓模块：当主力合约更换时，平当前持仓，更换为最新主力合约
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


# 获取交易手数函数
def get_lots(cash,symbol):
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
    RealFuture = get_dominant_future(symbol)
    IndexFuture = get_future_code(symbol)
    # 获取价格list
    Price_dict = attribute_history(IndexFuture,10,'1d',['open'])
    # 如果没有数据，返回
    if len(Price_dict) == 0:
        return
    else:
        open_future = Price_dict.iloc[-1]
    # 返回手数
    if IndexFuture in g.ATR.keys():
    # 这里的交易手数，使用了ATR倒数头寸
        return cash*0.1/(g.ATR[IndexFuture]*future_coef_list[symbol])
    else:# 函数运行之初会出现没将future写入ATR字典当中的情况
        return cash*0.0001/future_coef_list[symbol]


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