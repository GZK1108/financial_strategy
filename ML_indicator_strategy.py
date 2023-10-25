import pandas as pd
import numpy as np
import math
import json
import jqdata
from jqfactor import standardlize
from jqfactor import winsorize_med
from jqdata import *
from sklearn.model_selection import KFold
from jqfactor import get_factor_values  # 因子库
from jqlib.optimizer import *
import warnings
warnings.filterwarnings("ignore")


def initialize(context):
    set_params()
    set_backtest()
    
    if g.short == True:
        ## 设置单个账户
        # 获取初始资金
        init_cash = context.portfolio.starting_cash 
        ## 设置多个账户
        # 获取初始资金，并等分为两份
        # 设定subportfolios[0]为 股票和基金仓位，初始资金为 init_cash 变量代表的数值
        # 设定subportfolios[1]为 卖空仓位，初始资金为 init_cash 变量代表的数值
        init_cash = context.portfolio.starting_cash/2
        set_subportfolios([SubPortfolioConfig(cash=init_cash, type='stock'),\
                           SubPortfolioConfig(cash=init_cash, type='stock_margin')])
        
        # 设定融资利率: 年化8%
        set_option('margincash_interest_rate', 0.08)
        # 设置融资保证金比率: 150%
        set_option('margincash_margin_rate', 1.5)
        # 设定融券利率: 年化10%
        set_option('marginsec_interest_rate', 0.10)
        # 设定融券保证金比率: 150%
        set_option('marginsec_margin_rate', 1.5)

## 参数设置函数
def set_params():
    # 记录回测运行的天数
    g.days = 0
    # 当天是否交易
    g.if_trade = False                       

    ## 可变参数
    # 股票池
    # g.secCode = '000985.XSHG'  #中证全指
    g.secCode = '000300.XSHG'  # 沪深300
    # g.secCode = '000905.XSHG' #中证500
    # 调仓天数
    g.refresh_rate = 30 
    ## 机器学习算法
    # 线性回归：lr
    # 岭回归：ridge
    # 线性向量机：svr
    # 随机森林：rf
    # xgboost
    # lightgbm
    g.method = 'xgboost'
    
    ## 分组测试之用 ####
    # True:开启分组测试（g.stocknum失效,g.group有效，g.quantile有效）
    # False:关闭分组测试（g.stocknum有效，g.group有效，g.quantile失效）
    g.invest_by_group = False
    # 每组（占所有股票中的）百分比
    # g.group（MAX）* g.quantile = 1， 即包含全部分组
    g.quantile = 0.1
    # 分组
    # 第1组：1
    # 第2组：2
    # ... ...
    # 第n组：n
    g.group = 1
    # 持仓数（分组时失效）
    g.stocknum = 6
    
    #是否进行融资融券,False为不融资融券，True为融资融券
    g.short = False
    #是否进行组合优化，False为不进行组合优化，True为组合优化
    g.optimization = True

##
def set_backtest():
    # set_benchmark('000905.XSHG')   #中证500
    set_benchmark('000300.XSHG')  # 沪深300
    set_option('use_real_price', True)
    log.set_level('order', 'error')
    
## 保存不能被序列化的对象, 进程每次重启都初始化,
def process_initialize(context):
    
    # 交易次数记录
    g.__tradeCount = 0
    # 删除建仓日或者重新建仓日停牌的股票后剩余的可选股
    g.__feasible_stocks = [] 
    # 网格搜索是否开启
    g.__gridserach = False

    ## 机器学习验证集及测试集评分记录之用（实际交易策略中不需要，请设定为False）#####
    # True：开启（写了到研究模块，文件名：score.json）
    # False：关闭
    g.__scoreWrite = False
    g.__valscoreSum = 0
    g.__testscoreSum = 0
    ## 机器学习验证集及测试集评分记录之用（实际交易策略中不需要，请设定为False）#####

    # 训练集长度
    g.__trainlength = 4
    # 训练集合成间隔周期（交易日）
    g.__intervals = 21
    
    # 离散值处理列表
    g.__winsorizeList = []

    # 标准化处理列表
    g.__standardizeList = ['log_mcap',
                           'leverage',
                           'book_to_price_ratio',
                           'circulating_market_cap',
                           'fixed_asset_ratio',
                           'net_profit_to_total_operate_revenue_ttm',
                           'gross_income_ratio',
                           'roa_ttm',
                           'PEG',
                           'operating_revenue_growth_rate',
                           'net_profit_growth_rate']
                        
    # 聚宽一级行业
    g.__industry_set = ['HY001', 'HY002', 'HY003', 'HY004', 'HY005', 'HY006', 'HY007', 'HY008', 'HY009', 
          'HY010', 'HY011']

    # 因子组合
    g.__factorList = ['size',
                     'leverage',
                     'book_to_price_ratio',
                     'circulating_market_cap',
                     'fixed_asset_ratio',
                     'net_profit_to_total_operate_revenue_ttm',
                     'gross_income_ratio',
                     'roa_ttm',
                     'PEG',
                     'operating_revenue_growth_rate',
                     'net_profit_growth_rate',
                     'total_asset_growth_rate',
                     'net_operate_cashflow_growth_rate',
                     'total_profit_growth_rate',
                     'total_operating_revenue_ttm',
                     'operating_profit_ttm',
                     'gross_profit_ttm',
                     'EBIT',
                     'net_profit_ttm',
                     'market_cap',
                     'cash_flow_to_price_ratio',
                     'sales_to_price_ratio',
                     'net_profit_ratio',
                     'quick_ratio',
                     'current_ratio',
                     'operating_profit_ratio',
                     'SGI',
                     'roe_ttm',
                     'VOL10',
                     'VOL20',
                     'VOL60',
                     'VOL120',
                     'AR',
                     'BR',
                     'ARBR',
                     'VEMA5',
                     'DAVOL10',
                     'DAVOL5',
                     'Variance20',
                     'Variance60',
                     'Variance120',
                     'ATR6',
                     'ATR14',
                     'total_operating_revenue_per_share',
                     'eps_ttm',
                     'BIAS10',
                     'BIAS20',
                     'BIAS60',
                     'Volume1M',
                     'momentum',
                     'liquidity',
                     'earnings_yield',
                     'growth',
                    'HY001', 'HY002', 'HY003', 'HY004', 'HY005', 'HY006', 'HY007', 'HY008', 'HY009', 'HY010', 'HY011']

'''
================================================================================
每天开盘前
================================================================================
'''
#每天开盘前要做的事情
def before_trading_start(context):
    # 当天是否交易
    g.if_trade = False                       
    # 每g.refresh_rate天，调仓一次
    if g.days % g.refresh_rate == 0:
        g.if_trade = True                           
        # 设置手续费与手续费
        set_slip_fee(context)                       
        # 设置初始股票池
        sample = get_index_stocks(g.secCode)
        # 设置可交易股票池
        #g.feasible_stocks = set_feasible_stocks(sample,context)
        g.__feasible_stocks = set_feasible_stocks(sample,context)
        # 因子获取Query
        g.__q = get_q_Factor(g.__feasible_stocks)
    g.days += 1

#5
# 根据不同的时间段设置滑点与手续费
# 输入：context（见API）
# 输出：none
def set_slip_fee(context):
    # 将滑点设置为0
    set_slippage(FixedSlippage(0)) 
    # 根据不同的时间段设置手续费
    dt = context.current_dt
    if dt > datetime.datetime(2013,1, 1):
        set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, close_commission=0.0003, close_today_commission=0, min_commission=5), type='stock')
    elif dt > datetime.datetime(2008,9, 18):
        set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.001, close_commission=0.002, close_today_commission=0, min_commission=5), type='stock')
    elif dt > datetime.datetime(2008,4, 24):
        set_order_cost(OrderCost(open_tax=0.001, close_tax=0.001, open_commission=0.002, close_commission=0.003, close_today_commission=0, min_commission=5), type='stock')
    else:
        set_order_cost(OrderCost(open_tax=0.003, close_tax=0.003, open_commission=0.003, close_commission=0.004, close_today_commission=0, min_commission=5), type='stock')

#4
# 设置可行股票池：过滤掉当日停牌的股票
# 输入：initial_stocks为list类型,表示初始股票池； context（见API）
# 输出：unsuspened_stocks为list类型，表示当日未停牌的股票池，即：可行股票池
def set_feasible_stocks(initial_stocks,context):
    # 判断初始股票池的股票是否停牌，返回list
    paused_info = []
    current_data = get_current_data()
    for i in initial_stocks:
        paused_info.append(current_data[i].paused)
    df_paused_info = pd.DataFrame({'paused_info':paused_info},index = initial_stocks)
    unsuspened_stocks =list(df_paused_info.index[df_paused_info.paused_info == False])
    return unsuspened_stocks

'''
================================================================================
每天交易时
================================================================================
'''
# 每天回测时做的事情
def handle_data(context,data):
    if g.if_trade == True:
        # 记录交易次数
        g.__tradeCount = g.__tradeCount + 1

        # 训练集合成
        yesterday = context.previous_date

        df_train = get_df_train(g.__q,yesterday,g.__trainlength,g.__intervals) # 之前的数据
        # print('df_train',df_train.columns)
        # df_train = initialize_df(df_train)
        df_train = indicator_df(df_train,yesterday,g.__intervals)
        #print('df_train_new',df_train_new)
        
        # print('df_train',df_train)
        

        # T日截面数据（测试集）
        df = get_fundamentals(g.__q, date = None) # 昨天的数据
        # print(df)
        # df = initialize_df(df)
        df = indicator_df(df,yesterday,g.__intervals)
        

        # 离散值处理
        for fac in g.__winsorizeList:
            df_train[fac] = winsorize_med(df_train[fac], scale=5, inclusive=True, inf2nan=True, axis=0)    
            df[fac] = winsorize_med(df[fac], scale=5, inclusive=True, inf2nan=True, axis=0)    
        
        # 标准化处理        
        for fac in g.__standardizeList:
            df_train[fac] = standardlize(df_train[fac], inf2nan=True, axis=0)
            df[fac] = standardlize(df[fac], inf2nan=True, axis=0)

        # 中性化处理（行业中性化）
        df_train = neutralize(df_train,g.__industry_set)
        df = neutralize(df,g.__industry_set)

        #训练集（包括验证集）
        X_trainval = df_train[g.__factorList]
        X_trainval = X_trainval.fillna(0)
        
        #定义机器学习训练集输出
        y_trainval = df_train[['log_mcap']]
        y_trainval = y_trainval.fillna(0)
 
        #测试集
        X = df[g.__factorList]
        X = X.fillna(0)
        
        #定义机器学习测试集输出
        y = df[['log_mcap']]
        y.index = df['code']
        y = y.fillna(0)
 
        kfold = KFold(n_splits=4)
        if g.__gridserach == False:
            #不带网格搜索的机器学习
            if g.method == 'svr': #SVR
                from sklearn.svm import SVR
                model = SVR(C=100, gamma=1)
            elif g.method == 'lr':
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            elif g.method == 'ridge': #岭回归
                from sklearn.linear_model import Ridge
                model = Ridge(random_state=42,alpha=100)
            elif g.method == 'rf': #随机森林
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(random_state=42,n_estimators=500,n_jobs=-1)
            elif g.method == 'xgboost':  # 随机森林
                from xgboost import XGBRegressor
                model = XGBRegressor(booster='gbtree',
                                     n_estimators=500,
                                     learning_rate=0.1,
                                     min_child_weight=100,
                                     eta=0.02,
                                     max_depth=16,
                                     gamma=0.7,
                                     subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear',
                                     alpha=1,
                                     silent=1,
                                     verbose_eval=True,
                                     seed=12)
            elif g.method == 'lightgbm':
                import lightgbm as lgb
                model = lgb.LGBMRegressor()
            else:
                g.__scoreWrite = False
        else:
            # 带网格搜索的机器学习
            para_grid = {}
            if g.method == 'svr':
                from sklearn.svm import SVR  
                para_grid = {'C':[10,100],'gamma':[0.1,1,10]}
                grid_search_model = SVR()
            elif g.method == 'lr':
                from sklearn.linear_model import LinearRegression
                grid_search_model = LinearRegression()
            elif g.method == 'ridge':
                from sklearn.linear_model import Ridge
                para_grid = {'alpha':[1,10,100]}
                grid_search_model = Ridge()
            elif g.method == 'rf':
                from sklearn.ensemble import RandomForestRegressor
                para_grid = {'n_estimators':[100,500,1000]}
                grid_search_model = RandomForestRegressor()
            elif g.method == 'xgboost':  # 随机森林
                from xgboost import XGBRegressor
                para_grid = {'max_depth': [8, 12, 16]}
                grid_search_model = XGBRegressor()
            elif g.method == 'lightgbm':
                import lightgbm as lgb
                para_grid = {'max_depth': [8, 12, 16]}
                grid_search_model = lgb.LGBMRegressor()
            else:
                g.__scoreWrite = False
    
            from sklearn.model_selection import GridSearchCV
            model = GridSearchCV(grid_search_model, para_grid, cv=kfold, n_jobs=-1)
        
        # 拟合训练集，生成模型
        # print(X_trainval)
        model.fit(X_trainval,y_trainval)


        # 将特征打分与特征名称放进一个dataframe
        df_feature = pd.DataFrame({'factor':g.__factorList,'score':model.feature_importances_})
        # 按得分大小降序
        df_feature = df_feature.sort_values(by='score',ascending=False)
        # 输出模型特征打分
        # print("特征打分", df_feature)

        # 预测值
        y_pred = model.predict(X)

        # 新的因子：实际值与预测值之差    
        factor = y - pd.DataFrame(y_pred, index = y.index, columns = ['log_mcap'])
        
        # 对新的因子，即残差进行排序（按照从小到大）
        factor = factor.sort_index(by = 'log_mcap')

        # 区分上涨和下跌股票
        # 定位上涨股票
        factor_up = factor[factor['log_mcap'] < 0]
        # 定位下跌股票
        factor_down = factor[factor['log_mcap'] > 0]

        ###  分组测试用 ##############
        if g.invest_by_group == True:
            len_secCodeList = len(list(factor.index))
            g.stocknum = int(len_secCodeList * g.quantile)
        ###  分组测试用 ##############

        start = g.stocknum * (g.group-1)
        end = g.stocknum * g.group  # g.stocknum 设定的持仓数
        
        stockset = list(factor_up.index[start:end])  # 买入列表
        stockset_margin = list(factor_down.index[-end:])  # 卖出列表
        current_data = get_current_data()

        #优化组合权重
        if g.optimization == True:
            weight = portfolio_optimizer(date = yesterday, securities = stockset, target = MinVariance(count=250), 
            constraints = [WeightEqualConstraint(limit=1.0), WeightConstraint(low=0.0,high=1.0)],bounds = Bound(low=0.0, high=1.0),
            default_port_weight_range=[0.0, 1.0], 
            ftol=1e-9, 
            return_none_if_fail=False)
        else:
            weight = 1/len(stockset)
        amount = 1000 #设定卖空的固定股票数

       #卖出and卖空平仓
        if g.short == True:
            sell_list = list(context.subportfolios[0].long_positions.keys())
            stock_marginlist = list(context.subportfolios[1].short_positions.keys())
        else:
            sell_list = list(context.portfolio.positions.keys())
        
        for stock in sell_list:
            if stock not in stockset:
                if stock in g.__feasible_stocks:
                    if current_data[stock].last_price == current_data[stock].high_limit:#价格涨停
                        pass
                    else:
                        stock_sell = stock
                        order_target_value(stock_sell, 0,pindex=0)
        if g.short == True:               
            for stock in stock_marginlist:
                if stock in g.__feasible_stocks:
                    if current_data[stock].last_price == current_data[stock].low_limit:#价格跌停
                        pass
                    else:
                        marginsec_close(stock,amount,pindex=1)
                        print("卖空平仓"+stock)
                    

        #分配买入资金    
        if len(context.portfolio.positions) < g.stocknum:
            num = g.stocknum - len(context.portfolio.positions)
            if short == True:
                cash = context.portfolio.cash/2 #当需要融资融券时，一半用来买入卖出，一半卖空
            else:
                cash = context.portfolio.cash 
        else:
            cash = 0
            num = 0
            
        
        #买入and卖空
        for stock in stockset[:g.stocknum]:
            # 判断是否有组合优化
            if g.optimization == True:
                cash_stock = cash*weight[stock]
            else:
                cash_stock = cash*weight
                
            if stock in sell_list:
                pass
            else:
                if current_data[stock].last_price == current_data[stock].low_limit: #价格跌停
                   pass
                else:#买入
                    stock_buy = stock
                    order_target_value(stock_buy, cash_stock,pindex=0)
                    num = num - 1
                    if num == 0:
                        break   
        if g.short == True:
            for stock in stockset_margin[:g.stocknum]:
                if stock in stock_marginlist:
                    pass
                else:
                    if current_data[stock].last_price == current_data[stock].high_limit: #价格涨停
                        pass
                    else:#卖空
                        if stock in get_marginsec_stocks():
                            stock_margin = stock
                            marginsec_open(stock_margin, amount, style=None, pindex=1)
                            print("卖空"+stock)


# 获取初始特征值
def get_q_Factor(feasible_stocks):
    q = query(valuation.code, 
          valuation.market_cap,#市值
          valuation.circulating_market_cap,
          balance.total_assets - balance.total_liability,#净资产
          balance.total_assets / balance.total_liability, 
          indicator.net_profit_to_total_revenue, #净利润/营业总收入
          indicator.inc_revenue_year_on_year,  #营业收入增长率（同比）
          balance.development_expenditure, #RD
          valuation.pe_ratio, #市盈率（TTM）
          valuation.pb_ratio, #市净率（TTM）
          indicator.inc_net_profit_year_on_year,#净利润增长率（同比）
          balance.dividend_payable,
          indicator.roe,
          indicator.roa,
          income.operating_profit / income.total_profit, #OPTP
          indicator.gross_profit_margin, #销售毛利率GPM
          balance.fixed_assets / balance.total_assets, #FACR
          valuation.pcf_ratio, #CFP
          valuation.ps_ratio #PS
        ).filter(
            valuation.code.in_(feasible_stocks)
        )
    return q
    
# 训练集长度设置
def get_df_train(q,d,trainlength,interval):
    
    
    date1 = shift_trading_day(d,interval)
    # print('date1',date1)
    date2 = shift_trading_day(d,interval*2)
    date3 = shift_trading_day(d,interval*3)

    d1 = get_fundamentals(q, date = date1)
    d2 = get_fundamentals(q, date = date2)
    d3 = get_fundamentals(q, date = date3)

    if trainlength == 1:
        df_train = d1
    elif trainlength == 3:
        # 3个周期作为训练集    
        df_train = pd.concat([d1, d2, d3],ignore_index=True)
    elif trainlength == 4:
        date4 = shift_trading_day(d,interval*4)
        d4 = get_fundamentals(q, date = date4)
        # 4个周期作为训练集    
        df_train = pd.concat([d1, d2, d3, d4],ignore_index=True)
    elif trainlength == 6:
        date4 = shift_trading_day(d,interval*4)
        date5 = shift_trading_day(d,interval*5)
        date6 = shift_trading_day(d,interval*6)

        d4 = get_fundamentals(q, date = date4)
        d5 = get_fundamentals(q, date = date5)
        d6 = get_fundamentals(q, date = date6)

        # 6个周期作为训练集
        df_train = pd.concat([d1,d2,d3,d4,d5,d6],ignore_index=True)
    elif trainlength == 9:
        date4 = shift_trading_day(d,interval*4)
        date5 = shift_trading_day(d,interval*5)
        date6 = shift_trading_day(d,interval*6)
        date7 = shift_trading_day(d,interval*7)
        date8 = shift_trading_day(d,interval*8)
        date9 = shift_trading_day(d,interval*9)

        d4 = get_fundamentals(q, date = date4)
        d5 = get_fundamentals(q, date = date5)
        d6 = get_fundamentals(q, date = date6)
        d7 = get_fundamentals(q, date = date7)
        d8 = get_fundamentals(q, date = date8)
        d9 = get_fundamentals(q, date = date9)
    
        # 9个周期作为训练集
        df_train = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9],ignore_index=True)
    else:
        pass
    # print(df_train.columns)
    return df_train

# 某一日的前shift个交易日日期 
# 输入：date为datetime.date对象(是一个date，而不是datetime)；shift为int类型
# 输出：datetime.date对象(是一个date，而不是datetime)
def shift_trading_day(date,shift):
    # 获取所有的交易日，返回一个包含所有交易日的 list,元素值为 datetime.date 类型.
    tradingday = get_all_trade_days()
    # 得到date之后shift天那一天在列表中的行标号 返回一个数
    shiftday_index = list(tradingday).index(date) - shift
    # 根据行号返回该日日期 为datetime.date类型
    return tradingday[shiftday_index]


# 因子生成
def indicator_df(df,d,interval):
    # 设置时间点（可以放到if里面去生成，但是代码没这么好看）
    date1 = shift_trading_day(d,interval)
    date2 = shift_trading_day(d,interval*2)
    date3 = shift_trading_day(d,interval*3)
    date4 = shift_trading_day(d,interval*4)
    date5 = shift_trading_day(d,interval*5)
    date6 = shift_trading_day(d,interval*6)
    date7 = shift_trading_day(d,interval*7)
    date8 = shift_trading_day(d,interval*8)
    date9 = shift_trading_day(d,interval*9)
    # 设置因子列表
    factors = ['size',
                 'leverage',
                 'book_to_price_ratio',
                 'circulating_market_cap',
                 'fixed_asset_ratio',
                 'net_profit_to_total_operate_revenue_ttm',
                 'gross_income_ratio',
                 'roa_ttm',
                 'PEG',
                 'operating_revenue_growth_rate',
                 'net_profit_growth_rate',
                 'total_asset_growth_rate',
                 'net_operate_cashflow_growth_rate',
                 'total_profit_growth_rate',
                 'total_operating_revenue_ttm',
                 'operating_profit_ttm',
                 'gross_profit_ttm',
                 'EBIT',
                 'net_profit_ttm',
                 'market_cap',
                 'cash_flow_to_price_ratio',
                 'sales_to_price_ratio',
                 'net_profit_ratio',
                 'quick_ratio',
                 'current_ratio',
                 'operating_profit_ratio',
                 'SGI',
                 'roe_ttm',
                 'VOL10',
                 'VOL20',
                 'VOL60',
                 'VOL120',
                 'AR',
                 'BR',
                 'ARBR',
                 'VEMA5',
                 'DAVOL10',
                 'DAVOL5',
                 'Variance20',
                 'Variance60',
                 'Variance120',
                 'ATR6',
                 'ATR14',
                 'total_operating_revenue_per_share',
                 'eps_ttm',
                 'BIAS10',
                 'BIAS20',
                 'BIAS60',
                 'Volume1M',
                 'momentum',
                 'liquidity',
                 'earnings_yield',
                 'growth']
    
    # 对每个时间节点进行拆分，分别获取因子，然后进行组合
    codes = df['code'].drop_duplicates()
    if g.__trainlength == 1:
        # 先获取因子，然后字典转换为dataframe
        factor_data = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date1, end_date=date1))

    elif g.__trainlength == 3:
        factor_data_1 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date1, end_date=date1))
        factor_data_2 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date2, end_date=date2))
        factor_data_3 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date3, end_date=date3))
        factor_data = pd.concat([factor_data_1, factor_data_2,factor_data_3], axis=0, ignore_index=True)
    
    elif g.__trainlength == 4:
        factor_data_1 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date1, end_date=date1))
        factor_data_2 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date2, end_date=date2))
        factor_data_3 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date3, end_date=date3))
        factor_data_4 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date4, end_date=date4))
        factor_data = pd.concat([factor_data_1, factor_data_2,factor_data_3,factor_data_4], axis=0, ignore_index=True)
    
    elif g.__trainlength == 6:
        factor_data_1 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date1, end_date=date1))
        factor_data_2 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date2, end_date=date2))
        factor_data_3 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date3, end_date=date3))
        factor_data_4 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date4, end_date=date4))
        factor_data_5 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date5, end_date=date5))
        factor_data_6 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date6, end_date=date6))
        factor_data = pd.concat([factor_data_1, factor_data_2,factor_data_3,factor_data_4,factor_data_5,factor_data_6], axis=0, ignore_index=True)
        
    elif g.__trainlength == 9:
        factor_data_1 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date1, end_date=date1))
        factor_data_2 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date2, end_date=date2))
        factor_data_3 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date3, end_date=date3))
        factor_data_4 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date4, end_date=date4))
        factor_data_5 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date5, end_date=date5))
        factor_data_6 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date6, end_date=date6))
        factor_data_7 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date7, end_date=date7))
        factor_data_8 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date8, end_date=date8))
        factor_data_9 = dict_to_dataframe(get_factor_values(securities=codes.tolist(), factors=factors, start_date=date9, end_date=date9))
        factor_data = pd.concat([factor_data_1, factor_data_2,factor_data_3,factor_data_4,factor_data_5,factor_data_6,factor_data_7,factor_data_8,factor_data_9], axis=0, ignore_index=True)
    else:
        pass
    
    # 新建一个DataFrame，存储结果
    new_df = pd.DataFrame()
    new_df['code'] = df['code']
    new_df['log_mcap'] = np.log(df['market_cap'])
    for f in factors:
        new_df[f] = factor_data[f]
    new_df = new_df.fillna(0)
    del factor_data
    return new_df

# 字典转DataFrame
def dict_to_dataframe(dic):
    # 新建一个dataframe
    df = pd.DataFrame()
    for key in dic.keys():
        df[key] = dic[key].iloc[0,:].values
    return df

    
# 中性化
def neutralize(df,industry_set):
    lenth = df.shape[1]

    for i in range(len(industry_set)):
        s = pd.Series([0]*len(df), index=df.index)
        df[industry_set[i]] = s
        industry = get_industry_stocks(industry_set[i])
        for j in range(len(df)):
            if df.iloc[j,0] in industry:
                df.iloc[j,lenth + i] = 1
            
    return df    
    
'''
================================================================================
每天收盘后
================================================================================
'''
# 每天收盘后做的事情
# 进行长运算（本策略中不需要）
def after_trading_end(context):
    return