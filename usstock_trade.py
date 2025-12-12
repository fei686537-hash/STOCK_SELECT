# -*- coding: utf-8 -*-
from futu import *
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import pickle

############################ 1. 全局配置 ############################
FUTUOPEND_ADDRESS = '127.0.0.1'
FUTUOPEND_PORT = 11111

TRADING_ENVIRONMENT = TrdEnv.SIMULATE  # TrdEnv.REAL (真实) / TrdEnv.SIMULATE (模拟)
TRADING_MARKET = TrdMarket.US 
TRADING_PWD = '123456'
TRADING_PERIOD = KLType.K_1M 

save_dir = os.path.join(os.getcwd(), 'result')

# === M7 财报日期 (示例) ===
M7_EARNINGS_DATES = [
    '2024-05-01', '2024-05-05', '2024-05-10'
]

# === 策略参数 ===
STRATEGY_CONFIG = {
    'MAX_GLOBAL_HOLDINGS': 10,    # 最大持仓只数
    'MAX_BREAK_ATTEMPTS': 2,      # 突破策略最大尝试次数
    'STOP_LOSS_FIXED': 0.01,      # 1% 固定止损 (跌破买入价/均线)
    'TP1_THRESHOLD': 0.10,        # 10% 止盈触发线
    'TP2_THRESHOLD': 0.30,        # 30% 止盈触发线
    'TP_BUFFER': 0.01,            # 均线下方 1% 止盈
    'BREAK_VOL_RATIO': 2.0,       # 突破量比阈值 > 2.0
    'EOD_CHECK_TIME': "15:55",    # 尾盘检查时间
    'RE_ENTRY_COOLDOWN': 5,       # 止损后冷却时间(分钟)，防止瞬间重复买入
    # === 低吸策略专用参数 ===
    'DIP_STOP_LOSS': 0.03,        # 3% 强制止损
    'DIP_TP1_BUFFER': 0.002,      # 10%盈利后，5日线下方0.2%止盈
    'DIP_TP2_BUFFER': 0.002,      # 30%盈利后，10日线下方0.2%止盈
    # === 回调策略专用参数 ===
    'PULLBACK_STOP_LOSS': 0.03,   # 3% 强制止损
    'PULLBACK_TP1_BUFFER': 0.002, # 10%盈利后，5日线下方0.2%止盈
    'PULLBACK_TP2_BUFFER': 0.002, # 30%盈利后，10日线下方0.2%止盈
}

# === 全局数据容器 ===
CTX_DATA = {}
signals = None



def df_to_settings(df, market_prefix='US'):
    """
    将 base_snap DataFrame 转换为策略配置字典
    :param df: 传入的 DataFrame (base_snap)
    :param market_prefix: 股票市场前缀，美股为 'US'，港股为 'HK'
    :return: 格式化后的 STOCK_SETTINGS 字典
    """
    settings = {}
    
    # 如果 symbol 在索引中而不是列中，先重置索引
    if 'symbol' not in df.columns and df.index.name == 'symbol':
        df = df.reset_index()
    elif 'symbol' not in df.columns and 'symbol' in df.index.names:
        df = df.reset_index()

    for index, row in df.iterrows():
        # 1. 检查是否在股票池中
        if 'in_pool' in row and not row['in_pool']:
            continue

        # 2. 构建策略列表
        strategies = []
        if row.get('low_buy_candidate', False):  # 如果是低吸候选
            strategies.append('DIP')
        if row.get('breakout_candidate', False): # 如果是突破候选
            strategies.append('BREAK')
        if row.get('pullback_candidate', False): # 如果是回调候选
            strategies.append('PULLBACK')
        
        # 3. 只有当存在允许的策略时才加入配置
        if strategies:
            # 拼接富途格式代码，例如 US.TSLA
            full_code = f"{market_prefix}.{row['symbol']}"
            settings[full_code] = strategies
            
    return settings

def get_demo_stock(demo):
    current_date = datetime.now().strftime("%Y-%m-%d")

    if demo == 'HK':
        hk_data = {
            'time': [current_date] * 5,
            'symbol': [
                '00700',  # 腾讯控股 
                '09988',  # 阿里巴巴 
                '03690',  # 美团 
                '01211',  # 比亚迪股份 
                '00005'   # 汇丰控股 
            ],
            'in_pool': [True, True, True, True, True],        
            'low_buy_candidate': [True, True, False, True, False],         
            'breakout_candidate': [True, False, True, True, True] 
            }
        signals = pd.DataFrame(hk_data)

        # === 股票池配置 ===
        # 格式: '股票代码': ['允许的策略类型']
        return df_to_settings(signals, market_prefix='HK')
    
    if demo == 'US':
        us_data = {
            'time': [current_date] * 5,
            'symbol': [
                'AAPL',  # 苹果 
                'MSFT',  # 微软 
                'GOOGL', # 谷歌 
                'AMZN',  # 亚马逊 
                'TSLA'   # 特斯拉 
            ],
            'in_pool': [True, True, True, True, True],
            'low_buy_candidate': [True, True, False, True, False],
            'breakout_candidate': [True, False, True, True, True]
        }
        signals = pd.DataFrame(us_data)

        # === 股票池配置 ===
        # 格式: '股票代码': ['允许的策略类型']
        return df_to_settings(signals, market_prefix='US')


############################ 2. 上下文对象 ############################
quote_context = OpenQuoteContext(host=FUTUOPEND_ADDRESS, port=FUTUOPEND_PORT)
trade_context = OpenSecTradeContext(filter_trdmarket=TRADING_MARKET, host=FUTUOPEND_ADDRESS, port=FUTUOPEND_PORT, security_firm=SecurityFirm.FUTUSECURITIES)

############################ 3. 辅助逻辑函数 ############################

def is_m7_blackout():
    """M7财报避险检查：第一只财报前10天开始停止买入"""
    if not M7_EARNINGS_DATES: return False
    today = datetime.now().date()
    dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in M7_EARNINGS_DATES]
    start_date = min(dates) - timedelta(days=10)
    end_date = max(dates)
    return start_date <= today <= end_date

def get_holding_count():
    """获取当前持仓股票数量"""
    count = 0
    for code in CTX_DATA:
        if len(CTX_DATA[code]['positions']) > 0:
            count += 1
    return count

def get_worst_position_to_swap():
    """仓位满时，找到表现最差(盈亏比最低)的持仓用于换仓"""
    worst_code = None
    worst_pnl = 999.0
    worst_pos_idx = -1
    
    for code, data in CTX_DATA.items():
        if not data['positions']: continue
        # 假设每只股票只持有一个主仓位，取第一个
        pos = data['positions'][0]
        curr_price = data['last_price']
        if curr_price <= 0: continue
        
        pnl = (curr_price - pos['entry']) / pos['entry']
        if pnl < worst_pnl:
            worst_pnl = pnl
            worst_code = code
            worst_pos_idx = 0
            
    return worst_code, worst_pos_idx, worst_pnl

def refresh_static_data(code):
    """
    每日更新静态数据：
    1. 计算静态均线 (MA5, MA10, MA20)
    2. 准备昨日成交量 (用于低吸)
    3. 准备5日平均成交量 (用于突破)
    4. 重置当日标志位
    """
    print(f"🔄 [数据刷新] 正在更新 {code} 的静态数据...")
    ctx = CTX_DATA[code]
    
    # 重置当日标志位
    ctx['flags'] = {
        'dip_stopped_today': False, 
        'break_fail_count': 0, 
        'break_stopped_today': False,
        'pullback_stopped_today': False,  # 回调策略当日停止标志
        'pullback_bought_today': False,   # 回调策略当日已买入标志
        'last_sell_time': None # 用于冷却
    }
    
    # 1. 获取日K历史计算静态均线
    ret, k_day, page_req_key = quote_context.request_history_kline(code, start='2025-01-01', end=datetime.now().strftime("%Y-%m-%d"), ktype=SubType.K_DAY)
    if ret == RET_OK and not k_day.empty:
        # 必须排除今天(如果已开盘)，只用昨天及以前的数据
        today_str = datetime.now().strftime("%Y-%m-%d")
        hist_k = k_day[k_day['time_key'].str.contains(today_str) == False]
        
        if len(hist_k) >= 30:
            closes = hist_k['close'].values
            highs = hist_k['high'].values
            lows = hist_k['low'].values
            ctx['daily_mas']['ma5'] = closes[-5:].mean()
            ctx['daily_mas']['ma10'] = closes[-10:].mean()
            ctx['daily_mas']['ma20'] = closes[-20:].mean()
            ctx['daily_mas']['ma30'] = closes[-30:].mean()
            ctx['prev_close'] = closes[-1]
            # === 回调策略需要的数据 ===
            ctx['prev_high'] = highs[-1]     # 昨日最高价
            ctx['prev_low'] = lows[-1]       # 昨日最低价
            ctx['prev2_low'] = lows[-2] if len(lows) >= 2 else lows[-1]  # 前日最低价
            print(f"   [{code}] 静态MA30: {ctx['daily_mas']['ma30']:.2f} | 昨收: {ctx['prev_close']} | 昨高: {ctx['prev_high']} | 前日低: {ctx['prev2_low']}")
        else:
            print(f"⚠️ [{code}] 历史数据不足30天，无法计算MA30")

    # 2. 获取1分钟K线，分离成交量数据
    # 获取最近2000根1分钟K线
    ret, k_1m = quote_context.get_cur_kline(code, 2000, SubType.K_1M)
    if ret == RET_OK and not k_1m.empty:
        k_1m['date'] = k_1m['time_key'].apply(lambda x: x.split(' ')[0])
        k_1m['time'] = k_1m['time_key'].apply(lambda x: x.split(' ')[1][:5]) # HH:MM
        
        unique_dates = sorted(k_1m['date'].unique())
        today_str = datetime.now().strftime("%Y-%m-%d")
        # 排除今天
        past_dates = [d for d in unique_dates if d != today_str]
        
        if past_dates:
            # A. 提取昨日数据 (用于低吸策略对比)
            yesterday_date = past_dates[-1]
            yesterday_df = k_1m[k_1m['date'] == yesterday_date]
            ctx['yesterday_vol_map'] = dict(zip(yesterday_df['time'], yesterday_df['volume']))
            
            # B. 提取过去5日数据 (用于突破策略量比)
            target_dates = past_dates[-5:]
            avg_df = k_1m[k_1m['date'].isin(target_dates)]
            # 按时间分组求平均
            avg_vol_series = avg_df.groupby('time')['volume'].mean()
            ctx['avg_vol_map'] = avg_vol_series.to_dict()
            
            print(f"   [{code}] 量能数据: 昨日({yesterday_date}) + 5日均量({len(target_dates)}天)")
    
    ctx['last_update_date'] = datetime.now().date()

############################ 4. 交易执行 ############################

def execute_buy(code, price, strategy_type, breakout_ref=0.0):
    # M7 避险检查
    if is_m7_blackout():
        print(f"⚠️ [M7避险] 财报期间，暂停买入: {code}")
        return

    # 冷却期检查 (防止止损后立刻买入)
    last_sell = CTX_DATA[code]['flags']['last_sell_time']
    if last_sell:
        if (datetime.now() - last_sell).seconds < STRATEGY_CONFIG['RE_ENTRY_COOLDOWN'] * 60:
            return

    # 仓位管理：满仓则换仓
    if get_holding_count() >= STRATEGY_CONFIG['MAX_GLOBAL_HOLDINGS']:
        w_code, w_idx, w_pnl = get_worst_position_to_swap()
        if w_code:
            print(f"⚖️ [换仓] 卖出最差持仓: {w_code} (当前盈亏 {w_pnl*100:.2f}%)")
            execute_sell(w_code, w_idx, CTX_DATA[w_code]['last_price'], "force_swap")
            time.sleep(1) # 等待卖出成交
        else:
            print("⚠️ 仓位已满且无法换仓")
            return

    # 计算买入数量 (总资产/10)
    ret, data = trade_context.accinfo_query(trd_env=TRADING_ENVIRONMENT, acc_index=0)
    if ret != RET_OK: return
    total_assets = data['total_assets'][0]
    target_val = total_assets / STRATEGY_CONFIG['MAX_GLOBAL_HOLDINGS']
    
    lot_size = CTX_DATA[code]['lot_size']
    qty = int(target_val / price / lot_size) * lot_size
    if qty == 0: return

    print(f"🚀 [买入] {code} ({strategy_type}) 价格:{price} 数量:{qty}")
    
    if TRADING_ENVIRONMENT == TrdEnv.REAL:
        trade_context.place_order(price=price, qty=qty, code=code, trd_side=TrdSide.BUY, trd_env=TRADING_ENVIRONMENT)
    
    # 记录持仓
    CTX_DATA[code]['positions'].append({
        'entry': price,
        'qty': qty,
        'type': strategy_type,
        'time': datetime.now(),
        'breakout_price': breakout_ref, # 仅突破策略使用
        'highest_pnl_pct': -1.0,
        # === 低吸策略专用字段 ===
        'dip_target_ma': CTX_DATA[code].get('dip_target_ma', 'ma20'),  # 低吸目标均线
        'dip_target_ma_price': CTX_DATA[code].get('dip_target_ma_price', 0),  # 低吸目标均线价格
        # === 回调策略专用字段 ===
        'prev2_low': CTX_DATA[code].get('prev2_low', 0),  # 买入时的前日低点(当日止损用)
        'sold_half': False,  # 是否已卖出一半(30%止盈时触发)
    })

    #把持仓记录持久化
    with open('ctx_data.pkl', 'wb') as f:
        pickle.dump(CTX_DATA, f)


def execute_sell_partial(code, pos_idx, price, sell_ratio, reason):
    """
    部分卖出：用于回调策略30%盈利时卖出一半
    :param sell_ratio: 卖出比例，如0.5表示卖出一半
    """
    if pos_idx >= len(CTX_DATA[code]['positions']): return
    
    pos = CTX_DATA[code]['positions'][pos_idx]
    sell_qty = int(pos['qty'] * sell_ratio)
    lot_size = CTX_DATA[code]['lot_size']
    sell_qty = (sell_qty // lot_size) * lot_size  # 取整到每手
    
    if sell_qty == 0: return
    
    pnl_pct = (price - pos['entry']) / pos['entry'] * 100
    print(f"🔶 [部分卖出] {code} 原因:{reason} 价格:{price} 数量:{sell_qty}/{pos['qty']} 盈亏:{pnl_pct:.2f}%")
    
    if TRADING_ENVIRONMENT == TrdEnv.REAL:
        trade_context.place_order(price=price, qty=sell_qty, code=code, trd_side=TrdSide.SELL, trd_env=TRADING_ENVIRONMENT)
    
    # 更新剩余持仓数量
    pos['qty'] -= sell_qty
    pos['sold_half'] = True
    
    #把持仓记录持久化
    with open('ctx_data.pkl', 'wb') as f:
        pickle.dump(CTX_DATA, f)


def execute_sell(code, pos_idx, price, reason):
    if pos_idx >= len(CTX_DATA[code]['positions']): return
    
    pos = CTX_DATA[code]['positions'][pos_idx]
    pnl_pct = (price - pos['entry']) / pos['entry'] * 100
    print(f"🛑 [卖出] {code} 原因:{reason} 价格:{price} 盈亏:{pnl_pct:.2f}%")
    
    if TRADING_ENVIRONMENT == TrdEnv.REAL:
        trade_context.place_order(price=price, qty=pos['qty'], code=code, trd_side=TrdSide.SELL, trd_env=TRADING_ENVIRONMENT)
    
    # 移除持仓记录
    CTX_DATA[code]['positions'].pop(pos_idx)
    CTX_DATA[code]['flags']['last_sell_time'] = datetime.now()
    
    # === 策略状态更新 (核心逻辑) ===
    
    # 1. 低吸策略止损 -> 当日停止
    if reason.startswith("dip_"):
        CTX_DATA[code]['flags']['dip_stopped_today'] = True
        print(f"🚫 {code} 低吸策略今日已终止 (触发: {reason})")
        
    # 2. 突破策略止损 -> 允许重试1次，第2次失败则停止
    elif reason == "break_stop_loss":
        CTX_DATA[code]['flags']['break_fail_count'] += 1
        fail_count = CTX_DATA[code]['flags']['break_fail_count']
        print(f"⚠️ {code} 突破策略失败次数: {fail_count}/{STRATEGY_CONFIG['MAX_BREAK_ATTEMPTS']}")
        
        if fail_count >= STRATEGY_CONFIG['MAX_BREAK_ATTEMPTS']:
            CTX_DATA[code]['flags']['break_stopped_today'] = True
            print(f"🚫 {code} 突破策略今日已终止 (达到最大失败次数)")
    
    # 3. 回调策略止损 -> 当日停止
    elif reason.startswith("pullback_"):
        CTX_DATA[code]['flags']['pullback_stopped_today'] = True
        print(f"🚫 {code} 回调策略今日已终止 (触发止损)")

    #把持仓记录持久化
    with open('ctx_data.pkl', 'wb') as f:
        pickle.dump(CTX_DATA, f)

############################ 5. 核心策略逻辑 ############################

def process_one_bar(row):
    code = row['code']
    if code not in CTX_DATA: return
    
    ctx = CTX_DATA[code]
    
    # === 每日数据自动刷新逻辑 ===
    # 如果当前K线日期与上次更新日期不同，说明跨天了，需要重新计算静态均线
    current_date = datetime.strptime(row['time_key'], "%Y-%m-%d %H:%M:%S").date()
    if ctx['last_update_date'] != current_date:
        refresh_static_data(code)
    
    close_price = row['close']
    open_price = row['open']
    low_price = row['low']
    volume = row['volume']
    time_str = row['time_key'].split(' ')[1][:5] # "HH:MM"
    
    ctx['last_price'] = close_price
    
    # === 获取静态均线 (全天固定) ===
    ma5 = ctx['daily_mas']['ma5']
    ma10 = ctx['daily_mas']['ma10']
    ma20 = ctx['daily_mas']['ma20']
    ma30 = ctx['daily_mas']['ma30']
    
    if ma30 == 0: return # 数据未准备好
    
    # ================= A. 持仓监控 (卖出逻辑) =================
    # 倒序遍历，防止删除元素影响索引
    for i in range(len(ctx['positions']) - 1, -1, -1):
        pos = ctx['positions'][i]
        is_today = pos['time'].date() == current_date
        pnl_pct = (close_price - pos['entry']) / pos['entry']
        
        # 更新最高盈亏比例
        if pnl_pct > pos['highest_pnl_pct']:
            pos['highest_pnl_pct'] = pnl_pct
        
        should_sell = False
        reason = ""
        
        # --- 1. 低吸持仓 (DIP) ---
        if pos['type'] == 'DIP':
            # 获取低吸目标均线价格（用于尾盘检查）
            dip_target_ma = pos.get('dip_target_ma', 'ma20')
            dip_ma_price = ctx['daily_mas'].get(dip_target_ma.replace('ma', 'ma'), ma20)
            
            if is_today:
                # 规则1: 跌破买入点3%强制卖出
                if close_price < pos['entry'] * (1 - STRATEGY_CONFIG['DIP_STOP_LOSS']):
                    should_sell, reason = True, "dip_stop_3pct"
                # 规则2: 尾盘检查，如果跌破目标均线则卖出
                elif datetime.now().strftime("%H:%M") >= STRATEGY_CONFIG['EOD_CHECK_TIME']:
                    if close_price < dip_ma_price:
                        should_sell, reason = True, "dip_eod_check"
            else:
                # 次日起: 跌破买入点3%强制卖出
                if close_price < pos['entry'] * (1 - STRATEGY_CONFIG['DIP_STOP_LOSS']):
                    should_sell, reason = True, "dip_next_day_stop_3pct"
            
            # === 低吸策略专用止盈逻辑（使用动态均线）===
            if not should_sell:
                # 止盈规则1: >30%盈利，卖出一半，剩余设置10日线下方0.2%止盈
                if pos['highest_pnl_pct'] > STRATEGY_CONFIG['TP2_THRESHOLD']:
                    # 先检查是否已卖出一半
                    if not pos.get('sold_half', False):
                        execute_sell_partial(code, i, close_price, 0.5, "dip_tp_30pct_half")
                    # 剩余部分：跌破10日线下方0.2%全部卖出
                    if close_price < ma10 * (1 - STRATEGY_CONFIG['DIP_TP2_BUFFER']):
                        should_sell, reason = True, "dip_tp_30pct_ma10"
                # 止盈规则2: >10%盈利，设置5日线下方0.2%止盈
                elif pos['highest_pnl_pct'] > STRATEGY_CONFIG['TP1_THRESHOLD']:
                    if close_price < ma5 * (1 - STRATEGY_CONFIG['DIP_TP1_BUFFER']):
                        should_sell, reason = True, "dip_tp_10pct_ma5"
        
        # --- 2. 突破持仓 (BREAK) ---
        elif pos['type'] == 'BREAK':
            # 止损基准：使用买入时的突破点(通常即买入价)或买入价本身
            ref_price = pos['breakout_price'] if pos['breakout_price'] > 0 else pos['entry']
            
            if is_today:
                # 规则: 买入后如果跌破突破点1%就卖出
                if close_price < ref_price * (1 - STRATEGY_CONFIG['STOP_LOSS_FIXED']):
                    should_sell, reason = True, "break_stop_loss"
            else:
                # 次日: 跌破买入点1%止损
                if close_price < pos['entry'] * (1 - STRATEGY_CONFIG['STOP_LOSS_FIXED']):
                    should_sell, reason = True, "break_next_day_stop"
        
        # --- 3. 回调持仓 (PULLBACK) ---
        elif pos['type'] == 'PULLBACK':
            if is_today:
                # 当日规则1: 跌破前日低点止损
                prev2_low = pos.get('prev2_low', 0)
                if prev2_low > 0 and close_price < prev2_low:
                    should_sell, reason = True, "pullback_prev2_low_stop"
                # 当日规则2: 买入点跌3%强制止损
                elif close_price < pos['entry'] * (1 - STRATEGY_CONFIG['PULLBACK_STOP_LOSS']):
                    should_sell, reason = True, "pullback_3pct_stop"
            else:
                # 次日起: 跌破买入时前日低点止损 (这里前日低点是动态更新的ctx['prev_low'])
                # 注意：次日的"前日低点"指的是昨天的低点，即ctx['prev_low']
                if ctx.get('prev_low', 0) > 0 and close_price < ctx['prev_low']:
                    should_sell, reason = True, "pullback_prev_low_stop"
            
            # === 回调策略专用止盈逻辑 ===
            if not should_sell:
                # 止盈规则1: >30%盈利，卖出一半，剩余设置10日线下方0.2%止盈
                if pos['highest_pnl_pct'] > STRATEGY_CONFIG['TP2_THRESHOLD']:
                    # 先检查是否已卖出一半
                    if not pos.get('sold_half', False):
                        execute_sell_partial(code, i, close_price, 0.5, "pullback_tp_30pct_half")
                    # 剩余部分：跌破10日线下方0.2%全部卖出
                    if close_price < ma10 * (1 - STRATEGY_CONFIG['PULLBACK_TP2_BUFFER']):
                        should_sell, reason = True, "pullback_tp_30pct_ma10"
                # 止盈规则2: >10%盈利，设置5日线下方0.2%止盈
                elif pos['highest_pnl_pct'] > STRATEGY_CONFIG['TP1_THRESHOLD']:
                    if close_price < ma5 * (1 - STRATEGY_CONFIG['PULLBACK_TP1_BUFFER']):
                        should_sell, reason = True, "pullback_tp_10pct_ma5"
        
        # --- 4. 止盈逻辑 (仅BREAK策略，使用静态均线) ---
        if not should_sell and pos['type'] == 'BREAK':
            # 规则: >30%盈利，止盈点为10日线下方1%
            if pos['highest_pnl_pct'] > STRATEGY_CONFIG['TP2_THRESHOLD']: 
                if close_price < ma10 * (1 - STRATEGY_CONFIG['TP_BUFFER']):
                    should_sell, reason = True, "tp_30pct_ma10"
            # 规则: >10%盈利，止盈点为5日线下方1%
            elif pos['highest_pnl_pct'] > STRATEGY_CONFIG['TP1_THRESHOLD']: 
                if close_price < ma5 * (1 - STRATEGY_CONFIG['TP_BUFFER']):
                    should_sell, reason = True, "tp_10pct_ma5"
        
        if should_sell:
            execute_sell(code, i, close_price, reason)
            continue

    # ================= B. 开仓逻辑 (买入逻辑) =================
    if len(ctx['positions']) > 0: return # 已有持仓不加仓
    if is_m7_blackout(): return # 财报避险
    
    # --- 策略1: 低吸 (DIP) ---
    # 允许且今日未止损
    if 'DIP' in ctx['allowed'] and not ctx['flags']['dip_stopped_today']:
        # 新逻辑: 前一天最低价与均线(MA5/MA10/MA20/MA30)比较，找离它最近的均线作为买入触发点
        prev_low = ctx.get('prev_low', 0)
        
        if prev_low > 0 and ma5 > 0 and ma10 > 0 and ma20 > 0 and ma30 > 0:
            # 计算前日低点与各均线的距离（只考虑下方的均线）
            ma_candidates = []
            if prev_low > ma5:
                ma_candidates.append(('ma5', ma5, prev_low - ma5))
            if prev_low > ma10:
                ma_candidates.append(('ma10', ma10, prev_low - ma10))
            if prev_low > ma20:
                ma_candidates.append(('ma20', ma20, prev_low - ma20))
            if prev_low > ma30:
                ma_candidates.append(('ma30', ma30, prev_low - ma30))
            
            # 如果有下方的均线，选择距离最近的
            if ma_candidates:
                # 按距离排序，取最近的
                ma_candidates.sort(key=lambda x: x[2])
                target_ma_name, target_ma_price, _ = ma_candidates[0]
                
                # 保存目标均线信息（供买入后使用）
                ctx['dip_target_ma'] = target_ma_name
                ctx['dip_target_ma_price'] = target_ma_price
                
                # 买入条件: 当天价格触达目标均线附近，且同时段交易量小于前日
                # 触达条件: 当前最低价跌破均线，但收盘价收回均线之上
                if low_price < target_ma_price and close_price > target_ma_price:
                    # 对比【前日同时间】缩量
                    yesterday_vol = ctx['yesterday_vol_map'].get(time_str, 0)
                    
                    # 只有当昨日该分钟有量且今日量更小时才买入
                    if yesterday_vol > 0 and volume < yesterday_vol:
                        print(f"⚡ [信号] {code} 低吸: 回踩{target_ma_name.upper()}({target_ma_price:.2f})+缩量")
                        print(f"   前日低:{prev_low:.2f} | 目标均线:{target_ma_name}={target_ma_price:.2f} | 现量:{volume} < 昨日量:{int(yesterday_vol)}")
                        execute_buy(code, close_price, 'DIP')
                        return

    # --- 策略2: 突破 (BREAK) ---
    # 允许且今日未彻底停止(失败次数<2)
    if 'BREAK' in ctx['allowed'] and not ctx['flags']['break_stopped_today']:
        # 逻辑: 突破昨收 (作为突破基准)
        if close_price > ctx['prev_close']:
            # 逻辑: 量比 > 2 (对比【5日均量】)
            avg_vol_5d = ctx['avg_vol_map'].get(time_str, 1) # 默认为1防止除0
            
            vol_ratio = 0
            if avg_vol_5d > 0:
                vol_ratio = volume / avg_vol_5d
            
            if vol_ratio > STRATEGY_CONFIG['BREAK_VOL_RATIO']:
                print(f"⚡ [信号] {code} 突破: 量比 {vol_ratio:.2f} > 2.0 (现量:{volume} / 5日均量:{int(avg_vol_5d)})")
                # 记录 close_price 作为本次突破点
                execute_buy(code, close_price, 'BREAK', breakout_ref=close_price)
                return

    # --- 策略3: 回调 (PULLBACK) ---
    # 允许且今日未止损且今日未买入过
    if 'PULLBACK' in ctx['allowed'] and not ctx['flags'].get('pullback_stopped_today', False) and not ctx['flags'].get('pullback_bought_today', False):
        prev_close = ctx.get('prev_close', 0)
        
        # 逻辑: 当天开盘价比前一天收盘价高 (高开)
        # 只在开盘附近检查 (09:30-09:35 美股开盘时段)
        if time_str >= "09:30" and time_str <= "09:35":
            if prev_close > 0 and open_price > prev_close:
                print(f"⚡ [信号] {code} 回调买入: 高开突破昨收 (开盘:{open_price:.2f} > 昨收:{prev_close:.2f})")
                execute_buy(code, close_price, 'PULLBACK')
                ctx['flags']['pullback_bought_today'] = True
                return

############################ 6. 框架回调 ############################

class OnBarClass(CurKlineHandlerBase):
    def on_recv_rsp(self, rsp_pb):
        ret_code, data = super(OnBarClass, self).on_recv_rsp(rsp_pb)
        if ret_code == RET_OK:
            for index, row in data.iterrows():
                if row['k_type'] == TRADING_PERIOD:
                    process_one_bar(row)

class OnOrderClass(TradeOrderHandlerBase):
    def on_recv_rsp(self, rsp_pb):
        ret, data = super(OnOrderClass, self).on_recv_rsp(rsp_pb)
        if ret == RET_OK:
            print(f"📦 订单更新: {data['code'][0]} {data['order_status'][0]}")

############################ 7. 初始化 ############################

def init(STOCK_SETTINGS):
    if TRADING_ENVIRONMENT == TrdEnv.REAL:
        trade_context.unlock_trade(TRADING_PWD)

    if os.path.exists('ctx_data.pkl'):
        with open('ctx_data.pkl', 'rb') as f:
            global CTX_DATA
            CTX_DATA = pickle.load(f)

    codes = list(STOCK_SETTINGS.keys())
    print(f"⏳ 初始化 {len(codes)} 只股票数据...")
    
    # 获取每手股数
    ret, snap = quote_context.get_market_snapshot(codes)
    lot_map = {row['code']: row['lot_size'] for _, row in snap.iterrows()} if ret == RET_OK else {}
    
    # 订阅实时K线
    quote_context.subscribe(codes, [SubType.K_1M, SubType.TICKER])
    
    for code in codes:
        CTX_DATA[code] = {
            'allowed': STOCK_SETTINGS[code],
            'lot_size': lot_map.get(code, 100),
            'positions': [],
            'flags': {},
            'daily_mas': {'ma5': 0, 'ma10': 0, 'ma20': 0, 'ma30': 0},
            'yesterday_vol_map': {}, 
            'avg_vol_map': {},       
            'prev_close': 0.0,
            'prev_high': 0.0,    # 昨日最高价 (回调策略用)
            'prev_low': 0.0,     # 昨日最低价 (低吸/回调策略用)
            'prev2_low': 0.0,    # 前日最低价 (回调策略当日止损用)
            'dip_target_ma': 'ma20',      # 低吸目标均线名称
            'dip_target_ma_price': 0.0,   # 低吸目标均线价格
            'last_price': 0.0,
            'last_update_date': None # 用于检测跨天
        }
        
        # 首次加载静态数据
        refresh_static_data(code)

    print('✅ 策略已启动，等待行情...')
    return True

if __name__ == '__main__':
    DEMO_HK =  get_demo_stock('HK')
    DEMO_US =  get_demo_stock('US')

    # 读取昨天的选股结果（因为是昨天盘后生成的）
    # 交易日T应该读取T-1日盘后生成的选股文件
    yesterday = datetime.now() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    file_path = os.path.join(save_dir, yesterday_str + '_选股.json')
    
    # 如果昨天的文件不存在，尝试读取今天的（可能是测试场景）
    if not os.path.exists(file_path):
        today_str = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(save_dir, today_str + '_选股.json')
    
    if os.path.exists(file_path):
        signals = pd.read_json(file_path,orient='table')
        REAL_US = df_to_settings(signals,'US')
        print(f'✅ 读取选股文件: {os.path.basename(file_path)}')
        print(f'✅ 候选股票数: {len(signals)}')
        print(f'✅ 策略配置: {REAL_US}')


    #执行哪个表单就初始化哪个
    if init(REAL_US): #DEMO_US  #REAL_US
        quote_context.set_handler(OnBarClass())
        trade_context.set_handler(OnOrderClass())
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            quote_context.close()
            trade_context.close()