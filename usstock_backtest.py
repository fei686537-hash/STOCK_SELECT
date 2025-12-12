#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美股多策略回测系统
整合 usstock_select.py (选股) + usstock_trade.py (交易) + Backtrader (回测)

支持三种策略回测：
1. 低吸 (DIP) - 缩量回调买入
2. 突破 (BREAK) - 放量突破买入  
3. 回调 (PULLBACK) - 四点结构买入

使用方法：
    python usstock_backtest.py
"""

import backtrader as bt
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入选股模块
from usstock_select import (
    build_base_universe,
    add_derived_features,
    low_buy_candidates,
    breakout_buy_candidates,
    pullback_buy_candidates,
    get_data_single_stock_with_cache
)


# ============================================================
# 1. 配置参数
# ============================================================

BACKTEST_CONFIG = {
    # 回测时间范围
    'START_DATE': '2025-01-01',
    'END_DATE': '2025-12-01',
    
    # 资金配置
    'INITIAL_CASH': 100000,      # 初始资金
    'MAX_HOLDINGS': 10,          # 最大持仓股票数
    'COMMISSION': 0.001,         # 手续费 0.1%
    
    # 策略选择 (可多选)
    'ENABLE_DIP': False,         # 禁用低吸策略
    'ENABLE_BREAK': False,       # 禁用突破策略
    'ENABLE_PULLBACK': True,     # 启用回调策略
    
    # === 突破策略止损止盈 ===
    'STOP_LOSS_FIXED': 0.01,     # 1%固定止损
    'TP1_THRESHOLD': 0.10,       # 10%触发第一档止盈
    'TP2_THRESHOLD': 0.30,       # 30%触发第二档止盈
    'TP_BUFFER': 0.01,           # 均线下方1%止盈
    
    # === 低吸策略专用 (与实盘一致) ===
    'DIP_STOP_LOSS': 0.03,       # 3%强制止损
    'DIP_TP1_BUFFER': 0.002,     # 10%盈利后，5日线下方0.2%止盈
    'DIP_TP2_BUFFER': 0.002,     # 30%盈利后，10日线下方0.2%止盈
    
    # === 回调策略专用 ===
    'PULLBACK_STOP_LOSS': 0.03,  # 3%强制止损
    'PULLBACK_TP1_BUFFER': 0.002, # 5日线下方0.2%
    'PULLBACK_TP2_BUFFER': 0.002, # 10日线下方0.2%
}


# ============================================================
# 2. 自定义数据源 (Pandas -> Backtrader)
# ============================================================

class PandasData(bt.feeds.PandasData):
    """
    扩展 Backtrader 数据源，增加选股系统需要的字段
    """
    lines = (
        'ma5', 'ma10', 'ma20', 'ma50',
        'vol_ma5', 'vol_ratio',
        'tr_value', 'avg_tr_value_30',
        'swing_60', 'high_60', 'high_60_ex10',
        'turnover_value', 'float_mktcap',
        'prev_low', 'prev_high',  # 前日低点/高点（回调策略用）
        # 选股信号
        'low_buy_signal', 'breakout_signal', 'pullback_signal',
    )
    
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
        # 自定义字段映射
        ('ma5', 'ma5'),
        ('ma10', 'ma10'),
        ('ma20', 'ma20'),
        ('ma50', 'ma50'),
        ('vol_ma5', 'vol_ma5'),
        ('vol_ratio', 'vol_ratio'),
        ('tr_value', 'tr_value'),
        ('avg_tr_value_30', 'avg_tr_value_30'),
        ('swing_60', 'swing_60'),
        ('high_60', 'high_60'),
        ('high_60_ex10', 'high_60_ex10'),
        ('turnover_value', 'turnover_value'),
        ('float_mktcap', 'float_mktcap'),
        ('prev_low', 'prev_low'),
        ('prev_high', 'prev_high'),
        ('low_buy_signal', 'low_buy_signal'),
        ('breakout_signal', 'breakout_signal'),
        ('pullback_signal', 'pullback_signal'),
    )


# ============================================================
# 3. 多策略交易逻辑
# ============================================================

class MultiStrategy(bt.Strategy):
    """
    多策略交易系统
    整合低吸、突破、回调三种策略
    """
    
    params = (
        ('max_holdings', 10),
        ('stop_loss', 0.01),
        ('tp1_threshold', 0.10),
        ('tp2_threshold', 0.30),
        ('tp_buffer', 0.01),
        ('dip_stop_loss', 0.03),
        ('dip_tp1_buffer', 0.002),
        ('dip_tp2_buffer', 0.002),
        ('pullback_stop_loss', 0.03),
        ('pullback_tp1_buffer', 0.002),
        ('pullback_tp2_buffer', 0.002),
        ('enable_dip', True),
        ('enable_break', True),
        ('enable_pullback', True),
    )
    
    def __init__(self):
        # 记录每个持仓的信息
        self.positions_info = {}  # {data_name: {'entry': price, 'type': strategy, 'highest_pnl': 0, ...}}
        
        # 低吸策略目标均线信息 (临时存储)
        self.dip_target_ma_info = {}
        
        # 统计数据
        self.trade_log = []  # 完整交易记录 (买入+卖出配对)
        self.buy_records = []  # 买入记录
        self.sell_records = []  # 卖出记录
        self.total_trades = 0
        self.winning_trades = 0
        
        # 追踪当天已提交的订单金额（用于防止超买）
        self.pending_buy_value = 0
        self.last_order_date = None
        
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'[{dt}] {txt}')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            current_date = self.datas[0].datetime.date(0)
            if order.isbuy():
                self.log(f'🚀 买入 {order.data._name}: 价格={order.executed.price:.2f}, 数量={order.executed.size:.0f}')
                # 更新实际成交价格
                name = order.data._name
                if name in self.positions_info:
                    self.positions_info[name]['entry'] = order.executed.price
                    self.positions_info[name]['date'] = current_date
                    strategy_type = self.positions_info[name].get('type', 'UNKNOWN')
                else:
                    strategy_type = 'UNKNOWN'
                
                # 记录买入
                self.buy_records.append({
                    '日期': str(current_date),
                    '股票': name,
                    '策略': strategy_type,
                    '买入价': order.executed.price,
                    '数量': int(order.executed.size),
                    '金额': order.executed.price * order.executed.size,
                })
            else:
                self.log(f'🛑 卖出 {order.data._name}: 价格={order.executed.price:.2f}, 数量={order.executed.size:.0f}')
                name = order.data._name
                info = self.positions_info.get(name, {})
                entry_price = info.get('entry', 0)
                strategy_type = info.get('type', 'UNKNOWN')
                sell_reason = info.get('sell_reason', '')
                buy_date = info.get('date', '')
                
                # 记录卖出
                pnl_pct = (order.executed.price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                self.sell_records.append({
                    '卖出日期': str(current_date),
                    '股票': name,
                    '策略': strategy_type,
                    '买入日期': str(buy_date),
                    '买入价': entry_price,
                    '卖出价': order.executed.price,
                    '数量': int(abs(order.executed.size)),
                    '盈亏%': round(pnl_pct, 2),
                    '卖出原因': sell_reason,
                })
                
                # 卖出完成后清理持仓信息
                if name in self.positions_info:
                    del self.positions_info[name]
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            status_name = {order.Canceled: '取消', order.Margin: '保证金不足', order.Rejected: '拒绝'}
            self.log(f'⚠️ 订单{status_name.get(order.status, "失败")} {order.data._name}')
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.total_trades += 1
            if trade.pnl > 0:
                self.winning_trades += 1
            self.log(f'📊 交易结束 {trade.data._name}: 盈亏={trade.pnl:.2f} ({trade.pnlcomm:.2f}含手续费)')
            
            # 记录交易日志
            self.trade_log.append({
                'symbol': trade.data._name,
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'date': self.datas[0].datetime.date(0)
            })
    
    def get_holding_count(self):
        """获取当前持仓数量"""
        return len([d for d in self.datas if self.getposition(d).size > 0])
    
    def next(self):
        """每个bar执行的主逻辑"""
        
        current_date = self.datas[0].datetime.date(0)
        
        # ========== A. 持仓监控 (卖出逻辑) ==========
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size <= 0:
                continue
                
            name = data._name
            if name not in self.positions_info:
                continue
            
            info = self.positions_info[name]
            current_price = data.close[0]
            entry_price = info['entry']
            strategy_type = info['type']
            pnl_pct = (current_price - entry_price) / entry_price
            
            # 更新最高盈亏
            if pnl_pct > info.get('highest_pnl', 0):
                info['highest_pnl'] = pnl_pct
            
            highest_pnl = info.get('highest_pnl', 0)
            should_sell = False
            reason = ""
            
            # 获取均线 (使用数据中的预计算值)
            ma5 = data.ma5[0] if hasattr(data, 'ma5') and not np.isnan(data.ma5[0]) else current_price
            ma10 = data.ma10[0] if hasattr(data, 'ma10') and not np.isnan(data.ma10[0]) else current_price
            ma20 = data.ma20[0] if hasattr(data, 'ma20') and not np.isnan(data.ma20[0]) else current_price
            
            # 打印持仓监控信息
            self.log(f'📊 持仓监控 {name}: 价格=${current_price:.2f} | 成本=${entry_price:.2f} | 盈亏={pnl_pct*100:.2f}% | 最高={highest_pnl*100:.2f}% | MA5=${ma5:.2f} MA10=${ma10:.2f}')
            
            # --- 1. 低吸策略止损止盈 (与实盘一致) ---
            if strategy_type == 'DIP':
                # 获取低吸目标均线 (用于尾盘检查)
                dip_target_ma = info.get('dip_target_ma', 'ma20')
                if dip_target_ma == 'ma5':
                    dip_ma_price = ma5
                elif dip_target_ma == 'ma10':
                    dip_ma_price = ma10
                else:
                    dip_ma_price = ma20
                
                is_entry_day = (current_date == info['date'])
                
                # === 止损逻辑 ===
                if is_entry_day:
                    # 当日规则1: 跌破买入点3%强制卖出
                    if current_price < entry_price * (1 - self.p.dip_stop_loss):
                        should_sell, reason = True, f"DIP当日止损(跌破买入价3%)"
                else:
                    # 次日起: 跌破买入点3%强制卖出
                    if current_price < entry_price * (1 - self.p.dip_stop_loss):
                        should_sell, reason = True, f"DIP止损(跌破买入价3%)"
                
                # === 止盈逻辑 (使用动态均线) ===
                if not should_sell:
                    sold_half = info.get('sold_half', False)
                    
                    # 30%盈利后的处理
                    if highest_pnl > self.p.tp2_threshold:
                        # 先卖一半（如果还没卖过）
                        if not sold_half:
                            info['sold_half'] = True
                            self.log(f'📊 {name} 盈利超30%，标记半仓止盈（日K回测简化处理）')
                        # 剩余部分：跌破10日线下方0.2%全部卖出
                        target_price = ma10 * (1 - self.p.dip_tp2_buffer)
                        if current_price < target_price:
                            should_sell, reason = True, f"DIP_30%止盈(${current_price:.2f} < MA10 ${target_price:.2f})"
                    
                    # 10%盈利后，跌破5日线下方0.2%止盈
                    elif highest_pnl > self.p.tp1_threshold:
                        target_price = ma5 * (1 - self.p.dip_tp1_buffer)
                        if current_price < target_price:
                            should_sell, reason = True, f"DIP_10%止盈(${current_price:.2f} < MA5 ${target_price:.2f})"
            
            # --- 2. 突破策略止损止盈 ---
            elif strategy_type == 'BREAK':
                # 止损: 跌破买入价1%
                if current_price < entry_price * (1 - self.p.stop_loss):
                    should_sell, reason = True, "BREAK止损"
            
            # --- 3. 回调策略止损止盈 (与实盘一致) ---
            elif strategy_type == 'PULLBACK':
                # 获取前日低点 (用于止损判断)
                prev_low = data.prev_low[0] if hasattr(data, 'prev_low') and not np.isnan(data.prev_low[0]) else 0
                entry_prev_low = info.get('entry_prev_low', 0)  # 买入时记录的前日低点
                
                is_entry_day = (current_date == info['date'])  # 是否是买入当天
                holding_days = (current_date - info['date']).days
                
                # === 止损逻辑 (与实盘一致) ===
                if is_entry_day:
                    # 当日规则1: 跌破买入时的前日低点止损
                    if entry_prev_low > 0 and current_price < entry_prev_low:
                        should_sell, reason = True, f"PULLBACK当日止损(跌破前日低点${entry_prev_low:.2f})"
                    # 当日规则2: 跌破买入价3%强制止损
                    elif current_price < entry_price * (1 - self.p.pullback_stop_loss):
                        should_sell, reason = True, f"PULLBACK当日止损(跌破买入价3%)"
                else:
                    # 次日起: 跌破昨日低点止损
                    if prev_low > 0 and current_price < prev_low:
                        should_sell, reason = True, f"PULLBACK止损(跌破昨日低点${prev_low:.2f})"
                    # 或跌破买入价3%强制止损
                    elif current_price < entry_price * (1 - self.p.pullback_stop_loss):
                        should_sell, reason = True, f"PULLBACK止损(跌破买入价3%)"
                
                # === 止盈逻辑 (与实盘一致) ===
                if not should_sell:
                    sold_half = info.get('sold_half', False)
                    
                    # 30%盈利后的处理
                    if highest_pnl > self.p.tp2_threshold:
                        # 先卖一半（如果还没卖过）
                        if not sold_half:
                            # 标记为已卖半仓，下次触发MA10止盈时全部卖出
                            info['sold_half'] = True
                            self.log(f'📊 {name} 盈利超30%，标记半仓止盈（日K回测简化处理）')
                        # 剩余部分：跌破10日线下方0.2%全部卖出
                        target_price = ma10 * (1 - self.p.pullback_tp2_buffer)
                        if current_price < target_price:
                            should_sell, reason = True, f"PULLBACK_30%止盈(${current_price:.2f} < MA10 ${target_price:.2f})"
                    
                    # 10%盈利后，跌破5日线下方0.2%止盈
                    elif highest_pnl > self.p.tp1_threshold:
                        target_price = ma5 * (1 - self.p.pullback_tp1_buffer)
                        if current_price < target_price:
                            should_sell, reason = True, f"PULLBACK_10%止盈(${current_price:.2f} < MA5 ${target_price:.2f})"
            
            # --- 4. BREAK策略止盈 (使用静态均线) ---
            if not should_sell and strategy_type == 'BREAK':
                # 30%盈利后，跌破10日线下方1%止盈
                if highest_pnl > self.p.tp2_threshold:
                    if current_price < ma10 * (1 - self.p.tp_buffer):
                        should_sell, reason = True, "BREAK_30%止盈MA10"
                # 10%盈利后，跌破5日线下方1%止盈
                elif highest_pnl > self.p.tp1_threshold:
                    if current_price < ma5 * (1 - self.p.tp_buffer):
                        should_sell, reason = True, "BREAK_10%止盈MA5"
            
            # 执行卖出
            if should_sell:
                self.log(f'💰 触发卖出信号: {name} | 原因={reason} | 持仓天数={(current_date - info["date"]).days}天')
                # 记录卖出原因到 positions_info，供 notify_order 使用
                info['sell_reason'] = reason
                self.close(data)
                # 注意：不在这里 del，等 notify_order 记录完卖出信息后再清理
            else:
                # 未触发卖出，打印当前状态
                if pnl_pct >= 0.10:
                    self.log(f'✅ {name} 持仓良好 盈利{pnl_pct*100:.1f}%')
        
        # ========== B. 开仓逻辑 (买入逻辑) ==========
        # 检查是否可以开仓
        current_holdings = self.get_holding_count()
        if current_holdings >= self.p.max_holdings:
            return
        
        available_slots = self.p.max_holdings - current_holdings
        
        # 重置当天的待执行订单金额追踪
        if self.last_order_date != current_date:
            self.pending_buy_value = 0
            self.last_order_date = current_date
        
        # 计算可用现金（扣除已提交但未成交的订单）
        available_cash = self.broker.getcash() - self.pending_buy_value
        
        # 统计当天的信号
        signals_today = []
        
        for data in self.datas:
            if available_slots <= 0:
                break
                
            # 已有持仓跳过
            if self.getposition(data).size > 0:
                continue
            
            name = data._name
            
            # 检查选股信号
            try:
                low_signal = data.low_buy_signal[0] if hasattr(data, 'low_buy_signal') else 0
                break_signal = data.breakout_signal[0] if hasattr(data, 'breakout_signal') else 0
                pullback_signal = data.pullback_signal[0] if hasattr(data, 'pullback_signal') else 0
            except:
                continue
            
            strategy_type = None
            
            # 策略优先级: 回调 > 突破 > 低吸
            if self.p.enable_pullback and pullback_signal == 1:
                # 回调策略额外条件: 当天开盘价 > 昨收 (高开)
                try:
                    today_open = data.open[0]
                    
                    # 🔧 修复BUG: 确保获取的是当前股票的前一交易日收盘价
                    # 原问题: data.close[-1] 在多股票回测时可能索引错误
                    # 解决方案: 验证数据长度并使用正确的前一日收盘价
                    if len(data.close) > 1:
                        prev_close = data.close[-1]  # 当前股票的前一天收盘价
                    else:
                        # 如果当前股票数据不足，跳过
                        self.log(f'跳过回调信号: {name} | 数据不足，无前一日收盘价')
                        continue
                    
                    # 🔧 数据验证：确保获取到的是正确的前一交易日数据
                    current_datetime = data.datetime.date(0)  # 当前股票的当前日期
                    prev_datetime = data.datetime.date(-1)    # 当前股票的前一交易日
                    
                    # 🔧 调试日志：输出关键数据（仅NVTS在6月30日）
                    if name == 'NVTS' and str(current_datetime) == '2025-06-30':
                        self.log(f'🔍 [DEBUG] {name} | 当前日期:{current_datetime}')
                        self.log(f'🔍 [DEBUG] {name} | 前一日期:{prev_datetime}')
                        self.log(f'🔍 [DEBUG] 当日开盘 data.open[0] = {today_open:.4f}')
                        self.log(f'🔍 [DEBUG] 前日收盘 data.close[-1] = {prev_close:.4f}')
                        # 输出更多历史收盘价来验证数据正确性
                        self.log(f'🔍 [DEBUG] 数据长度: {len(data.close)}')
                        for i in range(-min(5, len(data.close)-1), 1):
                            try:
                                hist_date = data.datetime.date(i)
                                hist_close = data.close[i]
                                self.log(f'🔍 [DEBUG]   close[{i:2d}]: {hist_date} = ${hist_close:.4f}')
                            except Exception as e:
                                self.log(f'🔍 [DEBUG]   close[{i:2d}]: 错误 - {e}')
                        diff = today_open - prev_close
                        diff_pct = (diff / prev_close) * 100
                        self.log(f'🔍 [DEBUG] 差额: ${diff:+.4f} ({diff_pct:+.2f}%)')
                    
                    # 高开条件检查
                    if today_open > prev_close:
                        strategy_type = 'PULLBACK'
                        signals_today.append(f'{name}(回调)')
                    else:
                        self.log(f'跳过回调信号: {name} | 开盘{today_open:.2f} <= 昨收{prev_close:.2f}')
                except Exception as e:
                    self.log(f'跳过回调信号: {name} | 错误: {e}')
                    pass
            elif self.p.enable_break and break_signal == 1:
                strategy_type = 'BREAK'
                signals_today.append(f'{name}(突破)')
            elif self.p.enable_dip and low_signal == 1:
                # 低吸策略: 需要找到最近的下方均线
                try:
                    prev_low_price = data.prev_low[0] if hasattr(data, 'prev_low') and not np.isnan(data.prev_low[0]) else 0
                    today_low = data.low[0]
                    today_close = data.close[0]
                    
                    # 获取均线
                    ma5_val = data.ma5[0] if hasattr(data, 'ma5') and not np.isnan(data.ma5[0]) else 0
                    ma10_val = data.ma10[0] if hasattr(data, 'ma10') and not np.isnan(data.ma10[0]) else 0
                    ma20_val = data.ma20[0] if hasattr(data, 'ma20') and not np.isnan(data.ma20[0]) else 0
                    
                    if prev_low_price > 0 and ma5_val > 0 and ma10_val > 0 and ma20_val > 0:
                        # 计算前日低点与各均线的距离（只考虑下方的均线）
                        ma_candidates = []
                        if prev_low_price > ma5_val:
                            ma_candidates.append(('ma5', ma5_val, prev_low_price - ma5_val))
                        if prev_low_price > ma10_val:
                            ma_candidates.append(('ma10', ma10_val, prev_low_price - ma10_val))
                        if prev_low_price > ma20_val:
                            ma_candidates.append(('ma20', ma20_val, prev_low_price - ma20_val))
                        
                        if ma_candidates:
                            # 按距离排序，取最近的
                            ma_candidates.sort(key=lambda x: x[2])
                            target_ma_name, target_ma_price, _ = ma_candidates[0]
                            
                            # 买入条件: 当天最低价跌破目标均线，但收盘价收回均线之上
                            if today_low < target_ma_price and today_close > target_ma_price:
                                strategy_type = 'DIP'
                                signals_today.append(f'{name}(低吸-{target_ma_name.upper()})')
                                # 保存目标均线信息
                                self.dip_target_ma_info = {name: {'ma_name': target_ma_name, 'ma_price': target_ma_price}}
                except Exception as e:
                    pass
            
            if strategy_type:
                # 计算买入数量 (等权分配，基于可用现金)
                # 每只股票分配: min(总市值/max_holdings, 可用现金/剩余槽位)
                target_value = min(
                    self.broker.getvalue() / self.p.max_holdings,
                    available_cash / available_slots if available_slots > 0 else 0
                )
                price = data.close[0]
                size = int(target_value / price)
                # 获取前一日低点(PULLBACK止损用)
                prev_low = data.prev_low[0] if hasattr(data, 'prev_low') and len(data.prev_low) > 0 and not np.isnan(data.prev_low[0]) else 0
                
                order_value = size * price
                
                # 检查是否有足够现金
                if size > 0 and order_value <= available_cash:
                    self.log(f'触发买入信号: {name} | 策略={strategy_type} | 价格=${price:.2f} | 数量={size} | 金额=${order_value:.0f}')
                    self.buy(data=data, size=size)
                    
                    # 更新追踪变量
                    self.pending_buy_value += order_value
                    available_cash -= order_value
                    
                    self.positions_info[name] = {
                        'entry': price,
                        'type': strategy_type,
                        'highest_pnl': 0,
                        'date': current_date,
                        'entry_prev_low': prev_low,  # 入场时的前一日低点(PULLBACK止损用)
                        'sold_half': False,  # 是否已卖出一半(30%止盈标记)
                        # 低吸策略专用: 目标均线信息
                        'dip_target_ma': getattr(self, 'dip_target_ma_info', {}).get(name, {}).get('ma_name', 'ma20'),
                        'dip_target_ma_price': getattr(self, 'dip_target_ma_info', {}).get(name, {}).get('ma_price', 0),
                    }
                    
                    available_slots -= 1
                elif size > 0:
                    self.log(f'⚠️ 现金不足跳过: {name} | 需要${order_value:.0f} | 可用${available_cash:.0f}')
        
        # 如果今天有信号，打印汇总
        if signals_today:
            self.log(f'📋 今日选股信号({len(signals_today)}个): {", ".join(signals_today)}')
    
    def stop(self):
        """回测结束时的统计"""
        win_rate = self.winning_trades / self.total_trades * 100 if self.total_trades > 0 else 0
        self.log(f'')
        self.log(f'========== 回测结果统计 ==========')
        self.log(f'总交易次数: {self.total_trades}')
        self.log(f'盈利次数: {self.winning_trades}')
        self.log(f'胜率: {win_rate:.2f}%')
        self.log(f'最终资金: {self.broker.getvalue():.2f}')


# ============================================================
# 4. 数据准备函数
# ============================================================

def prepare_backtest_data(symbols, start_date, end_date, cache_dir='cache'):
    """
    准备回测数据：
    1. 获取历史行情数据
    2. 计算技术指标
    3. 生成选股信号
    
    返回: dict {symbol: DataFrame}
    """
    print("📊 正在准备回测数据...")
    
    # 检查是否有缓存的回测数据
    cache_file = os.path.join(cache_dir, f'backtest_data_{start_date}_{end_date}.pkl')
    if os.path.exists(cache_file):
        print(f"   从缓存加载: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # 获取所有股票数据
    all_data = []
    for symbol in symbols:
        try:
            df = get_data_single_stock_with_cache(symbol, days=365)
            if df is not None and not df.empty:
                df['symbol'] = symbol
                all_data.append(df)
        except Exception as e:
            print(f"   获取 {symbol} 失败: {e}")
    
    if not all_data:
        raise ValueError("没有获取到任何数据")
    
    # 合并数据
    combined = pd.concat(all_data, axis=0)
    combined = combined.reset_index()
    
    if 'time' not in combined.columns and 'Date' in combined.columns:
        combined['time'] = combined['Date']
    
    combined = combined.set_index(['time', 'symbol'])
    combined = combined.sort_index()
    
    # 过滤日期范围
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    combined = combined.loc[(combined.index.get_level_values('time') >= start_dt) & 
                           (combined.index.get_level_values('time') <= end_dt)]
    
    print(f"   数据范围: {combined.index.get_level_values('time').min()} ~ {combined.index.get_level_values('time').max()}")
    print(f"   股票数量: {combined.index.get_level_values('symbol').nunique()}")
    
    # 添加技术指标
    combined = add_derived_features(combined)
    
    # 生成选股信号 (逐日生成)
    print("   生成选股信号...")
    dates = combined.index.get_level_values('time').unique().sort_values()
    
    signal_records = []
    
    for i, date in enumerate(dates):
        if i < 60:  # 需要60天历史数据
            continue
        
        # 获取截止到当天的数据
        hist_data = combined.loc[combined.index.get_level_values('time') <= date].copy()
        
        try:
            # 构建基础股票池
            base_snap = build_base_universe(hist_data)
            
            # 低吸候选
            low_df = low_buy_candidates(hist_data, base_snap)
            low_symbols = low_df[low_df['low_buy_candidate'] == True].index.tolist()
            
            # 突破候选
            break_df = breakout_buy_candidates(hist_data, base_snap)
            break_symbols = break_df[break_df['breakout_candidate'] == True].index.tolist()
            
            # 回调候选
            pullback_df = pullback_buy_candidates(hist_data, base_snap)
            pullback_symbols = pullback_df[pullback_df['pullback_candidate'] == True].index.tolist()
            
            for symbol in combined.loc[date].index.get_level_values('symbol').unique():
                signal_records.append({
                    'time': date,
                    'symbol': symbol,
                    'low_buy_signal': 1 if symbol in low_symbols else 0,
                    'breakout_signal': 1 if symbol in break_symbols else 0,
                    'pullback_signal': 1 if symbol in pullback_symbols else 0
                })
                
        except Exception as e:
            # 某些日期可能数据不足，跳过
            pass
    
    # 合并信号到数据
    signal_df = pd.DataFrame(signal_records)
    signal_df = signal_df.set_index(['time', 'symbol'])
    
    combined = combined.join(signal_df, how='left')
    combined['low_buy_signal'] = combined['low_buy_signal'].fillna(0)
    combined['breakout_signal'] = combined['breakout_signal'].fillna(0)
    combined['pullback_signal'] = combined['pullback_signal'].fillna(0)
    
    # 拆分为单股票DataFrame
    result = {}
    for symbol in combined.index.get_level_values('symbol').unique():
        stock_data = combined.xs(symbol, level='symbol').copy()
        stock_data = stock_data.sort_index()
        result[symbol] = stock_data
    
    # 缓存结果
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"   数据已缓存: {cache_file}")
    
    return result


def prepare_backtest_data_simple(cache_file, start_date, end_date):
    """
    从现有缓存文件准备回测数据（简化版，不重新计算选股信号）
    
    直接使用 usstock_select.py 已经计算好的数据
    支持信号缓存，避免每次重新生成信号
    """
    print("📊 正在从缓存准备回测数据...")
    
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"缓存文件不存在: {cache_file}")
    
    # 构建信号缓存文件名
    cache_basename = os.path.basename(cache_file).replace('.pkl', '')
    signal_cache_file = os.path.join(
        os.path.dirname(cache_file),
        f"signals_{cache_basename}_{start_date}_{end_date}.pkl"
    )
    
    # 检查是否有信号缓存
    if os.path.exists(signal_cache_file):
        print(f"   💾 发现信号缓存，直接加载: {os.path.basename(signal_cache_file)}")
        with open(signal_cache_file, 'rb') as f:
            result = pickle.load(f)
        print(f"   ✅ 加载完成，共 {len(result)} 只股票")
        return result
    
    print(f"   ⏳ 未找到信号缓存，需要生成信号（首次运行较慢，后续会使用缓存）")
    
    with open(cache_file, 'rb') as f:
        datas = pickle.load(f)
    
    print(f"   加载缓存: {cache_file}")
    print(f"   数据形状: {datas.shape}")
    
    # 添加技术指标
    datas = add_derived_features(datas)
    
    # 过滤日期范围 - 转换为 date 类型进行比较
    start_dt = pd.Timestamp(start_date).date()
    end_dt = pd.Timestamp(end_date).date()
    time_index = datas.index.get_level_values('time')
    # 如果索引是 Timestamp 类型，转换为 date
    if hasattr(time_index[0], 'date'):
        time_dates = pd.Index([t.date() if hasattr(t, 'date') else t for t in time_index])
    else:
        time_dates = time_index
    datas = datas.loc[(time_dates >= start_dt) & (time_dates <= end_dt)]
    
    print(f"   日期范围: {datas.index.get_level_values('time').min()} ~ {datas.index.get_level_values('time').max()}")
    
    # 构建基础股票池
    base_snap = build_base_universe(datas)
    valid_symbols = base_snap[base_snap["in_pool"]].index.get_level_values("symbol").unique()
    print(f"   有效股票数: {len(valid_symbols)}")
    
    # 生成选股信号 (逐日)
    print("   生成选股信号...")
    dates = sorted(datas.index.get_level_values('time').unique())
    
    # 为数据添加信号列
    datas['low_buy_signal'] = 0
    datas['breakout_signal'] = 0  
    datas['pullback_signal'] = 0
    
    # 计算要处理的日期
    dates_to_process = dates[60:]
    total_days = len(dates_to_process)
    print(f"   总共需要处理 {total_days} 个交易日")
    
    # 信号统计
    low_buy_count = 0
    breakout_count = 0
    pullback_count = 0
    
    # 记录所有产生过信号的股票（修复：不再依赖最后一天的 valid_symbols）
    symbols_with_signals = set()
    
    import time
    start_time = time.time()
    
    for i, date in enumerate(dates_to_process):
        date_str = str(date)[:10]  # 兼容各种日期格式
        
        # 每10天打印一次进度
        if i % 10 == 0:
            elapsed = time.time() - start_time
            if i > 0:
                eta = (elapsed / i) * (total_days - i)
                print(f"   进度: {i}/{total_days} ({100*i/total_days:.1f}%) | 已用时: {elapsed:.1f}s | 预计剩余: {eta:.1f}s | 当前日期: {date_str} | 信号数: 低吸={low_buy_count} 突破={breakout_count} 回调={pullback_count}")
            else:
                print(f"   进度: {i}/{total_days} | 当前日期: {date_str}")
        
        # 记录本日产生的信号
        day_low = 0
        day_break = 0
        day_pullback = 0
        
        try:
            hist = datas.loc[datas.index.get_level_values('time') <= date]
            base = build_base_universe(hist)
            
            # 低吸信号
            if BACKTEST_CONFIG.get('ENABLE_DIP', True):
                low_df = low_buy_candidates(hist, base)
                if low_df is not None and 'low_buy_candidate' in low_df.columns:
                    for sym in low_df[low_df['low_buy_candidate']].index:
                        if (date, sym) in datas.index:
                            datas.loc[(date, sym), 'low_buy_signal'] = 1
                            low_buy_count += 1
                            day_low += 1
                            symbols_with_signals.add(sym)  # 记录产生信号的股票
            
            # 突破信号
            if BACKTEST_CONFIG.get('ENABLE_BREAK', True):
                break_df = breakout_buy_candidates(hist, base)
                if break_df is not None and 'breakout_candidate' in break_df.columns:
                    for sym in break_df[break_df['breakout_candidate']].index:
                        if (date, sym) in datas.index:
                            datas.loc[(date, sym), 'breakout_signal'] = 1
                            breakout_count += 1
                            day_break += 1
                            symbols_with_signals.add(sym)  # 记录产生信号的股票
            
            # 回调信号
            if BACKTEST_CONFIG.get('ENABLE_PULLBACK', True):
                pullback_df = pullback_buy_candidates(hist, base)
                if pullback_df is not None and 'pullback_candidate' in pullback_df.columns:
                    candidates = pullback_df[pullback_df['pullback_candidate']].index.tolist()
                    for sym in candidates:
                        if (date, sym) in datas.index:
                            datas.loc[(date, sym), 'pullback_signal'] = 1
                            pullback_count += 1
                            day_pullback += 1
                            symbols_with_signals.add(sym)  # 记录产生信号的股票
            
            # 如果今天有信号，打印详细信息
            if day_low > 0 or day_break > 0 or day_pullback > 0:
                signals = []
                if day_low > 0:
                    signals.append(f"低吸{day_low}个")
                if day_break > 0:
                    signals.append(f"突破{day_break}个")
                if day_pullback > 0:
                    signals.append(f"回调{day_pullback}个")
                print(f"   ⭐ {date_str} 产生信号: {', '.join(signals)}")
                if day_pullback > 0 and pullback_df is not None:
                    pullback_symbols = [sym for sym in candidates if (date, sym) in datas.index]
                    if pullback_symbols:
                        print(f"      回调股票: {', '.join(pullback_symbols[:5])}{'...' if len(pullback_symbols) > 5 else ''}")
                    
        except Exception as e:
            print(f"   ⚠️ 日期 {date_str} 处理失败: {str(e)[:50]}")
    
    total_time = time.time() - start_time
    print(f"   ✅ 信号生成完成! 用时: {total_time:.1f}秒")
    print(f"   📊 信号统计: 低吸={low_buy_count}, 突破={breakout_count}, 回调={pullback_count}")
    print(f"   📊 产生信号的股票数: {len(symbols_with_signals)}")
    
    # 拆分为单股票 (修复：使用产生过信号的股票，而不是最后一天的valid_symbols)
    result = {}
    for symbol in symbols_with_signals:
        try:
            stock_data = datas.xs(symbol, level='symbol').copy()
            if len(stock_data) > 60:
                # 确保索引是 DatetimeIndex (Backtrader需要)
                if not isinstance(stock_data.index, pd.DatetimeIndex):
                    stock_data.index = pd.to_datetime(stock_data.index)
                result[symbol] = stock_data
        except:
            pass
    
    print(f"   回测股票数: {len(result)}")
    
    # 保存信号缓存
    print(f"   💾 保存信号缓存: {os.path.basename(signal_cache_file)}")
    with open(signal_cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"   ✅ 缓存保存成功，下次运行将直接加载")
    
    return result


# ============================================================
# 5. 主程序
# ============================================================

def run_backtest():
    """运行回测"""
    
    print("=" * 60)
    print("       美股多策略回测系统")
    print("=" * 60)
    
    # 1. 查找可用的缓存文件
    cache_dir = os.path.join(os.getcwd(), 'cache')
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')] if os.path.exists(cache_dir) else []
    
    if not cache_files:
        print("❌ 未找到缓存数据，请先运行 usstock_select.py 获取数据")
        return
    
    # 使用最新的缓存文件
    cache_file = os.path.join(cache_dir, sorted(cache_files)[-1])
    print(f"📁 使用缓存: {cache_file}")
    
    # 2. 准备数据
    try:
        stock_data = prepare_backtest_data_simple(
            cache_file,
            BACKTEST_CONFIG['START_DATE'],
            BACKTEST_CONFIG['END_DATE']
        )
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not stock_data:
        print("❌ 没有有效的回测数据")
        return
    
    # 3. 创建 Backtrader 引擎
    cerebro = bt.Cerebro()
    
    # 4. 添加数据源
    print(f"\n📈 添加 {len(stock_data)} 只股票数据...")
    for symbol, df in stock_data.items():
        # 确保必要的列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            continue
        
        # 填充缺失值
        for col in ['ma5', 'ma10', 'ma20', 'ma50', 'vol_ma5', 'vol_ratio', 
                    'tr_value', 'avg_tr_value_30', 'swing_60', 
                    'high_60', 'high_60_ex10', 'turnover_value', 'float_mktcap',
                    'low_buy_signal', 'breakout_signal', 'pullback_signal',
                    'prev_low', 'prev_high']:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)
        
        data = PandasData(
            dataname=df,
            name=symbol,
            fromdate=datetime.strptime(BACKTEST_CONFIG['START_DATE'], '%Y-%m-%d'),
            todate=datetime.strptime(BACKTEST_CONFIG['END_DATE'], '%Y-%m-%d'),
        )
        cerebro.adddata(data)
    
    # 5. 配置策略
    cerebro.addstrategy(
        MultiStrategy,
        max_holdings=BACKTEST_CONFIG['MAX_HOLDINGS'],
        stop_loss=BACKTEST_CONFIG['STOP_LOSS_FIXED'],
        tp1_threshold=BACKTEST_CONFIG['TP1_THRESHOLD'],
        tp2_threshold=BACKTEST_CONFIG['TP2_THRESHOLD'],
        tp_buffer=BACKTEST_CONFIG['TP_BUFFER'],
        dip_stop_loss=BACKTEST_CONFIG['DIP_STOP_LOSS'],
        dip_tp1_buffer=BACKTEST_CONFIG['DIP_TP1_BUFFER'],
        dip_tp2_buffer=BACKTEST_CONFIG['DIP_TP2_BUFFER'],
        pullback_stop_loss=BACKTEST_CONFIG['PULLBACK_STOP_LOSS'],
        pullback_tp1_buffer=BACKTEST_CONFIG['PULLBACK_TP1_BUFFER'],
        pullback_tp2_buffer=BACKTEST_CONFIG['PULLBACK_TP2_BUFFER'],
        enable_dip=BACKTEST_CONFIG['ENABLE_DIP'],
        enable_break=BACKTEST_CONFIG['ENABLE_BREAK'],
        enable_pullback=BACKTEST_CONFIG['ENABLE_PULLBACK'],
    )
    
    # 6. 配置资金和手续费
    cerebro.broker.setcash(BACKTEST_CONFIG['INITIAL_CASH'])
    cerebro.broker.setcommission(commission=BACKTEST_CONFIG['COMMISSION'])
    
    # 7. 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 8. 运行回测
    print(f"\n🚀 开始回测...")
    print(f"   初始资金: ${BACKTEST_CONFIG['INITIAL_CASH']:,.2f}")
    print(f"   回测区间: {BACKTEST_CONFIG['START_DATE']} ~ {BACKTEST_CONFIG['END_DATE']}")
    print(f"   启用策略: ", end="")
    if BACKTEST_CONFIG['ENABLE_DIP']: print("低吸 ", end="")
    if BACKTEST_CONFIG['ENABLE_BREAK']: print("突破 ", end="")
    if BACKTEST_CONFIG['ENABLE_PULLBACK']: print("回调 ", end="")
    print("\n")
    
    results = cerebro.run()
    strat = results[0]
    
    # 9. 输出结果
    final_value = cerebro.broker.getvalue()
    returns_pct = (final_value - BACKTEST_CONFIG['INITIAL_CASH']) / BACKTEST_CONFIG['INITIAL_CASH'] * 100
    
    print("\n" + "=" * 60)
    print("              📊 回测结果")
    print("=" * 60)
    print(f"  初始资金:     ${BACKTEST_CONFIG['INITIAL_CASH']:>15,.2f}")
    print(f"  最终资金:     ${final_value:>15,.2f}")
    print(f"  总收益率:     {returns_pct:>15.2f}%")
    
    # 分析器结果
    try:
        sharpe = strat.analyzers.sharpe.get_analysis()
        print(f"  夏普比率:     {sharpe.get('sharperatio', 0) or 0:>15.2f}")
    except:
        pass
    
    try:
        dd = strat.analyzers.drawdown.get_analysis()
        print(f"  最大回撤:     {dd.get('max', {}).get('drawdown', 0):>15.2f}%")
    except:
        pass
    
    try:
        trades = strat.analyzers.trades.get_analysis()
        total = trades.get('total', {}).get('total', 0)
        won = trades.get('won', {}).get('total', 0)
        lost = trades.get('lost', {}).get('total', 0)
        print(f"  总交易次数:   {total:>15}")
        print(f"  盈利次数:     {won:>15}")
        print(f"  亏损次数:     {lost:>15}")
        if total > 0:
            print(f"  胜率:         {won/total*100:>15.2f}%")
    except:
        pass
    
    print("=" * 60)
    
    # 10. 导出交易记录到 Excel
    import pandas as pd
    from datetime import datetime as dt
    
    output_file = f"result/回测交易记录_{BACKTEST_CONFIG['START_DATE']}_{BACKTEST_CONFIG['END_DATE']}.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 买入记录
        if strat.buy_records:
            buy_df = pd.DataFrame(strat.buy_records)
            buy_df.to_excel(writer, sheet_name='买入记录', index=False)
        
        # 卖出记录
        if strat.sell_records:
            sell_df = pd.DataFrame(strat.sell_records)
            sell_df.to_excel(writer, sheet_name='卖出记录', index=False)
        
        # 汇总统计
        summary_data = {
            '指标': ['初始资金', '最终资金', '总收益率', '总交易次数', '盈利次数', '亏损次数', '胜率'],
            '值': [
                f"${BACKTEST_CONFIG['INITIAL_CASH']:,.2f}",
                f"${final_value:,.2f}",
                f"{returns_pct:.2f}%",
                total,
                won,
                lost,
                f"{won/total*100:.2f}%" if total > 0 else "N/A"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='汇总', index=False)
    
    print(f"\n📁 交易记录已导出到: {output_file}")


if __name__ == '__main__':
    run_backtest()
