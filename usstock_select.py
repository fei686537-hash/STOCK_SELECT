#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import pytz
import json
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
import yfinance as yf
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# 🔄 切换到6783股票池
file_path = os.path.join(os.getcwd(), 'usstock_all.txt')  # 从 usstock.txt 改为 usstock_all.txt
save_dir = os.path.join(os.getcwd(), 'result')
cache_dir = os.path.join(os.getcwd(), 'cache')  # 数据缓存目录


def read_stock_list(file_path):
    try:
        # 1. 读取文件
        df = pd.read_csv(file_path, encoding='utf-8')
        
        bp500_list = df['代码'].tolist()
        
        print(f"成功读取 {len(bp500_list)} 个股票代码")
        print(bp500_list)
        return bp500_list

    except Exception as e:
        print(f"读取出错: {e}")
        return None



def _apply_proxy(use_proxy=False, http_proxy=None, https_proxy=None):
    if use_proxy:
        if http_proxy:
            os.environ["HTTP_PROXY"] = http_proxy
            os.environ["http_proxy"] = http_proxy
        if https_proxy:
            os.environ["HTTPS_PROXY"] = https_proxy
            os.environ["https_proxy"] = https_proxy
    else:
        for k in ["HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy"]:
            os.environ.pop(k, None)


def _to_naive_datetime(idx):
    dt = pd.to_datetime(idx)
    if hasattr(dt, "tz") and dt.tz is not None:
        try:
            dt = dt.tz_convert(None)
        except Exception:
            try:
                dt = dt.tz_localize(None)
            except Exception:
                pass
    return dt


def _safe_get_info(tk: yf.Ticker):
    try:
        return tk.get_info() or {}
    except Exception:
        return {}


def _get_shares_history_df(tk: yf.Ticker):
    """
    返回 DataFrame(index: 披露日, col: shares_outstanding)，若无则 None
    """
    try:
        sh = tk.get_shares_full()
    except Exception:
        sh = None
    if isinstance(sh, pd.DataFrame) and not sh.empty:
        sh = sh.copy()
        sh.index = _to_naive_datetime(sh.index)
        if "Shares" in sh.columns:
            sh = sh[["Shares"]].rename(columns={"Shares": "shares_outstanding"})
        elif "shares_outstanding" in sh.columns:
            sh = sh[["shares_outstanding"]]
        else:
            return None
        sh = sh.sort_index()
        return sh
    return None


def _normalize_ratio(x):
    """
    将 x 归一化为 [0,1] 比例:
    - 若 > 1.5（例如 70 表示 70%），则除以 100
    - 若是 None/NaN 返回 NaN
    """
    try:
        if x is None:
            return np.nan
        v = float(x)
        if np.isnan(v):
            return np.nan
        return v/100.0 if v > 1.5 else v
    except Exception:
        return np.nan


def fetch_one(
    sym,
    start=None,
    end=None,
    use_proxy=False,
    http_proxy=None,
    https_proxy=None
):
    # 确保代理生效
    _apply_proxy(use_proxy, http_proxy, https_proxy)

    tk = yf.Ticker(sym)

    # 价格历史
    px = tk.history(start=start, end=end, auto_adjust=False)
    if px is None or px.empty:
        return None
    px = px.rename(columns=str.lower)
    need_cols = ["open","high","low","close","volume","country"]
    px = px[[c for c in need_cols if c in px.columns]]
    px.index = _to_naive_datetime(px.index)
    px = px.sort_index()

    # 低频信息：行业/板块、float/shares 快照 + 机构/内部持股比例
    sector = None
    industry = None
    country = None  # 新增：公司所在国家
    shares_out_snap = None
    float_snap = None

    # 新增：机构与内部持股比例（目标统一到 0-1）
    institution_pct = np.nan
    insider_pct = np.nan

    info = _safe_get_info(tk)
    if isinstance(info, dict):
        sector = info.get("sector")
        industry = info.get("industry") or info.get("industryKey") or info.get("industryDisp")
        country = info.get("country")  # 获取公司所在国家
        shares_out_snap = info.get("sharesOutstanding")
        float_snap = info.get("floatShares") or info.get("float") or info.get("float_shares")

        # 常见键兼容：institutionPercent/institutionsPercent/heldPercentInstitutions
        inst_keys = [
            "institutionPercent",
            "institutionsPercent",
            "heldPercentInstitutions",
            "institutionOwnership"
        ]
        for k in inst_keys:
            if k in info and pd.notna(info[k]):
                institution_pct = _normalize_ratio(info[k])
                break

        # 内部持股比例：heldPercentInsiders/insiderOwnership
        insider_keys = [
            "heldPercentInsiders",
            "insiderOwnership",
            "insidersPercent"
        ]
        for k in insider_keys:
            if k in info and pd.notna(info[k]):
                insider_pct = _normalize_ratio(info[k])
                break

    # fast_info 兜底
    try:
        fi = getattr(tk, "fast_info", None)
    except Exception:
        fi = None
    if fi is not None:
        try:
            sector = sector or getattr(fi, "sector", None)
        except Exception:
            pass
        try:
            industry = industry or getattr(fi, "industry", None)
        except Exception:
            pass
        try:
            float_snap = float_snap or getattr(fi, "float_shares", None)
        except Exception:
            pass
        # 某些版本 fast_info 也可能有 held_percent_* 字段
        for attr in ["held_percent_institutions", "institution_percent", "institutions_percent"]:
            try:
                v = getattr(fi, attr, None)
                if pd.notna(v) and np.isnan(institution_pct):
                    institution_pct = _normalize_ratio(v)
            except Exception:
                pass
        for attr in ["held_percent_insiders", "insider_percent", "insiders_percent"]:
            try:
                v = getattr(fi, attr, None)
                if pd.notna(v) and np.isnan(insider_pct):
                    insider_pct = _normalize_ratio(v)
            except Exception:
                pass

    # 季度 shares 历史（优先使用）
    shares_hist = _get_shares_history_df(tk)

    # 构造“每日 float_shares”并仅向前填充
    df = px.copy()
    df["symbol"] = sym
    df = df.reset_index().rename(columns={"index":"date", "Date":"date"})
    if "date" not in df.columns:
        df.insert(0, "date", _to_naive_datetime(px.index))
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    float_daily = None
    if shares_hist is not None and not shares_hist.empty:
        rhs = shares_hist.reset_index().rename(columns={"index":"date"})
        rhs["date"] = pd.to_datetime(rhs["date"]).dt.tz_localize(None)
        rhs = rhs.sort_values("date")

        merged = pd.merge_asof(
            left=df[["date"]].sort_values("date"),
            right=rhs[["date","shares_outstanding"]],
            on="date",
            direction="backward",
            allow_exact_matches=True
        )
        float_daily = merged["shares_outstanding"].astype("float64")

    # 如果历史缺失或全空，使用快照（float_snap 优先，其次 shares_out_snap）
    if float_daily is None or float_daily.isna().all():
        seed = np.nan
        if pd.notna(float_snap):
            seed = float(float_snap)
        elif pd.notna(shares_out_snap):
            seed = float(shares_out_snap)
        float_daily = pd.Series(seed, index=df.index, dtype="float64")

    # 仅前向填充，避免未来信息泄露
    float_daily = float_daily.ffill()
    df["float_shares"] = float_daily.values

    # 计算
    df["turnover_value"] = df.get("close") * df.get("volume")
    # 换手率：成交量 / 流通股本
    df["turnover_rate"] = np.where(
        (df["float_shares"].notna()) & (df["float_shares"] > 0),
        df["volume"] / df["float_shares"],
        np.nan
    )
    # 若使用成交额/流通市值：
    # df["turnover_rate"] = df["turnover_value"] / (df["close"] * df["float_shares"])

    df["float_mktcap"] = np.where(
        df["float_shares"].notna(),
        df["close"] * df["float_shares"],
        np.nan
    )

    # 行业/板块 + 机构/内部持股 + 国家（快照广播）
    df["sector"] = sector
    df["industry"] = industry
    df["country"] = country  # 新增：公司所在国家
    df["institution_pct"] = institution_pct
    df["insider_pct"] = insider_pct

    keep = [
        "symbol","open","high","low","close","volume",
        "turnover_value","float_mktcap","turnover_rate",
        "sector","industry","institution_pct","insider_pct",
        "date","country"
    ]
    df = df[keep]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.set_index("date")
    return df


def yahoo_datas(
    symbols,
    n_days=252,
    use_proxy=False,
    http_proxy=None,
    https_proxy=None,
    max_workers=8,
    batch_size=500,      # 每批获取的股票数量
    batch_pause=300,     # 每批之间暂停的秒数（5分钟）
    use_cache=True,      # 是否使用缓存
    force_refresh=False  # 是否强制刷新缓存
):
    """
    分批获取股票数据，避免被 Yahoo Finance 限流。
    每获取 batch_size 只股票后暂停 batch_pause 秒。
    
    参数:
        use_cache: 是否使用本地缓存（默认True）
        force_refresh: 是否强制刷新缓存，忽略已有数据（默认False）
    """
    # 生成缓存文件名（基于日期和股票数量）
    today_str = datetime.now().strftime("%Y-%m-%d")
    cache_filename = f"stock_data_{today_str}_{len(symbols)}stocks_{n_days}days.pkl"
    cache_filepath = os.path.join(cache_dir, cache_filename)
    
    # 尝试从缓存加载
    if use_cache and not force_refresh and os.path.exists(cache_filepath):
        print(f"\n{'='*60}")
        print(f"📦 发现缓存文件: {cache_filename}")
        print(f"{'='*60}")
        try:
            cached_data = pd.read_pickle(cache_filepath)
            print(f"✅ 成功从缓存加载 {len(cached_data.index.get_level_values('symbol').unique())} 只股票数据")
            print(f"   数据时间范围: {cached_data.index.get_level_values('time').min()} 至 {cached_data.index.get_level_values('time').max()}")
            print(f"{'='*60}\n")
            return cached_data
        except Exception as e:
            print(f"⚠️ 缓存加载失败: {e}")
            print(f"   将重新获取数据...\n")
    
    # 如果没有缓存或强制刷新，则从API获取
    if force_refresh:
        print(f"\n{'='*60}")
        print(f"🔄 强制刷新模式：忽略缓存，重新获取数据")
        print(f"{'='*60}\n")
    
    # 计算起止日期：取最近 n_days 个自然日的开始，交给 yfinance 自己做交易日筛选
    end_ts = pd.Timestamp.today().normalize()
    start_ts = end_ts - pd.Timedelta(days=int(n_days*2))  # 放宽窗口，防止非交易日不足
    start = start_ts.strftime("%Y-%m-%d")
    end = None  # 到今天

    results = []
    total_symbols = len(symbols)
    
    # 将股票列表分成多个批次
    batches = [symbols[i:i + batch_size] for i in range(0, total_symbols, batch_size)]
    total_batches = len(batches)
    
    print(f"\n{'='*60}")
    print(f"📊 开始分批获取数据")
    print(f"   总股票数: {total_symbols}")
    print(f"   批次大小: {batch_size}")
    print(f"   总批次数: {total_batches}")
    print(f"   批次间隔: {batch_pause} 秒 ({batch_pause//60} 分钟)")
    print(f"{'='*60}\n")
    
    for batch_idx, batch_symbols in enumerate(batches, 1):
        batch_start_time = time.time()
        print(f"\n📦 正在获取第 {batch_idx}/{total_batches} 批 ({len(batch_symbols)} 只股票)...")
        
        batch_results = []
        success_count = 0
        fail_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(
                    fetch_one,
                    s,
                    start,
                    end,
                    use_proxy,
                    http_proxy,
                    https_proxy
                ): s for s in batch_symbols
            }
            for fut in as_completed(futs):
                sym = futs[fut]
                try:
                    df = fut.result()
                    if df is not None and not df.empty:
                        df["symbol"] = sym
                        batch_results.append(df)
                        success_count += 1
                        print(f"  ✅ [{sym}] 获取 {len(df)} 行数据")
                    else:
                        fail_count += 1
                        print(f"  ⚠️ [{sym}] 无数据")
                except Exception as e:
                    fail_count += 1
                    print(f"  ❌ [{sym}] 错误: {e}")
        
        results.extend(batch_results)
        batch_elapsed = time.time() - batch_start_time
        
        print(f"\n📊 第 {batch_idx} 批完成: 成功 {success_count}, 失败 {fail_count}, 耗时 {batch_elapsed:.1f}秒")
        print(f"   累计获取: {len(results)} 只股票")
        
        # 如果不是最后一批，暂停等待
        if batch_idx < total_batches:
            print(f"\n⏳ 暂停 {batch_pause} 秒 ({batch_pause//60} 分钟) 以避免限流...")
            for remaining in range(batch_pause, 0, -30):
                print(f"   剩余等待时间: {remaining} 秒...")
                time.sleep(min(30, remaining))
            print("   ✅ 继续获取下一批...")

    if not results:
        print("\n❌ 没有获取到任何股票数据！")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"✅ 全部获取完成！共获取 {len(results)} 只股票数据")
    print(f"{'='*60}\n")

    out = pd.concat(results).sort_index()
    # 只保留最近 n_days 个“交易日”（每个 symbol 自己的尾部 n 天）
    out = out.groupby("symbol", group_keys=False).apply(lambda x: x.tail(n_days))
    out.index.name = "time"
    out = out.set_index("symbol", append=True)

    # 行业/板块/机构/内部持股兜底（transform 避免 index 级别错配）
    for col in ["sector", "industry", "institution_pct", "insider_pct"]:
        if col in out.columns:
            s = out[col]
            filled = (
                s.groupby(level=1)
                 .transform(lambda x: x.ffill().bfill())
            )
            out[col] = filled

    # 保存到缓存
    if use_cache:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            # 使用 Pickle 格式保存（更稳定，兼容性好）
            out.to_pickle(cache_filepath)
            file_size_mb = os.path.getsize(cache_filepath) / (1024 * 1024)
            print(f"\n{'='*60}")
            print(f"💾 数据已缓存到本地")
            print(f"   文件: {cache_filename}")
            print(f"   大小: {file_size_mb:.2f} MB")
            print(f"   位置: {cache_dir}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"\n⚠️ 缓存保存失败: {e}\n")

    return out


def get_data_single_stock_with_cache(symbol: str, n_days: int = 252) -> pd.DataFrame:
    """
    从缓存中获取单只股票的数据，用于回测。
    
    参数:
        symbol: 股票代码
        n_days: 数据天数
        
    返回:
        DataFrame，包含该股票的OHLCV数据，索引为日期
    """
    import glob
    
    # 查找最新的缓存文件
    cache_files = glob.glob(os.path.join(cache_dir, "stock_data_*.pkl"))
    if not cache_files:
        print(f"⚠️ 没有找到缓存文件，无法获取 {symbol} 数据")
        return pd.DataFrame()
    
    # 使用最新的缓存文件
    latest_cache = sorted(cache_files)[-1]
    
    try:
        all_data = pd.read_pickle(latest_cache)
        
        # 检查 symbol 是否存在
        symbols_in_cache = all_data.index.get_level_values("symbol").unique()
        if symbol not in symbols_in_cache:
            print(f"⚠️ 缓存中没有 {symbol} 的数据")
            return pd.DataFrame()
        
        # 提取该股票的数据
        stock_data = all_data.xs(symbol, level="symbol").copy()
        stock_data = stock_data.sort_index()
        
        # 只保留最近 n_days 天
        stock_data = stock_data.tail(n_days)
        
        return stock_data
        
    except Exception as e:
        print(f"❌ 读取缓存失败: {e}")
        return pd.DataFrame()


def build_base_universe(datas: pd.DataFrame) -> pd.DataFrame:
    """
    构建基础选股池 ：
    现在返回的 DataFrame 索引为 MultiIndex (time, symbol)。
    """
    # 1. 拷贝并确保索引有序 (time, symbol)
    df = datas.copy()
    
    # 索引标准化处理
    if "symbol" in df.columns and "time" not in df.index.names:
        if "time" in df.columns:
            df = df.set_index(["time", "symbol"])
    
    # 强制排序，这对 rolling/ewm 计算至关重要
    df = df.sort_index()

    # -------------------------------------------------------------------------
    # 2. 计算指标 (使用 transform 保持索引对齐)
    # -------------------------------------------------------------------------
    
    # 30日平均成交额
    df["avg_turnover_30"] = (
        df.groupby(level="symbol")["turnover_value"]
        .transform(lambda x: x.rolling(window=30, min_periods=20).mean())
    )
    
    # 30日平均换手率
    df["avg_turnover_rate_30"] = (
        df.groupby(level="symbol")["turnover_rate"]
        .transform(lambda x: x.rolling(window=30, min_periods=20).mean())
    )
    
    # EMA 50 & 150
    df["ema50"] = (
        df.groupby(level="symbol")["close"]
        .transform(lambda x: x.ewm(span=50, adjust=False, min_periods=40).mean())
    )
    
    df["ema150"] = (
        df.groupby(level="symbol")["close"]
        .transform(lambda x: x.ewm(span=150, adjust=False, min_periods=120).mean())
    )

    # -------------------------------------------------------------------------
    # 3. 截取最后一天进行选股 (Snapshot) - 【修复点】
    # -------------------------------------------------------------------------
    last_day = df.index.get_level_values("time").max()
    
    # 【修改前】 snap = df.xs(last_day, level="time").copy() -> 会丢失 time 索引
    # 【修改后】 使用布尔索引，或者 xs(..., drop_level=False)
    # 这里使用布尔索引，确保结果依然是 MultiIndex: (time, symbol)
    snap = df[df.index.get_level_values("time") == last_day].copy()
    
    # -------------------------------------------------------------------------
    # 4. 执行筛选逻辑
    # -------------------------------------------------------------------------

    # --- 0. 国家过滤（只保留美国公司）---
    # 如果缓存中没有 country 字段，则默认全部通过（兼容旧缓存）
    if "country" in snap.columns:
        country_ok = snap["country"].astype(str).str.strip().str.lower() == "united states"
    else:
        country_ok = pd.Series(True, index=snap.index)  # 旧缓存无 country，默认通过
        snap["country"] = None  # 添加空的 country 列

    # --- A. 基础门槛 ---
    mktcap_ok = snap["float_mktcap"] >= 30_000_000
    avg_turnover_ok = snap["avg_turnover_30"] >= 10_000_000
    price_ok = snap["close"] > 1.0

    # --- B. 行业黑名单 ---
    exclude_keywords = [
        "Biotech", "Bio-tech", 
        "Healthcare", "Health Care", 
        "Regional Bank", "Banks - Regional", 
        "Shell Company", "Blank Check", "SPAC", 
        "REIT", "Real Estate",
        # 空壳公司类型
        "Shell Companies", "Acquisition", "Special Purpose",
        "Merger", "Holdings Company"
    ]
    
    s_sector = snap["sector"].astype(str).fillna("").str.lower()
    s_industry = snap["industry"].astype(str).fillna("").str.lower()
    
    def is_blacklisted(series, keywords):
        mask = pd.Series(False, index=series.index)
        for k in keywords:
            mask |= series.str.contains(k.lower(), regex=False)
        return mask

    is_excluded = is_blacklisted(s_sector, exclude_keywords) | \
                  is_blacklisted(s_industry, exclude_keywords)
    
    sector_ok = ~is_excluded

    # --- C. 持仓结构 ---
    def normalize_pct(s):
        """
        将比例归一化到 [0, 1] 区间
        修复：改为逐元素判断，而不是整列判断
        - 如果值 > 1.5 (例如 70 表示 70%)，则除以 100
        - 如果值 <= 1.5 (例如 0.7 表示 70%)，则保持不变
        """
        s = pd.to_numeric(s, errors='coerce')
        # 逐元素判断：值 > 1.5 的才除以 100
        return s.apply(lambda x: x / 100.0 if pd.notna(x) and x > 1.5 else x)

    inst_ratio = normalize_pct(snap.get("institution_pct", pd.Series(0, index=snap.index)))
    inst_ok = inst_ratio < 0.70

    insider_ratio = normalize_pct(snap.get("insider_pct", pd.Series(0, index=snap.index)))
    insider_ok = insider_ratio < 0.30
    
    # 如果30日平均换手率高于2%，忽略机构持股和内部持股条件
    high_turnover = snap["avg_turnover_rate_30"] > 0.02
    structure_ok = (inst_ok & insider_ok) | high_turnover

    # --- D. 技术形态 ---
    ema_ok = ((snap["close"] > snap["ema50"]) & (snap["ema50"] > snap["ema150"])).fillna(False)

    # --- E. 换手率活跃度 ---
    # 换手率使用原始值（已经是小数形式，如 0.035 表示 3.5%）
    # 不需要 normalize_pct，因为换手率不会出现 "70表示70%" 这种情况
    tr_30 = snap["avg_turnover_rate_30"]
    turnover_rate_ok = tr_30 > 0.01

    # -------------------------------------------------------------------------
    # 5. 汇总结果
    # -------------------------------------------------------------------------
    snap["cond_country"] = country_ok  # 新增：国家条件
    snap["cond_mktcap"] = mktcap_ok
    snap["cond_liq"] = avg_turnover_ok
    snap["cond_price"] = price_ok
    snap["cond_sector"] = sector_ok
    snap["cond_structure"] = structure_ok
    snap["cond_trend"] = ema_ok
    snap["cond_activity"] = turnover_rate_ok

    cond_cols = [
        "cond_country",  # 国家条件
        "cond_mktcap", "cond_liq", "cond_price", 
        "cond_sector", "cond_structure", 
        "cond_trend", "cond_activity"
    ]
    snap["in_pool"] = snap[cond_cols].all(axis=1)

    output_cols = [
        "close", "volume", "float_mktcap", "turnover_value",
        "sector", "industry", "country",
        "avg_turnover_30", "avg_turnover_rate_30",
        "ema50", "ema150",
        "institution_pct", "insider_pct",
        "in_pool"
    ] + cond_cols
    
    return snap[output_cols].sort_values(["in_pool", "float_mktcap"], ascending=[False, False])


def add_derived_features(datas: pd.DataFrame) -> pd.DataFrame:
    """
    在原始 datas 上补充技术指标，全程使用 transform 保持索引对齐，
    避免 reset_index 导致的潜在错位。
    """
    df = datas.copy()

    # 1. 确保索引是 (time, symbol) 且排序
    if "symbol" in df.columns and "time" in df.columns:
        df = df.set_index(["time", "symbol"])
    
    if df.index.names != ["time", "symbol"]:
        # 尝试自动修复
        if "symbol" in df.columns:
            df = df.reset_index().set_index(["time", "symbol"])
    
    df = df.sort_index()

    # 2. 基础计算 (使用 transform 效率更高且安全)
    # 均线 (使用 EMA 指数移动平均，对价格变化反应更敏感)
    df["ma5"] = df.groupby(level="symbol")["close"].transform(lambda x: x.ewm(span=5, adjust=False, min_periods=3).mean())
    df["ma10"] = df.groupby(level="symbol")["close"].transform(lambda x: x.ewm(span=10, adjust=False, min_periods=5).mean())
    df["ma20"] = df.groupby(level="symbol")["close"].transform(lambda x: x.ewm(span=20, adjust=False, min_periods=10).mean())
    df["ma50"] = df.groupby(level="symbol")["close"].transform(lambda x: x.ewm(span=50, adjust=False, min_periods=25).mean())

    # 量比分母：5日均量
    df["vol_ma5"] = df.groupby(level="symbol")["volume"].transform(lambda x: x.rolling(5, min_periods=3).mean())

    # 换手率 (基于成交额 / 流通市值)
    # 注意：float_mktcap 可能为0或NaN，需处理
    denom = df["float_mktcap"].replace(0, np.nan)
    df["tr_value"] = df["turnover_value"] / denom

    # 30日平均换手率
    df["avg_tr_value_30"] = df.groupby(level="symbol")["tr_value"].transform(lambda x: x.rolling(30, min_periods=15).mean())

    # 60日振幅: (High_60 / Low_60) - 1
    roll_high_60 = df.groupby(level="symbol")["high"].transform(lambda x: x.rolling(60, min_periods=30).max())
    roll_low_60 = df.groupby(level="symbol")["low"].transform(lambda x: x.rolling(60, min_periods=30).min())
    df["swing_60"] = (roll_high_60 / roll_low_60) - 1.0
    
    # 60日最高价 (用于突破策略)
    df["high_60"] = roll_high_60
    
    # 排除最近10天的历史最高点 (用于突破策略)
    # 逻辑：使用 expanding() 找到10天前及更早的所有历史最高点
    # shift(10) 确保排除最近10天，expanding() 确保取到真正的历史最高点
    df["high_60_ex10"] = df.groupby(level="symbol")["high"].transform(
        lambda x: x.shift(10).expanding(min_periods=1).max()
    )

    # 52周最高价 (252日)
    df["high_252"] = df.groupby(level="symbol")["high"].transform(lambda x: x.rolling(252, min_periods=60).max())

    # 辅助：前一日成交额 (用于低吸逻辑) - 保留兼容
    df["turnover_value_prev"] = df.groupby(level="symbol")["turnover_value"].shift(1)
    
    # 辅助：前一日成交量 (用于缩量判断，比 turnover_value 更准确)
    df["volume_prev"] = df.groupby(level="symbol")["volume"].shift(1)
    
    # 辅助：前一日收盘价 (用于计算涨幅)
    df["close_prev"] = df.groupby(level="symbol")["close"].shift(1)
    
    # 辅助：前一日的前一日收盘价 (用于判断前一天是否下跌)
    df["close_prev2"] = df.groupby(level="symbol")["close"].shift(2)
    
    # 辅助：前一日MA5、MA20、MA50 (用于回调买入判断)
    df["ma5_prev"] = df.groupby(level="symbol")["ma5"].shift(1)
    df["ma20_prev"] = df.groupby(level="symbol")["ma20"].shift(1)
    df["ma50_prev"] = df.groupby(level="symbol")["ma50"].shift(1)
    
    # 辅助：前一日低点 (用于PULLBACK止损)
    df["prev_low"] = df.groupby(level="symbol")["low"].shift(1)
    
    # 辅助：前一日高点 (预留)
    df["prev_high"] = df.groupby(level="symbol")["high"].shift(1)

    # 【突破策略辅助字段】从10天前高点到昨日的最低价
    # 逻辑：使用 shift(1) 后取 rolling(9)，即 T-9 到 T-1 这9天的最低价
    df["low_since_high_to_yesterday"] = df.groupby(level="symbol")["low"].transform(
        lambda x: x.shift(1).rolling(9, min_periods=5).min()
    )
    
    # 【突破策略辅助字段】从最低点以来的最高价（用于判断今天是否是反弹最高点）
    # 逻辑：取最近10天的最高价（包含今天）
    df["high_since_low"] = df.groupby(level="symbol")["high"].transform(
        lambda x: x.rolling(10, min_periods=5).max()
    )

    return df


def low_buy_candidates(datas: pd.DataFrame, base_snap: pd.DataFrame) -> pd.DataFrame:
    """
    【低吸买入策略】
    逻辑：
    1. 缩量：日换手 < 30日均值*0.8 且 量比 < 1
    2. 活跃：60日振幅 > 50%
    3. 企稳：当日涨幅 (-3%, 3%)，且成交额小于昨日
    4. 当天下跌
    5. 当天价格比60日高点回撤 > 15%
    """
    # 1. 准备全量数据计算指标
    df = add_derived_features(datas)
    
    # 2. 锁定第一轮入选的股票
    valid_symbols = base_snap[base_snap["in_pool"]].index.get_level_values("symbol").unique()
    # 只取这些股票的数据（为了计算 rolling 必须取历史数据，不能只取最后一天）
    sub = df.loc[df.index.get_level_values("symbol").isin(valid_symbols)].copy()

    # 3. 计算当日逻辑
    # 取出最后一天
    last_day = sub.index.get_level_values("time").max()
    today = sub.xs(last_day, level="time").copy()

    # --- 条件 A: 缩量 ---
    # 换手率(交易额) < 30日均值 * 0.8
    cond_turn = today["tr_value"] < (today["avg_tr_value_30"] * 0.8)
    # 量比 < 1
    today["vol_ratio"] = today["volume"] / today["vol_ma5"]
    cond_volr = today["vol_ratio"] < 1.0

    # --- 条件 B: 活跃度 ---
    cond_swing = today["swing_60"] > 0.50

    # --- 条件 C: 价格与成交额形态 ---
    # 涨幅 (-3%, 3%)
    today["chg"] = (today["close"] / today["close_prev"] - 1.0)
    cond_chg = today["chg"].between(-0.03, 0.03)
    # 成交量缩量 (今天 < 昨天) - 使用 volume 比 turnover_value 更准确
    cond_tvo = today["volume"] < today["volume_prev"]

    # --- 条件 D: 当天下跌 ---
    cond_down = today["close"] < today["close_prev"]
    
    # --- 条件 E: 当天价格比60日高点回撤 > 15% ---
    today["drawdown_from_high60"] = (today["high_60"] - today["close"]) / today["high_60"]
    cond_drawdown = today["drawdown_from_high60"] > 0.15
    
    # --- 条件 F: 当天收盘价在5日均线之下（确保价格已回调到位，而非高位缩量）---
    cond_below_ma5 = today["close"] < today["ma5"]

    # 4. 汇总
    cond_cols = {
        "cond_swing60": cond_swing,
        "cond_turn_shrink": cond_turn,
        "cond_vol_ratio": cond_volr,
        "cond_chg_range": cond_chg,
        "cond_tvo_shrink": cond_tvo,
        "cond_down": cond_down,
        "cond_drawdown_15pct": cond_drawdown,
        "cond_below_ma5": cond_below_ma5
    }
    
    for k, v in cond_cols.items():
        today[k] = v

    today["low_buy_candidate"] = today[list(cond_cols.keys())].all(axis=1)

    out_cols = [
        "close", "chg", "vol_ratio", "swing_60", 
        "ma5", "ma10", "ma20", "tr_value", "high_60", "drawdown_from_high60",
        "volume", "volume_prev",  # 新增：用于调试成交量
        *cond_cols.keys(), "low_buy_candidate"
    ]
    return today[out_cols].sort_values("low_buy_candidate", ascending=False)


def breakout_buy_candidates(datas: pd.DataFrame, base_snap: pd.DataFrame) -> pd.DataFrame:
    """
    【突破买入策略】
    """
    # 1. 基础指标计算
    df = add_derived_features(datas)
    
    # 2. 锁定第一轮入选的股票
    valid_symbols = base_snap[base_snap["in_pool"]].index.get_level_values("symbol").unique()
    sub = df.loc[df.index.get_level_values("symbol").isin(valid_symbols)].copy()

    # ============================================================
    # ============================================================
    
    # A. 计算过去252天的滚动最大回撤
    # 滚动最高价 (High Watermark)
    sub["roll_max_252"] = sub.groupby(level="symbol")["high"].transform(lambda x: x.rolling(252, min_periods=60).max())
    # 每日回撤幅度
    sub["dd_pct"] = 1.0 - (sub["low"] / sub["roll_max_252"])
    # 过去252天内的最大回撤 (Max Drawdown)
    sub["max_dd_252"] = sub.groupby(level="symbol")["dd_pct"].transform(lambda x: x.rolling(252, min_periods=60).max())

    # B. 计算盘整判定辅助列 (T-1 和 T-11 的长期高点比较)
    # 逻辑：如果 T-1 的 252日高点 == T-11 的 252日高点，说明最近10天没有刷新长期高点 -> 盘整中
    high_252_series = sub["high_252"]
    sub["h252_prev"] = high_252_series.groupby(level="symbol").shift(1)
    sub["h252_prev_11"] = high_252_series.groupby(level="symbol").shift(11)
    
    # C. 【新增】计算从10天前高点到昨日的最低价
    # 逻辑：
    # - high_60_ex10 是 T-10 到 T-60 之间的最高价（即10天以上前的高点）
    # - 我们需要找到该高点之后到昨日之间的最低价（不包含今天）
    # - 使用 shift(1) 后取 rolling(9)，即 T-9 到 T-1 这9天的最低价
    sub["low_since_high_to_yesterday"] = sub.groupby(level="symbol")["low"].transform(
        lambda x: x.shift(1).rolling(9, min_periods=5).min()
    )
    
    # D. 【新增】计算从最低点以来的最高价（用于判断今天是否是反弹最高点）
    # 逻辑：取最近10天的最高价（包含今天），如果今天的收盘价等于这个最高价，说明今天是反弹最高点
    sub["high_since_low"] = sub.groupby(level="symbol")["high"].transform(
        lambda x: x.rolling(10, min_periods=5).max()
    )
    
    # ============================================================
    # 3. 切片取出“最后一天” (此时 today 已包含上述计算的新列)
    # ============================================================
    last_day = sub.index.get_level_values("time").max()
    today = sub.xs(last_day, level="time").copy()

    # 4. 计算当日逻辑判断
    
    # --- 条件 A: 活跃与位置 ---
    # 当前价格接近 60日高点 (90%~100%之间，即距离高点10%以内但还未突破)
    # 使用 high_60_ex10：排除最近10天的高点，确保取到的高点是10天前形成的
    cond_near_high = (today["close"] >= (today["high_60_ex10"] * 0.90)) & (today["close"] < today["high_60_ex10"])
    
    # 新增条件：最近10天内不能创新高（即 high_60 ≈ high_60_ex10，允许1%误差）
    # 如果 high_60 > high_60_ex10，说明最近10天创了新高，应该排除
    cond_no_recent_high = today["high_60"] <= (today["high_60_ex10"] * 1.01)
    
    # 【新增】严格排除当天已突破的股票：当天最高价不能超过10天前的高点
    cond_not_breakout_today = today["high"] <= today["high_60_ex10"]

    # --- 条件 B: 回撤与盘整 ---
    # 最大回撤 < 30%
    cond_dd = today["max_dd_252"] < 0.30
    
    # 盘整时间 > 10天
    # 比较刚才算好的 shift 列
    # 容差比较，防止浮点数微小差异
    consolidation_mask = (today["h252_prev"] <= today["h252_prev_11"] * 1.0001) & \
                         (today["h252_prev"] >= today["h252_prev_11"] * 0.9999)
    
    # 填充 False (防止刚上市数据不足导致 NaN)
    cond_consol = consolidation_mask.fillna(False)
    
    # --- 条件 C: 【新增】从10天前高点到昨日最低点的回撤在15%~40%之间 ---
    # 回撤幅度 = (高点 - 最低点) / 高点
    today["drawdown_from_high"] = (today["high_60_ex10"] - today["low_since_high_to_yesterday"]) / today["high_60_ex10"]
    cond_drawdown_range = (today["drawdown_from_high"] >= 0.15) & (today["drawdown_from_high"] <= 0.40)
    
    # --- 条件 D: 【新增】今天是低点以来的最高点（反弹最高点）---
    # 逻辑：今天的最高价应该是最近10天内的最高价，说明今天是反弹阶段的最强一天
    # 使用 high 而不是 close，允许盘中创新高的情况
    cond_is_rebound_high = today["high"] >= today["high_since_low"]

    # 5. 汇总结果
    cond_cols = {
        "cond_near_60d": cond_near_high,  # 接近60日高点
        "cond_no_recent_high": cond_no_recent_high,  # 最近10天没有创新高
        "cond_not_breakout_today": cond_not_breakout_today,  # 当天没有突破
        "cond_drawdown_range": cond_drawdown_range,  # 回撤在15%~40%之间
        "cond_is_rebound_high": cond_is_rebound_high  # 今天是反弹最高点
    }
    
    for k, v in cond_cols.items():
        today[k] = v

    today["breakout_candidate"] = today[list(cond_cols.keys())].all(axis=1)

    # 输出列
    out_cols = [
        "close", "high", "high_60", "high_60_ex10", "high_252", "swing_60", "max_dd_252",
        "low_since_high_to_yesterday", "drawdown_from_high", "high_since_low",  # 输出新计算的列
        *cond_cols.keys(), "breakout_candidate"
    ]
    
    return today[out_cols].sort_values("breakout_candidate", ascending=False)


def pullback_buy_candidates(datas: pd.DataFrame, base_snap: pd.DataFrame) -> pd.DataFrame:
    """
    【回调买入策略】- 四点结构
    
    寻找经典的"W底变形"结构：
    
    价格
      │     ⭐ 点1 (60日最高点)
      │    /\
      │   /  \
      │  /    \        ⭐ 点3 (反弹高点)
      │ /      \      /\
      │/        \    /  \
      │          \  /    \  ← 点4区域（买入点）
      │           \/      \/
      │         ⭐ 点2    ⭐ 点4
      │        (最低点)  (更高的低点)
      └─────────────────────────────→ 时间
    
    条件：
    1. 点1: 60日内的最高点
    2. 点2: 点1之后的最低点，回撤 > 20%（相对于点1）
    3. 点3: 点2之后的反弹最高点，反弹 > 15%（相对于点2）
    4. 点4: 点3之后的回调低点，且 点4 > 点2（形成更高的低点）
    5. 当前处于点4区域，准备再次上攻
    """
    # 1. 基础指标计算
    df = add_derived_features(datas)
    
    # 2. 锁定第一轮入选的股票
    valid_symbols = base_snap[base_snap["in_pool"]].index.get_level_values("symbol").unique()
    sub = df.loc[df.index.get_level_values("symbol").isin(valid_symbols)].copy()

    # 新增：补充换手率、量比、成交额缩量等列
    # 量比分母：5日均量
    sub["vol_ma5"] = sub.groupby(level="symbol")["volume"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    # 换手率(交易额/流通市值)
    denom = sub["float_mktcap"].replace(0, np.nan)
    sub["tr_value"] = sub["turnover_value"] / denom
    # 30日平均换手率
    sub["avg_tr_value_30"] = sub.groupby(level="symbol")["tr_value"].transform(lambda x: x.rolling(30, min_periods=15).mean())
    # 前一日成交额
    sub["turnover_value_prev"] = sub.groupby(level="symbol")["turnover_value"].shift(1)
    # 量比
    sub["vol_ratio"] = sub["volume"] / sub["vol_ma5"]
    
    # ============================================================
    # 3. 逐股票计算四点结构（精确计算）
    # ============================================================
    
    def calc_four_points(group):
        """
        计算单只股票的四点结构
        返回最后一天的四点信息
        
        【重要修正】点4定义：
        - 点4必须是最后一日（信号日/昨天）
        - 且最后一日必须是点3之后的最低点
        - 如果点3和最后一日之间有更低的点，图形被破坏，不满足条件
        """
        # 只取最近60天
        if len(group) < 30:
            return pd.Series({
                "point1_high": np.nan, "point1_days_ago": np.nan,
                "point2_low": np.nan, "point2_days_ago": np.nan,
                "point3_high": np.nan, "point3_days_ago": np.nan,
                "point4_low": np.nan, "point4_days_ago": np.nan,
                "point4_is_last_day": False,  # 新增
                "drawdown_p1_p2": np.nan, "rebound_p2_p3": np.nan,
                "drawdown_p3_p4": np.nan, "dist_to_point4": np.nan
            })
        
        recent = group.tail(60).copy()
        recent = recent.reset_index(drop=True)  # 使用数字索引便于计算
        n = len(recent)
        
        # 点1: 60日最高点
        p1_idx = recent["high"].idxmax()
        p1_high = recent["high"].iloc[p1_idx]
        p1_days_ago = n - 1 - p1_idx
        
        # 点1必须在5天以上前
        if p1_days_ago < 5:
            return pd.Series({
                "point1_high": p1_high, "point1_days_ago": p1_days_ago,
                "point2_low": np.nan, "point2_days_ago": np.nan,
                "point3_high": np.nan, "point3_days_ago": np.nan,
                "point4_low": np.nan, "point4_days_ago": np.nan,
                "point4_is_last_day": False,  # 新增
                "drawdown_p1_p2": np.nan, "rebound_p2_p3": np.nan,
                "drawdown_p3_p4": np.nan, "dist_to_point4": np.nan
            })
        
        # 点2: 点1之后的最低点
        after_p1 = recent.iloc[p1_idx+1:]
        if len(after_p1) < 3:
            return pd.Series({
                "point1_high": p1_high, "point1_days_ago": p1_days_ago,
                "point2_low": np.nan, "point2_days_ago": np.nan,
                "point3_high": np.nan, "point3_days_ago": np.nan,
                "point4_low": np.nan, "point4_days_ago": np.nan,
                "point4_is_last_day": False,  # 新增
                "drawdown_p1_p2": np.nan, "rebound_p2_p3": np.nan,
                "drawdown_p3_p4": np.nan, "dist_to_point4": np.nan
            })
        
        p2_idx = after_p1["low"].idxmin()
        p2_low = after_p1["low"].iloc[after_p1.index.get_loc(p2_idx)]
        p2_days_ago = n - 1 - p2_idx
        
        drawdown = (p1_high - p2_low) / p1_high
        
        # 点2必须在3天以上前（留出反弹空间）
        if p2_days_ago < 2:
            return pd.Series({
                "point1_high": p1_high, "point1_days_ago": p1_days_ago,
                "point2_low": p2_low, "point2_days_ago": p2_days_ago,
                "point3_high": np.nan, "point3_days_ago": np.nan,
                "point4_low": np.nan, "point4_days_ago": np.nan,
                "point4_is_last_day": False,  # 新增
                "drawdown_p1_p2": drawdown, "rebound_p2_p3": np.nan,
                "drawdown_p3_p4": np.nan, "dist_to_point4": np.nan
            })
        
        # 点3: 点2之后的反弹最高点
        after_p2 = recent.iloc[p2_idx+1:]
        if len(after_p2) < 2:
            return pd.Series({
                "point1_high": p1_high, "point1_days_ago": p1_days_ago,
                "point2_low": p2_low, "point2_days_ago": p2_days_ago,
                "point3_high": np.nan, "point3_days_ago": np.nan,
                "point4_low": np.nan, "point4_days_ago": np.nan,
                "point4_is_last_day": False,  # 新增
                "drawdown_p1_p2": drawdown, "rebound_p2_p3": np.nan,
                "drawdown_p3_p4": np.nan, "dist_to_point4": np.nan
            })
        
        p3_idx = after_p2["high"].idxmax()
        p3_high = after_p2["high"].iloc[after_p2.index.get_loc(p3_idx)]
        p3_days_ago = n - 1 - p3_idx
        
        rebound = (p3_high - p2_low) / p2_low
        
        # 点3必须在2天以上前（留出回调空间，点3和点4之间>=2天）
        if p3_days_ago < 2:
            return pd.Series({
                "point1_high": p1_high, "point1_days_ago": p1_days_ago,
                "point2_low": p2_low, "point2_days_ago": p2_days_ago,
                "point3_high": p3_high, "point3_days_ago": p3_days_ago,
                "point4_low": np.nan, "point4_days_ago": np.nan,
                "point4_is_last_day": False,  # 新增
                "drawdown_p1_p2": drawdown, "rebound_p2_p3": rebound,
                "drawdown_p3_p4": np.nan, "dist_to_point4": np.nan
            })
        
        # 点4: 【修正】点4必须是最后一日（昨天/信号日），且是点3之后的最低点
        # 逻辑：首先检查最后一日是否是点3之后的最低点
        # 如果点3和最后一日之间有更低的点，说明图形被破坏，不满足条件
        after_p3 = recent.iloc[p3_idx+1:]
        if len(after_p3) < 1:
            return pd.Series({
                "point1_high": p1_high, "point1_days_ago": p1_days_ago,
                "point2_low": p2_low, "point2_days_ago": p2_days_ago,
                "point3_high": p3_high, "point3_days_ago": p3_days_ago,
                "point4_low": np.nan, "point4_days_ago": np.nan,
                "point4_is_last_day": False,  # 新增：标记点4是否是最后一日
                "drawdown_p1_p2": drawdown, "rebound_p2_p3": rebound,
                "drawdown_p3_p4": np.nan, "dist_to_point4": np.nan
            })
        
        # 【关键修改】点4定义为最后一日（信号日/昨天）
        # 取最后一日的最低价作为点4
        last_day_low = recent["low"].iloc[-1]  # 最后一日的最低价
        
        # 检查最后一日是否是点3之后的最低点
        min_low_after_p3 = after_p3["low"].min()  # 点3之后的最低价
        point4_is_last_day = (last_day_low <= min_low_after_p3 * 1.001)  # 允许0.1%误差
        
        # 点4就是最后一日
        p4_low = last_day_low
        p4_days_ago = 0  # 点4就是最后一日，距今0天
        
        # 点3到点4的回撤幅度
        drawdown_p3_p4 = (p3_high - p4_low) / p3_high
        
        # 当前价格与点4的距离（点4就是最后一日，所以用收盘价）
        current_close = recent["close"].iloc[-1]
        dist_to_p4 = (current_close - p4_low) / p4_low
        
        return pd.Series({
            "point1_high": p1_high, "point1_days_ago": p1_days_ago,
            "point2_low": p2_low, "point2_days_ago": p2_days_ago,
            "point3_high": p3_high, "point3_days_ago": p3_days_ago,
            "point4_low": p4_low, "point4_days_ago": p4_days_ago,
            "point4_is_last_day": point4_is_last_day,  # 新增：标记点4是否是最后一日的最低点
            "drawdown_p1_p2": drawdown, "rebound_p2_p3": rebound,
            "drawdown_p3_p4": drawdown_p3_p4, "dist_to_point4": dist_to_p4
        })
    
    # 对每只股票计算四点结构
    last_day = sub.index.get_level_values("time").max()
    
    # 按symbol分组计算
    results = []
    for symbol in valid_symbols:
        try:
            group = sub.xs(symbol, level="symbol")
            four_points = calc_four_points(group)
            four_points["symbol"] = symbol
            results.append(four_points)
        except Exception as e:
            continue
    
    if not results:
        # 返回空DataFrame
        return pd.DataFrame()
    
    # 合并结果
    points_df = pd.DataFrame(results).set_index("symbol")
    
    # 获取最后一天的价格数据
    today = sub.xs(last_day, level="time").copy()

    # 合并四点数据
    today = today.join(points_df, how="left")

    # 新增：回调买入附加条件
    # 换手率(交易额) < 30日均值 * 0.8
    cond_turn = today["tr_value"] < (today["avg_tr_value_30"] * 0.8)
    # 量比 < 1
    cond_volr = today["vol_ratio"] < 1.0
    # 成交量缩量 (今天 < 昨天) - 使用 volume 比 turnover_value 更准确
    cond_tvo = today["volume"] < today["volume_prev"]
    # 前一天收盘价低于前一天的MA5（确保点4当天价格已回调到位）
    cond_below_ma5 = today["close_prev"] < today["ma5_prev"]
    # 前一天收盘价高于前一天的MA50（确保趋势向上，提高筛选标准）
    cond_above_ma50 = today["close_prev"] > today["ma50_prev"]
    
    # ============================================================
    # 4. 条件判断
    # ============================================================
    
    # 条件1: 点1在10天以上前（确保有足够的回调空间）
    cond_point1_timing = today["point1_days_ago"] >= 10
    
    # 条件2: 点2在点1之后（点2距今天数 < 点1距今天数）
    cond_point2_after_point1 = today["point2_days_ago"] < today["point1_days_ago"]
    
    # 条件3: 回撤幅度 > 20%
    cond_drawdown = today["drawdown_p1_p2"] >= 0.20
    
    # 条件4: 点3在点2之后（点3距今天数 < 点2距今天数）
    cond_point3_after_point2 = today["point3_days_ago"] < today["point2_days_ago"]
    
    # 条件5: 反弹幅度 > 15%
    cond_rebound = today["rebound_p2_p3"] >= 0.15
    
    # 条件6: 点4 > 点2（形成更高的低点）
    cond_higher_low = today["point4_low"] > today["point2_low"]
    
    # 条件7: 【修改】点4必须是最后一日，且是点3之后的最低点
    # 如果点3和最后一日之间有更低的点，图形被破坏，不满足条件
    cond_point4_is_last_day = today["point4_is_last_day"].fillna(False)
    
    # 条件8: 【删除旧的"接近点4"条件，因为点4现在就是最后一日】
    # 保留dist_to_point4字段用于参考，但不作为筛选条件
    # cond_near_point4 = today["dist_to_point4"] <= 0.10  # 已删除
    
    # 条件9: 点3不能超过点1（否则就是新高突破，不是回调）
    cond_point3_below_point1 = today["point3_high"] < today["point1_high"]
    
    # 条件10: 前一天（点4当天/最后一日）是下跌的
    # 修改逻辑：检查前一天收盘价 < 前一天的前一天收盘价
    cond_last_day_down = today["close_prev"] < today["close_prev2"]
    
    # 条件11: 点3和点4之间至少间隔2天（点4是最后一日，所以就是点3距今>=2天）
    cond_p3_p4_gap = today["point3_days_ago"] >= 2
    
    # 条件12: 点3到点4的回撤幅度 >= 5%
    cond_p3_p4_drawdown = today["drawdown_p3_p4"] >= 0.05

    # ============================================================
    # 5. 汇总结果
    # ============================================================
    cond_cols = {
        "cond_point1_timing": cond_point1_timing,           # 点1在10天以上前
        "cond_point2_after_p1": cond_point2_after_point1,   # 点2在点1之后
        "cond_drawdown_20pct": cond_drawdown,               # 回撤 > 20%
        "cond_point3_after_p2": cond_point3_after_point2,   # 点3在点2之后
        "cond_rebound_15pct": cond_rebound,                 # 反弹 > 15%
        "cond_higher_low": cond_higher_low,                 # 点4 > 点2
        "cond_point4_is_last_day": cond_point4_is_last_day, # 【新增】点4是最后一日且是点3后最低点
        "cond_p3_below_p1": cond_point3_below_point1,       # 点3 < 点1
        "cond_last_day_down": cond_last_day_down,           # 最后一天下跌
        "cond_p3_p4_gap": cond_p3_p4_gap,                   # 点3和点4间隔>=2天
        "cond_p3_p4_drawdown": cond_p3_p4_drawdown,         # 点3到点4回撤>=5%
        # 新增四项
        "cond_turn_shrink": cond_turn,
        "cond_vol_ratio": cond_volr,
        "cond_tvo_shrink": cond_tvo,
        "cond_below_ma5": cond_below_ma5,                   # 收盘价低于EMA5
        "cond_above_ma50": cond_above_ma50                  # 收盘价高于EMA50
    }

    for k, v in cond_cols.items():
        today[k] = v.fillna(False)

    today["pullback_candidate"] = today[list(cond_cols.keys())].all(axis=1)

    # 输出列
    out_cols = [
        "close", "high", "low",
        "point1_high", "point1_days_ago",
        "point2_low", "point2_days_ago", "drawdown_p1_p2",
        "point3_high", "point3_days_ago", "rebound_p2_p3",
        "point4_low", "point4_days_ago", "point4_is_last_day", "drawdown_p3_p4", "dist_to_point4",  # 新增 point4_is_last_day
        "tr_value", "avg_tr_value_30", "vol_ratio", "turnover_value", "turnover_value_prev",
        *cond_cols.keys(), "pullback_candidate"
    ]

    # 确保所有列都存在
    for col in out_cols:
        if col not in today.columns:
            today[col] = np.nan

    return today[out_cols].sort_values("pullback_candidate", ascending=False)
# 使用示例
if __name__ == "__main__":
    symbols = read_stock_list(file_path)
    if not symbols is None:
        # symbol 列表，bp500_list
        datas = yahoo_datas(
            symbols,
            n_days=252,
            use_proxy=True,
            http_proxy="http://127.0.0.1:4780",
            https_proxy="http://127.0.0.1:4780",
            max_workers=4
        )

        # 第一轮筛选
        pool = build_base_universe(datas)
        
        print(f"\n{'='*80}")
        print(f"📊 基础选股池统计")
        print(f"{'='*80}")
        print(f"   总股票数: {len(pool)}")
        print(f"   入选股票数: {pool['in_pool'].sum()}")
        print(f"   入选率: {pool['in_pool'].sum()/len(pool)*100:.2f}%")
        print(f"{'='*80}\n")
        
        # 打印入选的股票列表
        selected_pool = pool[pool["in_pool"]].copy()
        if len(selected_pool) > 0:
            print(f"\n{'='*80}")
            print(f"✅ 入选基础股票池的股票 (共 {len(selected_pool)} 只)")
            print(f"{'='*80}\n")
            
            # 重置索引以便打印
            selected_display = selected_pool.reset_index()
            
            for idx, row in selected_display.iterrows():
                symbol = row['symbol']
                print(f"\n【{idx+1}】 {symbol}")
                print(f"   股价: ${row['close']:.2f}")
                print(f"   行业: {row['sector'] if pd.notna(row['sector']) else 'N/A'}")
                print(f"   板块: {row['industry'] if pd.notna(row['industry']) else 'N/A'}")
                print(f"   流通市值: ${row['float_mktcap']:,.0f}" if pd.notna(row['float_mktcap']) else "   流通市值: N/A")
                print(f"   30日均成交额: ${row['avg_turnover_30']:,.0f}" if pd.notna(row['avg_turnover_30']) else "   30日均成交额: N/A")
                print(f"   30日均换手率: {row['avg_turnover_rate_30']*100:.2f}%" if pd.notna(row['avg_turnover_rate_30']) else "   30日均换手率: N/A")
                print(f"   EMA50: ${row['ema50']:.2f}" if pd.notna(row['ema50']) else "   EMA50: N/A")
                print(f"   EMA150: ${row['ema150']:.2f}" if pd.notna(row['ema150']) else "   EMA150: N/A")
                print(f"   机构持股: {row['institution_pct']*100:.1f}%" if pd.notna(row['institution_pct']) else "   机构持股: N/A")
                print(f"   内部持股: {row['insider_pct']*100:.1f}%" if pd.notna(row['insider_pct']) else "   内部持股: N/A")
            
            print(f"\n{'='*80}")
            print(f"📋 基础股票池汇总表")
            print(f"{'='*80}\n")
            
            # 打印汇总表格
            summary_cols = ["close", "sector", "industry", "float_mktcap", "avg_turnover_30", "avg_turnover_rate_30", "ema50", "ema150"]
            summary_display = selected_display[["symbol"] + summary_cols].copy()
            summary_display["avg_turnover_rate_30"] = summary_display["avg_turnover_rate_30"] * 100  # 转换为百分比
            print(summary_display.to_string(index=False))
            print(f"\n{'='*80}\n")
        else:
            print("\n⚠️ 没有股票通过基础筛选！\n")
        
        # 查看被淘汰的原因统计
        print(f"\n{'='*80}")
        print(f"📉 筛选条件通过率统计")
        print(f"{'='*80}")
        print(f"   美国公司: {pool['cond_country'].sum()} / {len(pool)} ({pool['cond_country'].sum()/len(pool)*100:.1f}%) 🇺🇸")
        print(f"   市值门槛: {pool['cond_mktcap'].sum()} / {len(pool)} ({pool['cond_mktcap'].sum()/len(pool)*100:.1f}%)")
        print(f"   流动性: {pool['cond_liq'].sum()} / {len(pool)} ({pool['cond_liq'].sum()/len(pool)*100:.1f}%)")
        print(f"   股价门槛: {pool['cond_price'].sum()} / {len(pool)} ({pool['cond_price'].sum()/len(pool)*100:.1f}%)")
        print(f"   行业筛选: {pool['cond_sector'].sum()} / {len(pool)} ({pool['cond_sector'].sum()/len(pool)*100:.1f}%)")
        print(f"   EMA趋势: {pool['cond_trend'].sum()} / {len(pool)} ({pool['cond_trend'].sum()/len(pool)*100:.1f}%) ⭐ 关键")
        print(f"   换手率活跃: {pool['cond_activity'].sum()} / {len(pool)} ({pool['cond_activity'].sum()/len(pool)*100:.1f}%)")
        print(f"{'='*80}\n")
        
        # 查看被淘汰的原因示例
        print(f"\n{'='*80}")
        print(f"❌ 被淘汰股票示例 (趋势不符合)")
        print(f"{'='*80}")
        rejected = pool[~pool["in_pool"] & pool["cond_mktcap"] & ~pool["cond_trend"]]
        if len(rejected) > 0:
            rejected_display = rejected.head(5).reset_index()
            print(rejected_display[["symbol", "close", "ema50", "ema150"]].to_string(index=False))
        else:
            print("无示例")
        print(f"{'='*80}\n")

        # 假设：
        # datas = 这些股票的历史数据（(time,symbol) MultiIndex）
        # pool  = “第一轮当日快照”，索引是 (time, symbol)，包含 in_pool 列

        last_day = pool.index.get_level_values("time").max()
        #print(pool)
        base_snap = pool.xs(last_day, level="time")[["in_pool"]]
        base_snap = base_snap.reset_index()
        base_snap["time"] = last_day
        base_snap = base_snap.set_index(["time","symbol"])[["in_pool"]]
        #print(base_snap)

        # 第二轮筛选
        low_buy = low_buy_candidates(datas, base_snap)
        breakout_buy = breakout_buy_candidates(datas, base_snap)
        pullback_buy = pullback_buy_candidates(datas, base_snap)  # 新增：回调买入

        # 结果
        low_list = low_buy.index[low_buy["low_buy_candidate"]].tolist()
        brk_list = breakout_buy.index[breakout_buy["breakout_candidate"]].tolist()
        pullback_list = pullback_buy.index[pullback_buy["pullback_candidate"]].tolist()  # 新增
        
        # 打印详细的买入候选信息
        print(f"\n{'='*80}")
        print(f"📈 低吸买入候选 (共 {len(low_list)} 只)")
        print(f"{'='*80}")
        if low_list:
            for sym in low_list:
                row = low_buy.loc[sym]
                print(f"  【{sym}】 买入价: ${row['close']:.2f} | MA20: ${row['ma20']:.2f} | 回撤: {row['drawdown_from_high60']*100:.1f}%")
                print(f"        今日量: {int(row['volume']):,} | 昨日量: {int(row['volume_prev']):,} | 量比: {row['vol_ratio']:.2f} | 缩量: {row['cond_tvo_shrink']}")
        else:
            print("  无")
        
        print(f"\n{'='*80}")
        print(f"🚀 突破买入候选 (共 {len(brk_list)} 只)")
        print(f"{'='*80}")
        if brk_list:
            for sym in brk_list:
                row = breakout_buy.loc[sym]
                print(f"  【{sym}】 买入价: ${row['close']:.2f} | 突破高点: ${row['high_60_ex10']:.2f} | 距高点: {(row['high_60_ex10']-row['close'])/row['high_60_ex10']*100:.1f}%")
        else:
            print("  无")
        
        print(f"\n{'='*80}")
        print(f"🔄 回调买入候选 (共 {len(pullback_list)} 只)")
        print(f"{'='*80}")
        if pullback_list:
            for sym in pullback_list:
                row = pullback_buy.loc[sym]
                print(f"  【{sym}】 买入价: ${row['close']:.2f} | 点1高: ${row['point1_high']:.2f} | 点4低: ${row['point4_low']:.2f} | 距点4: {row['dist_to_point4']*100:.1f}%")
        else:
            print("  无")
        print(f"{'='*80}\n")
        
        # 合并信号表
        signals = (
            low_buy[["low_buy_candidate"]]
            .join(breakout_buy[["breakout_candidate"]], how="outer")
            .join(pullback_buy[["pullback_candidate"]], how="outer")  # 新增
            .fillna(False)
            .sort_index()
        )

        today_str = datetime.now().strftime("%Y-%m-%d")
        save_file = today_str + '_选股.json'
        print("保存选股文件...",save_file)
        os.makedirs(save_dir, exist_ok=True)
        signals.to_json(os.path.join(save_dir, save_file),orient='table',indent=2)
