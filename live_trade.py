from __future__ import annotations

import os
import sys
import time
import math
import warnings
import datetime
import contextlib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bridge import mt5_bridge as mt5

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    from torch.amp import autocast as _amp_autocast
    USE_AMP = DEVICE.type == 'cuda'
except ImportError:
    _amp_autocast = None
    USE_AMP = False

def _amp_ctx():
    if USE_AMP:
        return _amp_autocast(device_type='cuda', enabled=True)
    return contextlib.nullcontext()

try:
    import yfinance as yf
except ImportError:
    import subprocess
    print("[SYSTEM] yfinance not detected. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

# -----------------------------------------------------------------------------
# 1. LIVE TRADING CONFIGURATION
# -----------------------------------------------------------------------------
MT5_LOGIN = 10011686584               # <--- ENTER YOUR MT5 ACCOUNT LOGIN HERE
MT5_PASSWORD = "SmWIN_X1"       # <--- ENTER YOUR MT5 PASSWORD HERE
MT5_SERVER = "MetaQuotes-Demo"   # <--- ENTER YOUR MT5 SERVER NAME HERE

# Suffix or prefix for broker symbols (e.g. "EURUSD.m" -> set suffix to ".m")
SYMBOL_PREFIX = ""
SYMBOL_SUFFIX = ""

# Sizing parameters
MAX_LOT_SIZE = 0.1                   # Max trade size per asset when allocation = 1.0 or -1.0
MAGIC_NUMBER = 881005                # Unique identifier for APEX orders
DEVIATION = 20                       # Max price slippage in points

# Model configuration matching training run
@dataclass
class OmegaConfig:
    PAIRS_MAP: Dict[str, str] = field(default_factory=lambda: {
        "EURUSD=X": "EURUSD", "GBPUSD=X": "GBPUSD",
        "JPY=X":    "USDJPY", "AUDUSD=X": "AUDUSD",
        "GC=F":     "XAUUSD",
    })
    MACRO_MAP: Dict[str, str] = field(default_factory=lambda: {
        "^TNX": "US10Y", "^VIX": "VIX", "DX-Y.NYB": "DXY", "CL=F": "CRUDE",
    })
    YIELD_MAP: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        "EURUSD": ("DXY", "US10Y"),
        "GBPUSD": ("DXY", "US10Y"),
        "USDJPY": ("US10Y", "VIX"),
        "AUDUSD": ("CRUDE", "US10Y"),
        "XAUUSD": ("VIX", "DXY"),
    })
    CURRENCY_MAP: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        "EURUSD": ("EUR", "USD"),
        "GBPUSD": ("GBP", "USD"),
        "USDJPY": ("USD", "JPY"),
        "AUDUSD": ("AUD", "USD"),
        "XAUUSD": ("XAU", "USD"),
    })
    PAIRS:           List[str] = field(init=False)
    NUM_ASSETS:      int       = field(init=False)

    CONTEXT_5m:   int  = 60
    CONTEXT_15m:  int  = 48
    CONTEXT_1h:   int  = 24
    CONTEXT_4h:   int  = 12
    FUTURE_5m:    int  = 12

    D_LATENT:     int  = 128
    D_MICRO:      int  = 64
    D_MACRO:      int  = 128

    def __post_init__(self):
        self.PAIRS      = list(self.PAIRS_MAP.values())
        self.NUM_ASSETS = len(self.PAIRS)

# Aligned features produced by _engineer_features to match slices correctly
PAIR_FEATS = ['Close', 'High', 'Low', 'Vol', 'Imb', 'ATR14', 'MACD', 'BB', 'RoC5', 'RoC20', 'Thrust', 'SwingHi', 'SwingLo']
N_PAIR_FEATS = len(PAIR_FEATS)

EVENT_TYPES = [
    "NFP", "CPI", "GDP", "RATE", "PMI", "RETAIL", "EMPLOYMENT",
    "TRADE_BAL", "HOUSING", "CONSUMER", "PPI", "SPEECH",
    "MINUTES", "MANUFACTURING", "OTHER_HIGH",
]
N_EVENT_TYPES = len(EVENT_TYPES)

# -----------------------------------------------------------------------------
# HELPER UTILITIES
# -----------------------------------------------------------------------------
def _to_series(col_data) -> pd.Series:
    if isinstance(col_data, pd.DataFrame):
        return col_data.iloc[:, 0]
    return col_data

def _flatten_yf(df_t: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df_t.columns, pd.MultiIndex):
        df_t = df_t.copy()
        df_t.columns = df_t.columns.get_level_values(0)
    return df_t

def classify_event(event_name: str) -> int:
    if not isinstance(event_name, str):
        return N_EVENT_TYPES - 1
    e = event_name.lower()
    if 'non-farm' in e or 'nonfarm' in e or 'nfp' in e:
        return 0
    if 'consumer price' in e or 'cpi' in e or 'inflation' in e:
        return 1
    if 'gdp' in e or 'gross domestic' in e:
        return 2
    if 'interest rate' in e or 'rate decision' in e or 'cash rate' in e or 'bank rate' in e or 'refinancing rate' in e:
        return 3
    if 'pmi' in e or 'purchasing manager' in e:
        return 4
    if 'retail sale' in e:
        return 5
    if 'employment' in e or 'unemployment' in e or 'jobless' in e or 'jobs' in e or 'payroll' in e:
        return 6
    if 'trade balance' in e or 'current account' in e:
        return 7
    if 'housing' in e or 'building permit' in e or 'home sale' in e or 'existing home' in e:
        return 8
    if 'consumer confidence' in e or 'consumer sentiment' in e or 'michigan' in e:
        return 9
    if 'producer price' in e or 'ppi' in e:
        return 10
    if 'speak' in e or 'speech' in e or 'testimony' in e or 'press conference' in e:
        return 11
    if 'minute' in e or 'meeting' in e or 'fomc' in e:
        return 12
    if 'industrial' in e or 'manufacturing' in e or 'factory' in e or 'production' in e:
        return 13
    return 14

def fetch_yield_data_aligned(cfg: OmegaConfig, target_df: pd.DataFrame, interval: str, period: str) -> None:
    tickers = ["^TNX", "^VIX", "DX-Y.NYB", "CL=F"]
    for tick in tickers:
        name = cfg.MACRO_MAP.get(tick)
        try:
            df_y = yf.download(tick, period=period, interval=interval, progress=False)
            if not df_y.empty:
                df_y = _flatten_yf(df_y)
                if df_y.index.tz is not None:
                    df_y.index = df_y.index.tz_convert("UTC").tz_localize(None)
                close = _to_series(df_y["Close"]).reindex(target_df.index).ffill().bfill()
                target_df[f"{name}_Close"] = close.values
            else:
                df_y_alt = yf.download(tick, period="30d", interval="1h", progress=False)
                if not df_y_alt.empty:
                    df_y_alt = _flatten_yf(df_y_alt)
                    if df_y_alt.index.tz is not None:
                        df_y_alt.index = df_y_alt.index.tz_convert("UTC").tz_localize(None)
                    close = _to_series(df_y_alt["Close"]).reindex(target_df.index, method='ffill').ffill().bfill()
                    target_df[f"{name}_Close"] = close.values
                else:
                    target_df[f"{name}_Close"] = 0.0
        except Exception as e:
            print(f"[YIELD] Warning: Error fetching yield {tick}: {e}. Setting 0.0.")
            target_df[f"{name}_Close"] = 0.0

def get_nfp_proximity(timestamps: pd.DatetimeIndex) -> np.ndarray:
    prox = np.zeros(len(timestamps), dtype=np.float32)
    for i, t in enumerate(timestamps):
        first_day = datetime.datetime(t.year, t.month, 1)
        w = first_day.weekday()
        first_friday_day = 1 + (4 - w) if w <= 4 else 1 + (11 - w)
        nfp_time = datetime.datetime(t.year, t.month, first_friday_day, 13, 30)
        
        if t > nfp_time:
            nm = t.month + 1
            ny = t.year
            if nm > 12:
                nm = 1
                ny += 1
            first_day_next = datetime.datetime(ny, nm, 1)
            w_next = first_day_next.weekday()
            first_friday_next = 1 + (4 - w_next) if w_next <= 4 else 1 + (11 - w_next)
            nfp_time = datetime.datetime(ny, nm, first_friday_next, 13, 30)
            
        dt_hours = (nfp_time - t).total_seconds() / 3600.0
        if dt_hours >= 0:
            prox[i] = float(np.exp(-dt_hours / 12.0))
    return prox

def precompute_calendar_features(dfs: Dict[str, pd.DataFrame], calendar_df: pd.DataFrame, cfg: OmegaConfig):
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "XAU"]
    impacts = ["High", "Medium"]
    for tf, df in dfs.items():
        if df.empty:
            continue
        pd_times = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df["NEWS_USD_NFP_Prox"] = get_nfp_proximity(pd_times)
        for curr in currencies:
            for imp in impacts:
                df[f"NEWS_{curr}_{imp}_Type"] = 0.0
                df[f"NEWS_{curr}_{imp}_Prox"] = 0.0
                df[f"NEWS_{curr}_{imp}_Surp"] = 0.0
    if calendar_df.empty:
        return
    def clean_val(val):
        if pd.isna(val) or val == "":
            return 0.0
        val_str = str(val).strip().replace("%", "").replace(",", "")
        multiplier = 1.0
        if val_str.endswith("M") or val_str.endswith("m"):
            multiplier = 1000000.0
            val_str = val_str[:-1]
        elif val_str.endswith("K") or val_str.endswith("k"):
            multiplier = 1000.0
            val_str = val_str[:-1]
        elif val_str.endswith("B") or val_str.endswith("b"):
            multiplier = 1000000000.0
            val_str = val_str[:-1]
        try:
            return float(val_str) * multiplier
        except ValueError:
            return 0.0
    calendar_df = calendar_df.copy()
    calendar_df['act_val'] = calendar_df['Actual'].apply(clean_val) if 'Actual' in calendar_df.columns else 0.0
    calendar_df['fore_val'] = calendar_df['Forecast'].apply(clean_val) if 'Forecast' in calendar_df.columns else 0.0
    calendar_df['surprise'] = (calendar_df['act_val'] - calendar_df['fore_val']) / (calendar_df['fore_val'].abs() + 1e-8)
    calendar_df['surprise'] = calendar_df['surprise'].clip(-5.0, 5.0).fillna(0.0)
    if 'Event' in calendar_df.columns:
        calendar_df['event_type'] = calendar_df['Event'].apply(classify_event)
    else:
        calendar_df['event_type'] = N_EVENT_TYPES - 1
    calendar_df['parsed_time'] = pd.to_datetime(calendar_df['DateTime'], errors='coerce', utc=True).dt.tz_localize(None)
    calendar_df = calendar_df.dropna(subset=['parsed_time']).sort_values('parsed_time')
    for tf, df in dfs.items():
        if df.empty:
            continue
        pd_times = pd.to_datetime(df.index, utc=True).tz_localize(None)
        for curr in currencies:
            for imp in impacts:
                sub_cal = calendar_df[(calendar_df['Currency'] == curr) & (calendar_df['Impact'] == imp)]
                if sub_cal.empty:
                    continue
                cal_times = sub_cal['parsed_time'].values
                surprises = sub_cal['surprise'].values
                event_types = sub_cal['event_type'].values
                idx_next = np.searchsorted(cal_times, pd_times, side='left')
                type_vals = np.zeros(len(pd_times), dtype=np.float32)
                prox_vals = np.zeros(len(pd_times), dtype=np.float32)
                surp_vals = np.zeros(len(pd_times), dtype=np.float32)
                for i, t in enumerate(pd_times):
                    n_idx = idx_next[i]
                    if n_idx < len(cal_times):
                        dt_minutes = (cal_times[n_idx] - t).total_seconds() / 60.0
                        if dt_minutes >= 0:
                            prox_vals[i] = float(np.exp(-dt_minutes / 60.0))
                            type_vals[i] = float(event_types[n_idx]) / float(N_EVENT_TYPES)
                    p_idx = n_idx - 1
                    while p_idx >= 0:
                        dt_past_min = (t - cal_times[p_idx]).total_seconds() / 60.0
                        if dt_past_min <= 15.0:
                            surp_vals[i] = float(surprises[p_idx])
                            break
                        else:
                            break
                df[f"NEWS_{curr}_{imp}_Type"] = type_vals
                df[f"NEWS_{curr}_{imp}_Prox"] = prox_vals
                df[f"NEWS_{curr}_{imp}_Surp"] = surp_vals

def precompute_trade_state_features(dfs: Dict[str, pd.DataFrame], cfg: OmegaConfig):
    for tf, df in dfs.items():
        if df.empty:
            continue
        for pair in cfg.PAIRS:
            close_col = f"{pair}_Close"
            if close_col not in df.columns:
                df[f"{pair}_SIM_POS"] = 0.0
                df[f"{pair}_SIM_AGE"] = 0.0
                df[f"{pair}_SIM_PNL"] = 0.0
                continue
            close = df[close_col]
            ema = close.ewm(span=20, adjust=False).mean()
            close_vals = close.values
            ema_vals = ema.values
            n_bars = len(close_vals)
            pos_arr = np.zeros(n_bars, dtype=np.float32)
            age_arr = np.zeros(n_bars, dtype=np.float32)
            pnl_arr = np.zeros(n_bars, dtype=np.float32)
            curr_pos = 0.0
            entry_idx = 0
            entry_price = 0.0
            for i in range(n_bars):
                if close_vals[i] > ema_vals[i]:
                    new_pos = 1.0
                else:
                    new_pos = -1.0
                if new_pos != curr_pos:
                    curr_pos = new_pos
                    entry_idx = i
                    entry_price = close_vals[i] if close_vals[i] > 1e-9 else 1.0
                pos_arr[i] = curr_pos
                age_arr[i] = float(i - entry_idx) / 12.0
                pnl_arr[i] = float((close_vals[i] - entry_price) / entry_price * 1e4 * curr_pos)
            df[f"{pair}_SIM_POS"] = pos_arr
            df[f"{pair}_SIM_AGE"] = age_arr
            df[f"{pair}_SIM_PNL"] = pnl_arr

def get_mt5_position_state(symbol: str, current_bar_time: pd.Timestamp) -> Tuple[float, float, float]:
    positions = mt5.positions_get(symbol=symbol)
    if positions is not None and len(positions) > 0:
        # Filter matching current strategy execution framework
        strategy_positions = [p for p in positions if p.magic == MAGIC_NUMBER]
        if len(strategy_positions) > 0:
            pos_info = strategy_positions[0]
            direction = 1.0 if pos_info.type == mt5.POSITION_TYPE_BUY else -1.0
            entry_time = pd.to_datetime(pos_info.time, unit='s')
            age_seconds = (current_bar_time - entry_time).total_seconds()
            age_bars = max(0.0, age_seconds / 300.0)
            trade_age = float(age_bars / 12.0)
            entry_price = float(pos_info.price_open)
            current_price = float(pos_info.price_current)
            trade_pnl = float((current_price - entry_price) / (entry_price + 1e-9) * 1e4 * direction)
            return direction, trade_age, trade_pnl
    return 0.0, 0.0, 0.0

# -----------------------------------------------------------------------------
# 2. NEURAL NETWORK ARCHITECTURE DEFINITION
# -----------------------------------------------------------------------------
class NestedTimescaleEncoder(nn.Module):
    def __init__(self, in_features, d_micro, d_macro):
        super().__init__()
        self.rnn_5m = nn.GRU(in_features, d_micro, batch_first=True)
        self.rnn_15m = nn.GRU(in_features + d_micro, d_micro, batch_first=True)
        self.rnn_1h = nn.GRU(in_features + d_micro, d_macro, batch_first=True)
        self.rnn_4h = nn.GRU(in_features + d_macro, d_macro, batch_first=True)

    def forward(self, x_5m, x_15m, x_1h, x_4h):
        _, h_5m = self.rnn_5m(x_5m)
        c_5m = h_5m[-1]
        c_5m_15m = c_5m.unsqueeze(1).expand(-1, x_15m.shape[1], -1)
        in_15m = torch.cat([x_15m, c_5m_15m], dim=-1)
        _, h_15m = self.rnn_15m(in_15m)
        c_15m = h_15m[-1]
        c_15m_1h = c_15m.unsqueeze(1).expand(-1, x_1h.shape[1], -1)
        in_1h = torch.cat([x_1h, c_15m_1h], dim=-1)
        _, h_1h = self.rnn_1h(in_1h)
        c_1h = h_1h[-1]
        c_1h_4h = c_1h.unsqueeze(1).expand(-1, x_4h.shape[1], -1)
        in_4h = torch.cat([x_4h, c_1h_4h], dim=-1)
        _, h_4h = self.rnn_4h(in_4h)
        c_4h = h_4h[-1]
        return c_4h

class LeWM_Encoder(nn.Module):
    def __init__(self, d_macro, d_latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_macro, d_macro),
            nn.GELU(),
            nn.Linear(d_macro, d_latent)
        )
    def forward(self, h_macro):
        return self.net(h_macro)

class LeWM_Predictor(nn.Module):
    def __init__(self, d_latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, d_latent)
        )
    def forward(self, z_t):
        return self.net(z_t)

class LearnedAllocator(nn.Module):
    def __init__(self, d_latent, num_assets):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_latent // 2),
            nn.GELU(),
            nn.Linear(d_latent // 2, 1),
            nn.Tanh(),
        )
        self.num_assets = num_assets
    def forward(self, z_t, B):
        return self.net(z_t).squeeze(-1).view(B, self.num_assets)

class DirectionHead(nn.Module):
    def __init__(self, d_latent, num_assets):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_latent // 2),
            nn.GELU(),
            nn.Linear(d_latent // 2, 1),
        )
        self.num_assets = num_assets
    def forward(self, z, B):
        return self.net(z).squeeze(-1).view(B, self.num_assets)

class ReturnProbe(nn.Module):
    def __init__(self, d_latent, num_assets):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_latent // 2),
            nn.GELU(),
            nn.Linear(d_latent // 2, 1),
        )
        self.num_assets = num_assets
    def forward(self, z, B):
        return self.net(z).squeeze(-1).view(B, self.num_assets)


class DGDFastMemory(nn.Module):
    """
    Online Sherman-Morrison DGD (Dual Gradient Descent) fast weight adapter.
    Maintains a fast-weight matrix W_fast that adapts online without backprop.
    Momentum buffer prevents oscillation on shock events.
    """
    def __init__(self, d: int, lr: float = 0.01, clamp: float = 0.05, spectral: float = 5.0, momentum: float = 0.90):
        super().__init__()
        self.d, self.lr, self.clamp = d, lr, clamp
        self.spectral, self.momentum = spectral, momentum
        self.W_slow       = nn.Parameter(torch.eye(d))
        self.W_fast: Optional[torch.Tensor]  = None
        self._mom_buf: Optional[torch.Tensor]= None

    def reset_fast(self) -> None:
        self.W_fast   = self.W_slow.detach().clone()
        self._mom_buf = torch.zeros_like(self.W_fast)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.W_fast is None:
            self.reset_fast()
        W = self.W_slow if self.training else self.W_fast
        return z @ W.to(dtype=z.dtype, device=z.device).T

    @torch.no_grad()
    def online_update(self, z_prev: torch.Tensor, error: torch.Tensor) -> None:
        if self.W_fast is None:
            self.reset_fast()
        x_t    = z_prev.view(-1).to(dtype=self.W_fast.dtype, device=self.W_fast.device)
        grad_y = error.view(-1).to(dtype=self.W_fast.dtype, device=self.W_fast.device)
        nxsq   = x_t.pow(2).sum().clamp(min=1e-6)
        eta_p  = self.lr / (1.0 + self.lr * nxsq)
        Wx     = torch.mv(self.W_fast, x_t)
        raw_upd= eta_p * (torch.outer(Wx, x_t) + torch.outer(grad_y, x_t))
        raw_upd= raw_upd.clamp(-self.clamp, self.clamp)
        self._mom_buf = self.momentum * self._mom_buf + (1 - self.momentum) * raw_upd
        self.W_fast.sub_(self._mom_buf)
        self._clamp_spectral()

    @torch.no_grad()
    def _clamp_spectral(self) -> None:
        try:
            smax = torch.linalg.svdvals(self.W_fast)[0]
            if smax > self.spectral:
                self.W_fast.mul_(self.spectral / smax)
        except Exception:
            fro = self.W_fast.norm()
            cap = self.spectral * math.sqrt(self.d)
            if fro > cap:
                self.W_fast.mul_(cap / fro)


class NestedLeWM_MasterSystem(nn.Module):
    def __init__(self, cfg: OmegaConfig, in_features: int):
        super().__init__()
        self.cfg = cfg
        self.nested_encoder = NestedTimescaleEncoder(in_features, cfg.D_MICRO, cfg.D_MACRO)
        self.lewm_encoder = LeWM_Encoder(cfg.D_MACRO, cfg.D_LATENT)
        self.lewm_predictor = LeWM_Predictor(cfg.D_LATENT)
        self.return_probe = ReturnProbe(cfg.D_LATENT, cfg.NUM_ASSETS)
        self.direction_head = DirectionHead(cfg.D_LATENT, cfg.NUM_ASSETS)
        self.allocator = LearnedAllocator(cfg.D_LATENT, cfg.NUM_ASSETS)
        self.dgd_mem = DGDFastMemory(cfg.D_LATENT, lr=0.001, momentum=0.90)

    def _encode(self, x_5m, x_15m, x_1h, x_4h):
        B, T5, N, F = x_5m.shape
        x_5m_flat  = x_5m.transpose(1, 2).reshape(B * N, T5, F)
        x_15m_flat = x_15m.transpose(1, 2).reshape(B * N, x_15m.shape[1], F)
        x_1h_flat  = x_1h.transpose(1, 2).reshape(B * N, x_1h.shape[1], F)
        x_4h_flat  = x_4h.transpose(1, 2).reshape(B * N, x_4h.shape[1], F)
        h_macro = self.nested_encoder(x_5m_flat, x_15m_flat, x_1h_flat, x_4h_flat)
        return self.lewm_encoder(h_macro), B

    def forward(self, x_5m, x_15m, x_1h, x_4h):
        z_t, B = self._encode(x_5m, x_15m, x_1h, x_4h)
        z_t_next_pred = self.lewm_predictor(z_t)
        y_pred     = self.return_probe(z_t, B)
        dir_logits = self.direction_head(z_t, B)
        alloc      = self.allocator(z_t, B)
        return z_t, z_t_next_pred, alloc, y_pred, dir_logits

# -----------------------------------------------------------------------------
# 3. LIVE FEATURE ENGINEERING & DATALOADER FUNCTIONS
# -----------------------------------------------------------------------------
def _engineer_features(df: pd.DataFrame, name: str) -> None:
    close = df[f"{name}_RawClose"]
    high = df[f"{name}_RawHigh"]
    low = df[f"{name}_RawLow"]
    opn = df[f"{name}_RawOpen"]
    
    ret = close.pct_change().fillna(0)
    df[f"{name}_Close"] = close.values
    df[f"{name}_High"]  = high.values
    df[f"{name}_Low"]   = low.values
    df[f"{name}_Vol"]   = ret.rolling(20, min_periods=1).std().fillna(0.001).values
    
    spread = (high - low).clip(lower=1e-8)
    df[f"{name}_Imb"]   = ((close - opn) / spread).clip(-1, 1).values
    
    c_prev = close.shift(1)
    tr = pd.concat([
        (high - low).rename("hl"),
        (high - c_prev).abs().rename("hcp"),
        (low  - c_prev).abs().rename("lcp"),
    ], axis=1).max(axis=1)
    df[f"{name}_ATR14"] = (tr.rolling(14, min_periods=1).mean() / (close.abs() + 1e-9)).values
    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    df[f"{name}_MACD"]  = ((macd - sig) / (close.abs() + 1e-9)).values
    
    bm  = close.rolling(20, min_periods=1).mean()
    bs  = close.rolling(20, min_periods=1).std().fillna(1e-8)
    df[f"{name}_BB"]    = (((close - (bm - 2 * bs)) / ((4 * bs) + 1e-9)) - 0.5).clip(-1, 1).values
    
    df[f"{name}_RoC5"]   = close.pct_change(5).fillna(0).values
    df[f"{name}_RoC20"]  = close.pct_change(20).fillna(0).values
    df[f"{name}_Thrust"] = ((close.rolling(5, min_periods=1).max() - close.rolling(5, min_periods=1).min()) / (close.abs() + 1e-9) / 5).values
    
    h_arr = high.values
    l_arr = low.values
    shi   = np.zeros(len(h_arr))
    slo   = np.zeros(len(l_arr))
    for i in range(2, len(h_arr) - 2):
        if (h_arr[i] > h_arr[i-1] and h_arr[i] > h_arr[i-2] and h_arr[i] > h_arr[i+1] and h_arr[i] > h_arr[i+2]):
            shi[i] = 1.0
        if (l_arr[i] < l_arr[i-1] and l_arr[i] < l_arr[i-2] and l_arr[i] < l_arr[i+1] and l_arr[i] < l_arr[i+2]):
            slo[i] = 1.0
    df[f"{name}_SwingHi"] = shi
    df[f"{name}_SwingLo"] = slo

def _get_pair_slice(df: pd.DataFrame, pair: str, t_start: int, t_end: int, cfg: OmegaConfig, tgt_len: int) -> np.ndarray:
    feats = []
    
    # 1. Standard technical indicators (13 columns)
    for fname in PAIR_FEATS:
        col = f"{pair}_{fname}"
        if col not in df.columns:
            feats.append(np.zeros(tgt_len, dtype=np.float32))
            continue
        arr = df[col].values[t_start:t_end].astype(np.float32)
        if len(arr) < tgt_len:
            arr = np.pad(arr, (tgt_len - len(arr), 0), mode="edge")
        elif len(arr) > tgt_len:
            arr = arr[-tgt_len:]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        mu = arr.mean(); sig = arr.std() + 1e-8
        norm_arr = (arr - mu) / sig
        feats.append(np.nan_to_num(norm_arr, nan=0.0, posinf=0.0, neginf=0.0))
        
    # 2. Sovereign yields (2 columns)
    base_yield, quote_yield = cfg.YIELD_MAP.get(pair, ("US10Y", "US10Y"))
    for yield_name in [base_yield, quote_yield]:
        col = f"{yield_name}_Close"
        if col not in df.columns:
            feats.append(np.zeros(tgt_len, dtype=np.float32))
            continue
        arr = df[col].values[t_start:t_end].astype(np.float32)
        if len(arr) < tgt_len:
            arr = np.pad(arr, (tgt_len - len(arr), 0), mode="edge")
        elif len(arr) > tgt_len:
            arr = arr[-tgt_len:]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        mu = arr.mean(); sig = arr.std() + 1e-8
        norm_arr = (arr - mu) / sig
        feats.append(np.nan_to_num(norm_arr, nan=0.0, posinf=0.0, neginf=0.0))
        
    # 3. Programmatic NFP proximity (1 column)
    col = "NEWS_USD_NFP_Prox"
    if col not in df.columns:
        feats.append(np.zeros(tgt_len, dtype=np.float32))
    else:
        arr = df[col].values[t_start:t_end].astype(np.float32)
        if len(arr) < tgt_len:
            arr = np.pad(arr, (tgt_len - len(arr), 0), mode="edge")
        elif len(arr) > tgt_len:
            arr = arr[-tgt_len:]
        feats.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
        
    # 4. Simulated/Actual trade state (3 columns)
    trade_cols = [
        f"{pair}_SIM_POS",
        f"{pair}_SIM_AGE",
        f"{pair}_SIM_PNL"
    ]
    for col in trade_cols:
        if col not in df.columns:
            feats.append(np.zeros(tgt_len, dtype=np.float32))
            continue
        arr = df[col].values[t_start:t_end].astype(np.float32)
        if len(arr) < tgt_len:
            arr = np.pad(arr, (tgt_len - len(arr), 0), mode="edge")
        elif len(arr) > tgt_len:
            arr = arr[-tgt_len:]
        feats.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
        
    return np.stack(feats, axis=-1)

def _align_tf(df_tf: pd.DataFrame, t_ref: pd.Timestamp, ctx_bars: int, pairs: List[str], cfg: OmegaConfig) -> np.ndarray:
    n_pairs = len(pairs)
    if df_tf.empty:
        return np.zeros((ctx_bars, n_pairs, 24), dtype=np.float32)
    
    loc = int(df_tf.index.searchsorted(t_ref, side="right")) - 1
    loc = max(loc, ctx_bars)
    
    slices = []
    for n, pair in enumerate(pairs):
        arr = _get_pair_slice(df_tf, pair, loc - ctx_bars, loc, cfg, ctx_bars)
        one_hot = np.zeros((ctx_bars, n_pairs), dtype=np.float32)
        one_hot[:, n] = 1.0
        arr = np.concatenate([arr, one_hot], axis=-1)
        slices.append(arr)
    return np.stack(slices, axis=1)

# -----------------------------------------------------------------------------
# 4. MT5 LIVE DATA COPY & PROCESSING
# -----------------------------------------------------------------------------
def get_mt5_symbol(pair: str) -> str:
    return f"{SYMBOL_PREFIX}{pair}{SYMBOL_SUFFIX}"

def fetch_timeframe_data(symbol_map: Dict[str, str], timeframe: int, count: int) -> pd.DataFrame:
    """Downloads historical OHLCV from MT5 and engineers features."""
    dfs_raw = []
    union_idx = None
    
    for pair, mt5_sym in symbol_map.items():
        # Ensure symbol is active and selected in market watch
        mt5.symbol_select(mt5_sym, True)
        
        rates = mt5.copy_rates_from_pos(mt5_sym, timeframe, 1, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Failed to fetch {timeframe} rates for {mt5_sym}. Error: {mt5.last_error()}")
            
        df_sym = pd.DataFrame(rates)
        df_sym.index = pd.to_datetime(df_sym['time'], unit='s')
        
        # Keep raw OHLC for feature engineering
        df_sym[f"{pair}_RawClose"] = df_sym['close'].astype(float)
        df_sym[f"{pair}_RawHigh"]  = df_sym['high'].astype(float)
        df_sym[f"{pair}_RawLow"]   = df_sym['low'].astype(float)
        df_sym[f"{pair}_RawOpen"]  = df_sym['open'].astype(float)
        
        dfs_raw.append(df_sym[[f"{pair}_RawClose", f"{pair}_RawHigh", f"{pair}_RawLow", f"{pair}_RawOpen"]])
        union_idx = df_sym.index if union_idx is None else union_idx.union(df_sym.index)
        
    union_idx = union_idx.sort_values()
    df_aligned = pd.DataFrame(index=union_idx)
    
    for df in dfs_raw:
        df_aligned = df_aligned.join(df, how='left')
        
    df_aligned = df_aligned.ffill().bfill()
    
    # Run feature engineering per pair
    for pair in symbol_map.keys():
        _engineer_features(df_aligned, pair)
        
    return df_aligned.ffill().bfill()

# -----------------------------------------------------------------------------
# 5. MT5 POSITION REBALANCING EXECUTION
# -----------------------------------------------------------------------------
def get_filling_type(symbol: str) -> int:
    """Determines broker supported order execution filling mode."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_IOC
    filling = info.filling_mode
    if filling & 1:
        return mt5.ORDER_FILLING_FOK
    elif filling & 2:
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN

def close_all_positions(symbol: str) -> bool:
    """Closes all open buys and sells for the symbol with a retry loop."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return True
        
    success = True
    for pos in positions:
        if pos.magic != MAGIC_NUMBER:
            continue
            
        closed_successfully = False
        for attempt in range(3):
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                print(f"[TRADE] Error getting tick prices for {symbol} (Attempt {attempt+1}/3).")
                time.sleep(2)
                continue
                
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            
            request = {
                "action": int(mt5.TRADE_ACTION_DEAL),
                "symbol": str(symbol),
                "volume": float(pos.volume),
                "type": int(order_type),
                "position": int(pos.ticket),
                "price": float(price),
                "deviation": int(DEVIATION),
                "magic": int(MAGIC_NUMBER),
                "comment": "APEX Portfolio Close",
                "type_time": int(mt5.ORDER_TIME_GTC),
                "type_filling": int(get_filling_type(symbol)),
            }
            
            result = mt5.order_send(request)
            if result is None:
                print(f"[TRADE] Close ticket {pos.ticket} returned None (Attempt {attempt+1}/3). Error={mt5.last_error()}")
            elif result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[TRADE] Closed position ticket {pos.ticket} for {symbol} ({pos.volume:.2f} lots).")
                closed_successfully = True
                break
            else:
                if result.retcode == 10018:
                    print(f"[TRADE] Close request failed for ticket {pos.ticket}: Market is closed (broker rollover gap). Rebalancing deferred to next candle.")
                    break
                else:
                    print(f"[TRADE] Close ticket {pos.ticket} failed: retcode={result.retcode} (Attempt {attempt+1}/3). Error={mt5.last_error()}")
            time.sleep(2)
            
        if not closed_successfully:
            success = False
            
    return success

def execute_rebalance(symbol: str, target_weight: float) -> None:
    """Compares current net open volume with target lot size and executes rebalancing with retries."""
    target_lots = target_weight * MAX_LOT_SIZE
    
    positions = mt5.positions_get(symbol=symbol)
    current_volume = 0.0
    if positions:
        for pos in positions:
            if pos.magic == MAGIC_NUMBER:
                if pos.type == mt5.POSITION_TYPE_BUY:
                    current_volume += pos.volume
                elif pos.type == mt5.POSITION_TYPE_SELL:
                    current_volume -= pos.volume
                    
    if abs(target_lots - current_volume) < 0.005:
        print(f"[TRADE] {symbol} allocation is aligned. Target: {target_lots:+.2f} lots | Current: {current_volume:+.2f} lots.")
        return
        
    print(f"[TRADE] Rebalancing {symbol}: Current: {current_volume:+.2f} lots -> Target: {target_lots:+.2f} lots.")
    
    if not close_all_positions(symbol):
        print(f"[TRADE] Error clearing old positions for {symbol}. Rebalancing aborted.")
        return
        
    if abs(target_lots) >= 0.01:
        opened_successfully = False
        for attempt in range(3):
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                print(f"[TRADE] Failed to fetch price ticks for {symbol} (Attempt {attempt+1}/3).")
                time.sleep(2)
                continue
                
            order_type = mt5.ORDER_TYPE_BUY if target_lots > 0 else mt5.ORDER_TYPE_SELL
            price = tick.ask if target_lots > 0 else tick.bid
            lots = round(abs(target_lots), 2)
            
            request = {
                "action": int(mt5.TRADE_ACTION_DEAL),
                "symbol": str(symbol),
                "volume": float(lots),
                "type": int(order_type),
                "price": float(price),
                "deviation": int(DEVIATION),
                "magic": int(MAGIC_NUMBER),
                "comment": f"APEX Alloc={target_weight:+.2f}",
                "type_time": int(mt5.ORDER_TIME_GTC),
                "type_filling": int(get_filling_type(symbol)),
            }
            
            result = mt5.order_send(request)
            if result is None:
                print(f"[TRADE] Open request returned None for {symbol} (Attempt {attempt+1}/3). Error={mt5.last_error()}")
            elif result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"[TRADE] Opened position for {symbol}: {'BUY' if target_lots > 0 else 'SELL'} of {lots:.2f} lots.")
                opened_successfully = True
                break
            else:
                if result.retcode == 10018:
                    print(f"[TRADE] Open request failed for {symbol}: Market is closed (broker rollover gap). Execution deferred to next candle.")
                    break
                else:
                    print(f"[TRADE] Open request failed for {symbol} ({lots:.2f} lots): retcode={result.retcode} (Attempt {attempt+1}/3). Error={mt5.last_error()}")
            time.sleep(2)

# -----------------------------------------------------------------------------
# 6. MAIN LIVE TRADING PIPELINE
# -----------------------------------------------------------------------------
def run_live_inference():
    print("=" * 80)
    print("  APEX: MetaTrader 5 Live Inference & Rebalancing Pipeline")
    print("=" * 80)
    
    cfg = OmegaConfig()
    
    in_features = 24
    model = NestedLeWM_MasterSystem(cfg, in_features).to(DEVICE)
    
    model_path = 'apex_model.pt'
    if not os.path.exists(model_path):
        print(f"[APEX] Error: weights file '{model_path}' not found in the current directory.")
        print("Please place the trained weights file in this folder and try again.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    model.dgd_mem.reset_fast()

    probe_optimizer = torch.optim.AdamW(model.return_probe.parameters(), lr=0.001)

    prev_z_t_adapted = None
    prev_z_t_next_pred = None
    adaptive_queue = []
    print(f"[APEX] Model architecture loaded successfully. Weights: '{model_path}'")
    
    print(f"[MT5] Connecting to server {MT5_SERVER}...")
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"[MT5] Connection failed. Error code: {mt5.last_error()}")
        return
        
    account_info = mt5.account_info()
    if account_info is None:
        print("[MT5] Failed to fetch account information.")
        mt5.shutdown()
        return
        
    print(f"[MT5] Connected successfully! Account: {account_info.login} | Balance: {account_info.balance:.2f} {account_info.currency}")
    
    symbol_map = {pair: get_mt5_symbol(pair) for pair in cfg.PAIRS}
    print(f"[MT5] Symbol Map: {symbol_map}")
    
    fetch_bars = 120
    last_processed_timestamp = None
    
    try:
        while True:
            rates_5m_ref = mt5.copy_rates_from_pos(list(symbol_map.values())[0], mt5.TIMEFRAME_M5, 1, 1)
            if rates_5m_ref is None or len(rates_5m_ref) == 0:
                print(f"[MT5] Warning: Failed to read reference tick. Retrying in 10s...")
                time.sleep(10)
                continue
                
            current_bar_time = pd.to_datetime(rates_5m_ref[0]['time'], unit='s')
            
            if last_processed_timestamp is None or current_bar_time > last_processed_timestamp:
                print() 
                print("=" * 80)
                print(f"  NEW BAR DETECTED | Open Time: {current_bar_time} | Executing Inference...")
                print("=" * 80)
                
                try:
                    df5  = fetch_timeframe_data(symbol_map, mt5.TIMEFRAME_M5,  fetch_bars)
                    df15 = fetch_timeframe_data(symbol_map, mt5.TIMEFRAME_M15, fetch_bars)
                    df1h = fetch_timeframe_data(symbol_map, mt5.TIMEFRAME_H1,  fetch_bars)
                    df4h = fetch_timeframe_data(symbol_map, mt5.TIMEFRAME_H4,  fetch_bars)
                    
                    fetch_yield_data_aligned(cfg, df5, "5m", "60d")
                    fetch_yield_data_aligned(cfg, df15, "15m", "60d")
                    fetch_yield_data_aligned(cfg, df1h, "1h", "730d")
                    fetch_yield_data_aligned(cfg, df4h, "1h", "730d")
                    
                    calendar_df = pd.DataFrame()
                    if os.path.exists('calendar.csv'):
                        try:
                            calendar_df = pd.read_csv('calendar.csv')
                        except Exception as e:
                            print(f"[CALENDAR] Warning: Failed to read calendar.csv: {e}")
                            
                    precompute_calendar_features({"5m": df5, "15m": df15, "1h": df1h, "4h": df4h}, calendar_df, cfg)
                    precompute_trade_state_features({"5m": df5, "15m": df15, "1h": df1h, "4h": df4h}, cfg)
                    
                    for pair in cfg.PAIRS:
                        symbol = symbol_map[pair]
                        pos, age, pnl = get_mt5_position_state(symbol, current_bar_time)
                        
                        df5.loc[df5.index[-1], f"{pair}_SIM_POS"] = pos
                        df5.loc[df5.index[-1], f"{pair}_SIM_AGE"] = age
                        df5.loc[df5.index[-1], f"{pair}_SIM_PNL"] = pnl
                        
                        for df_tf in [df15, df1h, df4h]:
                            if not df_tf.empty:
                                loc = int(df_tf.index.searchsorted(current_bar_time, side="right")) - 1
                                if 0 <= loc < len(df_tf):
                                    df_tf.loc[df_tf.index[loc], f"{pair}_SIM_POS"] = pos
                                    df_tf.loc[df_tf.index[loc], f"{pair}_SIM_AGE"] = age
                                    df_tf.loc[df_tf.index[loc], f"{pair}_SIM_PNL"] = pnl
                                    
                    t_ref = current_bar_time
                    
                    n_pairs = len(cfg.PAIRS)
                    forex_5m = []
                    for n, pair in enumerate(cfg.PAIRS):
                        arr = _get_pair_slice(df5, pair, len(df5) - cfg.CONTEXT_5m, len(df5), cfg, cfg.CONTEXT_5m)
                        one_hot = np.zeros((cfg.CONTEXT_5m, n_pairs), dtype=np.float32)
                        one_hot[:, n] = 1.0
                        arr = np.concatenate([arr, one_hot], axis=-1)
                        forex_5m.append(arr)
                    x_5m = np.stack(forex_5m, axis=1)
                    
                    x_15m = _align_tf(df15, t_ref, cfg.CONTEXT_15m, cfg.PAIRS, cfg)
                    x_1h  = _align_tf(df1h,  t_ref, cfg.CONTEXT_1h,  cfg.PAIRS, cfg)
                    x_4h  = _align_tf(df4h,  t_ref, cfg.CONTEXT_4h,  cfg.PAIRS, cfg)
                    
                    bx_5m  = torch.tensor(x_5m,  dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    bx_15m = torch.tensor(x_15m, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    bx_1h  = torch.tensor(x_1h,  dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    bx_4h  = torch.tensor(x_4h,  dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    
                    if prev_z_t_adapted is not None and prev_z_t_next_pred is not None:
                        with torch.no_grad():
                            z_target, _ = model._encode(bx_5m, bx_15m, bx_1h, bx_4h)
                            z_mean = prev_z_t_adapted.mean(dim=0)
                            err_mean = (prev_z_t_next_pred - z_target).mean(dim=0)
                            model.dgd_mem.online_update(z_mean, err_mean)
                            drift = torch.norm(model.dgd_mem.W_fast - model.dgd_mem.W_slow).item()
                            print(f"[ADAPT] Level 1 Synaptic Update. Latent Drift: {drift:.4f}")
                            
                    with torch.no_grad():
                        z_t, z_t_next_pred, alloc, y_pred, dir_logits = model(bx_5m, bx_15m, bx_1h, bx_4h)
                        
                        alloc_weights = alloc.cpu().numpy()[0]
                        denom = np.sum(np.abs(alloc_weights))
                        if denom > 1e-6:
                            alloc_weights = alloc_weights / denom
                        else:
                            alloc_weights = np.zeros_like(alloc_weights)
                            
                        predicted_ret = y_pred.cpu().numpy()[0]
                        dir_probs     = torch.sigmoid(dir_logits).cpu().numpy()[0]
                        
                    prev_z_t_adapted = z_t.detach().clone()
                    prev_z_t_next_pred = z_t_next_pred.detach().clone()
                    
                    prices_then = {pair: df5[f"{pair}_Close"].iloc[-1] for pair in cfg.PAIRS}
                    adaptive_queue.append({
                        'z': z_t.detach().clone(),
                        'time': current_bar_time,
                        'prices': prices_then
                    })
                    
                    while len(adaptive_queue) > 0:
                        oldest = adaptive_queue[0]
                        dt_min = (current_bar_time - oldest['time']).total_seconds() / 60.0
                        if dt_min >= 60.0:
                            entry = adaptive_queue.pop(0)
                            z_then = entry['z']
                            realized_returns_bps = []
                            for pair in cfg.PAIRS:
                                p_then = entry['prices'][pair]
                                p_now = df5[f"{pair}_Close"].iloc[-1]
                                ret_bps = (p_now - p_then) / (p_then + 1e-9) * 1e4
                                realized_returns_bps.append(ret_bps)
                            
                            y_actual = torch.tensor([realized_returns_bps], dtype=torch.float32).to(DEVICE)
                            
                            model.train()
                            probe_optimizer.zero_grad()
                            y_pred_then = model.return_probe(z_then, 1)
                            loss_l2 = F.huber_loss(y_pred_then, y_actual)
                            loss_l2.backward()
                            probe_optimizer.step()
                            model.eval()
                            print(f"[CONSOLIDATE] Level 2 Return Probe updated. Huber Loss: {loss_l2.item():.6f}")
                        else:
                            break
                            
                    print(f"\n[INFERENCE] Output Summary:")
                    for n, pair in enumerate(cfg.PAIRS):
                        direction = "UP" if dir_probs[n] > 0.5 else "DOWN"
                        print(f"  [OK] {pair} -> Pred Return: {predicted_ret[n]:+5.2f} bps | Dir: {direction} ({dir_probs[n]*100:4.1f}%) | Weight: {alloc_weights[n]:+5.2f}")
                        
                    print(f"\n[PORTFOLIO] Rebalancing positions based on predicted allocations...")
                    for n, pair in enumerate(cfg.PAIRS):
                        symbol = symbol_map[pair]
                        target_weight = alloc_weights[n]
                        
                        base_curr, quote_curr = cfg.CURRENCY_MAP.get(pair, ("USD", "USD"))
                        base_prox = df5[f"NEWS_{base_curr}_High_Prox"].iloc[-1] if f"NEWS_{base_curr}_High_Prox" in df5.columns else 0.0
                        quote_prox = df5[f"NEWS_{quote_curr}_High_Prox"].iloc[-1] if f"NEWS_{quote_curr}_High_Prox" in df5.columns else 0.0
                        nfp_prox = df5["NEWS_USD_NFP_Prox"].iloc[-1] if "NEWS_USD_NFP_Prox" in df5.columns else 0.0
                        base_type = df5[f"NEWS_{base_curr}_High_Type"].iloc[-1] if f"NEWS_{base_curr}_High_Type" in df5.columns else 0.0
                        
                        if base_prox > 0.5 or quote_prox > 0.5 or nfp_prox > 0.5:
                            max_prox = max(base_prox, quote_prox, nfp_prox)
                            evt_idx = int(round(base_type * N_EVENT_TYPES))
                            evt_name = EVENT_TYPES[min(evt_idx, len(EVENT_TYPES)-1)]
                            print(f"  [NEWS INFO] {pair}: Event={evt_name} Prox={max_prox:.2f} — Managed by neural policy.")
                            
                        execute_rebalance(symbol, target_weight)
                        
                    last_processed_timestamp = current_bar_time
                    print(f"\n[APEX] Rebalancing complete. Next check in 10s...")
                    
                except Exception as e:
                    print(f"[APEX] Error during processing cycle: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                sys.stdout.write(f"\r[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} | Monitoring bar: {current_bar_time} | Waiting for next candle...")
                sys.stdout.flush()
                    
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n[SYSTEM] Keyboard interrupt received. Exiting...")
    finally:
        mt5.shutdown()
        print("[SYSTEM] MetaTrader 5 interface closed.")

if __name__ == '__main__':
    run_live_inference()
