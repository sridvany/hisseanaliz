import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

st.set_page_config(layout="wide", page_title="AI Teknik Analiz Terminali")


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def calc_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index, dtype=float)
    minus_dm = pd.Series(minus_dm, index=high.index, dtype=float)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = 100 * (smooth_plus / atr.replace(0, np.nan))
    minus_di = 100 * (smooth_minus / atr.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


def calc_nadaraya_watson(close, bandwidth=8, window=100):
    n = len(close)
    nw_line = np.full(n, np.nan)
    start = max(0, n - window)
    y = close.values[start:].astype(float)
    m = len(y)
    for i in range(m):
        weights = np.array([
            np.exp(-((i - j) ** 2) / (2 * bandwidth ** 2))
            for j in range(m)
        ])
        nw_line[start + i] = np.sum(weights * y) / np.sum(weights)
    nw_series = pd.Series(nw_line, index=close.index)
    residuals = close.values[start:] - nw_line[start:]
    mae = np.nanmean(np.abs(residuals))
    nw_upper = nw_series + 2 * mae
    nw_lower = nw_series - 2 * mae
    return nw_series, nw_upper, nw_lower


def calc_linear_regression_channel(close, period=50, std_mult=2.0):
    n = len(close)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period - 1, n):
        y = close.values[i - period + 1:i + 1].astype(float)
        x = np.arange(period)
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        residuals = y - y_pred
        std = np.std(residuals)
        mid[i] = y_pred[-1]
        upper[i] = y_pred[-1] + std_mult * std
        lower[i] = y_pred[-1] - std_mult * std
    return (
        pd.Series(mid, index=close.index),
        pd.Series(upper, index=close.index),
        pd.Series(lower, index=close.index),
    )


def create_complete_trading_chart(ticker, start, end, per, k_len, s_mult, srsi_len, v_bins, f_look,
                                   show_kama, show_supertrend, show_stochrsi, div_lookback, show_fib, show_vrvp,
                                   show_sma, sma1_len, sma2_len, show_ema, ema1_len, ema2_len, show_bb, bb_len, bb_std,
                                   show_ichimoku, show_poc, chart_type,
                                   show_rsi, rsi_period, rsi_lower, rsi_upper,
                                   show_macd,
                                   show_adx, adx_period, adx_threshold,
                                   show_obv,
                                   show_zscore, z_period, z_threshold,
                                   show_lrc, lrc_period, lrc_std,
                                   show_nw, nw_bandwidth, nw_window):

    # 1. Veri Çekme
    resample_map = {"2h": "2h", "4h": "4h", "8h": "8h"}
    raw_p = "1h" if per in resample_map else per

    if end >= datetime.now().date():
        df = yf.download(ticker, start=start, interval=raw_p, auto_adjust=True)
    else:
        end_with_today = end + timedelta(days=1)
        df = yf.download(ticker, start=start, end=end_with_today, interval=raw_p, auto_adjust=True)

    if df.empty:
        st.error("Veri bulunamadı. Lütfen tarih sınırlarını veya sembolü kontrol edin.")
        return None, None, None, [], []

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.strip().title() for c in df.columns]

    try:
        anlik_gercek_fiyat = float(yf.Ticker(ticker).fast_info['lastPrice'])
    except Exception:
        anlik_gercek_fiyat = float(df['Close'].iloc[-1])

    # 2. Resampling
    if per in resample_map:
        df = df.resample(resample_map[per]).agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

    # ============================================================
    # 3. Teknik Gösterge Hesaplamaları
    # ============================================================

    # KAMA
    if show_kama:
        try:
            kama_result = ta.kama(df['Close'], length=k_len)
            if kama_result is not None:
                df['KAMA'] = kama_result
            else:
                show_kama = False
        except Exception as e:
            st.warning(f"KAMA hatası: {e}")
            show_kama = False

    # SuperTrend
    if show_supertrend:
        try:
            sti = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=s_mult)
            if sti is not None and hasattr(sti, 'iloc'):
                df['ST_Line'], df['ST_Dir'] = sti.iloc[:, 0], sti.iloc[:, 1]
                df['Buy']  = (df['ST_Dir'] == 1)  & (df['ST_Dir'].shift(1) == -1)
                df['Sell'] = (df['ST_Dir'] == -1) & (df['ST_Dir'].shift(1) == 1)
            else:
                show_supertrend = False
        except Exception as e:
            st.warning(f"SuperTrend hatası: {e}")
            show_supertrend = False

    # Divergence Osilatörü
    if show_stochrsi:
        try:
            rsi_raw = ta.rsi(df['Close'], length=srsi_len)
            if rsi_raw is not None:
                df['Mom']        = rsi_raw - 50
                df['Mom_Signal'] = ta.ema(df['Mom'], length=9)
                df['Mom_Hist']   = df['Mom'] - df['Mom_Signal']
                lookback = div_lookback
                df['Swing_Low']      = df['Close'][(df['Close'].shift(lookback) > df['Close']) & (df['Close'].shift(-lookback) > df['Close'])]
                df['Swing_High']     = df['Close'][(df['Close'].shift(lookback) < df['Close']) & (df['Close'].shift(-lookback) < df['Close'])]
                df['Mom_Swing_Low']  = df['Mom'][(df['Mom'].shift(lookback) > df['Mom'])   & (df['Mom'].shift(-lookback) > df['Mom'])]
                df['Mom_Swing_High'] = df['Mom'][(df['Mom'].shift(lookback) < df['Mom'])   & (df['Mom'].shift(-lookback) < df['Mom'])]
                df['Bull_Div'] = False
                df['Bear_Div'] = False
                sl_idx = df.dropna(subset=['Swing_Low']).index
                if len(sl_idx) > 1:
                    sl_close = df['Close'].reindex(sl_idx)
                    sl_mom   = df['Mom'].reindex(sl_idx)
                    bull_mask = (sl_close.values[1:] < sl_close.values[:-1]) & \
                                (sl_mom.values[1:]   > sl_mom.values[:-1])
                    df.loc[sl_idx[1:][bull_mask], 'Bull_Div'] = True
                sh_idx = df.dropna(subset=['Swing_High']).index
                if len(sh_idx) > 1:
                    sh_close = df['Close'].reindex(sh_idx)
                    sh_mom   = df['Mom'].reindex(sh_idx)
                    bear_mask = (sh_close.values[1:] > sh_close.values[:-1]) & \
                                (sh_mom.values[1:]   < sh_mom.values[:-1])
                    df.loc[sh_idx[1:][bear_mask], 'Bear_Div'] = True
            else:
                show_stochrsi = False
        except Exception as e:
            st.warning(f"Divergence osilatörü hatası: {e}")
            show_stochrsi = False

    # SMA
    if show_sma:
        try:
            sma1_result = ta.sma(df['Close'], length=sma1_len)
            sma2_result = ta.sma(df['Close'], length=sma2_len)
            if sma1_result is not None: df['SMA_1'] = sma1_result
            if sma2_result is not None: df['SMA_2'] = sma2_result
            if sma1_result is None and sma2_result is None:
                show_sma = False
        except Exception as e:
            st.warning(f"SMA hatası: {e}")
            show_sma = False

    # EMA
    if show_ema:
        try:
            ema1_result = ta.ema(df['Close'], length=ema1_len)
            ema2_result = ta.ema(df['Close'], length=ema2_len)
            if ema1_result is not None: df['EMA_1'] = ema1_result
            if ema2_result is not None: df['EMA_2'] = ema2_result
            if ema1_result is None and ema2_result is None:
                show_ema = False
        except Exception as e:
            st.warning(f"EMA hatası: {e}")
            show_ema = False

    # Bollinger Bands
    if show_bb:
        try:
            bbands = ta.bbands(df['Close'], length=bb_len, std=bb_std)
            if bbands is not None and hasattr(bbands, 'columns'):
                for c in bbands.columns:
                    if c.startswith('BBU'):   df['BB_Upper'] = bbands[c]
                    elif c.startswith('BBM'): df['BB_Mid']   = bbands[c]
                    elif c.startswith('BBL'): df['BB_Lower'] = bbands[c]
            else:
                show_bb = False
        except Exception as e:
            st.warning(f"Bollinger Bands hatası: {e}")
            show_bb = False

    # Ichimoku
    if show_ichimoku:
        try:
            ichi_result = ta.ichimoku(df['High'], df['Low'], df['Close'])
            ichi_df = ichi_result[0] if isinstance(ichi_result, tuple) else ichi_result
            if ichi_df is None or not hasattr(ichi_df, 'columns'):
                show_ichimoku = False
            else:
                cols = ichi_df.columns.tolist()
                for c in cols:
                    cl = c.upper()
                    if 'ITS' in cl:   df['Tenkan']   = ichi_df[c]
                    elif 'IKS' in cl: df['Kijun']    = ichi_df[c]
                    elif 'ISA' in cl: df['Senkou_A'] = ichi_df[c]
                    elif 'ISB' in cl: df['Senkou_B'] = ichi_df[c]
                    elif 'ICS' in cl: df['Chikou']   = ichi_df[c]
                required = ['Tenkan', 'Kijun', 'Senkou_A', 'Senkou_B', 'Chikou']
                if not all(c in df.columns for c in required):
                    show_ichimoku = False
        except Exception as e:
            st.warning(f"Ichimoku hatası: {e}")
            show_ichimoku = False

    # Fibonacci
    fib = {}
    if show_fib:
        f_df = df if f_look is None else df.tail(f_look)
        hi, lo = f_df['High'].max(), f_df['Low'].min()
        diff = hi - lo
        fib = {
            '23.6%': hi - 0.236 * diff,
            '38.2%': hi - 0.382 * diff,
            '50.0%': hi - 0.500 * diff,
            '61.8%': hi - 0.618 * diff,
            '78.6%': hi - 0.786 * diff
        }

    # ── YENİ İNDİKATÖRLER ──────────────────────────────────────

    # RSI
    if show_rsi:
        try:
            df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
        except Exception as e:
            st.warning(f"RSI hatası: {e}")
            show_rsi = False

    # MACD
    if show_macd:
        try:
            macd_result = ta.macd(df['Close'])
            if macd_result is not None:
                df['MACD']   = macd_result.iloc[:, 0]
                df['MACD_S'] = macd_result.iloc[:, 1]
                df['MACD_H'] = macd_result.iloc[:, 2]
            else:
                show_macd = False
        except Exception as e:
            st.warning(f"MACD hatası: {e}")
            show_macd = False

    # ADX
    if show_adx:
        try:
            df['ADX'], df['PLUS_DI'], df['MINUS_DI'] = calc_adx(df['High'], df['Low'], df['Close'], period=adx_period)
        except Exception as e:
            st.warning(f"ADX hatası: {e}")
            show_adx = False

    # OBV
    if show_obv:
        try:
            obv_sign = np.sign(df['Close'].diff()).fillna(0)
            df['OBV'] = (df['Volume'] * obv_sign).cumsum()
            df['OBV_SMA'] = df['OBV'].rolling(window=20).mean()
        except Exception as e:
            st.warning(f"OBV hatası: {e}")
            show_obv = False

    # Z-Score
    if show_zscore:
        try:
            z_mean = df['Close'].rolling(z_period).mean()
            z_std  = df['Close'].rolling(z_period).std().replace(0, np.nan)
            df['Z_Score'] = (df['Close'] - z_mean) / z_std
        except Exception as e:
            st.warning(f"Z-Score hatası: {e}")
            show_zscore = False

    # Linear Regression Channel
    if show_lrc:
        try:
            df['LRC_Mid'], df['LRC_Upper'], df['LRC_Lower'] = calc_linear_regression_channel(
                df['Close'], period=lrc_period, std_mult=lrc_std
            )
        except Exception as e:
            st.warning(f"LRC hatası: {e}")
            show_lrc = False

    # Nadaraya-Watson
    if show_nw:
        try:
            df['NW_Line'], df['NW_Upper'], df['NW_Lower'] = calc_nadaraya_watson(
                df['Close'], bandwidth=nw_bandwidth, window=nw_window
            )
        except Exception as e:
            st.warning(f"Nadaraya-Watson hatası: {e}")
            show_nw = False

    # ============================================================
    # LEGEND İÇİN SINYAL DURUM HESABI (skor sistemi kaldırıldı)
    # ============================================================
    last = df.iloc[-1]

    def _fmt(val, decimals=2):
        try:
            return f"{float(val):.{decimals}f}"
        except Exception:
            return "N/A"

    legend_signals = []

    # RSI
    if show_rsi and 'RSI' in df.columns:
        rsi_val = float(last['RSI']) if not pd.isna(last['RSI']) else None
        if rsi_val is not None:
            if rsi_val < rsi_lower:
                durum, renk = f"Aşırı Satım ({_fmt(rsi_val, 1)})", "#00c853"
            elif rsi_val > rsi_upper:
                durum, renk = f"Aşırı Alım ({_fmt(rsi_val, 1)})", "#ff1744"
            else:
                durum, renk = f"Nötr ({_fmt(rsi_val, 1)})", "#aaaaaa"
            legend_signals.append(('RSI', durum, renk))

    # MACD
    if show_macd and 'MACD' in df.columns and 'MACD_S' in df.columns:
        try:
            macd_v = float(last['MACD'])
            macd_s = float(last['MACD_S'])
            if macd_v > macd_s:
                durum, renk = f"Yükseliş ({_fmt(macd_v)} > {_fmt(macd_s)})", "#00c853"
            else:
                durum, renk = f"Düşüş ({_fmt(macd_v)} < {_fmt(macd_s)})", "#ff1744"
            legend_signals.append(('MACD', durum, renk))
        except Exception:
            pass

    # ADX
    if show_adx and 'ADX' in df.columns:
        try:
            adx_v = float(last['ADX'])
            pdi   = float(last['PLUS_DI'])
            mdi   = float(last['MINUS_DI'])
            if adx_v > adx_threshold:
                if pdi > mdi:
                    durum, renk = f"Güçlü Yükseliş Trendi (ADX:{_fmt(adx_v, 1)})", "#00c853"
                else:
                    durum, renk = f"Güçlü Düşüş Trendi (ADX:{_fmt(adx_v, 1)})", "#ff1744"
            else:
                durum, renk = f"Zayıf Trend (ADX:{_fmt(adx_v, 1)})", "#aaaaaa"
            legend_signals.append(('ADX', durum, renk))
        except Exception:
            pass

    # OBV
    if show_obv and 'OBV' in df.columns and 'OBV_SMA' in df.columns:
        try:
            obv_v   = float(last['OBV'])
            obv_sma = float(last['OBV_SMA'])
            if obv_v > obv_sma:
                durum, renk = "Hacim Trendi Yükseliş", "#00c853"
            else:
                durum, renk = "Hacim Trendi Düşüş", "#ff1744"
            legend_signals.append(('OBV', durum, renk))
        except Exception:
            pass

    # Stoch RSI / Divergence
    if show_stochrsi and 'Mom' in df.columns:
        try:
            mom_v = float(last['Mom'])
            mom_s = float(last['Mom_Signal'])
            bull  = bool(last['Bull_Div'])
            bear  = bool(last['Bear_Div'])
            if bull:
                durum, renk = f"Yükseliş Uyumsuzluğu (Mom:{_fmt(mom_v, 1)})", "#00c853"
            elif bear:
                durum, renk = f"Düşüş Uyumsuzluğu (Mom:{_fmt(mom_v, 1)})", "#ff1744"
            elif mom_v < -20:
                durum, renk = f"Aşırı Satım (Mom:{_fmt(mom_v, 1)})", "#00c853"
            elif mom_v > 20:
                durum, renk = f"Aşırı Alım (Mom:{_fmt(mom_v, 1)})", "#ff1744"
            elif mom_v > mom_s:
                durum, renk = f"Momentum Yükseliş (Mom:{_fmt(mom_v, 1)})", "#00c853"
            else:
                durum, renk = f"Momentum Düşüş (Mom:{_fmt(mom_v, 1)})", "#aaaaaa"
            legend_signals.append(('Stoch RSI / Div', durum, renk))
        except Exception:
            pass

    # Z-Score
    if show_zscore and 'Z_Score' in df.columns:
        try:
            z_val = float(last['Z_Score'])
            if z_val < -z_threshold:
                durum, renk = f"Aşırı Satım (Z:{_fmt(z_val)})", "#00c853"
            elif z_val > z_threshold:
                durum, renk = f"Aşırı Alım (Z:{_fmt(z_val)})", "#ff1744"
            else:
                durum, renk = f"Nötr (Z:{_fmt(z_val)})", "#aaaaaa"
            legend_signals.append(('Z-Score', durum, renk))
        except Exception:
            pass

    # LRC
    if show_lrc and 'LRC_Mid' in df.columns:
        try:
            close_v   = float(last['Close'])
            lrc_up    = float(last['LRC_Upper'])
            lrc_lo    = float(last['LRC_Lower'])
            lrc_mid   = float(last['LRC_Mid'])
            if close_v > lrc_up:
                durum, renk = f"Üst Kanal Dışı (Aşırı Alım)", "#ff1744"
            elif close_v < lrc_lo:
                durum, renk = f"Alt Kanal Dışı (Aşırı Satım)", "#00c853"
            else:
                durum, renk = f"Kanal İçi (Orta:{_fmt(lrc_mid)})", "#aaaaaa"
            legend_signals.append(('LR Channel', durum, renk))
        except Exception:
            pass

    # Nadaraya-Watson
    if show_nw and 'NW_Line' in df.columns:
        try:
            close_v  = float(last['Close'])
            nw_up    = float(last['NW_Upper'])
            nw_lo    = float(last['NW_Lower'])
            nw_mid   = float(last['NW_Line'])
            if close_v > nw_up:
                durum, renk = f"Üst Zarf Dışı (Aşırı Alım)", "#ff1744"
            elif close_v < nw_lo:
                durum, renk = f"Alt Zarf Dışı (Aşırı Satım)", "#00c853"
            else:
                durum, renk = f"Zarf İçi (NW:{_fmt(nw_mid)})", "#aaaaaa"
            legend_signals.append(('Nadaraya-Watson', durum, renk))
        except Exception:
            pass

    # ============================================================
    # 4. Görselleştirme
    # ============================================================
    has_oscillator = show_stochrsi
    row_heights = [0.65, 0.35] if has_oscillator else [1.0, 0.001]

    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                        column_widths=[0.85, 0.15], row_heights=row_heights,
                        vertical_spacing=0.05, horizontal_spacing=0.01)

    # ── LEGEND: İNDİKATÖR DURUMLARI ──────────────────────────────
    if legend_signals:
        # Başlık
        fig.add_trace(go.Scatter(
            x=[df.index[0]], y=[df['Close'].iloc[0]],
            mode='lines', line=dict(color='rgba(0,0,0,0)', width=0),
            name='<span style="font-size:13px; font-weight:bold; color:#333;">── İndikatör Durumları ──</span>',
            showlegend=True, hoverinfo='skip'
        ), row=1, col=1)

        for isim, durum, renk in legend_signals:
            fig.add_trace(go.Scatter(
                x=[df.index[0]], y=[df['Close'].iloc[0]],
                mode='lines', line=dict(color='rgba(0,0,0,0)', width=0),
                name=f'<span style="color:{renk}; font-size:11px;"><b>{isim}:</b> {durum}</span>',
                showlegend=True, hoverinfo='skip'
            ), row=1, col=1)

        # Ayırıcı boşluk
        fig.add_trace(go.Scatter(
            x=[df.index[0]], y=[df['Close'].iloc[0]],
            mode='lines', line=dict(color='rgba(0,0,0,0)', width=0),
            name=' ', showlegend=True, hoverinfo='skip'
        ), row=1, col=1)

    # Grafik tipi
    if chart_type == "Mum (Candlestick)":
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                      low=df['Low'], close=df['Close'], name='Fiyat'), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines',
                                  line=dict(color='#ff1744', width=2), name='Fiyat (Kapanış)'), row=1, col=1)

    # KAMA
    if show_kama:
        fig.add_trace(go.Scatter(x=df.index, y=df['KAMA'],
                                  line=dict(color='#2962ff', width=2), name='KAMA'), row=1, col=1)

    # SuperTrend
    if show_supertrend:
        fig.add_trace(go.Scatter(x=df.index, y=df['ST_Line'],
                                  line=dict(color='rgba(80,80,80,0.7)', width=2.5),
                                  line_shape='hv', name='Trend Sınırı'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df[df['Buy']].index, y=df[df['Buy']]['Low'] * 0.98,
                                  mode='markers+text', text="AL",
                                  marker=dict(symbol='triangle-up', size=15, color='green'), name='AL'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df[df['Sell']].index, y=df[df['Sell']]['High'] * 1.02,
                                  mode='markers+text', text="SAT",
                                  marker=dict(symbol='triangle-down', size=15, color='red'), name='SAT'), row=1, col=1)

    # Divergence okları
    if show_stochrsi:
        bull_div = df[df['Bull_Div'] == True]
        bear_div = df[df['Bear_Div'] == True]
        if len(bull_div) > 0:
            fig.add_trace(go.Scatter(x=bull_div.index, y=bull_div['Low'] * 0.99,
                                      mode='markers+text', text="▲D",
                                      textposition='bottom center', textfont=dict(size=9, color='#00e676'),
                                      marker=dict(symbol='triangle-up', size=12, color='#00e676'),
                                      name='Yükseliş Uyumsuzluğu', visible='legendonly'), row=1, col=1)
        if len(bear_div) > 0:
            fig.add_trace(go.Scatter(x=bear_div.index, y=bear_div['High'] * 1.01,
                                      mode='markers+text', text="▼D",
                                      textposition='top center', textfont=dict(size=9, color='#ff1744'),
                                      marker=dict(symbol='triangle-down', size=12, color='#ff1744'),
                                      name='Düşüş Uyumsuzluğu', visible='legendonly'), row=1, col=1)

    # SMA
    if show_sma:
        if 'SMA_1' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_1'],
                                      line=dict(color='#ff9800', width=2), name=f'SMA 1 ({sma1_len})', visible='legendonly'), row=1, col=1)
        if 'SMA_2' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_2'],
                                      line=dict(color='#2196f3', width=2), name=f'SMA 2 ({sma2_len})', visible='legendonly'), row=1, col=1)

    # EMA
    if show_ema:
        if 'EMA_1' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_1'],
                                      line=dict(color='#ab47bc', width=2), name=f'EMA 1 ({ema1_len})', visible='legendonly'), row=1, col=1)
        if 'EMA_2' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_2'],
                                      line=dict(color='#26a69a', width=2), name=f'EMA 2 ({ema2_len})', visible='legendonly'), row=1, col=1)

    # Bollinger Bands
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'],
                                  line=dict(color='rgba(174,134,255,0.6)', width=1), name='BB Üst', visible='legendonly'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'],
                                  line=dict(color='rgba(174,134,255,0.9)', width=1, dash='dot'), name='BB Orta', visible='legendonly'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'],
                                  line=dict(color='rgba(174,134,255,0.6)', width=1),
                                  fill='tonexty', fillcolor='rgba(174,134,255,0.07)', name='BB Alt', visible='legendonly'), row=1, col=1)

    # Ichimoku
    if show_ichimoku:
        fig.add_trace(go.Scatter(x=df.index, y=df['Tenkan'],
                                  line=dict(color='#0496ff', width=1), name='Tenkan-sen', visible='legendonly'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Kijun'],
                                  line=dict(color='#991515', width=1), name='Kijun-sen', visible='legendonly'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_A'],
                                  line=dict(color='rgba(67,160,71,0.5)', width=1), name='Senkou A', visible='legendonly'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_B'],
                                  line=dict(color='rgba(244,67,54,0.5)', width=1),
                                  fill='tonexty', fillcolor='rgba(67,160,71,0.06)', name='Senkou B', visible='legendonly'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Chikou'],
                                  line=dict(color='#9c27b0', width=1, dash='dot'), name='Chikou', visible='legendonly'), row=1, col=1)

    # Linear Regression Channel
    if show_lrc:
        fig.add_trace(go.Scatter(x=df.index, y=df['LRC_Mid'],
                                  line=dict(color='white', width=1, dash='dash'), name='LRC Orta', visible='legendonly'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['LRC_Upper'],
                                  line=dict(color='rgba(200,200,200,0.5)', width=1, dash='dot'), name='LRC Üst', visible='legendonly'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['LRC_Lower'],
                                  line=dict(color='rgba(200,200,200,0.5)', width=1, dash='dot'),
                                  fill='tonexty', fillcolor='rgba(150,150,150,0.05)', name='LRC Alt', visible='legendonly'), row=1, col=1)

    # Nadaraya-Watson
    if show_nw:
        fig.add_trace(go.Scatter(x=df.index, y=df['NW_Line'],
                                  line=dict(color='gold', width=1.5), name='NW Orta', visible='legendonly'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['NW_Upper'],
                                  line=dict(color='rgba(255,215,0,0.5)', width=1, dash='dot'), name='NW Üst', visible='legendonly'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['NW_Lower'],
                                  line=dict(color='rgba(255,215,0,0.5)', width=1, dash='dot'),
                                  fill='tonexty', fillcolor='rgba(255,215,0,0.04)', name='NW Alt', visible='legendonly'), row=1, col=1)

    # Fibonacci
    if show_fib:
        for l, p in fib.items():
            fig.add_hline(y=p, line_dash="dash", line_color="rgba(128,128,128,0.5)",
                          annotation_text=f"{l} ({p:.2f})",
                          annotation_position="right",
                          annotation_bgcolor="orange",
                          annotation_font_color="black",
                          row=1, col=1)

    # Son fiyat çizgisi
    prev_close = None
    if not df.empty:
        gosterilecek_fiyat = anlik_gercek_fiyat if anlik_gercek_fiyat is not None else float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else float(df['Open'].iloc[-1])
        price_color = "#00e676" if gosterilecek_fiyat >= prev_close else "#ff1744"
        fig.add_hline(y=gosterilecek_fiyat, line_dash="dot", line_width=1, line_color=price_color,
                      annotation_text=f"Anlık: {gosterilecek_fiyat:.2f}",
                      annotation_position="right",
                      annotation_bgcolor=price_color,
                      annotation_font_color="black",
                      row=1, col=1)

    # VRVP & POC
    top3_hacim = []
    if show_vrvp:
        bins = pd.cut(df['Close'], bins=v_bins, retbins=True)[1]
        df['V_T'] = df.apply(lambda r: 'B' if r['Close'] >= r['Open'] else 'S', axis=1)
        max_total_vol = -1
        poc_price_low = 0
        poc_price_high = 0
        hacim_listesi = []

        for i in range(v_bins):
            m = (df['Close'] >= bins[i]) & (df['Close'] < bins[i + 1])
            vb = df[m & (df['V_T'] == 'B')]['Volume'].sum()
            vs = df[m & (df['V_T'] == 'S')]['Volume'].sum()
            total_vol = vb + vs
            orta_fiyat = (bins[i] + bins[i + 1]) / 2
            hacim_listesi.append((orta_fiyat, total_vol))
            if total_vol > max_total_vol:
                max_total_vol = total_vol
                poc_price_low = bins[i]
                poc_price_high = bins[i + 1]
            fig.add_trace(go.Bar(x=[vs], y=[(bins[i] + bins[i + 1]) / 2], orientation='h',
                                  marker_color='rgba(239,83,80,0.4)', showlegend=False), row=1, col=2)
            fig.add_trace(go.Bar(x=[vb], y=[(bins[i] + bins[i + 1]) / 2], orientation='h',
                                  marker_color='rgba(38,166,154,0.4)', showlegend=False), row=1, col=2)

        ref_fiyat = anlik_gercek_fiyat if anlik_gercek_fiyat else float(df['Close'].iloc[-1])
        destekler = sorted([x for x in hacim_listesi if x[0] < ref_fiyat], key=lambda x: x[1], reverse=True)[:3]
        direncler = sorted([x for x in hacim_listesi if x[0] >= ref_fiyat], key=lambda x: x[1], reverse=True)[:3]
        top3_hacim = (destekler, direncler)

        if show_poc and max_total_vol > 0:
            x_coords = [df.index[0], df.index[-1], df.index[-1], df.index[0], df.index[0]]
            poc_mid = (poc_price_low + poc_price_high) / 2
            offset = (poc_price_high - poc_price_low) * 0.1
            y_coords = [poc_mid - offset, poc_mid - offset, poc_mid + offset, poc_mid + offset, poc_mid - offset]
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                fill="toself",
                fillcolor='rgba(255, 255, 0, 0.25)',
                line=dict(color='yellow', width=1.5),
                name='POC (Destek)',
                showlegend=True,
                hoverinfo='skip'
            ), row=1, col=1)

    # Osilatör paneli
    if show_stochrsi:
        fig.add_trace(go.Scatter(x=df.index, y=df['Mom'],
                                  line=dict(color='#00c853', width=1.5), name='Momentum'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Mom_Signal'],
                                  line=dict(color='#ff1744', width=1.5), name='Sinyal'), row=2, col=1)
        fig.add_hline(y=0,   line_dash="solid", line_color="rgba(128,128,128,0.5)", row=2, col=1)
        fig.add_hline(y=30,  line_dash="dash",  line_color="rgba(255,23,68,0.5)",  annotation_text="30", row=2, col=1)
        fig.add_hline(y=20,  line_dash="dot",   line_color="rgba(255,23,68,0.3)",  row=2, col=1)
        fig.add_hline(y=-30, line_dash="dash",  line_color="rgba(0,200,83,0.5)",   annotation_text="-30", row=2, col=1)
        fig.add_hline(y=-20, line_dash="dot",   line_color="rgba(0,200,83,0.3)",   row=2, col=1)

        bull_div = df[df['Bull_Div'] == True]
        bear_div = df[df['Bear_Div'] == True]
        if len(bull_div) > 0:
            fig.add_trace(go.Scatter(x=bull_div.index, y=bull_div['Mom'],
                                      mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00e676'),
                                      name='Bull Div', showlegend=False), row=2, col=1)
        if len(bear_div) > 0:
            fig.add_trace(go.Scatter(x=bear_div.index, y=bear_div['Mom'],
                                      mode='markers', marker=dict(symbol='triangle-down', size=10, color='#ff1744'),
                                      name='Bear Div', showlegend=False), row=2, col=1)

    # ── Layout ──────────────────────────────────────────────────
    visible_candles = 100
    if not df.empty and len(df) > visible_candles:
        view_start = df.index[-visible_candles]
        view_end   = df.index[-1]
    else:
        view_start = df.index[0]  if not df.empty else datetime.now() - timedelta(days=30)
        view_end   = df.index[-1] if not df.empty else datetime.now()

    fig.update_layout(
        template='plotly_white',
        height=1200,
        dragmode='pan',
        barmode='stack',
        title=f"<b>{ticker}</b> Teknik Analizi",
        margin=dict(l=10, r=60, t=50, b=10),
        legend=dict(
            font=dict(size=11),
            itemwidth=30,
            x=1.01,
            xanchor='left',
            y=1,
            yanchor='top',
            bgcolor='rgba(255,255,255,0.6)'
        ),
        xaxis=dict(
            range=[view_start, view_end],
            rangeslider=dict(visible=False),
            fixedrange=False,
            autorange=False
        ),
        yaxis=dict(
            side="right",
            fixedrange=False,
            autorange=True
        )
    )

    fig.update_xaxes(matches='x')

    return fig, anlik_gercek_fiyat, prev_close, top3_hacim


# ============================================================
# KONTROL PANELİ
# ============================================================
st.sidebar.header("🛠️ Analiz Ayarları")

Hisse           = st.sidebar.text_input("Varlık Sembolü", value="PAXG-USD")
col1, col2      = st.sidebar.columns(2)
Baslangic       = col1.date_input("Başlangıç", value=datetime.now() - timedelta(days=120))
Bitis           = col2.date_input("Bitiş",     value=datetime.now())
Secilen_Periyot = st.sidebar.selectbox("Periyot", ["15m", "30m", "1h", "2h", "4h", "8h", "1d", "1wk"], index=5)

st.sidebar.markdown("---")

oto_yenile = st.sidebar.checkbox("Otomatik Yenile (1 Dk)", value=False)
if oto_yenile:
    if st_autorefresh is not None:
        st_autorefresh(interval=60000, key="data_refresh")
    else:
        st.sidebar.warning("Otomatik yenileme için `pip install streamlit-autorefresh` çalıştırın.")

st.sidebar.markdown("---")

GRAFIK_TIPI = st.sidebar.radio("Grafik Görünümü", ["Çizgi (Line)", "Mum (Candlestick)"], horizontal=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Mevcut Göstergeler")
show_kama       = st.sidebar.checkbox("KAMA",                     value=True)
show_supertrend = st.sidebar.checkbox("SuperTrend (AL/SAT)",      value=True)
show_stochrsi   = st.sidebar.checkbox("Divergence Osilatörü",     value=True)
show_fib        = st.sidebar.checkbox("Fibonacci Seviyeleri",     value=False)
show_vrvp       = st.sidebar.checkbox("VRVP (Hacim Profili)",     value=True)
show_poc        = st.sidebar.checkbox("Sarı Dikdörtgen (POC)",    value=True)
show_sma        = st.sidebar.checkbox("SMA",                      value=True)
show_ema        = st.sidebar.checkbox("EMA",                      value=False)
show_bb         = st.sidebar.checkbox("Bollinger Bands",          value=True)
show_ichimoku   = st.sidebar.checkbox("Ichimoku Cloud",           value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Ek Göstergeler")
show_rsi    = st.sidebar.checkbox("RSI",                      value=True)
show_macd   = st.sidebar.checkbox("MACD",                     value=True)
show_adx    = st.sidebar.checkbox("ADX",                      value=True)
show_obv    = st.sidebar.checkbox("OBV",                      value=True)
show_zscore = st.sidebar.checkbox("Z-Score (Mean Reversion)", value=True)
show_lrc    = st.sidebar.checkbox("Linear Regression Channel",value=True)
show_nw     = st.sidebar.checkbox("Nadaraya-Watson",          value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Hassasiyet Ayarları")

KAMA_HIZI    = st.sidebar.slider("KAMA Hızı",              5,  50,  10)           if show_kama        else 10
TREND_CARPAN = st.sidebar.slider("Trend Çarpanı",          1.0, 5.0, 2.0, 0.5)   if show_supertrend  else 2.0
OSILATOR_PER = st.sidebar.slider("Divergence RSI Periyodu",7,  30,  14)           if show_stochrsi    else 14
DIV_LOOKBACK = st.sidebar.slider("Divergence Lookback",    2,  20,  5)            if show_stochrsi    else 5
HACIM_DETAY  = st.sidebar.slider("Hacim Detayı",           20, 100, 40)           if show_vrvp        else 40
FIB_BAKIS    = st.sidebar.number_input("Fib Geriye Bakış", value=100)             if show_fib         else 100
SMA_1_LEN    = st.sidebar.slider("SMA 1 Periyodu",         5,  200, 50)           if show_sma         else 50
SMA_2_LEN    = st.sidebar.slider("SMA 2 Periyodu",         5,  400, 200)          if show_sma         else 200
EMA_1_LEN    = st.sidebar.slider("EMA 1 Periyodu",         5,  200, 20)           if show_ema         else 20
EMA_2_LEN    = st.sidebar.slider("EMA 2 Periyodu",         5,  400, 50)           if show_ema         else 50
BB_LEN       = st.sidebar.slider("BB Periyodu",            5,  50,  20)           if show_bb          else 20
BB_STD       = st.sidebar.slider("BB Standart Sapma",      1.0, 4.0, 2.0, 0.5)   if show_bb          else 2.0

RSI_PERIOD   = st.sidebar.slider("RSI Periyodu",           5,  30,  14)           if show_rsi         else 14
RSI_LOWER    = st.sidebar.slider("RSI Aşırı Satım Eşiği",  10, 40,  30)           if show_rsi         else 30
RSI_UPPER    = st.sidebar.slider("RSI Aşırı Alım Eşiği",   60, 90,  70)           if show_rsi         else 70
ADX_PERIOD   = st.sidebar.slider("ADX Periyodu",           7,  30,  14)           if show_adx         else 14
ADX_THRESH   = st.sidebar.slider("ADX Trend Eşiği",        15, 40,  25)           if show_adx         else 25
Z_PERIOD     = st.sidebar.slider("Z-Score Periyodu",       10, 60,  30)           if show_zscore      else 30
Z_THRESH     = st.sidebar.slider("Z-Score Eşiği (±)",      1.0, 3.0, 2.0, 0.1)   if show_zscore      else 2.0
LRC_PERIOD   = st.sidebar.slider("LRC Periyodu",           20, 100, 50)           if show_lrc         else 50
LRC_STD      = st.sidebar.slider("LRC Std Çarpanı",        1.0, 3.0, 2.0, 0.5)   if show_lrc         else 2.0
NW_BW        = st.sidebar.slider("NW Bant Genişliği",      3,  20,  8)            if show_nw          else 8
NW_WINDOW    = st.sidebar.slider("NW Pencere (bar)",        50, 300, 100)          if show_nw          else 100

# ============================================================
# ANALİZİ BAŞLAT
# ============================================================
if st.sidebar.button("Analizi Başlat") or oto_yenile:
    with st.spinner('Veriler hesaplanıyor...'):
        fig, anlik_fiyat, onceki_fiyat, top3_hacim = create_complete_trading_chart(
            Hisse, Baslangic, Bitis, Secilen_Periyot,
            KAMA_HIZI, TREND_CARPAN, OSILATOR_PER, HACIM_DETAY, FIB_BAKIS,
            show_kama, show_supertrend, show_stochrsi, DIV_LOOKBACK, show_fib, show_vrvp,
            show_sma, SMA_1_LEN, SMA_2_LEN, show_ema, EMA_1_LEN, EMA_2_LEN, show_bb, BB_LEN, BB_STD,
            show_ichimoku, show_poc, GRAFIK_TIPI,
            show_rsi, RSI_PERIOD, RSI_LOWER, RSI_UPPER,
            show_macd,
            show_adx, ADX_PERIOD, ADX_THRESH,
            show_obv,
            show_zscore, Z_PERIOD, Z_THRESH,
            show_lrc, LRC_PERIOD, LRC_STD,
            show_nw, NW_BW, NW_WINDOW
        )
        if fig:
            st.markdown("""
            <style>
            div[data-testid="stMetric"] { padding: 8px 12px; }
            div[data-testid="stMetric"] label { font-size: 0.75rem !important; }
            div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
            div[data-testid="stMetric"] div[data-testid="stMetricDelta"] { font-size: 0.7rem !important; }
            </style>
            """, unsafe_allow_html=True)

            if anlik_fiyat is not None and onceki_fiyat is not None:
                fiyat_farki = anlik_fiyat - onceki_fiyat
                yuzde_fark  = (fiyat_farki / onceki_fiyat) * 100
                tr_saati    = datetime.now(timezone(timedelta(hours=3)))

                destekler, direncler = top3_hacim if top3_hacim else ([], [])
                kolon_sayisi = 3 + len(destekler) + len(direncler)
                tum_kolonlar = st.columns(kolon_sayisi)
                ki = 0

                tum_kolonlar[ki].metric(f"Anlık ({Hisse})", f"{anlik_fiyat:.2f}", f"{yuzde_fark:+.2f}%"); ki += 1
                tum_kolonlar[ki].metric("Periyot", Secilen_Periyot); ki += 1
                tum_kolonlar[ki].metric("Güncelleme", tr_saati.strftime("%H:%M:%S")); ki += 1

                for idx, (fiyat, hacim) in enumerate(destekler):
                    fark_yuzde = ((fiyat - anlik_fiyat) / anlik_fiyat) * 100
                    tum_kolonlar[ki].metric(f"↓ Destek {idx+1}", f"{fiyat:.2f}", f"{fark_yuzde:+.2f}%"); ki += 1

                for idx, (fiyat, hacim) in enumerate(direncler):
                    fark_yuzde = ((fiyat - anlik_fiyat) / anlik_fiyat) * 100
                    tum_kolonlar[ki].metric(f"↑ Direnç {idx+1}", f"{fiyat:.2f}", f"{fark_yuzde:+.2f}%"); ki += 1

            st.markdown("---")
            config = {'scrollZoom': True, 'displayModeBar': True}
            st.plotly_chart(fig, use_container_width=True, config=config)
else:
    st.info("Analiz yapmak için sol paneldeki 'Analizi Başlat' butonuna tıklayın.")
