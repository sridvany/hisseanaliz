import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone

# --- EKLENEN OTOMATİK YENİLEME KÜTÜPHANESİ ---
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None
# ---------------------------------------------

st.set_page_config(layout="wide", page_title="AI Teknik Analiz Terminali")


def hesapla_skorlar(df, show_kama, show_supertrend, show_stochrsi, show_bb, show_ichimoku, show_sma, show_ema):
    """Tüm AL/SAT skorlarını hesaplar ve dict olarak döner."""
    skorlar = {}

    # ── 1. Güçlü Momentum AL ──────────────────────────────────────
    if show_stochrsi and all(c in df.columns for c in ['Bull_Div', 'Mom', 'Mom_Signal']):
        last = df.iloc[-1]
        k1 = bool(last['Bull_Div'])
        k2 = float(last['Mom']) < -20
        k3 = float(last['Mom']) > float(last['Mom_Signal'])
        puan = int(k1) + int(k2) + int(k3)
        skorlar['🟢 Güçlü Momentum AL'] = {
            'puan': puan, 'max': 3,
            'detay': [
                ('Bullish Divergence', k1),
                ('Mom < -20 (Aşırı Satım)', k2),
                ('Mom > Sinyal Çizgisi', k3),
            ]
        }

    # ── 2. Trend Konfirmasyon AL ──────────────────────────────────
    cols_needed = []
    if show_supertrend: cols_needed.append('ST_Dir')
    if show_kama:       cols_needed.append('KAMA')
    if show_bb:         cols_needed.append('BB_Mid')
    if cols_needed and all(c in df.columns for c in cols_needed):
        last = df.iloc[-1]
        k1 = (float(last['ST_Dir']) == 1)       if 'ST_Dir' in df.columns else None
        k2 = (float(last['Close']) > float(last['KAMA'])) if 'KAMA'   in df.columns else None
        k3 = (float(last['Close']) > float(last['BB_Mid'])) if 'BB_Mid' in df.columns else None
        aktif = [x for x in [k1, k2, k3] if x is not None]
        puan  = sum(int(x) for x in aktif)
        detay = []
        if k1 is not None: detay.append(('SuperTrend Yukarı', k1))
        if k2 is not None: detay.append(('Fiyat > KAMA', k2))
        if k3 is not None: detay.append(('Fiyat > BB Orta', k3))
        skorlar['🟢 Trend Konfirmasyon AL'] = {'puan': puan, 'max': len(detay), 'detay': detay}

    # ── 3. Ichimoku Güçlü AL ─────────────────────────────────────
    if show_ichimoku and all(c in df.columns for c in ['Senkou_A', 'Senkou_B', 'Tenkan', 'Kijun', 'Chikou']):
        last = df.iloc[-1]
        k1 = float(last['Close']) > float(last['Senkou_A']) and float(last['Close']) > float(last['Senkou_B'])
        k2 = float(last['Tenkan']) > float(last['Kijun'])
        # Chikou karşılaştırması: 26 bar önceki fiyatla
        if len(df) > 26:
            k3 = float(last['Chikou']) > float(df['Close'].iloc[-26])
        else:
            k3 = False
        puan = int(k1) + int(k2) + int(k3)
        skorlar['🟢 Ichimoku Güçlü AL'] = {
            'puan': puan, 'max': 3,
            'detay': [
                ('Fiyat Bulut Üzerinde', k1),
                ('Tenkan > Kijun', k2),
                ('Chikou Güçlü', k3),
            ]
        }

    # ── 4. Hareketli Ortalama AL ──────────────────────────────────
    ma_cols = []
    if show_sma and 'SMA_1' in df.columns and 'SMA_2' in df.columns: ma_cols.append('sma')
    if show_ema and 'EMA_1' in df.columns: ma_cols.append('ema')
    if ma_cols:
        last = df.iloc[-1]
        detay = []
        if 'sma' in ma_cols:
            k1 = float(last['SMA_1']) > float(last['SMA_2'])
            k2 = float(last['Close']) > float(last['SMA_1'])
            detay += [('Golden Cross (SMA1>SMA2)', k1), ('Fiyat > SMA1', k2)]
        if 'ema' in ma_cols:
            k3 = float(last['Close']) > float(last['EMA_1'])
            detay.append(('Fiyat > EMA1', k3))
        puan = sum(int(d[1]) for d in detay)
        skorlar['🟢 Hareketli Ortalama AL'] = {'puan': puan, 'max': len(detay), 'detay': detay}

    # ── 5. Güçlü Momentum SAT ────────────────────────────────────
    if show_stochrsi and all(c in df.columns for c in ['Bear_Div', 'Mom', 'Mom_Signal']):
        last = df.iloc[-1]
        k1 = bool(last['Bear_Div'])
        k2 = float(last['Mom']) > 20
        k3 = float(last['Mom']) < float(last['Mom_Signal'])
        puan = int(k1) + int(k2) + int(k3)
        skorlar['🔴 Güçlü Momentum SAT'] = {
            'puan': puan, 'max': 3,
            'detay': [
                ('Bearish Divergence', k1),
                ('Mom > 20 (Aşırı Alım)', k2),
                ('Mom < Sinyal Çizgisi', k3),
            ]
        }

    # ── 6. BB Kırılım SAT ────────────────────────────────────────
    sat_cols = []
    if show_bb and 'BB_Upper' in df.columns: sat_cols.append('bb')
    if show_supertrend and 'ST_Dir' in df.columns: sat_cols.append('st')
    if show_stochrsi and 'Bear_Div' in df.columns: sat_cols.append('div')
    if sat_cols:
        last = df.iloc[-1]
        detay = []
        if 'bb' in sat_cols:
            k1 = float(last['Close']) > float(last['BB_Upper'])
            detay.append(('Fiyat > BB Üst', k1))
        if 'st' in sat_cols:
            k2 = float(last['ST_Dir']) == -1
            detay.append(('SuperTrend Aşağı', k2))
        if 'div' in sat_cols:
            k3 = bool(last['Bear_Div'])
            detay.append(('Bearish Divergence', k3))
        puan = sum(int(d[1]) for d in detay)
        skorlar['🔴 BB Kırılım SAT'] = {'puan': puan, 'max': len(detay), 'detay': detay}

    return skorlar


def skor_annotation_html(skorlar):
    """Skorları Plotly annotation için HTML string'e çevirir."""
    if not skorlar:
        return ""
    satirlar = ["<b>── Temel Sinyaller ──</b>"]
    for baslik, veri in skorlar.items():
        p, m = veri['puan'], veri['max']
        yuzde = int(p / m * 100) if m > 0 else 0
        bar = "█" * p + "░" * (m - p)
        renk = "#00c853" if "AL" in baslik else "#ff1744"
        satirlar.append(
            f'<span style="color:{renk}"><b>{baslik}</b></span> {p}/{m} [{bar}] %{yuzde}'
        )
        for aciklama, durum in veri['detay']:
            isaret = "✔" if durum else "✘"
            renk2 = "#00c853" if durum else "#888888"
            satirlar.append(f'  <span style="color:{renk2}">{isaret} {aciklama}</span>')
    return "<br>".join(satirlar)


def create_complete_trading_chart(ticker, start, end, per, k_len, s_mult, srsi_len, v_bins, f_look,
                                   show_kama, show_supertrend, show_stochrsi, div_lookback, show_fib, show_vrvp,
                                   show_sma, sma1_len, sma2_len, show_ema, ema1_len, ema2_len, show_bb, bb_len, bb_std,
                                   show_ichimoku, chart_type):

    # 1. Veri Çekme (GÜNCELLENDİ)
    resample_map = {"2h": "2h", "4h": "4h", "8h": "8h"}
    raw_p = "1h" if per in resample_map else per
    
    # Bitiş tarihi bugün veya sonrasındaysa end parametresini yollamıyoruz (Canlı veri için)
    if end >= datetime.now().date():
        df = yf.download(ticker, start=start, interval=raw_p, auto_adjust=True)
    else:
        end_with_today = end + timedelta(days=1)
        df = yf.download(ticker, start=start, end=end_with_today, interval=raw_p, auto_adjust=True)

    if df.empty:
        st.error("Veri bulunamadı. Lütfen tarih sınırlarını veya sembolü kontrol edin.")
        return None, None, None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.strip().title() for c in df.columns]

    # --- EKLENEN ANLIK FİYAT ÇEKME BLOĞU (GÜNCELLENDİ) ---
    try:
        canli_veri = yf.download(ticker, period="1d", interval="1m", progress=False)
        if not canli_veri.empty:
            if isinstance(canli_veri.columns, pd.MultiIndex):
                canli_veri.columns = canli_veri.columns.get_level_values(0)
            anlik_gercek_fiyat = float(canli_veri['Close'].iloc[-1])
        else:
            anlik_gercek_fiyat = float(df['Close'].iloc[-1])
    except Exception as e:
        anlik_gercek_fiyat = float(df['Close'].iloc[-1])
    # ------------------------------------------------------

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
                st.warning("Bollinger Bands hesaplanamadı.")
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
                st.warning("Ichimoku hesaplanamadı.")
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
                    st.warning(f"Ichimoku sütunları eşleşmedi: {cols}")
                    show_ichimoku = False
        except Exception as e:
            st.warning(f"Ichimoku hesaplama hatası: {e}")
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

    # ============================================================
    # SKOR HESAPLAMA
    # ============================================================
    skorlar = hesapla_skorlar(df, show_kama, show_supertrend, show_stochrsi, show_bb, show_ichimoku, show_sma, show_ema)

    # ============================================================
    # 4. Görselleştirme
    # ============================================================
    has_oscillator = show_stochrsi
    row_heights = [0.65, 0.35] if has_oscillator else [1.0, 0.001]

    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                        column_widths=[0.85, 0.15], row_heights=row_heights,
                        vertical_spacing=0.05, horizontal_spacing=0.01)

    # ============================================================
    # SINYAL SKORLARI — İLK EKLENEN = LEGEND'DA EN ALTTA
    # ============================================================
    if not df.empty:
        # 1) Skor detaylarını ters sırada ekle (en alttaki skor önce)
        for baslik, veri in list(skorlar.items()):
            p, m = veri['puan'], veri['max']
            yuzde = int(p / m * 100) if m > 0 else 0
            renk = "#00c853" if "AL" in baslik else "#ff1744"
            # Önce detayları ters sırada ekle
            for aciklama, durum in reversed(veri['detay']):
                isaret = "✔" if durum else "✘"
                renk2 = "#00c853" if durum else "#aaaaaa"
                fig.add_trace(go.Scatter(
                    x=[df.index[0]], y=[df['Close'].iloc[0]],
                    mode='lines', line=dict(color='rgba(0,0,0,0)', width=0),
                    name=f'<span style="color:{renk2}; font-size:10px;">  {isaret} {aciklama}</span>',
                    showlegend=True, hoverinfo='skip'
                ), row=1, col=1)
            # Sonra skor başlığı
            skor_label = f'<span style="color:{renk}; font-size:11px;"><b>{baslik}</b>  {p}/{m} [{"█"*p}{"░"*(m-p)}] %{yuzde}</span>'
            fig.add_trace(go.Scatter(
                x=[df.index[0]], y=[df['Close'].iloc[0]],
                mode='lines', line=dict(color='rgba(0,0,0,0)', width=0),
                name=skor_label, showlegend=True, hoverinfo='skip'
            ), row=1, col=1)

        # 2) "── Çoklu Sinyal Skorları ──" başlığı
        fig.add_trace(go.Scatter(
            x=[df.index[0]], y=[df['Close'].iloc[0]],
            mode='lines', line=dict(color='rgba(0,0,0,0)', width=0),
            name='<span style="font-size:13px; color:#333; font-weight:bold;">── Temel Sinyaller ──</span>',
            showlegend=True, hoverinfo='skip'
        ), row=1, col=1)

        # 3) En son eklenen → legend'da en üstte → indikatörlerin hemen altında
        fig.add_trace(go.Scatter(
            x=[df.index[0]], y=[df['Close'].iloc[0]],
            mode='lines', line=dict(color='rgba(0,0,0,0)', width=0),
            name='<span style="font-size:12px; color:red; font-weight:bold;"> </span>',
            showlegend=True, hoverinfo='skip'
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
        # --- EKLENEN ANLIK FİYAT ÇİZGİSİ BLOĞU ---
        gosterilecek_fiyat = anlik_gercek_fiyat if anlik_gercek_fiyat is not None else float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else float(df['Open'].iloc[-1])
        price_color = "#00e676" if gosterilecek_fiyat >= prev_close else "#ff1744"
        fig.add_hline(y=gosterilecek_fiyat, line_dash="dot", line_width=1, line_color=price_color,
                      annotation_text=f"Anlık: {gosterilecek_fiyat:.2f}",
                      annotation_position="right",
                      annotation_bgcolor=price_color,
                      annotation_font_color="black",
                      row=1, col=1)
        # ------------------------------------------

    # VRVP
    if show_vrvp:
        bins = pd.cut(df['Close'], bins=v_bins, retbins=True)[1]
        df['V_T'] = df.apply(lambda r: 'B' if r['Close'] >= r['Open'] else 'S', axis=1)
        for i in range(v_bins):
            m = (df['Close'] >= bins[i]) & (df['Close'] < bins[i + 1])
            vb = df[m & (df['V_T'] == 'B')]['Volume'].sum()
            vs = df[m & (df['V_T'] == 'S')]['Volume'].sum()
            fig.add_trace(go.Bar(x=[vs], y=[(bins[i] + bins[i + 1]) / 2], orientation='h',
                                  marker_color='rgba(239,83,80,0.4)', showlegend=False), row=1, col=2)
            fig.add_trace(go.Bar(x=[vb], y=[(bins[i] + bins[i + 1]) / 2], orientation='h',
                                  marker_color='rgba(38,166,154,0.4)', showlegend=False), row=1, col=2)

    # Osilatör paneli
    if show_stochrsi:
        fig.add_trace(go.Scatter(x=df.index, y=df['Mom'],
                                  line=dict(color='#00c853', width=1.5), name='Momentum', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Mom_Signal'],
                                  line=dict(color='#ff1744', width=1.5), name='Sinyal', visible='legendonly'), row=2, col=1)
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

    # ============================================================
    # Layout
    # ============================================================
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
    
    # Çoklu dönüş: figürü, anlık fiyatı ve hesaplanan önceki kapanışı döndürür
    return fig, anlik_gercek_fiyat, prev_close


# ============================================================
# KONTROL PANELİ
# ============================================================
st.sidebar.header("🛠️ Analiz Ayarları")

Hisse          = st.sidebar.text_input("Varlık Sembolü", value="GC=F")
col1, col2     = st.sidebar.columns(2)
Baslangic      = col1.date_input("Başlangıç", value=datetime.now() - timedelta(days=120))
Bitis          = col2.date_input("Bitiş",     value=datetime.now())
Secilen_Periyot = st.sidebar.selectbox("Periyot", ["15m", "30m", "1h", "2h", "4h", "8h", "1d", "1wk"], index=8)

st.sidebar.markdown("---")

# EKLENEN OTOMATİK YENİLEME AYARI
oto_yenile = st.sidebar.checkbox("Otomatik Yenile (1 Dk)", value=False)
if oto_yenile:
    if st_autorefresh is not None:
        st_autorefresh(interval=60000, key="data_refresh")
    else:
        st.sidebar.warning("Otomatik yenileme için terminalden `pip install streamlit-autorefresh` komutunu çalıştırın.")

st.sidebar.markdown("---")

GRAFIK_TIPI = st.sidebar.radio("Grafik Görünümü", ["Çizgi (Line)", "Mum (Candlestick)"], horizontal=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Göstergeler")

show_kama       = st.sidebar.checkbox("KAMA",                     value=True)
show_supertrend = st.sidebar.checkbox("SuperTrend (AL/SAT)",      value=True)
show_stochrsi   = st.sidebar.checkbox("Divergence Osilatörü",     value=True)
show_fib        = st.sidebar.checkbox("Fibonacci Seviyeleri",     value=False)
show_vrvp       = st.sidebar.checkbox("VRVP (Hacim Profili)",     value=True)
show_sma        = st.sidebar.checkbox("SMA",                      value=True)
show_ema        = st.sidebar.checkbox("EMA",                      value=False)
show_bb         = st.sidebar.checkbox("Bollinger Bands",          value=True)
show_ichimoku   = st.sidebar.checkbox("Ichimoku Cloud",           value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Hassasiyet Ayarları")

KAMA_HIZI    = st.sidebar.slider("KAMA Hızı",               5,  50,  10)          if show_kama        else 10
TREND_CARPAN = st.sidebar.slider("Trend Çarpanı",           1.0, 5.0, 2.0, 0.5) if show_supertrend else 2.0
OSILATOR_PER = st.sidebar.slider("Divergence RSI Periyodu", 7,  30,  14)          if show_stochrsi   else 14
DIV_LOOKBACK = st.sidebar.slider("Divergence Lookback",     2,  20,  5)           if show_stochrsi   else 5
HACIM_DETAY  = st.sidebar.slider("Hacim Detayı",            20, 100, 40)          if show_vrvp       else 40
FIB_BAKIS    = st.sidebar.number_input("Fib Geriye Bakış (Mum)", value=100)       if show_fib        else 100

SMA_1_LEN = st.sidebar.slider("SMA 1 Periyodu", 5, 200, 50)  if show_sma else 50
SMA_2_LEN = st.sidebar.slider("SMA 2 Periyodu", 5, 400, 200) if show_sma else 200

EMA_1_LEN = st.sidebar.slider("EMA 1 Periyodu", 5, 200, 20)  if show_ema else 20
EMA_2_LEN = st.sidebar.slider("EMA 2 Periyodu", 5, 400, 50)  if show_ema else 50

BB_LEN = st.sidebar.slider("BB Periyodu",        5,  50, 20)        if show_bb else 20
BB_STD = st.sidebar.slider("BB Standart Sapma", 1.0, 4.0, 2.0, 0.5) if show_bb else 2.0

# ============================================================
# ANALİZİ BAŞLAT
# ============================================================
if st.sidebar.button("Analizi Başlat") or oto_yenile:
    with st.spinner('Veriler hesaplanıyor...'):
        fig, anlik_fiyat, onceki_fiyat = create_complete_trading_chart(
            Hisse, Baslangic, Bitis, Secilen_Periyot,
            KAMA_HIZI, TREND_CARPAN, OSILATOR_PER, HACIM_DETAY, FIB_BAKIS,
            show_kama, show_supertrend, show_stochrsi, DIV_LOOKBACK, show_fib, show_vrvp,
            show_sma, SMA_1_LEN, SMA_2_LEN, show_ema, EMA_1_LEN, EMA_2_LEN, show_bb, BB_LEN, BB_STD,
            show_ichimoku, GRAFIK_TIPI
        )
        if fig:
            # --- EKLENEN CANLI METRİK KARTLARI ---
            if anlik_fiyat is not None and onceki_fiyat is not None:
                m1, m2, m3 = st.columns(3)
                fiyat_farki = anlik_fiyat - onceki_fiyat
                yuzde_fark = (fiyat_farki / onceki_fiyat) * 100
                m1.metric(f"Anlık Fiyat ({Hisse})", f"{anlik_fiyat:.2f}", f"{yuzde_fark:.2f}%")
                m2.metric("Seçilen Periyot", Secilen_Periyot)
                tr_saati = datetime.now(timezone(timedelta(hours=3)))
                m3.metric("Son Güncelleme Zamanı", tr_saati.strftime("%H:%M:%S"))
                st.markdown("---")
            # ------------------------------------

            config = {'scrollZoom': True, 'displayModeBar': True}
            st.plotly_chart(fig, use_container_width=True, config=config)
else:
    st.info("Analiz yapmak için sol paneldeki 'Analizi Başlat' butonuna tıklayın. Varlık sembolünü bilmiyorsanız Gemini'ye 'yfinance ...... tickerı nedir' yazın. Uygulama yatırım tavsiyesi içermez. Ücretsizdir.")

    st.markdown("""
    ---
    ### 📖 Teknik Analiz Klavuzu

    #### 🚀 Sinyaller ve Oklar
    * **Büyük Üçgenler (AL/SAT):** SuperTrend indikatörünün ana trend onay sinyalleridir.
    * **🟢 Yeşil ▲D (Yükseliş Uyumsuzluğu):** Fiyat daha düşük dip yaparken osilatör daha yüksek dip yapar. Bu, satış baskısının zayıfladığını ve potansiyel bir yukarı dönüşü işaret eder.
    * **🔴 Kırmızı ▼D (Düşüş Uyumsuzluğu):** Fiyat daha yüksek tepe yaparken osilatör daha düşük tepe yapar. Bu, alım gücünün tükendiğini ve potansiyel bir aşağı dönüşü işaret eder.

    #### 🎯 İndikatör Mantığı (Emniyet Kemeriniz)
    * **KAMA Hızı:** Değerini düşürseniz fiyatı daha yakından izler, artırırsanız ana trendi gösterir.
    * **Trend Çarpanı:** Değerini 1.5 gibi seviyelere düşürürseniz 'AL/SAT' sinyalleri çok daha erken gelir.
    * **Osilatör Periyodu:** Divergence hesaplamasının RSI periyodunu belirler. Düşük değer daha hassas, yüksek değer daha az gürültülüdür.
    * **SMA:** Basit hareketli ortalama. İki farklı SMA seçerek kısa (örn: 20) ve uzun (örn: 50) dönem trendlerini karşılaştırabilirsiniz. Kısa SMA, uzun SMA'yı yukarı kestiğinde alım gücü artıyor demektir.
    * **Bollinger Bands:** Fiyatın volatilite bandını gösterir. Bantlar daralırsa büyük hareket beklenir. 
       BB Periyodu hareketli ortalamanın kaç periyot üzerinden hesaplanacağını belirler. Artarsa uzun vadeli trend görülür ama erken sinyal kaçabilir. Azalırsa yanlış sinyal artar.
       BB Standart Sapma bantların ortalamanın ne kadar uzağına çizileceğini belirler. Artarsa Sinyaller azalır ama gelen sinyaller daha güçlü olur. Azalırsa yanlış sinyal artar.
    * **Ichimoku Cloud:** Bulut (Kumo) destek/direnç, Tenkan/Kijun kesişmeleri sinyal üretir.

    #### 📈 Divergence Osilatörü ve Hacim Okuma
    * **Yeşil Çizgi (Momentum):** RSI'ın sıfır merkezli hali. Sıfırın üstü yükseliş bölgesi, altı düşüş bölgesidir.
    * **Kırmızı Çizgi (Sinyal):** Momentumun 9 periyotluk ortalaması. Yeşil çizgi kırmızıyı yukarı keserse alım, aşağı keserse satım sinyalidir.
    * **Yeşil/Kırmızı Kesişme:** Yeşil çizgi kırmızının üzerindeyken histogram pozitiftir (alıcılar güçlü). Yeşil kırmızının altına düştüğünde histogram negatife döner (satıcılar güçlü).
    * **Kırmızı Yatay Çizgiler (+20/+30 — Aşırı Alım):** Momentum bu bölgeye çıktığında fiyat aşırı alım bölgesindedir. Yeşil çizgi bu bölgede kırmızıyı aşağı keserse güçlü satım sinyalidir.
    * **Yeşil Yatay Çizgiler (-20/-30 — Aşırı Satım):** Momentum bu bölgeye düştüğünde fiyat aşırı satım bölgesindedir. Yeşil çizgi bu bölgede kırmızıyı yukarı keserse güçlü alım sinyalidir.
    * **Uyumsuzluk (Divergence) — Yalan Dedektörü:** Bu gösterge fiyatın söylediği ile gerçekte olan arasındaki çelişkiyi yakalar. Fiyat yeni tepe yapıyorsa ama momentum yapmıyorsa (▼D), "Bu yükseliş yalan, güç tükeniyor" der. Fiyat yeni dip yapıyorsa ama momentum yapmıyorsa (▲D), "Bu düşüş yalan, satıcılar zayıflıyor" der. Aşırı bölgelerde (+30 üstü veya -30 altı) oluşan uyumsuzluklar en güvenilir yalan tespitleridir.
    * **VRVP (Hacim Profili):** Sağdaki barlar paranın en çok hangi fiyat seviyesinde maliyetlendiğini gösterir.
    * **Fibonacci Seviyeleri:** Fiyatın matematiksel olarak destek/direnç bulabileceği önemli oranları (%23.6, %38.2, %50, %61.8, %78.6) gösterir.

    #### 📐 SMA (Basit Hareketli Ortalama)
    * **Trend Yönü:** Fiyat SMA'nın üzerindeyse yükseliş trendi, altındaysa düşüş trendi hakimdir.
    * **Kesişme Sinyalleri:** Fiyat SMA'yı alttan yukarı keserse alım, üstten aşağı keserse satım sinyalidir.
    * **Golden Cross / Death Cross:** Kısa SMA (örn: 50), uzun SMA'yı (örn: 200) yukarı keserse "Golden Cross" (güçlü alım), aşağı keserse "Death Cross" (güçlü satım) oluşur.
    * **Periyot Seçimi:** Kısa periyot (10-20) kısa vadeli dalgalanmaları, uzun periyot (50-200) ana trendi gösterir.

    #### 🎸 Bollinger Bands (Volatilite Bantları)
    * **Aşırı Alım/Satım:** Fiyat üst banda yaklaşırsa aşırı alım bölgesi, alt banda yaklaşırsa aşırı satım bölgesidir.
    * **Squeeze (Daralma):** Bantlar birbirine yaklaşırsa büyük bir kırılım hareketi yakındır. Kırılımın yönü trendi belirler.
    * **Bant Dışı Dönüş:** Fiyat alt bandın dışına çıkıp tekrar içeri girerse potansiyel yukarı dönüş, üst bant için tersi geçerlidir.
    * **Orta Bant (SMA):** Orta çizgi destek/direnç görevi görür. Fiyat orta bandın üzerinde kalıyorsa trend güçlüdür.

    #### ☁️ Ichimoku Cloud (Bulut Sistemi)
    * **Bulut (Kumo) Yorumu:** Fiyat bulutun üstündeyse yükseliş, altındaysa düşüş trendi hakimdir. Bulutun içindeyse kararsız bölgedir.
    * **Tenkan/Kijun Kesişmesi:** Tenkan (mavi) Kijun'u (kırmızı) yukarı keserse alım sinyali, aşağı keserse satım sinyalidir.
    * **Bulut Kalınlığı:** Kalın bulut güçlü destek/direnç anlamına gelir, ince bulut ise zayıf bariyerdir.
    * **Chikou (Gecikmeli Çizgi):** Mor noktalı çizgi fiyatın 26 periyot gerisini gösterir. Fiyatın üzerindeyse trend güçlü, altındaysa trend zayıflıyor demektir.

    #### ⏱ Zaman Dilimi ve Geçmiş Veri Limitleri
    * **Seçenekler:** `15m`, `30m`, `1h`, `2h`, `4h`, `8h`, `1d`
    * **Maksimum Geriye Dönük Süreler:**
    * · **1m:** 7 gün
    * · **2m / 5m / 15m / 30m / 90m:** 60 gün
    * · **1h:** 730 gün
    * · **2h / 4h / 8h:** 60 gün (1h veriden türetilir)
    * · **1d:** Sınırsız
    * · **1w:** Sınırsız

    #### 🔍 Ticker (Sembol) Seçimi
    * **Borsa İstanbul:** Sembolün sonuna `.IS` ekleyin. Örn: `THYAO.IS`, `ASELS.IS`, `TUPRS.IS`
    * **Kripto:** Parite formatında yazın. Örn: `BTC-USD`, `ETH-USD`, `AVAX-USD`
    * **ABD Hisseleri:** Direkt sembol yeterli. Örn: `AAPL`, `TSLA`, `MSFT`
    * **Döviz / Emtia:** `EURUSD=X`, `GC=F` (Altın), `CL=F` (Petrol)

    #### 📊 Temel Sinyaller (Legend Alt Bölümü)
    * Grafik oluşturulduktan sonra sağdaki legend'ın altında her skor grubu için **puan/max** ve **%yüzde** gösterilir.
    * **✔** koşul sağlandı, **✘** koşul sağlanmadı anlamına gelir.
    * Skorlar yatırım tavsiyesi değildir; sadece teknik koşulların özetini sunar.
    ---
    *(Salih Rıdvan Yılmaz - sry@tahmin.ai)*
    """)




