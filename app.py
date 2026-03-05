import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ============================================================
# ⚙️ AYARLAR VE YARDIMCI FONKSİYONLAR
# ============================================================
st.set_page_config(layout="wide", page_title="Pro AI Trading Terminal")

@st.cache_data(ttl=600)
def fetch_data(ticker, start, end, interval):
    """Veri çekme ve temel temizlik işlemleri."""
    resample_map = {"2h": "1h", "4h": "1h", "8h": "1h"}
    target_p = interval
    download_p = resample_map.get(interval, interval)
    
    df = yf.download(ticker, start=start, end=end, interval=download_p, auto_adjust=True)
    if df.empty: return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.strip().title() for c in df.columns]

    if interval in resample_map:
        df = df.resample(interval).agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
    
    return df

# ============================================================
# 🧠 HESAPLAMA MOTORU (LOGIC)
# ============================================================
def get_technical_data(df, params):
    """Tüm teknik gösterge hesaplamalarını yapar."""
    # KAMA
    if params['show_kama']:
        df['KAMA'] = ta.kama(df['Close'], length=params['kama_len'])

    # SuperTrend
    if params['show_st_signals']:
        st_df = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=params['st_mult'])
        if st_df is not None:
            df['ST_Line'] = st_df.iloc[:, 0]
            df['ST_Dir'] = st_df.iloc[:, 1]
            df['Buy_Sig'] = (df['ST_Dir'] == 1) & (df['ST_Dir'].shift(1) == -1)
            df['Sell_Sig'] = (df['ST_Dir'] == -1) & (df['ST_Dir'].shift(1) == 1)

    # Momentum & Divergence (Vektörel)
    if params['show_osc']:
        df['RSI'] = ta.rsi(df['Close'], length=params['osc_len'])
        df['Mom'] = df['RSI'] - 50
        df['Mom_Sig'] = ta.ema(df['Mom'], length=9)
        
        # Swing tespiti (vektörel)
        lb = 5
        df['Is_Low'] = (df['Close'] == df['Close'].rolling(2*lb+1, center=True).min())
        df['Is_High'] = (df['Close'] == df['Close'].rolling(2*lb+1, center=True).max())
        
        # Basit uyumsuzluk mantığı
        df['Bull_Div'] = (df['Is_Low']) & (df['Close'] < df['Close'].shift(lb)) & (df['Mom'] > df['Mom'].shift(lb))
        df['Bear_Div'] = (df['Is_High']) & (df['Close'] > df['Close'].shift(lb)) & (df['Mom'] < df['Mom'].shift(lb))

    # Bollinger & SMA
    if params['show_bb']:
        bb = ta.bbands(df['Close'], length=params['bb_len'], std=params['bb_std'])
        df = pd.concat([df, bb], axis=1)
    if params['show_sma']:
        df['SMA'] = ta.sma(df['Close'], length=params['sma_len'])

    # Ichimoku (Sütun güvenliği sağlandı)
    if params['show_ichi']:
        ichi, _ = ta.ichimoku(df['High'], df['Low'], df['Close'])
        if ichi is not None:
            df['Tenkan'] = ichi.iloc[:, 0] # ITS_9
            df['Kijun'] = ichi.iloc[:, 1]  # IKS_26
            df['SpanA'] = ichi.iloc[:, 2]  # ISA_9
            df['SpanB'] = ichi.iloc[:, 3]  # ISB_26
            df['Chikou'] = ichi.iloc[:, 4] # ICS_26
            
    return df

# ============================================================
# 🎨 GÖRSELLEŞTİRME MOTORU (UI)
# ============================================================
def plot_technical_chart(df, ticker, params):
    """Plotly grafiğini oluşturur."""
    has_osc = params['show_osc']
    rows = 2 if has_osc else 1
    heights = [0.7, 0.3] if has_osc else [1.0]

    fig = make_subplots(rows=rows, cols=2, shared_xaxes=True, 
                        column_widths=[0.85, 0.15], row_heights=heights,
                        vertical_spacing=0.03, horizontal_spacing=0.01)

    # 1. Ana Fiyat Serisi
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)

    # 2. Gösterge Çizimleri
    if 'KAMA' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['KAMA'], line=dict(color='blue', width=1.5), name="KAMA"), row=1, col=1)
    
    if params['show_st_signals'] and 'ST_Line' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['ST_Line'], line=dict(color='gray', dash='dot'), name="ST Trend"), row=1, col=1)
        # Sinyaller
        buys = df[df['Buy_Sig']]
        sells = df[df['Sell_Sig']]
        fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name="Buy"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name="Sell"), row=1, col=1)

    # 3. VRVP (Hacim Profili) - Vektörel ve Performanslı
    if params['show_vrvp']:
        bins = pd.cut(df['Close'], bins=params['v_bins'])
        v_profile = df.groupby(bins)['Volume'].sum()
        y_centers = [b.mid for b in v_profile.index]
        fig.add_trace(go.Bar(x=v_profile.values, y=y_centers, orientation='h', marker_color='rgba(120,120,120,0.3)', showlegend=False), row=1, col=2)

    # 4. Alt Panel (Osilatör)
    if has_osc:
        fig.add_trace(go.Scatter(x=df.index, y=df['Mom'], line=dict(color='green'), name="Momentum"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Mom_Sig'], line=dict(color='red', width=1), name="Signal"), row=2, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-20, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(height=800, template="plotly_white", title=f"{ticker} Analysis", xaxis_rangeslider_visible=False)
    return fig

# ============================================================
# 🕹️ STREAMLIT ARAYÜZÜ
# ============================================================
st.sidebar.title("🔍 Terminal Settings")
symbol = st.sidebar.text_input("Ticker", "BTC-USD")
interval = st.sidebar.selectbox("Period", ["15m", "1h", "4h", "1d"], index=1)

# Parametre Sözlüğü (Tüm ayarları tek yerde toplar)
p = {
    'show_kama': st.sidebar.checkbox("KAMA", True),
    'kama_len': st.sidebar.slider("KAMA Length", 5, 50, 10),
    'show_st_signals': st.sidebar.checkbox("SuperTrend", True),
    'st_mult': st.sidebar.slider("ST Multiplier", 1.0, 5.0, 3.0),
    'show_osc': st.sidebar.checkbox("Divergence Osc", True),
    'osc_len': st.sidebar.slider("RSI Length", 7, 30, 14),
    'show_vrvp': st.sidebar.checkbox("VRVP", True),
    'v_bins': st.sidebar.slider("Volume Bins", 20, 100, 50),
    'show_bb': st.sidebar.checkbox("Bollinger", False),
    'bb_len': 20, 'bb_std': 2.0,
    'show_sma': st.sidebar.checkbox("SMA", False),
    'sma_len': 50,
    'show_ichi': st.sidebar.checkbox("Ichimoku", False)
}

if st.sidebar.button("Run Analysis"):
    with st.spinner("Processing..."):
        data = fetch_data(symbol, datetime.now()-timedelta(days=100), datetime.now(), interval)
        if data is not None:
            processed_df = get_technical_data(data, p)
            final_fig = plot_technical_chart(processed_df, symbol, p)
            st.plotly_chart(final_fig, use_container_width=True)
        else:
            st.error("Data not found!")
