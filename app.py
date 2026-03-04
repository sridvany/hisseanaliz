import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- 1. SAYFA AYARLARI ---
st.set_page_config(layout="wide", page_title="AI Teknik Analiz Terminali")

def create_complete_trading_chart(ticker, start, end, per, k_len, s_mult, srsi_len, v_bins, f_look):
    # Veri Çekme
    raw_p = "1h" if per == "8h" else per
    df = yf.download(ticker, start=start, end=end, interval=raw_p, auto_adjust=True)

    if df.empty:
        st.error("Veri bulunamadı. Lütfen sembolü veya tarihleri kontrol edin.")
        return None

    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.strip().title() for c in df.columns]

    # 8 Saatlik Resampling
    if per == "8h":
        df = df.resample('8h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

    # İndikatör Hesaplamaları
    df['KAMA'] = ta.kama(df['Close'], length=k_len)
    sti = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=s_mult)
    df['ST_Line'], df['ST_Dir'] = sti.iloc[:, 0], sti.iloc[:, 1]
    df['Buy'] = (df['ST_Dir'] == 1) & (df['ST_Dir'].shift(1) == -1)
    df['Sell'] = (df['ST_Dir'] == -1) & (df['ST_Dir'].shift(1) == 1)
    
    srsi = ta.stochrsi(df['Close'], length=srsi_len, rsi_length=srsi_len, k=3, d=3)
    df['stoch_k'], df['stoch_d'] = srsi.iloc[:, 0], srsi.iloc[:, 1]

    # K ve D Kesişme Okları
    df['Cross_Up'] = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
    df['Cross_Down'] = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))

    # Fibonacci Seviyeleri
    f_df = df if f_look is None else df.tail(f_look)
    hi, lo = f_df['High'].max(), f_df['Low'].min()
    fib = {'23.6%': hi-0.236*(hi-lo), '38.2%': hi-0.382*(hi-lo), '50%': hi-0.5*(hi-lo)}

    # --- GÖRSELLEŞTİRME ---
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=False,
                        column_widths=[0.85, 0.15], row_heights=[0.85, 0.15], 
                        vertical_spacing=0.03, horizontal_spacing=0.01)

    # ÜST PANEL: Mumlar, KAMA ve Trend
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['KAMA'], line=dict(color='#2962ff', width=2), name='KAMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ST_Line'], line=dict(color='rgba(80, 80, 80, 0.8)', width=2.5), line_shape='hv', name='Trend Sınırı'), row=1, col=1)

    # Fibonacci Çizgileri
    for l, p in fib.items():
        fig.add_hline(y=p, line_dash="dash", line_color="rgba(128,128,128,0.5)", annotation_text=l, row=1, col=1)

    # AL/SAT Sinyalleri ve Oklar
    fig.add_trace(go.Scatter(x=df[df['Buy']].index, y=df[df['Buy']]['Low']*0.98, mode='markers+text', text="AL", marker=dict(symbol='triangle-up', size=15, color='green'), name='AL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[df['Sell']].index, y=df[df['Sell']]['High']*1.02, mode='markers+text', text="SAT", marker=dict(symbol='triangle-down', size=15, color='red'), name='SAT'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[df['Cross_Up']].index, y=df[df['Cross_Up']]['Low']*0.99, mode='markers', marker=dict(symbol='arrow-up', size=8, color='#00ff00'), name='K/D Kes-Yukarı'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[df['Cross_Down']].index, y=df[df['Cross_Down']]['High']*1.01, mode='markers', marker=dict(symbol='arrow-down', size=8, color='#ff00ff'), name='K/D Kes-Aşağı'), row=1, col=1)

    # SAĞ PANEL: Hacim Profili (VRVP)
    bins = pd.cut(df['Close'], bins=v_bins, retbins=True)[1]
    df['V_T'] = df.apply(lambda r: 'B' if r['Close'] >= r['Open'] else 'S', axis=1)
    for i in range(v_bins):
        m = (df['Close'] >= bins[i]) & (df['Close'] < bins[i+1])
        vb, vs = df[m & (df['V_T']=='B')]['Volume'].sum(), df[m & (df['V_T']=='S')]['Volume'].sum()
        fig.add_trace(go.Bar(x=[vs], y=[(bins[i]+bins[i+1])/2], orientation='h', marker_color='rgba(239,83,80,0.4)', showlegend=False, yaxis='y2'), row=1, col=2)
        fig.add_trace(go.Bar(x=[vb], y=[(bins[i]+bins[i+1])/2], orientation='h', marker_color='rgba(38,166,154,0.4)', showlegend=False, yaxis='y2'), row=1, col=2)

    # ALT PANEL: Stoch RSI (yaxis3 kullanılır)
    fig.add_trace(go.Scatter(x=df.index, y=df['stoch_k'], line=dict(color='green', width=1.5), name='K'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['stoch_d'], line=dict(color='red', width=1.5), name='D'), row=2, col=1)

    # --- KRİTİK DÜZENLEME: SADECE ALT PANEL BEYAZ VE NET ---
    fig.add_shape(
        type='rect', xref='paper', yref='y3 domain',
        x0=0, y0=0, x1=1, y1=1,
        fillcolor='white', layer='below', line_width=0
    )

    fig.update_layout(
        template='plotly_dark', # Üst taraf koyu kalsın
        height=1000, 
        xaxis_rangeslider_visible=False, 
        barmode='stack', 
        title=f"<b>{ticker}</b> Analiz Terminali",
        
        # Alt Panelin (yaxis3) Görünürlük Ayarları
        yaxis3=dict(
            tickfont=dict(color='black', size=11), # Rakamlar siyah
            gridcolor='rgba(0, 0, 0, 0.2)',        # Izgaralar görünür gri
            zerolinecolor='black',                # Sıfır çizgisi net siyah
            showgrid=True
        ),
        xaxis3=dict(
            gridcolor='rgba(0, 0, 0, 0.2)', # Dikey ızgaralar siyah üzerinde görünür
            showgrid=True
        )
    )
    return fig

# --- STREAMLIT ARAYÜZÜ ---
st.sidebar.header("🛠️ Analiz Ayarları")
Hisse = st.sidebar.text_input("Hisse Sembolü", value="THYAO.IS")
col1, col2 = st.sidebar.columns(2)
Baslangic = col1.date_input("Başlangıç", value=datetime.now() - timedelta(days=60))
Bitis = col2.date_input("Bitiş", value=datetime.now())
Secilen_Periyot = st.sidebar.selectbox("Periyot", ["15m", "1h", "8h", "1d"], index=2)

st.sidebar.markdown("---")
KAMA_HIZI = st.sidebar.slider("KAMA Hızı", 5, 50, 10)
TREND_CARPAN = st.sidebar.slider("Trend Çarpanı", 1.0, 5.0, 2.0, 0.5)
OSILATOR_PER = st.sidebar.slider("Osilatör Periyodu", 7, 30, 14)
HACIM_DETAY = st.sidebar.slider("Hacim Detayı", 20, 100, 40)
FIB_BAKIS = st.sidebar.number_input("Fib Geriye Bakış (Mum)", value=100)

if st.sidebar.button("Analizi Başlat"):
    with st.spinner('Veriler hesaplanıyor...'):
        fig = create_complete_trading_chart(Hisse, Baslangic, Bitis, Secilen_Periyot, KAMA_HIZI, TREND_CARPAN, OSILATOR_PER, HACIM_DETAY, FIB_BAKIS)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Analiz yapmak için sol paneldeki 'Analizi Başlat' butonuna tıklayın.")
    st.markdown("""
    ---
    ### 📖 Strateji ve Kullanım Rehberi
    * **🚀 Büyük Üçgenler (AL/SAT):** SuperTrend ana trend onay sinyalleridir.
    * **🟢 Küçük Yeşil Oklar:** Stoch RSI dipten dönüş uyarısıdır. 'Hazırlan' mesajı verir.
    * **🟣 Küçük Mor Oklar:** Tepedeki yorulma sinyalidir. Kar almayı hatırlatır.
    * **📊 Alt Panel (Beyaz Bölge):** Osilatörün her hareketi net görünmesi için özel tasarlanmıştır. 20 altı kesişim alım, 80 üstü satım odaklıdır.
    * **🎯 KAMA & Trend:** Fiyat gri basamakların ve mavi çizginin üstündeyse 'Boğa' piyasası hakimdir.
    """)
