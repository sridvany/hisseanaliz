import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Sayfa Genişliği Ayarı
st.set_page_config(layout="wide", page_title="AI Teknik Analiz Terminali")

def create_complete_trading_chart(ticker, start, end, per, k_len, s_mult, srsi_len, v_bins, f_look):
    # 1. Veri Çekme (1h çekip 8h'e tamamlama mantığı)
    raw_p = "1h" if per == "8h" else per
    df = yf.download(ticker, start=start, end=end, interval=raw_p, auto_adjust=True)

    if df.empty:
        st.error("Veri bulunamadı. Lütfen tarih sınırlarını veya sembolü kontrol edin.")
        return None

    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.strip().title() for c in df.columns]

    # 2. 8 Saatlik Resampling
    if per == "8h":
        df = df.resample('8h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

    # 3. Teknik Gösterge Hesaplamaları
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

    # Fib ve VRVP
    f_df = df if f_look is None else df.tail(f_look)
    hi, lo = f_df['High'].max(), f_df['Low'].min()
    fib = {'23.6%': hi-0.236*(hi-lo), '38.2%': hi-0.382*(hi-lo), '50%': hi-0.5*(hi-lo)}

    # 4. Görselleştirme (Streamlit Uyumlulaştırılmış)
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                        column_widths=[0.85, 0.15], row_heights=[0.8, 0.2], 
                        vertical_spacing=0.05, horizontal_spacing=0.01)

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['KAMA'], line=dict(color='#2962ff', width=2), name='KAMA'), row=1, col=1)
    
    # Trend Sınırı (Geliştirilmiş Görünüm)
    fig.add_trace(go.Scatter(x=df.index, y=df['ST_Line'], line=dict(color='rgba(80, 80, 80, 0.7)', width=2.5), line_shape='hv', name='Trend Sınırı'), row=1, col=1)

    for l, p in fib.items():
        fig.add_hline(y=p, line_dash="dash", line_color="rgba(128,128,128,0.5)", annotation_text=l, row=1, col=1)

    fig.add_trace(go.Scatter(x=df[df['Buy']].index, y=df[df['Buy']]['Low']*0.98, mode='markers+text', text="AL", marker=dict(symbol='triangle-up', size=15, color='green'), name='AL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[df['Sell']].index, y=df[df['Sell']]['High']*1.02, mode='markers+text', text="SAT", marker=dict(symbol='triangle-down', size=15, color='red'), name='SAT'), row=1, col=1)
    
    # Oklar
    fig.add_trace(go.Scatter(x=df[df['Cross_Up']].index, y=df[df['Cross_Up']]['Low']*0.99, mode='markers', marker=dict(symbol='arrow-up', size=8, color='#00ff00'), name='K/D Kes-Yukarı'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[df['Cross_Down']].index, y=df[df['Cross_Down']]['High']*1.01, mode='markers', marker=dict(symbol='arrow-down', size=8, color='#ff00ff'), name='K/D Kes-Aşağı'), row=1, col=1)

    # Hacim Barları
    bins = pd.cut(df['Close'], bins=v_bins, retbins=True)[1]
    df['V_T'] = df.apply(lambda r: 'B' if r['Close'] >= r['Open'] else 'S', axis=1)
    for i in range(v_bins):
        m = (df['Close'] >= bins[i]) & (df['Close'] < bins[i+1])
        vb, vs = df[m & (df['V_T']=='B')]['Volume'].sum(), df[m & (df['V_T']=='S')]['Volume'].sum()
        fig.add_trace(go.Bar(x=[vs], y=[(bins[i]+bins[i+1])/2], orientation='h', marker_color='rgba(239,83,80,0.4)', showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=[vb], y=[(bins[i]+bins[i+1])/2], orientation='h', marker_color='rgba(38,166,154,0.4)', showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter(x=df.index, y=df['stoch_k'], line=dict(color='green'), name='K'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['stoch_d'], line=dict(color='red'), name='D'), row=2, col=1)

    fig.update_layout(template='plotly_white', height=900, xaxis_rangeslider_visible=False, barmode='stack', title=f"<b>{ticker}</b> Teknik Analizi")
    return fig

# ============================================================
# 🕹️ KONTROL PANELİ (WEB ARAYÜZÜ)
# ============================================================
st.sidebar.header("🛠️ Analiz Ayarları")

Hisse = st.sidebar.text_input("Varlık Sembolü", value="GC=F")
col1, col2 = st.sidebar.columns(2)
Baslangic = col1.date_input("Başlangıç", value=datetime.now() - timedelta(days=60))
Bitis = col2.date_input("Bitiş", value=datetime.now())
Secilen_Periyot = st.sidebar.selectbox("Periyot", ["15m", "1h", "8h", "1d"], index=2)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Hassasiyet Ayarları")
KAMA_HIZI = st.sidebar.slider("KAMA Hızı", 5, 50, 10)
TREND_CARPAN = st.sidebar.slider("Trend Çarpanı", 1.0, 5.0, 2.0, 0.5)
OSILATOR_PER = st.sidebar.slider("Osilatör Periyodu", 7, 30, 14)
HACIM_DETAY = st.sidebar.slider("Hacim Detayı", 20, 100, 40)
FIB_BAKIS = st.sidebar.number_input("Fib Geriye Bakış (Mum)", value=100)

# Butona tıklandığında analizi yap
if st.sidebar.button("Analizi Başlat"):
    with st.spinner('Veriler hesaplanıyor...'):
        fig = create_complete_trading_chart(Hisse, Baslangic, Bitis, Secilen_Periyot, KAMA_HIZI, TREND_CARPAN, OSILATOR_PER, HACIM_DETAY, FIB_BAKIS)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
# Butona henüz basılmadıysa kılavuzu göster
else:
    st.info("Analiz yapmak için sol paneldeki 'Analizi Başlat' butonuna tıklayın. Varlık sembolünü bilmiyorsanız gemini'ye yfinance -varlık- tickerı yazın")
    
    st.markdown("""
    ---
    ### 📖 Analiz ve Strateji Kılavuzu
    
    #### 🚀 Sinyaller ve Oklar
    * **Büyük Üçgenler (AL/SAT):** SuperTrend indikatörünün ana trend onay sinyalleridir.
    * **🟢 Küçük Yeşil Oklar (K/D Kes-Yukarı):** Alt paneldeki yeşil çizgi (K), kırmızıyı (D) alttan yukarı kestiğinde belirir. Genellikle büyük 'AL' üçgeninden önce gelerek size **'Hazırlan, trend dönebilir'** mesajı verir.
    * **🟣 Küçük Mor Oklar (K/D Kes-Aşağı):** Yeşil çizginin kırmızıyı üstten aşağı kestiği anlardır. Büyük 'SAT' sinyalinden önce gelen bir **'Kar almayı düşün veya temkinli ol'** uyarısıdır.
    
    #### 🎯 İndikatör Mantığı (Emniyet Kemeriniz)
    * **KAMA Hızı:** Değerini düşürürseniz fiyatı daha yakından izler, artırırsanız ana trendi gösterir. (Hızlı piyasada indikatörleri sakinleştirmek, sizi duygusal hatalardan korur.)
    * **Trend Çarpanı:** Değerini 1.5 gibi seviyelere düşürürseniz 'AL/SAT' sinyalleri çok daha erken gelir.
    * **Stoch RSI Periyodu:** Hızlı piyasalarda gürültüleri filtrelemek için artırılmalıdır.
    
    #### 📈 Osilatör ve Hacim Okuma
    * **Yükseliş Sinyali:** Yeşil çizgi (K), kırmızıyı (D) alttan yukarı kesiyorsa ve **20 seviyesinin altındaysa** (aşırı satım), bu güçlü bir yükseliş sinyalidir.
    * **Düşüş Sinyali:** Yeşil çizgi (K), kırmızıyı (D) üstten aşağı kesiyorsa ve **80 seviyesinin üzerindeyse** (aşırı alım), bu bir düşüş uyarısıdır.
    * **VRVP (Hacim Profili):** Sağdaki barlar paranın en çok hangi fiyat seviyesinde maliyetlendiğini gösterir.Değerini 100 yaparsan sağdaki barlar çok daha detaylı hale gelir.
    * **Fibonacci Seviyeleri:** Fiyatın matematiksel olarak destek bulabileceği %23.6, %38.2 ve %50 bölgelerini gösterir.

    #### ⏱ Zaman Dilimi ve Geçmiş Veri Limitleri
    * **Seçenekler:** `15m (minute)`, `1h (hour)`, `8h`, `1d (day)` arasından birini seçin.
    * **Maksimum Geriye Dönük Süreler:**
    *   · **1m:** 7 gün
    *   · **2m / 5m / 15m / 30m / 90m:** 60 gün
    *   · **1h:** 730 gün
    *   · **8h:** 60 gün
    *   · **1d:** Sınırsız
    *   · **1w:** Sınırsız

    #### 🔍 Ticker (Sembol) Seçimi
    * **Borsa İstanbul:** Sembolün sonuna `.IS` ekleyin. Örn: `THYAO.IS`, `ASELS.IS`, `TUPRS.IS`
    * **Kripto:** Parite formatında yazın. Örn: `BTC-USD`, `ETH-USD`, `AVAX-USD`
    * **ABD Hisseleri:** Direkt sembol yeterli. Örn: `AAPL`, `TSLA`, `MSFT`
    * **Döviz / Emtia:** `EURUSD=X`, `GC=F` (Altın), `CL=F` (Petrol)

    ---
    *Uygulama yatırım tavsiyesi içermez. Ücretli online sitelerdeki analizleri ücretsiz olarak sunar.*
    *(Salih Rıdvan Yılmaz)*
    """)


















