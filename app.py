import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ============================================================
# ⚙️ AYARLAR VE SAYFA YAPISI
# ============================================================
st.set_page_config(layout="wide", page_title="AI Teknik Analiz Terminali")

@st.cache_data(ttl=600)
def get_clean_data(ticker, start, end, interval):
    """Veri çekme ve resampling (Eleştirideki hızlandırma uygulandı)"""
    resample_map = {"2h": "1h", "4h": "1h", "8h": "1h"}
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
# 🧠 HESAPLAMA MOTORU (ELEŞTİRİLERE GÖRE İYİLEŞTİRİLDİ)
# ============================================================
def calculate_all_indicators(df, p):
    # KAMA
    if p['show_kama']:
        df['KAMA'] = ta.kama(df['Close'], length=p['k_len'])

    # SuperTrend
    if p['show_st']:
        st_df = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=p['s_mult'])
        if st_df is not None:
            df['ST_Line'] = st_df.iloc[:, 0]
            df['ST_Dir'] = st_df.iloc[:, 1]
            df['Buy'] = (df['ST_Dir'] == 1) & (df['ST_Dir'].shift(1) == -1)
            df['Sell'] = (df['ST_Dir'] == -1) & (df['ST_Dir'].shift(1) == 1)

    # Divergence (Vektörel Hesaplama - Hızlı ve Hatasız)
    if p['show_osc']:
        df['RSI'] = ta.rsi(df['Close'], length=p['osc_len'])
        df['Mom'] = df['RSI'] - 50
        df['Mom_Signal'] = ta.ema(df['Mom'], length=9)
        # Swing tespiti (Eleştirideki for döngüsü yerine vektörel çözüm)
        lb = 5
        df['Low_S'] = (df['Close'] == df['Close'].rolling(2*lb+1, center=True).min())
        df['High_S'] = (df['Close'] == df['Close'].rolling(2*lb+1, center=True).max())
        df['Bull_Div'] = (df['Low_S']) & (df['Close'] < df['Close'].shift(lb)) & (df['Mom'] > df['Mom'].shift(lb))
        df['Bear_Div'] = (df['High_S']) & (df['Close'] > df['Close'].shift(lb)) & (df['Mom'] < df['Mom'].shift(lb))

    # SMA & Bollinger
    if p['show_sma']:
        df['SMA'] = ta.sma(df['Close'], length=p['sma_len'])
    if p['show_bb']:
        bbands = ta.bbands(df['Close'], length=p['bb_len'], std=p['bb_std'])
        if bbands is not None:
            df['BB_Upper'] = bbands.iloc[:, 2]
            df['BB_Mid'] = bbands.iloc[:, 1]
            df['BB_Lower'] = bbands.iloc[:, 0]

    # Ichimoku (Sütun Güvenliği Sağlandı)
    if p['show_ichimoku']:
        ichi, _ = ta.ichimoku(df['High'], df['Low'], df['Close'])
        if ichi is not None:
            df['Tenkan'] = ichi.iloc[:, 0]
            df['Kijun'] = ichi.iloc[:, 1]
            df['Senkou_A'] = ichi.iloc[:, 2]
            df['Senkou_B'] = ichi.iloc[:, 3]
            df['Chikou'] = ichi.iloc[:, 4]

    # Fibonacci
    if p['show_fib']:
        f_df = df.tail(p['f_look'])
        hi, lo = f_df['High'].max(), f_df['Low'].min()
        p['fib_levels'] = {
            '23.6%': hi - 0.236 * (hi - lo),
            '38.2%': hi - 0.382 * (hi - lo),
            '50%': hi - 0.5 * (hi - lo)
        }
    
    return df

# ============================================================
# 🎨 GÖRSELLEŞTİRME (ORİJİNAL TASARIM VE RENKLER)
# ============================================================
def create_trading_plot(df, ticker, p):
    has_osc = p['show_osc']
    row_h = [0.65, 0.35] if has_osc else [1.0, 0.01]
    
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, 
                        column_widths=[0.85, 0.15], row_heights=row_h,
                        vertical_spacing=0.05, horizontal_spacing=0.01)

    # 1. Mum Grafiği
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                  low=df['Low'], close=df['Close'], name="Fiyat"), row=1, col=1)

    # 2. İndikatör Çizimleri
    if 'KAMA' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['KAMA'], line=dict(color='#2962ff', width=2), name='KAMA'), row=1, col=1)
    
    if 'SMA' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA'], line=dict(color='#ff9800', width=2), name='SMA'), row=1, col=1)

    if 'ST_Line' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['ST_Line'], line=dict(color='rgba(80,80,80,0.7)', width=2, line_shape='hv'), name='Trend'), row=1, col=1)
        buys, sells = df[df['Buy']], df[df['Sell']]
        fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.98, mode='markers+text', text="AL", marker=dict(symbol='triangle-up', size=15, color='green'), name='AL'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.02, mode='markers+text', text="SAT", marker=dict(symbol='triangle-down', size=15, color='red'), name='SAT'), row=1, col=1)

    if 'BB_Upper' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(174,134,255,0.6)', width=1), name='BB Üst'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(174,134,255,0.6)', width=1), fill='tonexty', fillcolor='rgba(174,134,255,0.07)', name='BB Alt'), row=1, col=1)

    # 3. Uyumsuzluk İşaretleri (▲D / ▼D)
    if has_osc:
        bull = df[df['Bull_Div']]
        bear = df[df['Bear_Div']]
        fig.add_trace(go.Scatter(x=bull.index, y=bull['Low']*0.99, mode='markers+text', text="▲D", textposition="bottom center", marker=dict(symbol='triangle-up', color='#00e676', size=12), name='Yük. Uyumsuzluk'), row=1, col=1)
        fig.add_trace(go.Scatter(x=bear.index, y=bear['High']*1.01, mode='markers+text', text="▼D", textposition="top center", marker=dict(symbol='triangle-down', color='#ff1744', size=12), name='Düş. Uyumsuzluk'), row=1, col=1)

    # 4. VRVP (Hacim Profili - Vektörel Hızlandırma)
    if p['show_vrvp']:
        bins = pd.cut(df['Close'], bins=p['v_bins'])
        v_profile = df.groupby(bins, observed=False)['Volume'].sum()
        fig.add_trace(go.Bar(x=v_profile.values, y=[b.mid for b in v_profile.index], orientation='h', marker_color='rgba(128,128,128,0.4)', showlegend=False), row=1, col=2)

    # 5. Osilatör Paneli
    if has_osc:
        fig.add_trace(go.Scatter(x=df.index, y=df['Mom'], line=dict(color='#00c853', width=1.5), name='Momentum'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Mom_Signal'], line=dict(color='#ff1744', width=1.5), name='Sinyal'), row=2, col=1)
        for val in [30, 20, 0, -20, -30]:
            fig.add_hline(y=val, line_dash="dash", line_color="rgba(128,128,128,0.3)", row=2, col=1)

    # Fibonacci Çizgileri
    if p['show_fib'] and 'fib_levels' in p:
        for label, val in p['fib_levels'].items():
            fig.add_hline(y=val, line_dash="dot", line_color="rgba(128,128,128,0.5)", annotation_text=label, row=1, col=1)

    fig.update_layout(height=1200, template='plotly_white', xaxis_rangeslider_visible=False, title=f"<b>{ticker}</b> Teknik Analizi")
    return fig

# ============================================================
# 🕹️ KONTROL PANELİ VE REHBER (TAM METİN)
# ============================================================
st.sidebar.header("🛠️ Analiz Ayarları")
Hisse = st.sidebar.text_input("Varlık Sembolü", value="GC=F")
col1, col2 = st.sidebar.columns(2)
Baslangic = col1.date_input("Başlangıç", value=datetime.now() - timedelta(days=60))
Bitis = col2.date_input("Bitiş", value=datetime.now())
Secilen_Periyot = st.sidebar.selectbox("Periyot", ["15m", "30m", "1h", "2h", "4h", "8h", "1d", "1wk"], index=4)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Göstergeler")

params = {
    'show_kama': st.sidebar.checkbox("KAMA", value=True),
    'k_len': st.sidebar.slider("KAMA Hızı", 5, 50, 10),
    'show_st': st.sidebar.checkbox("SuperTrend (AL/SAT)", value=True),
    's_mult': st.sidebar.slider("Trend Çarpanı", 1.0, 5.0, 2.0, 0.5),
    'show_osc': st.sidebar.checkbox("Divergence Osilatörü", value=True),
    'osc_len': st.sidebar.slider("Divergence RSI Periyodu", 7, 30, 14),
    'show_vrvp': st.sidebar.checkbox("VRVP (Hacim Profili)", value=True),
    'v_bins': st.sidebar.slider("Hacim Detayı", 20, 100, 40),
    'show_fib': st.sidebar.checkbox("Fibonacci Seviyeleri", value=True),
    'f_look': st.sidebar.number_input("Fib Geriye Bakış (Mum)", value=100),
    'show_sma': st.sidebar.checkbox("SMA", value=False), 'sma_len': 20,
    'show_bb': st.sidebar.checkbox("Bollinger Bands", value=False), 'bb_len': 20, 'bb_std': 2.0,
    'show_ichimoku': st.sidebar.checkbox("Ichimoku Cloud", value=False)
}

if st.sidebar.button("Analizi Başlat"):
    with st.spinner('Veriler hesaplanıyor...'):
        df = get_clean_data(Hisse, Baslangic, Bitis, Secilen_Periyot)
        if df is not None:
            df = calculate_all_indicators(df, params)
            fig = create_trading_plot(df, Hisse, params)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Veri bulunamadı!")
else:
    st.info("Analiz yapmak için sol paneldeki 'Analizi Başlat' butonuna tıklayın. Varlık sembolünü bilmiyorsanız gemini'ye yfinance tickerı nedir yazın. Uygulama yatırım tavsiyesi içermez. Ücretsizdir.")

    # REHBER KISMI - TAM VE EKSİKSİZ
    st.markdown("""
    ---
    ### 📖 Teknik Analiz Klavuzu

    #### 🚀 Sinyaller ve Oklar
    * **Büyük Üçgenler (AL/SAT):** SuperTrend indikatörünün ana trend onay sinyalleridir.
    * **🟢 Yeşil ▲D (Yükseliş Uyumsuzluğu):** Fiyat daha düşük dip yaparken osilatör daha yüksek dip yapar. Bu, satış baskısının zayıfladığını ve potansiyel bir yukarı dönüşü işaret eder.
    * **🔴 Kırmızı ▼D (Düşüş Uyumsuzluğu):** Fiyat daha yüksek tepe yaparken osilatör daha düşük tepe yapar. Bu, alım gücünün tükendiğini ve potansiyel bir aşağı dönüşü işaret eder.

    #### 🎯 İndikatör Mantığı (Emniyet Kemeriniz)
    * **KAMA Hızı:** Değerini düşürürseniz fiyatı daha yakından izler, artırırsanız ana trendi gösterir.
    * **Trend Çarpanı:** Değerini 1.5 gibi seviyelere düşürürseniz 'AL/SAT' sinyalleri çok daha erken gelir.
    * **Osilatör Periyodu:** Divergence hesaplamasının RSI periyodunu belirler. Düşük değer daha hassas, yüksek değer daha az gürültülüdür.
    * **SMA:** Basit hareketli ortalama. Kısa periyot (10-20) hızlı sinyal, uzun periyot (50-200) ana trend.
    * **Bollinger Bands:** Fiyatın volatilite bandını gösterir. Bantlar daralırsa büyük hareket beklenir.
    * **Ichimoku Cloud:** Bulut (Kumo) destek/direnç, Tenkan/Kijun kesişmeleri sinyal üretir.

    #### 📈 Divergence Osilatörü ve Hacim Okuma
    * **Yeşil Çizgi (Momentum):** RSI'ın sıfır merkezli hali. Sıfırın üstü yükseliş bölgesi, altı düşüş bölgesidir.
    * **Kırmızı Çizgi (Sinyal):** Momentumun 9 periyotluk ortalaması. Yeşil çizgi kırmızıyı yukarı keserse alım, aşağı keserse satım sinyalidir.
    * **Uyumsuzluk (Divergence) — Yalan Dedektörü:** Bu gösterge fiyatın söylediği ile gerçekte olan arasındaki çelişkiyi yakalar. Fiyat yeni tepe yapıyorsa ama momentum yapmıyorsa (▼D), "Bu yükseliş yalan" der. Fiyat yeni dip yapıyorsa ama momentum yapmıyorsa (▲D), "Bu düşüş yalan" der.
    * **VRVP (Hacim Profili):** Sağdaki barlar paranın en çok hangi fiyat seviyesinde maliyetlendiğini gösterir.
    * **Fibonacci Seviyeleri:** Fiyatın matematiksel olarak destek bulabileceği %23.6, %38.2 ve %50 bölgelerini gösterir.

    #### 🔍 Ticker (Sembol) Seçimi
    * **Borsa İstanbul:** Sembolün sonuna `.IS` ekleyin (Örn: `THYAO.IS`).
    * **Kripto:** Parite formatında yazın (Örn: `BTC-USD`).
    * **ABD Hisseleri:** Direkt sembol yeterli (Örn: `AAPL`).
    * **Döviz / Emtia:** `EURUSD=X`, `GC=F` (Altın).

    ---
    *(Salih Rıdvan Yılmaz - sry@tahmin.ai)*
    """)
