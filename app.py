import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Sayfa Genişliği Ayarı
st.set_page_config(layout="wide", page_title="AI Teknik Analiz Terminali")

def create_complete_trading_chart(ticker, start, end, per, k_len, s_mult, srsi_len, v_bins, f_look,
                                   show_kama, show_supertrend, show_stochrsi, show_fib, show_vrvp,
                                   show_sma, sma_len, show_bb, bb_len, bb_std,
                                   show_ichimoku):
    # 1. Veri Çekme (resampling gereken periyotlar için 1h çekip dönüştürme)
    resample_map = {"2h": "2h", "4h": "4h", "8h": "8h"}
    raw_p = "1h" if per in resample_map else per
    df = yf.download(ticker, start=start, end=end, interval=raw_p, auto_adjust=True)

    if df.empty:
        st.error("Veri bulunamadı. Lütfen tarih sınırlarını veya sembolü kontrol edin.")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.strip().title() for c in df.columns]

    # 2. Resampling (2h, 4h, 8h)
    if per in resample_map:
        df = df.resample(resample_map[per]).agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

    # ============================================================
    # 3. Teknik Gösterge Hesaplamaları
    # ============================================================

    # KAMA
    if show_kama:
        df['KAMA'] = ta.kama(df['Close'], length=k_len)

    # SuperTrend
    if show_supertrend:
        sti = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=s_mult)
        df['ST_Line'], df['ST_Dir'] = sti.iloc[:, 0], sti.iloc[:, 1]
        df['Buy'] = (df['ST_Dir'] == 1) & (df['ST_Dir'].shift(1) == -1)
        df['Sell'] = (df['ST_Dir'] == -1) & (df['ST_Dir'].shift(1) == 1)

    # Stoch RSI
    if show_stochrsi:
        srsi = ta.stochrsi(df['Close'], length=srsi_len, rsi_length=srsi_len, k=3, d=3)
        df['stoch_k'], df['stoch_d'] = srsi.iloc[:, 0], srsi.iloc[:, 1]
        df['Cross_Up'] = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        df['Cross_Down'] = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))

    # SMA
    if show_sma:
        df['SMA'] = ta.sma(df['Close'], length=sma_len)

    # Bollinger Bands
    if show_bb:
        bbands = ta.bbands(df['Close'], length=bb_len, std=bb_std)
        df['BB_Upper'] = bbands.iloc[:, 0]
        df['BB_Mid'] = bbands.iloc[:, 1]
        df['BB_Lower'] = bbands.iloc[:, 2]

    # Ichimoku
    if show_ichimoku:
        ichi_result = ta.ichimoku(df['High'], df['Low'], df['Close'])
        ichi_df = ichi_result[0]
        df['Tenkan'] = ichi_df['ITS_9']
        df['Kijun'] = ichi_df['IKS_26']
        df['Senkou_A'] = ichi_df['ISA_9']
        df['Senkou_B'] = ichi_df['ISB_26']
        df['Chikou'] = ichi_df['ICS_26']

    # Fibonacci
    fib = {}
    if show_fib:
        f_df = df if f_look is None else df.tail(f_look)
        hi, lo = f_df['High'].max(), f_df['Low'].min()
        fib = {'23.6%': hi - 0.236 * (hi - lo), '38.2%': hi - 0.382 * (hi - lo), '50%': hi - 0.5 * (hi - lo)}

    # ============================================================
    # 4. Görselleştirme
    # ============================================================
    has_oscillator = show_stochrsi
    row_heights = [0.8, 0.2] if has_oscillator else [1.0, 0.001]

    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                        column_widths=[0.85, 0.15], row_heights=row_heights,
                        vertical_spacing=0.05, horizontal_spacing=0.01)

    # Mum Grafiği
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                  low=df['Low'], close=df['Close'], name='Fiyat'), row=1, col=1)

    # KAMA
    if show_kama:
        fig.add_trace(go.Scatter(x=df.index, y=df['KAMA'],
                                  line=dict(color='#2962ff', width=2), name='KAMA'), row=1, col=1)

    # SuperTrend + AL/SAT Sinyalleri
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

    # Stoch RSI Okları (Ana grafik üzerinde)
    if show_stochrsi:
        fig.add_trace(go.Scatter(x=df[df['Cross_Up']].index, y=df[df['Cross_Up']]['Low'] * 0.99,
                                  mode='markers', marker=dict(symbol='arrow-up', size=8, color='#00ff00'),
                                  name='K/D Kes-Yukarı'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df[df['Cross_Down']].index, y=df[df['Cross_Down']]['High'] * 1.01,
                                  mode='markers', marker=dict(symbol='arrow-down', size=8, color='#ff00ff'),
                                  name='K/D Kes-Aşağı'), row=1, col=1)

    # SMA
    if show_sma:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA'],
                                  line=dict(color='#ff9800', width=2), name=f'SMA({sma_len})'), row=1, col=1)

    # Bollinger Bands
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'],
                                  line=dict(color='rgba(174,134,255,0.6)', width=1), name='BB Üst'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'],
                                  line=dict(color='rgba(174,134,255,0.9)', width=1, dash='dot'), name='BB Orta'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'],
                                  line=dict(color='rgba(174,134,255,0.6)', width=1),
                                  fill='tonexty', fillcolor='rgba(174,134,255,0.07)', name='BB Alt'), row=1, col=1)

    # Ichimoku
    if show_ichimoku:
        fig.add_trace(go.Scatter(x=df.index, y=df['Tenkan'],
                                  line=dict(color='#0496ff', width=1), name='Tenkan-sen'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Kijun'],
                                  line=dict(color='#991515', width=1), name='Kijun-sen'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_A'],
                                  line=dict(color='rgba(67,160,71,0.5)', width=1), name='Senkou A'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_B'],
                                  line=dict(color='rgba(244,67,54,0.5)', width=1),
                                  fill='tonexty', fillcolor='rgba(67,160,71,0.06)', name='Senkou B'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Chikou'],
                                  line=dict(color='#9c27b0', width=1, dash='dot'), name='Chikou'), row=1, col=1)

    # Fibonacci
    if show_fib:
        for l, p in fib.items():
            fig.add_hline(y=p, line_dash="dash", line_color="rgba(128,128,128,0.5)",
                          annotation_text=l, row=1, col=1)

    # VRVP (Hacim Profili)
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

    # Osilatör Paneli (Stoch RSI)
    if show_stochrsi:
        fig.add_trace(go.Scatter(x=df.index, y=df['stoch_k'],
                                  line=dict(color='green'), name='K'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['stoch_d'],
                                  line=dict(color='red'), name='D'), row=2, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="rgba(255,0,0,0.4)",
                      annotation_text="80", row=2, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="rgba(0,128,0,0.4)",
                      annotation_text="20", row=2, col=1)

    fig.update_layout(template='plotly_white', height=900,
                      xaxis_rangeslider_visible=False, barmode='stack',
                      title=f"<b>{ticker}</b> Teknik Analizi")
    return fig


# ============================================================
# 🕹️ KONTROL PANELİ (WEB ARAYÜZÜ)
# ============================================================
st.sidebar.header("🛠️ Analiz Ayarları")

Hisse = st.sidebar.text_input("Varlık Sembolü", value="GC=F")
col1, col2 = st.sidebar.columns(2)
Baslangic = col1.date_input("Başlangıç", value=datetime.now() - timedelta(days=60))
Bitis = col2.date_input("Bitiş", value=datetime.now())
Secilen_Periyot = st.sidebar.selectbox("Periyot", ["15m", "30m", "1h", "2h", "4h", "8h", "1d"], index=4)

# ============================================================
# 📊 Gösterge Açma/Kapama
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Göstergeler")

show_kama = st.sidebar.checkbox("KAMA", value=True)
show_supertrend = st.sidebar.checkbox("SuperTrend (AL/SAT)", value=True)
show_stochrsi = st.sidebar.checkbox("Stoch RSI + Oklar", value=True)
show_fib = st.sidebar.checkbox("Fibonacci Seviyeleri", value=True)
show_vrvp = st.sidebar.checkbox("VRVP (Hacim Profili)", value=True)
show_sma = st.sidebar.checkbox("SMA", value=False)
show_bb = st.sidebar.checkbox("Bollinger Bands", value=False)
show_ichimoku = st.sidebar.checkbox("Ichimoku Cloud", value=False)

# ============================================================
# 🎯 Hassasiyet Ayarları
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Hassasiyet Ayarları")

KAMA_HIZI = st.sidebar.slider("KAMA Hızı", 5, 50, 10) if show_kama else 10
TREND_CARPAN = st.sidebar.slider("Trend Çarpanı", 1.0, 5.0, 2.0, 0.5) if show_supertrend else 2.0
OSILATOR_PER = st.sidebar.slider("Osilatör Periyodu", 7, 30, 14) if show_stochrsi else 14
HACIM_DETAY = st.sidebar.slider("Hacim Detayı", 20, 100, 40) if show_vrvp else 40
FIB_BAKIS = st.sidebar.number_input("Fib Geriye Bakış (Mum)", value=100) if show_fib else 100

SMA_LEN = st.sidebar.slider("SMA Periyodu", 5, 200, 20) if show_sma else 20
BB_LEN = st.sidebar.slider("BB Periyodu", 5, 50, 20) if show_bb else 20
BB_STD = st.sidebar.slider("BB Standart Sapma", 1.0, 4.0, 2.0, 0.5) if show_bb else 2.0

# ============================================================
# 🚀 Analizi Başlat
# ============================================================
if st.sidebar.button("Analizi Başlat"):
    with st.spinner('Veriler hesaplanıyor...'):
        fig = create_complete_trading_chart(
            Hisse, Baslangic, Bitis, Secilen_Periyot,
            KAMA_HIZI, TREND_CARPAN, OSILATOR_PER, HACIM_DETAY, FIB_BAKIS,
            show_kama, show_supertrend, show_stochrsi, show_fib, show_vrvp,
            show_sma, SMA_LEN, show_bb, BB_LEN, BB_STD,
            show_ichimoku
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Analiz yapmak için sol paneldeki 'Analizi Başlat' butonuna tıklayın. Varlık sembolünü bilmiyorsanız gemini'ye yfinance -varlık- tickerı yazın. Uygulama yatırım tavsiyesi içermez. Ücretsizdir.")

    st.markdown("""
    ---
    ### 📖 Analiz ve Strateji Kılavuzu

    #### 🚀 Sinyaller ve Oklar
    * **Büyük Üçgenler (AL/SAT):** SuperTrend indikatörünün ana trend onay sinyalleridir.
    * **🟢 Küçük Yeşil Oklar (K/D Kes-Yukarı):** Alt paneldeki yeşil çizgi (K), kırmızıyı (D) alttan yukarı kestiğinde belirir. Genellikle büyük 'AL' üçgeninden önce gelerek size **'Hazırlan, trend dönebilir'** mesajı verir.
    * **🟣 Küçük Mor Oklar (K/D Kes-Aşağı):** Yeşil çizginin kırmızıyı üstten aşağı kestiği anlardır. Büyük 'SAT' sinyalinden önce gelen bir **'Kar almayı düşün veya temkinli ol'** uyarısıdır.

    #### 🎯 İndikatör Mantığı (Emniyet Kemeriniz)
    * **KAMA Hızı:** Değerini düşürürseniz fiyatı daha yakından izler, artırırsanız ana trendi gösterir.
    * **Trend Çarpanı:** Değerini 1.5 gibi seviyelere düşürürseniz 'AL/SAT' sinyalleri çok daha erken gelir.
    * **Stoch RSI Periyodu:** Hızlı piyasalarda gürültüleri filtrelemek için artırılmalıdır.
    * **SMA:** Basit hareketli ortalama. Kısa periyot (10-20) hızlı sinyal, uzun periyot (50-200) ana trend.
    * **Bollinger Bands:** Fiyatın volatilite bandını gösterir. Bantlar daralırsa büyük hareket beklenir.
    * **Ichimoku Cloud:** Bulut (Kumo) destek/direnç, Tenkan/Kijun kesişmeleri sinyal üretir.

    #### 📈 Osilatör ve Hacim Okuma
    * **Yükseliş Sinyali:** Yeşil çizgi (K), kırmızıyı (D) alttan yukarı kesiyorsa ve **20 seviyesinin altındaysa** (aşırı satım), bu güçlü bir yükseliş sinyalidir.
    * **Düşüş Sinyali:** Yeşil çizgi (K), kırmızıyı (D) üstten aşağı kesiyorsa ve **80 seviyesinin üzerindeyse** (aşırı alım), bu bir düşüş uyarısıdır.
    * **VRVP (Hacim Profili):** Sağdaki barlar paranın en çok hangi fiyat seviyesinde maliyetlendiğini gösterir.
    * **Fibonacci Seviyeleri:** Fiyatın matematiksel olarak destek bulabileceği %23.6, %38.2 ve %50 bölgelerini gösterir.

    #### ⏱ Zaman Dilimi ve Geçmiş Veri Limitleri
    * **Seçenekler:** `15m`, `30m`, `1h`, `2h`, `4h`, `8h`, `1d`
    * **Maksimum Geriye Dönük Süreler:**
    *   · **1m:** 7 gün
    *   · **2m / 5m / 15m / 30m / 90m:** 60 gün
    *   · **1h:** 730 gün
    *   · **2h / 4h / 8h:** 60 gün (1h veriden türetilir)
    *   · **1d:** Sınırsız
    *   · **1w:** Sınırsız

    #### 🔍 Ticker (Sembol) Seçimi
    * **Borsa İstanbul:** Sembolün sonuna `.IS` ekleyin. Örn: `THYAO.IS`, `ASELS.IS`, `TUPRS.IS`
    * **Kripto:** Parite formatında yazın. Örn: `BTC-USD`, `ETH-USD`, `AVAX-USD`
    * **ABD Hisseleri:** Direkt sembol yeterli. Örn: `AAPL`, `TSLA`, `MSFT`
    * **Döviz / Emtia:** `EURUSD=X`, `GC=F` (Altın), `CL=F` (Petrol)

    ---

    *(Salih Rıdvan Yılmaz - sry@tahmin.ai)*
    """)
