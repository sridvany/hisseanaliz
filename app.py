import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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
                df['Buy'] = (df['ST_Dir'] == 1) & (df['ST_Dir'].shift(1) == -1)
                df['Sell'] = (df['ST_Dir'] == -1) & (df['ST_Dir'].shift(1) == 1)
            else:
                show_supertrend = False
        except Exception as e:
            st.warning(f"SuperTrend hatası: {e}")
            show_supertrend = False

    # Divergence Osilatörü (Momentum + Uyumsuzluk Tespiti)
    if show_stochrsi:
        try:
            rsi_raw = ta.rsi(df['Close'], length=srsi_len)
            if rsi_raw is not None:
                df['Mom'] = rsi_raw - 50  # Sıfır merkezli momentum
                df['Mom_Signal'] = ta.ema(df['Mom'], length=9)  # Sinyal çizgisi
                df['Mom_Hist'] = df['Mom'] - df['Mom_Signal']  # Histogram

                # Swing noktaları ile uyumsuzluk tespiti
                lookback = 5
                df['Swing_Low'] = df['Close'][(df['Close'].shift(lookback) > df['Close']) & (df['Close'].shift(-lookback) > df['Close'])]
                df['Swing_High'] = df['Close'][(df['Close'].shift(lookback) < df['Close']) & (df['Close'].shift(-lookback) < df['Close'])]
                df['Mom_Swing_Low'] = df['Mom'][(df['Mom'].shift(lookback) > df['Mom']) & (df['Mom'].shift(-lookback) > df['Mom'])]
                df['Mom_Swing_High'] = df['Mom'][(df['Mom'].shift(lookback) < df['Mom']) & (df['Mom'].shift(-lookback) < df['Mom'])]

                # Bullish / Bearish Divergence — vektörel tespit
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
            sma_result = ta.sma(df['Close'], length=sma_len)
            if sma_result is not None:
                df['SMA'] = sma_result
            else:
                show_sma = False
        except Exception as e:
            st.warning(f"SMA hatası: {e}")
            show_sma = False

    # Bollinger Bands
    if show_bb:
        try:
            bbands = ta.bbands(df['Close'], length=bb_len, std=bb_std)
            if bbands is not None and hasattr(bbands, 'columns'):
                for c in bbands.columns:
                    if c.startswith('BBU'):
                        df['BB_Upper'] = bbands[c]
                    elif c.startswith('BBM'):
                        df['BB_Mid'] = bbands[c]
                    elif c.startswith('BBL'):
                        df['BB_Lower'] = bbands[c]
            else:
                st.warning("Bollinger Bands hesaplanamadı (veri yetersiz olabilir).")
                show_bb = False
        except Exception as e:
            st.warning(f"Bollinger Bands hatası: {e}")
            show_bb = False

    # Ichimoku
    if show_ichimoku:
        try:
            ichi_result = ta.ichimoku(df['High'], df['Low'], df['Close'])
            # tuple ise ilk elemanı al, değilse direkt kullan
            if isinstance(ichi_result, tuple):
                ichi_df = ichi_result[0]
            else:
                ichi_df = ichi_result
            # None kontrolü
            if ichi_df is None or not hasattr(ichi_df, 'columns'):
                st.warning("Ichimoku hesaplanamadı (veri yetersiz olabilir).")
                show_ichimoku = False
            else:
                cols = ichi_df.columns.tolist()
                # Sadece isim bazlı erişim — iloc çakışmasını önler
                for c in cols:
                    cl = c.upper()
                    if 'ITS' in cl:
                        df['Tenkan'] = ichi_df[c]
                    elif 'IKS' in cl:
                        df['Kijun'] = ichi_df[c]
                    elif 'ISA' in cl:
                        df['Senkou_A'] = ichi_df[c]
                    elif 'ISB' in cl:
                        df['Senkou_B'] = ichi_df[c]
                    elif 'ICS' in cl:
                        df['Chikou'] = ichi_df[c]
                # Atama kontrolü
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
        fib = {'23.6%': hi - 0.236 * (hi - lo), '38.2%': hi - 0.382 * (hi - lo), '50%': hi - 0.5 * (hi - lo)}

    # ============================================================
    # 4. Görselleştirme
    # ============================================================
    has_oscillator = show_stochrsi
    row_heights = [0.65, 0.35] if has_oscillator else [1.0, 0.001]

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

    # Divergence Okları (Ana grafik üzerinde)
    if show_stochrsi:
        bull_div = df[df['Bull_Div'] == True]
        bear_div = df[df['Bear_Div'] == True]
        if len(bull_div) > 0:
            fig.add_trace(go.Scatter(x=bull_div.index, y=bull_div['Low'] * 0.99,
                                      mode='markers+text', text="▲D",
                                      textposition='bottom center', textfont=dict(size=9, color='#00e676'),
                                      marker=dict(symbol='triangle-up', size=12, color='#00e676'),
                                      name='Yükseliş Uyumsuzluğu'), row=1, col=1)
        if len(bear_div) > 0:
            fig.add_trace(go.Scatter(x=bear_div.index, y=bear_div['High'] * 1.01,
                                      mode='markers+text', text="▼D",
                                      textposition='top center', textfont=dict(size=9, color='#ff1744'),
                                      marker=dict(symbol='triangle-down', size=12, color='#ff1744'),
                                      name='Düşüş Uyumsuzluğu'), row=1, col=1)

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

    # Fibonacci (Sağda, Turuncu Kutu, Siyah Yazı)
    if show_fib:
        for l, p in fib.items():
            fig.add_hline(y=p, line_dash="dash", line_color="rgba(128,128,128,0.5)",
                          annotation_text=f"{l} ({p:.2f})", 
                          annotation_position="right",
                          annotation_bgcolor="orange", 
                          annotation_font_color="black", 
                          row=1, col=1)

    # Son Fiyat Gösterimi (Sağda, Yön Rengine Göre Kutu, Siyah Yazı)
    if not df.empty:
        last_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else df['Open'].iloc[-1]
        
        # Yükseliş varsa yeşil, düşüş varsa kırmızı arka plan
        price_color = "#00e676" if last_close >= prev_close else "#ff1744" 
        
        fig.add_hline(y=last_close, line_dash="dot", line_width=1, line_color=price_color,
                      annotation_text=f"{last_close:.2f}", 
                      annotation_position="right",
                      annotation_bgcolor=price_color, 
                      annotation_font_color="black",
                      row=1, col=1)

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

    # Osilatör Paneli (Divergence Momentum)
    if show_stochrsi:
        # Histogram barları
        fig.add_trace(go.Scatter(x=df.index, y=df['Mom'],
                                  line=dict(color='#00c853', width=1.5), name='Momentum'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Mom_Signal'],
                                  line=dict(color='#ff1744', width=1.5), name='Sinyal'), row=2, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="rgba(128,128,128,0.5)", row=2, col=1)
        # Üst sınırlar (kırmızı - aşırı alım bölgesi)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(255,23,68,0.5)",
                      annotation_text="30", row=2, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="rgba(255,23,68,0.3)", row=2, col=1)
        # Alt sınırlar (yeşil - aşırı satım bölgesi)
        fig.add_hline(y=-30, line_dash="dash", line_color="rgba(0,200,83,0.5)",
                      annotation_text="-30", row=2, col=1)
        fig.add_hline(y=-20, line_dash="dot", line_color="rgba(0,200,83,0.3)", row=2, col=1)

        # Divergence işaretleri osilatör üzerinde
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

    fig.update_layout(template='plotly_white', height=1200,
                      xaxis_rangeslider_visible=False, barmode='stack',
                      title=f"<b>{ticker}</b> Teknik Analizi",
                      margin=dict(r=50)) # Sağ tarafta etiketlerin kesilmemesi için küçük bir boşluk eklendi
    return fig


# ============================================================
# 🕹️ KONTROL PANELİ (WEB ARAYÜZÜ)
# ============================================================
st.sidebar.header("🛠️ Analiz Ayarları")

Hisse = st.sidebar.text_input("Varlık Sembolü", value="GC=F")
col1, col2 = st.sidebar.columns(2)
Baslangic = col1.date_input("Başlangıç", value=datetime.now() - timedelta(days=60))
Bitis = col2.date_input("Bitiş", value=datetime.now())
Secilen_Periyot = st.sidebar.selectbox("Periyot", ["15m", "30m", "1h", "2h", "4h", "8h", "1d", "1wk"], index=4)

# ============================================================
# 📊 Gösterge Açma/Kapama
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Göstergeler")

show_kama = st.sidebar.checkbox("KAMA", value=True)
show_supertrend = st.sidebar.checkbox("SuperTrend (AL/SAT)", value=True)
show_stochrsi = st.sidebar.checkbox("Divergence Osilatörü", value=True)
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
OSILATOR_PER = st.sidebar.slider("Divergence RSI Periyodu", 7, 30, 14) if show_stochrsi else 14
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
    st.info("Analiz yapmak için sol paneldeki 'Analizi Başlat' butonuna tıklayın. Varlık sembolünü bilmiyorsanız gemini'ye yfinance ...... tickerı nedir yazın. Uygulama yatırım tavsiyesi içermez. Ücretsizdir.")

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
    * **Fibonacci Seviyeleri:** Fiyatın matematiksel olarak destek bulabileceği %23.6, %38.2 ve %50 bölgelerini gösterir.

    #### 📐 SMA (Basit Hareketli Ortalama)
    * **Trend Yönü:** Fiyat SMA'nın üzerindeyse yükseliş trendi, altındaysa düşüş trendi hakimdir.
    * **Kesişme Sinyalleri:** Fiyat SMA'yı alttan yukarı keserse alım, üstten aşağı keserse satım sinyalidir.
    * **Golden Cross / Death Cross:** SMA 50, SMA 200'ü yukarı keserse "Golden Cross" (güçlü alım), aşağı keserse "Death Cross" (güçlü satım) oluşur.
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

    ---

    *(Salih Rıdvan Yılmaz - sry@tahmin.ai)*
    """)
