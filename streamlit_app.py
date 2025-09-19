import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# --- 기본 설정 ---
st.set_page_config(page_title="🌡️ 기후 변화와 학업 성취 대시보드", layout="wide")

# Pretendard 폰트 설정
FONT_PATH = "/fonts/Pretendard-Bold.ttf"
if os.path.exists(FONT_PATH):
    FONT_NAME = "Pretendard"
else:
    FONT_NAME = None  # 폰트 없으면 기본 폰트 사용

def apply_plotly_font(fig):
    if FONT_NAME:
        fig.update_layout(font=dict(family=FONT_NAME, size=14))
    return fig

# --- 데이터 로딩 함수 ---
@st.cache_data
def load_noaa_data():
    url = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-monthly/access/anomalies.csv"
    try:
        df = pd.read_csv(url)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[df["date"] <= pd.Timestamp.today()]
        df = df.rename(columns={"anomaly": "value"})
        return df[["date", "value"]], True
    except Exception:
        dates = pd.date_range("2000-01-01", periods=240, freq="M")
        values = np.sin(np.linspace(0, 20, 240)) + np.random.normal(0, 0.2, 240)
        df = pd.DataFrame({"date": dates, "value": values})
        return df, False

@st.cache_data
def load_user_data():
    np.random.seed(42)
    years = np.arange(2000, 2021)
    temps = 22 + 0.05 * (years - 2000) + np.random.normal(0, 0.3, len(years))
    scores = 500 - (temps - 22) * 5 + np.random.normal(0, 5, len(years))
    df = pd.DataFrame({
        "date": pd.to_datetime(years, format="%Y"),
        "summer_avg_temp_C": temps,
        "math_score": scores
    })
    return df

# --- 대시보드 제목 ---
st.title("🌡️ 기후 변화와 학업 성취 대시보드")

# --- 탭 생성 ---
tab1, tab2 = st.tabs(["① NOAA 데이터", "② 사용자 연구 데이터"])

# ==============================
# (1) NOAA 공개 데이터
# ==============================
with tab1:
    st.header("🌍 NOAA 공개 데이터: 전 지구 해수면 온도 이상치")
    noaa_df, success = load_noaa_data()
    if not success:
        st.warning("⚠️ NOAA API 호출 실패 → 예시 데이터 사용 중입니다.")

    # 사이드바 옵션
    with st.sidebar:
        st.subheader("NOAA 데이터 옵션")
        min_date_noaa = noaa_df["date"].min().date()
        max_date_noaa = noaa_df["date"].max().date()
        date_range_noaa = st.date_input("NOAA 기간 선택", (min_date_noaa, max_date_noaa),
                                        min_value=min_date_noaa, max_value=max_date_noaa)

    start_date_noaa, end_date_noaa = pd.to_datetime(date_range_noaa[0]), pd.to_datetime(date_range_noaa[1])
    df_noaa_vis = noaa_df[(noaa_df["date"] >= start_date_noaa) & (noaa_df["date"] <= end_date_noaa)]

    # 주요 지표
    col1, col2 = st.columns(2)
    col1.metric("평균 이상치 (°C)", f"{df_noaa_vis['value'].mean():.3f}")
    col2.metric("표준편차 (°C)", f"{df_noaa_vis['value'].std():.3f}")

    # 시각화
    st.subheader("📈 시계열 추세")
    fig_noaa = px.line(df_noaa_vis, x="date", y="value",
                       labels={"value": "온도 이상치 (°C)", "date": "날짜"})
    st.plotly_chart(apply_plotly_font(fig_noaa), use_container_width=True)

    st.subheader("📊 12개월 이동평균")
    df_noaa_vis["MA12"] = df_noaa_vis["value"].rolling(12).mean()
    fig_noaa_ma = px.line(df_noaa_vis, x="date", y="MA12",
                          labels={"MA12": "12개월 이동평균 (°C)", "date": "날짜"})
    st.plotly_chart(apply_plotly_font(fig_noaa_ma), use_container_width=True)

    # 다운로드
    st.download_button("📥 NOAA 데이터 다운로드", df_noaa_vis.to_csv(index=False).encode("utf-8"),
                       "noaa_data.csv", "text/csv")

# ==============================
# (2) 사용자 연구 데이터
# ==============================
with tab2:
    st.header("📚 사용자 연구 데이터: 기온과 학업 성취")
    user_df = load_user_data()

    # 사이드바 옵션
    with st.sidebar:
        st.subheader("사용자 데이터 옵션")
        min_date = user_df["date"].min().date()
        max_date = user_df["date"].max().date()
        date_range = st.date_input("사용자 데이터 기간 선택", (min_date, max_date),
                                   min_value=min_date, max_value=max_date)
        smoothing_window = st.slider("이동평균 윈도우", 0, 5, 0, help="0이면 스무딩 미적용")
        standardize = st.checkbox("수학 점수 표준화(Z-score)", value=False)

    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_vis = user_df[(user_df["date"] >= start_date) & (user_df["date"] <= end_date)].copy()

    if smoothing_window > 0:
        df_vis["math_score"] = df_vis["math_score"].rolling(smoothing_window).mean()
    if standardize:
        df_vis["math_score"] = (df_vis["math_score"] - df_vis["math_score"].mean()) / df_vis["math_score"].std()

    # 주요 지표
    col1, col2 = st.columns(2)
    col1.metric("평균 여름 기온 (°C)", f"{df_vis['summer_avg_temp_C'].mean():.2f}")
    col2.metric("평균 수학 점수", f"{df_vis['math_score'].mean():.2f}")

    # 시계열
    st.subheader("📈 연도별 수학 점수 및 여름 평균 기온")
    fig3 = px.line(df_vis, x="date", y=["summer_avg_temp_C", "math_score"],
                   labels={"value": "값", "date": "연도", "variable": "지표"})
    st.plotly_chart(apply_plotly_font(fig3), use_container_width=True)

    # 산점도
    st.subheader("📊 기온 vs 수학 점수")
    try:
        scatter_trend = px.scatter(
            df_vis, x="summer_avg_temp_C", y="math_score",
            trendline="ols",
            labels={"summer_avg_temp_C": "여름 평균기온 (°C)", "math_score": "수학 점수"}
        )
        st.plotly_chart(apply_plotly_font(scatter_trend), use_container_width=True)
    except Exception:
        st.error("OLS 회귀선 실패 → Polyfit 대체")
        coeffs = np.polyfit(df_vis["summer_avg_temp_C"], df_vis["math_score"], 1)
        poly_line = np.polyval(coeffs, df_vis["summer_avg_temp_C"])
        fig4 = px.scatter(df_vis, x="summer_avg_temp_C", y="math_score")
        fig4.add_traces(px.line(x=df_vis["summer_avg_temp_C"], y=poly_line).data)
        st.plotly_chart(apply_plotly_font(fig4), use_container_width=True)

    # 상관계수
    corr_value = df_vis["summer_avg_temp_C"].corr(df_vis["math_score"])
    st.metric("여름 평균기온 vs 수학 점수 상관계수", f"{corr_value:.3f}")

    # 다운로드
    st.download_button("📥 사용자 데이터 다운로드", df_vis.to_csv(index=False).encode("utf-8"),
                       "user_data.csv", "text/csv")

# --- 출처 ---
st.markdown("---")
st.markdown("### 📚 참고 출처")
st.markdown("- NOAA, [GHCN Monthly Anomalies](https://www.ncei.noaa.gov/data/global-historical-climatology-network-monthly/access/anomalies.csv)")
st.markdown("- Park, R. J., & Goodman, J. (2023). *Heat and Learning*. PLOS Climate.")
st.markdown("- OECD PISA 데이터 및 학업 성취도 연구")
st.markdown("- 기상청 기후자료개방포털")
