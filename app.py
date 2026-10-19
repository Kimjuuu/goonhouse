import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 페이지 설정 (제목 및 아이콘 설정)
st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")

# 앱 제목
st.title("🎬 영화 유형 나누기")

# 데이터 로드
url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"


@st.cache_data
def load_data(data_url):
    df = pd.read_csv(data_url)
    return df


df_raw = load_data(url)

# 데이터 전처리 및 속성 생성
df = df_raw.copy()

# 필요 속성 생성
df["log_first_scrn"] = np.log10(df["first_scrn"].replace(0, np.nan))
df["log_total_audi"] = np.log10(df["total_audi"].replace(0, np.nan))
df["days_in_top10"] = df["days_in_top10"]

# 롱런 지수 = 누적 관객 / 첫 주 관객
df["long_run_index"] = df["total_audi"] / df["first_week_audi"].replace(
    0, np.nan
)

# 네 속성 중 결측값이 있거나 첫 주 관객이 0인 영화 제거
feature_cols_all = [
    "log_first_scrn",
    "log_total_audi",
    "days_in_top10",
    "long_run_index",
]
df_clean = df.dropna(subset=feature_cols_all + ["first_week_audi"]).copy()
df_clean = df_clean[df_clean["first_week_audi"] > 0].reset_index(drop=True)

# 속성 이름 매핑 (화면 표시용)
feature_names = {
    "log_first_scrn": "스크린 수(상용로그)",
    "log_total_audi": "누적 관객(상용로그)",
    "days_in_top10": "10위권 일수",
    "long_run_index": "롱런 지수",
}

# 군집화에 사용할 속성 선택 UI (기본값: 4개 모두, 최소 2개 선택)
selected_features_display = st.multiselect(
    "군집화에 사용할 속성을 선택하세요 (2개 이상 선택):",
    options=list(feature_names.values()),
    default=list(feature_names.values()),
)

if len(selected_features_display) < 2:
    st.warning("속성을 최소 2개 이상 선택해 주세요.")
    st.stop()

# 선택된 속성 컬럼명 변환
reverse_feature_names = {v: k for k, v in feature_names.items()}
selected_features = [
    reverse_feature_names[name] for name in selected_features_display
]

# 속성 표준화 및 K-평균 군집화 (K=3, random_state 고정)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[selected_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean["cluster_num"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 높은 순서로 묶음 라벨 지정 (㉮, ㉯, ㉰)
cluster_audi_mean = df_clean.groupby("cluster_num")["total_audi"].mean()
sorted_clusters = cluster_audi_mean.sort_values(ascending=False).index

label_map = {
    sorted_clusters[0]: "㉮",
    sorted_clusters[1]: "㉯",
    sorted_clusters[2]: "㉰",
}
df_clean["cluster"] = df_clean["cluster_num"].map(label_map)

# 전체 편수와 묶은 편수 한 줄로 출력
cluster_counts = df_clean["cluster"].value_counts().to_dict()
st.write(
    f"**전체 편수:** {len(df_clean)}편 | **묶음별 편수:** ㉮: {cluster_counts.get('㉮', 0)}편, ㉯: {cluster_counts.get('㉯', 0)}편, ㉰: {cluster_counts.get('㉰', 0)}편"
)

st.divider()

# 2차원 산점도
st.subheader("📌 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_axis_2d = st.selectbox(
        "2차원 X축 속성 선택:",
        options=selected_features_display,
        index=0,
        key="2d_x",
    )
with col2:
    y_axis_2d = st.selectbox(
        "2차원 Y축 속성 선택:",
        options=selected_features_display,
        index=min(1, len(selected_features_display) - 1),
        key="2d_y",
    )

fig_2d = px.scatter(
    df_clean,
    x=reverse_feature_names[x_axis_2d],
    y=reverse_feature_names[y_axis_2d],
    color="cluster",
    color_discrete_map={"㉮": "#EF553B", "㉯": "#00CC96", "㉰": "#636EFA"},
    hover_name="movieNm",
    labels={
        reverse_feature_names[x_axis_2d]: x_axis_2d,
        reverse_feature_names[y_axis_2d]: y_axis_2d,
        "cluster": "묶음",
    },
    category_orders={"cluster": ["㉮", "㉯", "㉰"]},
)
st.plotly_chart(fig_2d, use_container_width=True)

# 3차원 산점도
st.subheader("📌 3차원 산점도")
col3, col4, col5 = st.columns(3)
with col3:
    x_axis_3d = st.selectbox(
        "3차원 X축 속성 선택:",
        options=selected_features_display,
        index=0,
        key="3d_x",
    )
with col4:
    y_axis_3d = st.selectbox(
        "3차원 Y축 속성 선택:",
        options=selected_features_display,
        index=min(1, len(selected_features_display) - 1),
        key="3d_y",
    )
with col5:
    z_axis_3d = st.selectbox(
        "3차원 Z축 속성 선택:",
        options=selected_features_display,
        index=min(2, len(selected_features_display) - 1),
        key="3d_z",
    )

fig_3d = px.scatter_3d(
    df_clean,
    x=reverse_feature_names[x_axis_3d],
    y=reverse_feature_names[y_axis_3d],
    z=reverse_feature_names[z_axis_3d],
    color="cluster",
    color_discrete_map={"㉮": "#EF553B", "㉯": "#00CC96", "㉰": "#636EFA"},
    hover_name="movieNm",
    labels={
        reverse_feature_names[x_axis_3d]: x_axis_3d,
        reverse_feature_names[y_axis_3d]: y_axis_3d,
        reverse_feature_names[z_axis_3d]: z_axis_3d,
        "cluster": "묶음",
    },
    category_orders={"cluster": ["㉮", "㉯", "㉰"]},
)
st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

# 묶음별 통계 표 (편수 및 원래 단위 속성 평균)
st.subheader("📊 묶음별 편수 및 속성 평균 (원래 단위)")

summary_df = (
    df_clean.groupby("cluster")
    .agg(
        편수=("movieNm", "count"),
        스크린_수_평균=("first_scrn", "mean"),
        누적_관객_평균=("total_audi", "mean"),
        top10_일수_평균=("days_in_top10", "mean"),
        롱런_지수_평균=("long_run_index", "mean"),
    )
    .reindex(["㉮", "㉯", "㉰"])
)

# 컬럼명 정리
summary_df.columns = [
    "편수",
    "스크린 수 (평균)",
    "누적 관객 (평균)",
    "10위권 일수 (평균)",
    "롱런 지수 (평균)",
]
st.dataframe(summary_df.style.format("{:,.2f}", subset=summary_df.columns[1:]))

# 묶음별 누적 관객 Top 5 영화 제목
st.subheader("🏆 묶음별 누적 관객 Top 5 영화")
cols = st.columns(3)
for idx, cluster_label in enumerate(["㉮", "㉯", "㉰"]):
    with cols[idx]:
        st.markdown(f"### 묶음 {cluster_label}")
        top5_movies = (
            df_clean[df_clean["cluster"] == cluster_label]
            .sort_values(by="total_audi", ascending=False)
            .head(5)[["movieNm", "total_audi"]]
        )
        for rank, (_, row) in enumerate(top5_movies.iterrows(), start=1):
            st.write(f"{rank}. **{row['movieNm']}** ({row['total_audi']:,}명)")
