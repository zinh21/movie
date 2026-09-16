import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# 페이지 설정 (제목 + 아이콘, 브라우저 탭 표시)
# ---------------------------
st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")

st.title("🎬 영화 유형 나누기")

# ---------------------------
# 데이터 불러오기
# ---------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()
total_count = len(df)

# ---------------------------
# 파생 변수 만들기
# ---------------------------
work = df.copy()

# 결측치 및 first_week_audi == 0 인 행 제외
work = work.dropna(subset=["first_scrn", "total_audi", "first_week_audi", "days_in_top10"])
work = work[work["first_week_audi"] != 0]

# 상용로그 변환 (0 이하 값 있으면 로그 계산 안 되므로 안전하게 제외)
work = work[(work["first_scrn"] > 0) & (work["total_audi"] > 0)]

work["log_scrn"] = np.log10(work["first_scrn"])
work["log_audi"] = np.log10(work["total_audi"])
work["days_top10"] = work["days_in_top10"]

# 롱런 지수: 누적 관객 / 첫 주 관객, 20 초과시 20으로 자름
work["longrun_index"] = work["total_audi"] / work["first_week_audi"]
work["longrun_index"] = work["longrun_index"].clip(upper=20)

clustered_count = len(work)

st.write(f"전체 영화 편수: **{total_count}편** / 묶음에 사용된 편수: **{clustered_count}편**")

# ---------------------------
# 속성 선택 (2개 이상, 기본 4개 모두)
# ---------------------------
attr_map = {
    "스크린 수(log)": "log_scrn",
    "누적 관객(log)": "log_audi",
    "10위권 일수": "days_top10",
    "롱런 지수": "longrun_index",
}

st.subheader("1️⃣ 묶음에 사용할 속성 선택")
selected_labels = st.multiselect(
    "두 개 이상 선택하세요.",
    options=list(attr_map.keys()),
    default=list(attr_map.keys()),
)

if len(selected_labels) < 2:
    st.warning("⚠️ 최소 두 개 이상의 속성을 선택해야 합니다.")
    st.stop()

selected_cols = [attr_map[label] for label in selected_labels]

# ---------------------------
# 묶음 수 선택 (2~7, 기본 3)
# ---------------------------
st.subheader("2️⃣ 묶음 수 선택")
n_clusters = st.slider("묶음 수를 선택하세요.", min_value=2, max_value=7, value=3, step=1)

# ---------------------------
# 표준화
# ---------------------------
X = work[selected_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 선택한 묶음 수로 K-means 실행 (난수 고정)
# ---------------------------
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
raw_labels = kmeans.fit_predict(X_scaled)
work["묶음_번호"] = raw_labels

# ---------------------------
# 묶음 기호 준비: ㉮㉯㉰㉱㉲㉳㉴ (누적 관객 평균 큰 순서)
# ---------------------------
symbol_pool = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]
symbol_order = symbol_pool[:n_clusters]

group_order = (
    work.groupby("묶음_번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
symbol_map = {gid: symbol_order[i] for i, gid in enumerate(group_order)}
work["묶음"] = work["묶음_번호"].map(symbol_map)

# ---------------------------
# 묶음 이름표: ㉮=흥행대작형, ㉯=반짝흥행형, ㉰=군소흥행형
# (묶음 수가 3보다 크면 나머지는 "기타형N"으로 표시)
# ---------------------------
name_pool = ["흥행대작형", "반짝흥행형", "군소흥행형"]
symbol_to_name = {}
for i, sym in enumerate(symbol_order):
    if i < len(name_pool):
        symbol_to_name[sym] = name_pool[i]
    else:
        symbol_to_name[sym] = f"기타형{i+1}"

work["묶음이름"] = work["묶음"].map(symbol_to_name)

# 화면에 표시할 라벨: "㉮ (흥행대작형)" 형태로 통일
display_label = {sym: f"{sym} ({symbol_to_name[sym]})" for sym in symbol_order}
work["묶음표시"] = work["묶음"].map(display_label)
display_order = [display_label[sym] for sym in symbol_order]

# 색상은 자동 배정 (n_clusters 개수만큼)
palette = px.colors.qualitative.Set2
color_map = {display_label[sym]: palette[i % len(palette)] for i, sym in enumerate(symbol_order)}

# ---------------------------
# 2차원 산점도
# ---------------------------
st.subheader("3️⃣ 2차원 산점도")

col1, col2 = st.columns(2)
with col1:
    x_label_2d = st.selectbox("가로축 속성", options=selected_labels, index=0, key="x2d")
with col2:
    y_default_index = 1 if len(selected_labels) > 1 else 0
    y_label_2d = st.selectbox("세로축 속성", options=selected_labels, index=y_default_index, key="y2d")

x_col_2d = attr_map[x_label_2d]
y_col_2d = attr_map[y_label_2d]

fig_2d = px.scatter(
    work,
    x=x_col_2d,
    y=y_col_2d,
    color="묶음표시",
    category_orders={"묶음표시": display_order},
    color_discrete_map=color_map,
    hover_name="movieNm",
    labels={x_col_2d: x_label_2d, y_col_2d: y_label_2d, "묶음표시": "묶음"},
    title=f"{x_label_2d} vs {y_label_2d}",
)
st.plotly_chart(fig_2d, use_container_width=True)

# ---------------------------
# 3차원 산점도
# ---------------------------
st.subheader("4️⃣ 3차원 산점도")

if len(selected_labels) < 3:
    st.info("ℹ️ 3차원 산점도를 그리려면 속성을 3개 이상 선택해야 합니다.")
else:
    col3, col4, col5 = st.columns(3)
    with col3:
        x_label_3d = st.selectbox("X축 속성", options=selected_labels, index=0, key="x3d")
    with col4:
        idx_y = 1 if len(selected_labels) > 1 else 0
        y_label_3d = st.selectbox("Y축 속성", options=selected_labels, index=idx_y, key="y3d")
    with col5:
        idx_z = 2 if len(selected_labels) > 2 else 0
        z_label_3d = st.selectbox("Z축 속성", options=selected_labels, index=idx_z, key="z3d")

    x_col_3d = attr_map[x_label_3d]
    y_col_3d = attr_map[y_label_3d]
    z_col_3d = attr_map[z_label_3d]

    fig_3d = px.scatter_3d(
        work,
        x=x_col_3d,
        y=y_col_3d,
        z=z_col_3d,
        color="묶음표시",
        category_orders={"묶음표시": display_order},
        color_discrete_map=color_map,
        hover_name="movieNm",
        labels={x_col_3d: x_label_3d, y_col_3d: y_label_3d, z_col_3d: z_label_3d, "묶음표시": "묶음"},
        title=f"{x_label_3d} · {y_label_3d} · {z_label_3d}",
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

# ---------------------------
# 묶음별 편수 및 평균 (원래 단위)
# ---------------------------
st.subheader("5️⃣ 묶음별 편수 및 평균 (원래 단위)")

work["스크린 수"] = work["first_scrn"]
work["누적 관객"] = work["total_audi"]
work["10위권 일수"] = work["days_top10"]
work["롱런 지수(원본)"] = work["longrun_index"]

summary = (
    work.groupby("묶음표시")
    .agg(
        편수=("movieNm", "count"),
        평균_스크린수=("스크린 수", "mean"),
        평균_누적관객=("누적 관객", "mean"),
        평균_10위권일수=("10위권 일수", "mean"),
        평균_롱런지수=("롱런 지수(원본)", "mean"),
    )
    .reindex(display_order)
    .round(2)
)
st.dataframe(summary, use_container_width=True)

# ---------------------------
# 묶음별 누적 관객 상위 5편
# ---------------------------
st.subheader("6️⃣ 묶음별 누적 관객 상위 5편")

for disp in display_order:
    st.markdown(f"**{disp} 묶음**")
    top5 = (
        work[work["묶음표시"] == disp]
        .sort_values("total_audi", ascending=False)
        .head(5)["movieNm"]
        .tolist()
    )
    for i, name in enumerate(top5, start=1):
        st.write(f"{i}. {name}")

# ---------------------------
# 엘보우 방법: 묶음 수 1~7에 대한 WCSS 계산
# ---------------------------
st.subheader("7️⃣ 묶음 수에 따른 오차제곱합(WCSS) 변화")

k_range = list(range(1, 8))
wcss_list = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss_list.append(km.inertia_)

fig_elbow = go.Figure()
fig_elbow.add_trace(
    go.Scatter(
        x=k_range,
        y=wcss_list,
        mode="lines+markers",
        name="WCSS",
    )
)
fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="red")
fig_elbow.update_layout(
    xaxis_title="묶음 수",
    yaxis_title="오차제곱합(WCSS)",
    title="묶음 수에 따른 오차제곱합 변화",
)
st.plotly_chart(fig_elbow, use_container_width=True)

# ---------------------------
# WCSS 감소량 표
# ---------------------------
st.subheader("8️⃣ 묶음 수별 WCSS 및 감소량")

decrease_list = [None]  # 첫 줄은 비교 대상 없음
for i in range(1, len(wcss_list)):
    decrease_list.append(wcss_list[i - 1] - wcss_list[i])

wcss_table = pd.DataFrame(
    {
        "묶음 수": k_range,
        "WCSS": [round(v, 2) for v in wcss_list],
        "직전 대비 감소량": [round(v, 2) if v is not None else None for v in decrease_list],
    }
)
st.dataframe(wcss_table, use_container_width=True)

# ---------------------------
# 실루엣 점수 (선택한 묶음 수 기준)
# ---------------------------
st.subheader("9️⃣ 실루엣 점수")

if n_clusters >= 2:
    sil_score = silhouette_score(X_scaled, raw_labels)
    st.write(f"현재 묶음 수({n_clusters}개)의 실루엣 점수: **{sil_score:.3f}** (−1 ~ 1, 1에 가까울수록 묶음이 뚜렷함)")
else:
    st.info("ℹ️ 실루엣 점수는 묶음 수가 2 이상일 때만 계산할 수 있습니다.")
