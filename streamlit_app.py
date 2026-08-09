import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------
# Page config
# -----------------------------------------------------------------
st.set_page_config(page_title="Gulf Crisis Data Insights Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

# -----------------------------------------------------------------
# Load data
# -----------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("clean_data.csv", parse_dates=["date"])
    return df

df = load_data()

st.title("📊 Data Insights Dashboard — Gulf Crisis / Gulf War (1990-1991)")

# -----------------------------------------------------------------
# Sidebar: dataset description + filters
# -----------------------------------------------------------------
st.sidebar.header("ℹ️ About the Dataset")
st.sidebar.markdown(
    """
This dataset was manually built from verified sources (Wikipedia, CNN, UN Security Council,
academic reports) since no ready-made dataset exists for the Gulf Crisis / invasion of Kuwait.
It combines 5 types of records into one table:

- **Timeline Event**: daily military, political, and diplomatic events
- **UN Resolution**: UN Security Council resolutions
- **Coalition Force**: troop numbers and casualties per country
- **Iraqi Casualty Estimate**: multiple conflicting Iraqi casualty estimates
- **Saudi Battle Event**: Saudi Arabia-specific battles and missile strikes
"""
)

st.sidebar.header("🔎 Filters")

record_types = sorted(df["record_type"].dropna().unique())
selected_types = st.sidebar.multiselect(
    "Record type", options=record_types, default=record_types
)

categories = sorted(df["category"].dropna().unique())
selected_categories = st.sidebar.multiselect(
    "Category", options=categories, default=categories
)

min_year = int(df["year"].dropna().min())
max_year = int(df["year"].dropna().max())
year_range = st.sidebar.slider(
    "Year range", min_value=min_year, max_value=max_year,
    value=(min_year, max_year)
)

entities = sorted(df["actor_or_country"].dropna().unique())
selected_entities = st.sidebar.multiselect(
    "Actor / Country", options=entities, default=[]
)

# -----------------------------------------------------------------
# Apply filters
# -----------------------------------------------------------------
filtered = df[
    df["record_type"].isin(selected_types)
    & df["category"].isin(selected_categories)
]
filtered = filtered[
    filtered["year"].isna()
    | ((filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1]))
]
if selected_entities:
    filtered = filtered[filtered["actor_or_country"].isin(selected_entities)]

# -----------------------------------------------------------------
# Main page: data preview
# -----------------------------------------------------------------
st.subheader("👀 Data Preview")
st.dataframe(filtered.head(20), use_container_width=True)
st.caption(f"Rows after filtering: {len(filtered)} out of {len(df)}")

# -----------------------------------------------------------------
# Summary statistics
# -----------------------------------------------------------------
st.subheader("📈 Summary Statistics")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Numeric columns**")
    st.dataframe(filtered[["metric_value", "year"]].describe(), use_container_width=True)
with col2:
    st.markdown("**Record type distribution**")
    st.dataframe(filtered["record_type"].value_counts(), use_container_width=True)

# -----------------------------------------------------------------
# Interactive visualizations
# -----------------------------------------------------------------
st.subheader("📊 Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Category Distribution", "Timeline", "Top Forces", "Correlation Heatmap"]
)

with tab1:
    st.bar_chart(filtered["category"].value_counts())

with tab2:
    events = filtered[filtered["record_type"] == "Timeline Event"].copy()
    if not events.empty:
        events["month"] = events["date"].dt.to_period("M").astype(str)
        st.line_chart(events["month"].value_counts().sort_index())
    else:
        st.info("No timeline events match the current filters.")

with tab3:
    forces = filtered[
        (filtered["record_type"] == "Coalition Force")
        & (filtered["metric_name"] == "troops_deployed")
    ].sort_values("metric_value", ascending=False).head(10)
    if not forces.empty:
        st.bar_chart(forces.set_index("actor_or_country")["metric_value"])
    else:
        st.info("No force data matches the current filters.")

with tab4:
    heat_df = filtered.copy()
    heat_df["title_length"] = heat_df["title"].astype(str).str.len()
    heat_df["month_num"] = heat_df["date"].dt.month
    numeric_cols = ["metric_value", "year", "month_num", "title_length"]
    corr = heat_df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    st.pyplot(fig)

# -----------------------------------------------------------------
# Insights section
# -----------------------------------------------------------------
st.subheader("💡 Insights")
st.markdown(
    """
- Most timeline events cluster between January and February 1991 (the actual Desert Storm period).
- The United States dominates troop numbers far above every other coalition country.
- There is no strong linear relationship between troop numbers and casualties — losses were tied
  more to the nature of the mission than to the size of military contribution.
- Iraqi casualty estimates vary enormously between sources (from thousands to hundreds of thousands),
  reflecting the lack of a reliable official count for that side of the conflict.
- Saudi Arabia stands out as both the main coalition host and a direct target — it fought the only
  major ground battle of the war (Khafji) and absorbed the war's single deadliest strike (Al Khobar).
"""
)

st.caption("⚠️ Figures in this dataset are estimates drawn from multiple secondary sources, not a unified official census.")
