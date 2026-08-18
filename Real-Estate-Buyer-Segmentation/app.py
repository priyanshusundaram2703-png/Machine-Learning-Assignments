
import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("buyer_segmentation_final.csv")

# -----------------------------
# Title
# -----------------------------
st.title("🏠 Real Estate Buyer Segmentation & Investment Intelligence")
st.caption(
    "Machine Learning based Buyer Segmentation and Investment Profiling "
    "for Real Estate Market Intelligence"
)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🎛️ Filters")

countries = ["All"] + sorted(df["country"].dropna().unique().tolist())
regions = ["All"] + sorted(df["region"].dropna().unique().tolist())
purposes = ["All"] + sorted(df["acquisition_purpose"].dropna().unique().tolist())
client_types = ["All"] + sorted(df["client_type"].dropna().unique().tolist())

selected_country = st.sidebar.selectbox("Country", countries)
selected_region = st.sidebar.selectbox("Region", regions)
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", purposes)
selected_client = st.sidebar.selectbox("Client Type", client_types)

filtered_df = df.copy()

if selected_country != "All":
    filtered_df = filtered_df[
        filtered_df["country"] == selected_country
    ]

if selected_region != "All":
    filtered_df = filtered_df[
        filtered_df["region"] == selected_region
    ]

if selected_purpose != "All":
    filtered_df = filtered_df[
        filtered_df["acquisition_purpose"] == selected_purpose
    ]

if selected_client != "All":
    filtered_df = filtered_df[
        filtered_df["client_type"] == selected_client
    ]

# -----------------------------
# KPI Cards
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "👥 Buyers",
    f"{len(filtered_df):,}"
)

col2.metric(
    "🏠 Properties",
    f"{filtered_df['total_properties'].sum():,.0f}"
)

col3.metric(
    "💰 Property Value",
    f"${filtered_df['total_property_value'].sum()/1e6:.2f}M"
)

col4.metric(
    "⭐ Avg Satisfaction",
    f"{filtered_df['satisfaction_score'].mean():.2f}/5"
)

st.divider()

# -----------------------------
# Buyer Segmentation Overview
# -----------------------------
st.header("🎯 Buyer Segmentation Overview")

segment_counts = (
    filtered_df["buyer_segment"]
    .value_counts()
    .reset_index()
)

segment_counts.columns = ["Buyer Segment", "Buyers"]

fig_segment = px.bar(
    segment_counts,
    x="Buyer Segment",
    y="Buyers",
    title="Buyer Distribution by Segment",
    text="Buyers"
)

fig_segment.update_layout(
    xaxis_title="Buyer Segment",
    yaxis_title="Number of Buyers"
)

st.plotly_chart(fig_segment, use_container_width=True)

# -----------------------------
# Segment Property Value
# -----------------------------
st.header("💰 Investment & Property Analysis")

segment_value = (
    filtered_df.groupby("buyer_segment", as_index=False)
    .agg(
        Average_Property_Value=("total_property_value", "mean"),
        Average_Property_Price=("avg_property_price", "mean"),
        Average_Properties=("total_properties", "mean")
    )
)

fig_value = px.bar(
    segment_value,
    x="buyer_segment",
    y="Average_Property_Value",
    title="Average Property Value by Buyer Segment",
    text_auto=".2s"
)

st.plotly_chart(fig_value, use_container_width=True)

# -----------------------------
# Acquisition Purpose
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    purpose_data = pd.crosstab(
        filtered_df["buyer_segment"],
        filtered_df["acquisition_purpose"]
    ).reset_index()

    purpose_long = purpose_data.melt(
        id_vars="buyer_segment",
        var_name="Purpose",
        value_name="Buyers"
    )

    fig_purpose = px.bar(
        purpose_long,
        x="buyer_segment",
        y="Buyers",
        color="Purpose",
        barmode="group",
        title="Acquisition Purpose by Segment"
    )

    st.plotly_chart(fig_purpose, use_container_width=True)

with col2:
    loan_data = pd.crosstab(
        filtered_df["buyer_segment"],
        filtered_df["loan_applied"]
    ).reset_index()

    loan_long = loan_data.melt(
        id_vars="buyer_segment",
        var_name="Loan Applied",
        value_name="Buyers"
    )

    fig_loan = px.bar(
        loan_long,
        x="buyer_segment",
        y="Buyers",
        color="Loan Applied",
        barmode="group",
        title="Loan Behavior by Segment"
    )

    st.plotly_chart(fig_loan, use_container_width=True)

# -----------------------------
# Geographic Buyer Analysis
# -----------------------------
st.header("🌍 Geographic Buyer Analysis")

region_data = (
    filtered_df.groupby(
        ["region", "buyer_segment"]
    )
    .size()
    .reset_index(name="Buyers")
)

fig_region = px.bar(
    region_data,
    x="region",
    y="Buyers",
    color="buyer_segment",
    barmode="stack",
    title="Buyer Segments by Region"
)

st.plotly_chart(fig_region, use_container_width=True)

# -----------------------------
# Segment Insights
# -----------------------------
st.header("💡 Segment Insights")

for segment in filtered_df["buyer_segment"].unique():

    segment_df = filtered_df[
        filtered_df["buyer_segment"] == segment
    ]

    with st.expander(f"📌 {segment}"):

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Buyers",
            f"{len(segment_df):,}"
        )

        c2.metric(
            "Avg Age",
            f"{segment_df['age'].mean():.1f}"
        )

        c3.metric(
            "Avg Properties",
            f"{segment_df['total_properties'].mean():.2f}"
        )

        c4.metric(
            "Avg Property Value",
            f"${segment_df['total_property_value'].mean():,.0f}"
        )

        st.write(
            "Most common acquisition purpose:",
            segment_df["acquisition_purpose"].mode().iloc[0]
        )

        st.write(
            "Most common client type:",
            segment_df["client_type"].mode().iloc[0]
        )

        st.write(
            "Most common loan behavior:",
            segment_df["loan_applied"].mode().iloc[0]
        )

# -----------------------------
# Segment Summary Table
# -----------------------------
st.header("📊 Final Buyer Segment Summary")

summary = (
    filtered_df.groupby("buyer_segment")
    .agg(
        Buyers=("client_id", "count"),
        Avg_Age=("age", "mean"),
        Avg_Satisfaction=("satisfaction_score", "mean"),
        Avg_Properties=("total_properties", "mean"),
        Avg_Property_Value=("total_property_value", "mean"),
        Avg_Property_Price=("avg_property_price", "mean"),
        Avg_Area_Sqft=("avg_area_sqft", "mean")
    )
    .reset_index()
)

summary["Buyer %"] = (
    summary["Buyers"] / len(filtered_df) * 100
)

st.dataframe(
    summary.round(2),
    use_container_width=True
)

st.success(
    "✅ Buyer segmentation dashboard successfully generated using K-Means clustering results."
)

st.caption(
    "Developed for Real Estate Market Intelligence | Machine Learning Project"
)
