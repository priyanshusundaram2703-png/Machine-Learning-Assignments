import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Nassau Factory Optimization",
    page_icon="🏭",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🏭 Factory Reallocation & Shipping Optimization")
st.caption(
    "Decision Intelligence Dashboard for Nassau Candy Distributor"
)


# =========================================================
# CONSTANTS
# =========================================================

FACTORY_COORDINATES = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.119140, -96.181150),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107)
}

FACTORY_MAPPING = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory"
}

# Regional representative coordinates.
# Used because the source dataset provides region/city/ZIP,
# but does not provide customer latitude/longitude.
REGION_COORDINATES = {
    "Atlantic": (39.0, -75.0),
    "Pacific": (37.0, -120.0),
    "Interior": (39.0, -100.0),
    "Gulf": (31.0, -95.0)
}


# =========================================================
# HELPERS
# =========================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Return great-circle distance in kilometers."""

    r = 6371.0

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return r * c


def factory_region_distance(factory, region):
    """Distance from factory to regional representative point."""

    if region not in REGION_COORDINATES:
        return np.nan

    factory_lat, factory_lon = FACTORY_COORDINATES[factory]
    region_lat, region_lon = REGION_COORDINATES[region]

    return haversine_distance(
        factory_lat,
        factory_lon,
        region_lat,
        region_lon
    )


# =========================================================
# LOAD DATA
# =========================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload Nassau Candy Distributor CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

else:
    try:
        df = pd.read_csv("Nassau Candy Distributor.csv")
    except FileNotFoundError:
        st.warning(
            "Please upload 'Nassau Candy Distributor.csv' "
            "from the sidebar."
        )
        st.stop()


# =========================================================
# DATA PREPARATION
# =========================================================

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce"
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"],
    errors="coerce"
)

df["Lead Time Days"] = (
    df["Ship Date"] - df["Order Date"]
).dt.days

df["Current Factory"] = (
    df["Product Name"].map(FACTORY_MAPPING)
)

df["Customer Location"] = (
    df["City"].astype(str).str.strip()
    + ", "
    + df["State/Province"].astype(str).str.strip()
)

df["Order Month"] = df["Order Date"].dt.month

df["Order DayOfWeek"] = (
    df["Order Date"].dt.dayofweek
)

df["Profit Margin"] = np.where(
    df["Sales"] != 0,
    (df["Gross Profit"] / df["Sales"]) * 100,
    0
)


model_df = df.dropna(
    subset=["Current Factory", "Lead Time Days"]
).copy()


# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource
def train_models(data):

    features = [
        "Product Name",
        "Current Factory",
        "Region",
        "Ship Mode",
        "Division",
        "Units",
        "Sales",
        "Cost",
        "Gross Profit",
        "Order Month",
        "Order DayOfWeek"
    ]

    categorical_features = [
        "Product Name",
        "Current Factory",
        "Region",
        "Ship Mode",
        "Division"
    ]

    numerical_features = [
        "Units",
        "Sales",
        "Cost",
        "Gross Profit",
        "Order Month",
        "Order DayOfWeek"
    ]

    X = data[features]
    y = data["Lead Time Days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "num",
                "passthrough",
                numerical_features
            )
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }

    trained_models = {}
    results = {}

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        prediction = pipeline.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            prediction
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                prediction
            )
        )

        r2 = r2_score(
            y_test,
            prediction
        )

        trained_models[name] = pipeline

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

    results_df = pd.DataFrame(results).T

    best_model_name = results_df[
        "RMSE"
    ].idxmin()

    return (
        trained_models,
        results_df,
        best_model_name
    )


with st.spinner("Training ML models..."):

    (
        trained_models,
        model_results,
        best_model_name
    ) = train_models(model_df)


best_model = trained_models[best_model_name]


# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.header("🎛️ Scenario Controls")

products = sorted(
    model_df["Product Name"].dropna().unique()
)

regions = sorted(
    model_df["Region"].dropna().unique()
)

ship_modes = sorted(
    model_df["Ship Mode"].dropna().unique()
)

selected_product = st.sidebar.selectbox(
    "Product",
    products
)

selected_region = st.sidebar.selectbox(
    "Region",
    regions
)

selected_ship_mode = st.sidebar.selectbox(
    "Ship Mode",
    ship_modes
)

speed_priority = st.sidebar.slider(
    "Optimization Priority: Speed ↔ Profit",
    min_value=0,
    max_value=100,
    value=70,
    step=10
)

speed_weight = speed_priority / 100
profit_weight = 1 - speed_weight


# =========================================================
# CURRENT PRODUCT DATA
# =========================================================

product_data = model_df[
    model_df["Product Name"] == selected_product
].copy()

scenario_data = product_data[
    (product_data["Region"] == selected_region)
    & (
        product_data["Ship Mode"]
        == selected_ship_mode
    )
].copy()

if scenario_data.empty:

    scenario_data = product_data[
        product_data["Region"]
        == selected_region
    ].copy()

if scenario_data.empty:

    scenario_data = product_data.copy()


current_factory = FACTORY_MAPPING[
    selected_product
]

current_profit_margin = (
    scenario_data["Gross Profit"].sum()
    / scenario_data["Sales"].sum()
    * 100
    if scenario_data["Sales"].sum() != 0
    else 0
)


# =========================================================
# TOP KPI CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Factory",
        current_factory
    )

with col2:
    st.metric(
        "Product Sales",
        f"${scenario_data['Sales'].sum():,.0f}"
    )

with col3:
    st.metric(
        "Gross Profit",
        f"${scenario_data['Gross Profit'].sum():,.0f}"
    )

with col4:
    st.metric(
        "Profit Margin",
        f"{current_profit_margin:.1f}%"
    )


# =========================================================
# BUILD FACTORY SCENARIO TABLE
# =========================================================

scenario_rows = []

median_units = scenario_data["Units"].median()
median_sales = scenario_data["Sales"].median()
median_cost = scenario_data["Cost"].median()
median_profit = scenario_data["Gross Profit"].median()
median_month = scenario_data["Order Month"].median()
median_day = scenario_data["Order DayOfWeek"].median()

for factory in FACTORY_COORDINATES.keys():

    input_row = pd.DataFrame({
        "Product Name": [selected_product],
        "Current Factory": [factory],
        "Region": [selected_region],
        "Ship Mode": [selected_ship_mode],
        "Division": [
            scenario_data["Division"].mode().iloc[0]
            if not scenario_data["Division"].mode().empty
            else product_data["Division"].mode().iloc[0]
        ],
        "Units": [median_units],
        "Sales": [median_sales],
        "Cost": [median_cost],
        "Gross Profit": [median_profit],
        "Order Month": [median_month],
        "Order DayOfWeek": [median_day]
    })

    predicted_lead_time = float(
        best_model.predict(input_row)[0]
    )

    distance = factory_region_distance(
        factory,
        selected_region
    )

    scenario_rows.append({
        "Factory": factory,
        "Predicted Lead Time (days)": predicted_lead_time,
        "Estimated Distance (km)": distance,
        "Current Factory":
            factory == current_factory
    })


scenario_df = pd.DataFrame(
    scenario_rows
)


# =========================================================
# NORMALIZE SCENARIO SCORES
# =========================================================

lead_min = scenario_df[
    "Predicted Lead Time (days)"
].min()

lead_max = scenario_df[
    "Predicted Lead Time (days)"
].max()

distance_min = scenario_df[
    "Estimated Distance (km)"
].min()

distance_max = scenario_df[
    "Estimated Distance (km)"
].max()


if lead_max != lead_min:

    scenario_df["Speed Score"] = (
        (lead_max
         - scenario_df["Predicted Lead Time (days)"])
        / (lead_max - lead_min)
    )

else:

    scenario_df["Speed Score"] = 1.0


if distance_max != distance_min:

    scenario_df["Distance Score"] = (
        (distance_max
         - scenario_df["Estimated Distance (km)"])
        / (distance_max - distance_min)
    )

else:

    scenario_df["Distance Score"] = 1.0


# Use product historical margin as a financial stability
# indicator. It is not factory-specific because the dataset
# does not provide factory-specific shipping cost/profit.
scenario_df["Profit Stability Score"] = (
    current_profit_margin / 100
)

scenario_df["Recommendation Score"] = (
    (
        speed_weight
        * scenario_df["Speed Score"]
    )
    +
    (
        0.5
        * speed_weight
        * scenario_df["Distance Score"]
    )
    +
    (
        profit_weight
        * scenario_df["Profit Stability Score"]
    )
)


scenario_df = scenario_df.sort_values(
    "Recommendation Score",
    ascending=False
).reset_index(drop=True)


scenario_df["Rank"] = (
    np.arange(len(scenario_df))
    + 1
)


# =========================================================
# RECOMMENDATION
# =========================================================

best_factory = scenario_df.iloc[0]["Factory"]

if best_factory == current_factory:

    recommendation_text = (
        f"✅ Keep the current assignment: "
        f"**{current_factory}**"
    )

else:

    recommendation_text = (
        f"🔄 Recommended alternative factory: "
        f"**{best_factory}**"
    )

st.subheader("🎯 Recommendation")

st.info(
    recommendation_text
)


# =========================================================
# FACTORY OPTIMIZATION SIMULATOR
# =========================================================

st.subheader("🏭 Factory Optimization Simulator")

display_df = scenario_df[
    [
        "Rank",
        "Factory",
        "Predicted Lead Time (days)",
        "Estimated Distance (km)",
        "Speed Score",
        "Distance Score",
        "Recommendation Score",
        "Current Factory"
    ]
].copy()

st.dataframe(
    display_df.round(3),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# WHAT-IF SCENARIO
# =========================================================

st.subheader("🔮 What-If Scenario Analysis")

current_row = scenario_df[
    scenario_df["Factory"] == current_factory
].iloc[0]

recommended_row = scenario_df.iloc[0]

lead_time_change = (
    current_row["Predicted Lead Time (days)"]
    - recommended_row["Predicted Lead Time (days)"]
)

distance_change = (
    current_row["Estimated Distance (km)"]
    - recommended_row["Estimated Distance (km)"]
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Current Factory",
        current_factory
    )

with c2:
    st.metric(
        "Recommended Factory",
        recommended_row["Factory"]
    )

with c3:
    st.metric(
        "Distance Change (km)",
        f"{distance_change:,.1f}"
    )

st.write(
    f"Predicted lead-time difference: "
    f"**{lead_time_change:,.1f} days**"
)


# =========================================================
# CHARTS
# =========================================================

st.subheader("📊 Factory Comparison")

chart_data = scenario_df[
    ["Factory", "Predicted Lead Time (days)"]
].set_index("Factory")

st.bar_chart(chart_data)


distance_chart = scenario_df[
    ["Factory", "Estimated Distance (km)"]
].set_index("Factory")

st.subheader("🗺️ Estimated Factory-to-Region Distance")

st.bar_chart(distance_chart)


# =========================================================
# RISK & IMPACT PANEL
# =========================================================

st.subheader("⚠️ Risk & Impact Panel")

if recommended_row["Factory"] != current_factory:

    st.warning(
        "Alternative factory shows a better optimization score "
        "for the selected scenario."
    )

else:

    st.success(
        "Current factory remains the strongest option "
        "for the selected scenario."
    )


if current_profit_margin < 10:

    st.error(
        f"Low historical profit margin detected: "
        f"{current_profit_margin:.1f}%"
    )

elif current_profit_margin < 20:

    st.warning(
        f"Moderate profit margin: "
        f"{current_profit_margin:.1f}%"
    )

else:

    st.success(
        f"Healthy historical profit margin: "
        f"{current_profit_margin:.1f}%"
    )


st.caption(
    "Note: Customer coordinates are not included in the source "
    "dataset. Therefore, estimated distance uses representative "
    "regional coordinates. Factory-specific shipping cost data "
    "is also unavailable, so profit margin is shown as a "
    "financial stability indicator rather than a fabricated "
    "factory-specific profit estimate."
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

with st.expander("🤖 ML Model Performance"):

    model_display = model_results.copy()

    model_display.columns = [
        "MAE",
        "RMSE",
        "R²"
    ]

    st.dataframe(
        model_display.round(3),
        use_container_width=True
    )

    st.write(
        f"**Selected model:** {best_model_name}"
    )


# =========================================================
# DATASET OVERVIEW
# =========================================================

with st.expander("📁 Dataset Overview"):

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "Rows",
            f"{len(df):,}"
        )

    with b:
        st.metric(
            "Columns",
            f"{df.shape[1]:,}"
        )

    with c:
        st.metric(
            "Products",
            f"{df['Product Name'].nunique():,}"
        )

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Nassau Candy Distributor | Factory Reallocation & "
    "Shipping Optimization Recommendation System"
)
