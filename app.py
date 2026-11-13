import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


st.title("My Data Dashboard")

df = pd.read_csv("data/games_clean.csv")
target = "Average playtime forever"

st.write("Preview of the data:")
st.dataframe(df.head())


st.title("Q1: Which genres/tags lead to the highest playtimes (all time)?")

CORE_GENRES = ["Action", "Adventure", "Casual", "Indie", "Simulation", "Strategy",
               "RPG", "Sports", "Racing", "Massively Multiplayer"]

@st.cache_data
def prepare_data(target_col):
  release_year = pd.to_datetime(df["Release date"], format="%b %d, %Y").dt.year
  genre_flags = df["Genres"].str.get_dummies(sep=",")[CORE_GENRES]
  X = pd.concat([df[["Price", "Achievements"]],release_year.rename("Release year"),genre_flags,],axis=1,)
  y = np.log1p(df[target_col])
  return X, y

X, y = prepare_data(target)

with st.expander("Understanding the question", type="step"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=50, random_state=42).fit(X_train, y_train)

    baseline_r2 = round(r2_score(y_test, np.full(len(y_test), y_train.mean())), 3)
    tree_r2 = round(r2_score(y_test, tree.predict(X_test)), 3)

    col1, col2 = st.columns(2)
    col1.metric("Baseline R²", baseline_r2)
    col2.metric("Tree R²", tree_r2, delta=round(tree_r2 - baseline_r2, 3))

with st.expander("Feature Importance", type="step"):
    importance = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(importance, horizontal=True, sort=False)


hours = df[target] / 60
g = pd.DataFrame({"genre": df["Genres"].str.split(","), "hours": hours}).explode("genre")
genre_med = g[g["genre"].isin(CORE_GENRES)].groupby("genre")["hours"].median().sort_values()

t = pd.DataFrame({"tag": df["Tags"].str.split(","), "hours": hours}).explode("tag")
t["tag"] = t["tag"].str.strip()
tag_stats = t.groupby("tag")["hours"].agg(["median", "count"])
tag_med = tag_stats[tag_stats["count"] >= 200]["median"].sort_values()    # only tags with 200+ games

with st.expander("Median Playtime by Genre and Tag", type="step"):
    df_plot = genre_med.reset_index().sort_values(by="hours", ascending=False)
    df_plot.columns = ["Genre", "Median Playtime (Hours)"]
    st.bar_chart(data=df_plot, x="Genre", y="Median Playtime (Hours)", horizontal=True, sort=False)

    df_tag_plot = tag_med.tail(10).reset_index().sort_values(by="median", ascending=False)
    df_tag_plot.columns = ["Tag", "Median Playtime (Hours)"]
    st.bar_chart(data=df_tag_plot, x="Tag", y="Median Playtime (Hours)", horizontal=True, sort=False)

st.divider()

st.title("Q2: Can we predict a game's average playtime forever based on price factors?")

X2 = np.log1p(df[["Price", "Achievements"]])
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=42)

lin = LinearRegression().fit(X2_train, y2_train)
pred = lin.predict(X2_test)

Baseline_R2 = round(r2_score(y2_test, np.full(len(y2_test), y2_train.mean())), 3)
Linear_R2 = round(r2_score(y2_test, pred), 3)
Typical_Error_MAE = round(mean_absolute_error(y2_test, pred), 2)
Price_Coefficient = round(lin.coef_[0], 3)
Achievements_Coefficient = round(lin.coef_[1], 3)

with st.expander("Linear Regression Results", type="step"):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Baseline R²", Baseline_R2)
    col2.metric("Linear R²", Linear_R2, delta=round(Linear_R2 - Baseline_R2, 3))
    col3.metric("Typical Error (MAE, log scale)", Typical_Error_MAE)
    col4.metric("Price Coefficient", Price_Coefficient)
    col5.metric("Achievements Coefficient", Achievements_Coefficient)

# median playtime by price range
bands = pd.cut(df["Price"], [-0.01, 0, 5, 10, 20, 100], labels=["Free", "$0.01-5", "$5-10", "$10-20", "$20-100"])
price_med = hours.groupby(bands, observed=True).median()
with st.expander("Median Playtime by Price Range", type="step"):
    st.bar_chart(price_med, x_label="Price Range", y_label="Median Playtime (Hours)", sort=False)

st.divider()

st.title("Q3: Can we cluster games into distinct playtime archetypes based on playtime and peak concurrent users?")

with st.expander("Silhouette Analysis", type="step"):
    Z = StandardScaler().fit_transform(np.log1p(df[[target, "Peak CCU"]]))
    silhouette_cols = st.columns(5)
    for k in range(2, 7):
        labels_k = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Z)
        silhouette_cols[k - 2].metric(f"k = {k}", round(silhouette_score(Z, labels_k, sample_size=5000, random_state=42), 3))

with st.expander("Cluster Summary", type="step"):
    K = 4
    labels = pd.Series(KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(Z), index=df.index)
    cluster_summary = pd.DataFrame({
        "games": labels.value_counts(),
        "median playtime (hours)": hours.groupby(labels).median(),
        "median peak CCU": df["Peak CCU"].groupby(labels).median(),
    }).sort_values("median playtime (hours)")
    st.write(cluster_summary.round(2))

with st.expander("Cluster Visualization", type="step"):
    st.scatter_chart(
        df,
        x="Median Playtime Hours (Log10)",  
        y="Median Peak CCU (Log10)",  
        color="Archetypes", 
    )