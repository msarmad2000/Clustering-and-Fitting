# Import necessary libraries for system-level configuration, visualization, statistics, and machine learning
import os  # For environment configuration
import warnings  # To suppress known library warnings
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical computations
import pandas as pd  # For data handling
import scipy.stats as ss  # For statistical analysis
import seaborn as sns  # For advanced plotting
from sklearn.cluster import KMeans  # For clustering
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error  # Model evaluation
from sklearn.linear_model import LinearRegression  # Regression models
from sklearn.preprocessing import StandardScaler  # Feature scaling

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Fix KMeans memory leak on Windows
os.environ["OMP_NUM_THREADS"] = "1"

# Function to create a scatterplot showing relationship between two numeric variables

def plot_relational_plot(df):
    """Plot a relational scatterplot for usage rate vs pregnancy rate."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x="Contraceptive Usage Rate (%)", y="Teen Pregnancy Rate (per 1000 teens)", hue="Country", palette="coolwarm", s=80, ax=ax)
    ax.set_title("Usage Rate vs Teen Pregnancy Rate", fontsize=14, fontweight="bold")
    plt.grid(True)
    plt.savefig('relational_plot.png')
    plt.show()

# Function to plot a count of categorical responses by type

def plot_categorical_plot(df):
    """Plot a bar plot of most popular condom types."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x="Most Popular Condom Type", hue="Most Popular Condom Type", palette="viridis", legend=False)
    plt.title("Most Popular Condom Type Distribution", fontsize=16, weight='bold')
    plt.xlabel("Condom Type", fontsize=14)
    plt.ylabel("Number of Respondents", fontsize=14)
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=12, color='black') 
    plt.tight_layout()
    plt.savefig("categorical_plot.png")
    plt.show()

# Function to produce both a pairplot and heatmap to show variable relationships

def plot_statistical_plot(df):
    """Plot both a beautified pairplot and a correlation heatmap."""
    selected = df[["Contraceptive Usage Rate (%)", "Teen Pregnancy Rate (per 1000 teens)"]].apply(pd.to_numeric, errors='coerce').dropna()
    sns.set(style="ticks", font_scale=1.2)

    # Beautiful pairplot
    pair = sns.pairplot(
        selected,
        kind="scatter",
        diag_kind="hist",
        plot_kws={'alpha': 0.7, 's': 60, 'edgecolor': 'w'}
    )
    pair.fig.suptitle("Statistical Relationships", y=1.05, fontsize=16, weight="bold")
    pair.savefig("statistical_plot_pair_beautiful.png")
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(selected.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.show()

# Function to compute and print key statistics

def statistical_analysis(df, col):
    data = df[col].dropna()
    stats = {
        "mean": np.mean(data),
        "median": np.median(data),
        "mode": ss.mode(data, keepdims=True).mode[0],
        "variance": np.var(data),
        "std_dev": np.std(data),
        "skewness": ss.skew(data),
        "kurtosis": ss.kurtosis(data)
    }
    print(f"\nStatistics for {col}:")
    for key, value in stats.items():
        print(f"{key.capitalize()}: {value:.2f}")
    return stats

# Function to write stats to a file for documentation

def writing(stats, col):
    with open("statistics.txt", "w") as f:
        f.write(f"Statistics for {col}:\n")
        for key, value in stats.items():
            f.write(f"{key.capitalize()}: {value:.2f}\n")

# Function to run KMeans clustering with silhouette score and elbow method evaluation

def perform_clustering(df, col1, col2):
    data_original = df[[col1, col2]].dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_original)

    silhouette_scores = []
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_scaled)
        silhouette_scores.append(silhouette_score(data_scaled, labels))
        inertias.append(kmeans.inertia_)

    # Elbow Method Plot
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, marker='o')
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Within-cluster Sum of Squares)")
    plt.grid(True)
    plt.savefig("elbow_plot.png")
    plt.show()

    # Choose best_k based on silhouette score
    best_k = silhouette_scores.index(max(silhouette_scores)) + 2
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)
    return labels, data_original.values, centers_original[:, 0], centers_original[:, 1], best_k

# Function to visualize clustering results

def plot_clustered_data(labels, data, xkmeans, ykmeans, best_k):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette="Set1", s=100, alpha=0.7, ax=ax)
    ax.scatter(xkmeans, ykmeans, c="red", marker="X", s=200, label="Cluster Centers")
    ax.set_title(f"K-Means Clustering (k={best_k})", fontsize=14)
    ax.set_xlabel("Contraceptive Usage Rate (%)")
    ax.set_ylabel("Teen Pregnancy Rate (per 1000 teens)")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('clustering.png')
    plt.show()

# Function to fit and evaluate simple linear regression

def perform_fitting(df, x_col, y_col):
    df = df.dropna(subset=[x_col, y_col])
    X = df[[x_col]].values
    y = df[y_col].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label="Data", alpha=0.7)
    plt.plot(X, y_pred, color="red", label=f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}\nR² = {r2:.2f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.grid(True)
    plt.savefig('fitting.png')
    plt.show()

# Function to perform and visualize multiple linear regression

def perform_multiple_regression(df):
    features = [
        "Contraceptive Usage Rate (%)",
        "Awareness Index (0-10)",
        "HIV Prevention Awareness (%)",
        "Online Sales (%)",
        "Average Price per Condom (USD)"
    ]
    target = "Teen Pregnancy Rate (per 1000 teens)"
    data = df[features + [target]].dropna()
    X = data[features]
    y = data[target]

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Actual Teen Pregnancy Rate")
    plt.ylabel("Predicted Teen Pregnancy Rate")
    plt.title(f"Multiple Regression Fit (R² = {r2:.2f}, MAE = {mae:.2f}, RMSE = {rmse:.2f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("multivariate_regression_fit.png")
    plt.show()

    print("Intercept:", model.intercept_)
    print("\nFeature Coefficients:")
    for feature, coef in zip(features, model.coef_):
        print(f"{feature:40}: {coef:.4f}")

# Function to clean strings and strip whitespaces from the DataFrame

def preprocessing(df):
    df = df.copy()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return df

# Main function to orchestrate the full analysis pipeline

def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    col = "Teen Pregnancy Rate (per 1000 teens)"
    stats = statistical_analysis(df, col)
    writing(stats, col)

    col1, col2 = "Contraceptive Usage Rate (%)", "Teen Pregnancy Rate (per 1000 teens)"
    clustering_results = perform_clustering(df, col1, col2)
    if clustering_results:
        plot_clustered_data(*clustering_results)

    perform_fitting(df, col1, col2)
    perform_multiple_regression(df)

if __name__ == '__main__':
    main()