import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
import csv
from csv import DictReader
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    create_correlation_matrix()
    train_random_forest_model_dependencies()
    train_random_forest_model_score()
    visualize_categories()

def create_correlation_matrix():
    # Load the CSV file into a pandas DataFrame
    # average_population_metric("mining_project_data.csv", "updated_mining_project_data.csv")
    # print(new_file)
    df = pd.read_csv('updated msr_data(without-duplicates).csv')
    print(df)

    # Calculate the correlation matrix
    correlation_matrix = df[['total_dependencies', 'aggregate_score', 'normalized_speed_metric', 'normalized_popularity_metric', 'normalized_freshness_metric']].corr()
    # correlation_matrix = df[['popularity_score', 'total_popularity_1_year', 'average_freshness', 'average_speed', 'dependencies_count']].corr()

    # Plot the correlation matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Popularity Metrics and Dependencies Count')
    plt.show()


def average_population_metric(file, output_csv):
    with open(file, newline='') as f:
        reader = DictReader(f)
        rows = list(reader)

        for row in rows:
            popularity_str = row['popularity_metrics']
            popularity_list = eval(row['popularity_metrics'])
            average_popularity = sum(popularity_list) / len(popularity_list) if popularity_list else 0.0
            row['average_popularity_metric'] = (average_popularity)

    with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        fieldnames = reader.fieldnames + ['average_popularity_metric']  # Add the new column to the fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()  # Write the header row (including the new column)
        writer.writerows(rows)  # Write the updated rows


def train_random_forest_model_score():
    # load CSV file
    file_path = 'updated msr_data(without-duplicates).csv'
    data = pd.read_csv(file_path)

    # get relevant columns in csv file
    features = ['normalized_speed_metric', 'normalized_popularity_metric', 'normalized_freshness_metric']
    target = 'aggregate_score'

    # drop any rows with missing values - if exists
    data = data.dropna(subset=features + [target])

    # split data
    X = data[features]
    y = data[target]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train RF regressor
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    # get and visualize feature importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance',
                                                                                               ascending=False)

    print("Feature Importances for Aggregate Score:")
    print(importance_df)

    # plot feature importances
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Feature Importances for Aggregate Score')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def train_random_forest_model_dependencies():
    # load CSV file
    file_path = 'updated msr_data(without-duplicates).csv'
    data = pd.read_csv(file_path)

    # get relevant columns in csv file
    features = ['normalized_speed_metric', 'normalized_popularity_metric',
                'normalized_freshness_metric']
    target = 'total_dependencies'

    # drop any rows with missing values - if exists
    data = data.dropna(subset=features + [target])

    # split data
    X = data[features]
    y = data[target]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train RF classifer
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    # get and visualize feature importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance',
                                                                                               ascending=False)

    print("Feature Importances for Dependency:")
    print(importance_df)

    # plot feature importances
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Feature Importances for Dependency')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def visualize_categories():
    file_path = 'mining_project_results_categorized.csv'
    data = pd.read_csv(file_path)
    # print(data.head())

    # identify columns in data
    artifact_column = 'artifact_type'
    popularity_column = 'average_popularity_metric'
    dependency_column = 'total_dependencies'
    aggregate_column = 'aggregate_score'

    # Box Plot: Popularity by Artifact Type
    plt.figure(figsize=(5, 5))
    sns.boxplot(x=artifact_column, y=popularity_column, data=data, palette="Set2")
    plt.title("Popularity by Artifact Type")
    plt.xlabel("Artifact Type")
    plt.ylabel("Popularity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Scatter Plot: Popularity vs Dependency Count
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=dependency_column, y=popularity_column, hue=artifact_column, data=data, palette="Set2")
    sns.regplot(x=dependency_column, y=popularity_column, data=data, scatter=False, color="blue",
                line_kws={"label": "Trendline"})
    plt.title("Popularity vs Dependency Count")
    plt.xlabel("Dependency Count")
    plt.ylabel("Popularity")
    plt.legend(title="Artifact Type")
    plt.tight_layout()
    plt.show()

    # Scatter Plot: Aggregate Score vs Dependency
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=dependency_column, y=aggregate_column, hue=artifact_column, data=data, palette="Set2")
    sns.regplot(x=dependency_column, y=aggregate_column, data=data, scatter=False, color="blue",
                line_kws={"label": "Trendline"})

    plt.title("Aggregate Popularity Score vs Dependency by Artifact Type")
    plt.xlabel("Aggregate Popularity Score")
    plt.ylabel("Dependency")
    plt.legend(title="Artifact Type")
    plt.tight_layout()
    plt.show()


main()