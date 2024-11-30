import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
import csv
from csv import DictReader
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    create_correlation_matrix()
    train_random_forest_model()

def create_correlation_matrix():
    # Load the CSV file into a pandas DataFrame
    average_population_metric("mining_project_data.csv", "updated_mining_project_data.csv")
    # print(new_file)
    df = pd.read_csv('updated_mining_project_data.csv')
    print(df)

    # Calculate the correlation matrix
    correlation_matrix = df[['total_releases', 'aggregate_score', 'speed_metric', 'average_popularity_metric']].corr()
    # correlation_matrix = df[['popularity_score', 'total_popularity_1_year', 'average_freshness', 'average_speed', 'dependencies_count']].corr()
    #
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


def train_random_forest_model():
    # load CSV file
    file_path = 'mining_project_data.csv'
    data = pd.read_csv(file_path)

    # get relevant columns in csv file
    features = ['speed', 'aggregated_score', 'freshness', 'dependency_count']
    target = 'popularity_1_year'

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

    print("Feature Importances:")
    print(importance_df)

    # plot feature importances
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

main()