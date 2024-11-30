# MSR Mining Project
Software artifacts, such as libraries, frameworks, and API's play a huge role in the software development community. The popularity and usability of such artifacts plays a critical role in their adoption and sustainability. This research investigates whether an artifact’s number of dependents correlates with its popularity. Using a dataset of open-source artifacts containing software artifacts, releases, and associated metrics, we analyze the relationship between dependency count and popularity, using statistical methods and visualization techniques to identify trends. Popularity metrics include popularity-1-year, which tracks the frequency of new artifact versions released within a year, freshness, reflecting the recency of updates, and speed, indicating the pace of release cycles.

This code includes the data we used for our analysis, code for the correlation matrix, and Random Forest ML model for feature importance analysis.


## Getting Started

To run the program, please follow these steps:

1. **Clone the repository in an IDE:**

   ```bash
   git clone https://github.com/eunicehassan3/MSRMiningProject.git
   ```

2. **Run the program to view our results**


## Technologies Used in Project

This project incorporates the following technologies: 
- Maven Central (with metrics) Graph Dependency Database: where the dataset was sourced
- Neo4j Desktop: to open and query the database
- Python

## Cypher Used in Neo4J Desktop


## Contributors
Eunice Hassan (Sometimes referenced as Saheed Hassan in code) and Elisabeth Hassan