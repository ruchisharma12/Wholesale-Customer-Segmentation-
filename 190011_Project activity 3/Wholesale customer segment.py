# Step 1: Import the dataset and necessary libraries
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import BisectingKMeans

# Create Spark session
spark = SparkSession.builder.appName("WholesaleCustomerSegmentation").getOrCreate()

# Load the dataset
data = spark.read.csv("wholesale.csv", header=True, inferSchema=True)

# Step 2: Data Exploration
# Display the statistical description of the dataset
data.describe().show()

# Select relevant features for clustering
selected_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

# Assemble features into a single vector column
assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
data = assembler.transform(data)

# Step 3: Clustering using Hierarchical Method
# Apply Hierarchical Clustering
k = 5  # Number of clusters
bkm = BisectingKMeans(k=k, seed=1)
model = bkm.fit(data)

# Step 4: Choose the optimal number of clusters
# Evaluate clustering performance using the Within Set Sum of Squared Errors (WSSSE)
wssse = model.computeCost(data)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Step 5: Fit the model and make predictions
# Assign clusters to data points
predictions = model.transform(data)

# Display the predicted cluster for each data point
predictions.select("prediction").show()
