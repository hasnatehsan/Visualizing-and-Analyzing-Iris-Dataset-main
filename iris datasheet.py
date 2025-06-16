# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris = pd.read_csv(url)

# Display the first few rows of the dataset
print(iris.head())

# Perform basic data analysis
print("\nBasic Data Analysis:")
print(iris.describe())

# Visualize the data
# Pairplot
sns.pairplot(iris, hue="species", markers=["o", "s", "D"])
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=iris, width=0.5, palette="pastel")
plt.title("Boxplot of Iris Dataset Features")
plt.show()

# Heatmap of correlations
plt.figure(figsize=(10, 6))
sns.heatmap(iris.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Feature Correlations in Iris Dataset")
plt.show()
