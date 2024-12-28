# CodTech-Task-2
- **Name**: Vikram Singh Kushwah  
- **Company**: CODTECH IT SOLUTIONS  
- **ID**: CT12WDS90  
- **Domain**: Data Analytics  
- **Duration**: 5th December 2024 to 5th March 2025  
- **Mentor**: Neela Santhosh Kumar  

# Customer Segmentation and Analysis

This project focuses on customer segmentation using K-Means clustering to analyze customer spending patterns. The goal is to identify distinct customer groups based on their annual income and spending score, allowing businesses to tailor their marketing strategies.

## Project Overview

**Objective**:  
To segment customers into different groups based on their purchasing behavior, using K-Means clustering. This segmentation helps businesses identify target groups, enhance customer satisfaction, and improve marketing efficiency.

**Tools & Technologies**:  
- **Python**  
- **Pandas**  
- **NumPy**  
- **Matplotlib**  
- **Seaborn**  
- **Scikit-learn**  

---

## Dataset  
**Dataset**: Mall_Customers.csv  
The dataset contains information about customers, including gender, age, annual income, and spending score.

**Key Columns**:  
- **Annual Income** (in 1000s)  
- **Spending Score** (1-100)

---

## Project Workflow

### 1. Data Collection & Analysis
- Load the dataset using Pandas:  
  ```python
  customer_data = pd.read_csv('Mall_Customers.csv')
  ```
- Display the first 5 rows:  
  ```python
  customer_data.head()
  ```
- Shape of the dataset (rows and columns):  
  ```python
  customer_data.shape
  ```
- Dataset information:  
  ```python
  customer_data.info()
  ```
- Check for missing values:  
  ```python
  customer_data.isnull().sum()
  ```

---

### 2. Feature Selection  
- Select **Annual Income** and **Spending Score** for clustering:  
  ```python
  x = customer_data.iloc[:, [3, 4]].values
  ```

---

### 3. Determining Optimal Clusters (Elbow Method)
- Calculate Within-Cluster Sum of Squares (WCSS) for different cluster numbers:  
  ```python
  wcss = []
  for i in range(1, 11):
      kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
      kmeans.fit(x)
      wcss.append(kmeans.inertia_)
  ```
- Plot the Elbow Graph to find the optimal number of clusters:  
  ```python
  sns.set()
  plt.plot(range(1, 11), wcss)
  plt.title('The Elbow Point Graph')
  plt.xlabel('Number of Clusters')
  plt.ylabel('WCSS')
  plt.show()
  ```

---

### 4. Training the K-Means Model  
- Train the K-Means model with 5 clusters (based on the elbow point):  
  ```python
  kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
  y = kmeans.fit_predict(x)
  ```

---

### 5. Visualization of Clusters
- Plot the clusters and their centroids:  
  ```python
  plt.figure(figsize=(8, 8))
  plt.scatter(x[y == 0, 0], x[y == 0, 1], s=50, c='green', label='Cluster 1')
  plt.scatter(x[y == 1, 0], x[y == 1, 1], s=50, c='red', label='Cluster 2')
  plt.scatter(x[y == 2, 0], x[y == 2, 1], s=50, c='blue', label='Cluster 3')
  plt.scatter(x[y == 3, 0], x[y == 3, 1], s=50, c='yellow', label='Cluster 4')
  plt.scatter(x[y == 4, 0], x[y == 4, 1], s=50, c='black', label='Cluster 5')
  plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')
  plt.title('Customer Groups')
  plt.xlabel('Annual Income')
  plt.ylabel('Spending Score')
  plt.legend()
  plt.show()
  ```

---

## Results
- Customers are grouped into 5 distinct clusters based on their income and spending patterns.
- Each cluster represents a unique customer segment that businesses can target with customized marketing strategies.

---

## Visualization  

![image](https://github.com/user-attachments/assets/22c050f4-b369-4e6b-9360-157d0f24d34b)
![image](https://github.com/user-attachments/assets/e1bb3f0e-63b6-4d22-a8a0-e821708023db)

