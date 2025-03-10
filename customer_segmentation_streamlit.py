import streamlit as st  
import pickle  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.decomposition import PCA  
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import KMeans  

# Load the clustering model and scaler  
with open("Customer_Segmentation/kmeans_model.pkl", "rb") as f:  
    kmeans = pickle.load(f)  

# Load scaler if available  
scaler = None  
expected_features = []  
try:  
    with open("Customer_Segmentation/scaler.pkl", "rb") as f:  
        scaler = pickle.load(f)  
    with open("Customer_Segmentation/feature_names.pkl", "rb") as f:  
        expected_features = pickle.load(f)  # List of features used during training  
except FileNotFoundError:  
    st.warning("Scaler or feature names file not found. Using raw data for clustering.")  

# Streamlit UI  
st.title("📊 Clustering Model App")  
st.write("Upload a dataset to visualize clustering using the pre-trained model.")  

# Upload dataset  
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])  
if uploaded_file:  
    try:  
        # Dynamically handle file types  
        if uploaded_file.name.endswith(".csv"):  
            df = pd.read_csv(uploaded_file)  
        else:  
            df = pd.read_excel(uploaded_file)  

        st.success("Dataset loaded successfully!")  

        # Display the initial data  
        st.write("### Preview of Your Data")  
        st.write(df.head())  

        # Drop any non-numeric columns for clustering  
        df_numeric = df.select_dtypes(include=[np.number])  
        if df_numeric.empty:  
            st.error("The uploaded dataset does not contain numeric columns suitable for clustering.")  
        else:  
            # Align dataset features with expected features  
            common_features = [col for col in expected_features if col in df_numeric.columns]  
            
            if not common_features:  
                st.error("Uploaded dataset does not contain any expected features for clustering.")  
            else:  
                df_numeric = df_numeric[common_features].dropna()  # Drop missing values  
                
                # Display info about numeric features  
                st.write("### Numeric Features Used for Clustering")  
                st.write(df_numeric.describe())  

                # Apply scaling if available  
                if scaler:  
                    df_scaled = scaler.transform(df_numeric)  
                    st.write("Data has been scaled using the provided scaler.")  
                else:  
                    df_scaled = df_numeric.to_numpy()  
                
                # Elbow Method to find the best number of clusters  
                st.header("Elbow Method for Optimal K")  
                inertia = []  
                k_range = range(1, 11)  # Check K values from 1 to 10  
                for k in k_range:  
                    kmeans = KMeans(n_clusters=k, random_state=42)  
                    kmeans.fit(df_scaled)  
                    inertia.append(kmeans.inertia_)  
                
                # Plotting Elbow graph  
                plt.figure(figsize=(8, 5))  
                plt.plot(k_range, inertia, marker='o')  
                plt.title('Elbow Method for Optimal K')  
                plt.xlabel('Number of clusters (K)')  
                plt.ylabel('Inertia')  
                plt.xticks(k_range)  
                st.pyplot(plt)  

                # Check for minimal data for clustering  
                if df_scaled.shape[0] < 2:  
                    st.error("Not enough data points to perform clustering.")  
                else:  
                    # Predict clusters  
                    best_k = 3  # Set a default; ideally this would be input from user or determined from elbow plot  
                    df['Cluster'] = kmeans.predict(df_scaled)  
                    
                    # Visualize Clusters using PCA  
                    pca = PCA(n_components=2)  
                    df_pca = pca.fit_transform(df_scaled)  
                    df_pca = pd.DataFrame(df_pca, columns=["PC1", "PC2"])  
                    df_pca["Cluster"] = df['Cluster']  
                    
                    # Plot PCA Results  
                    plt.figure(figsize=(10, 6))  
                    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Cluster", palette='viridis', ax=plt.gca())  
                    plt.title("Cluster Visualization using PCA")  
                    st.pyplot(plt)  
                    
                    # Display clustered data  
                    st.write("### Clustered Data")  
                    st.write(df)  
    except ImportError:  
        st.error("Missing optional dependency 'openpyxl'. Please install it using `pip install openpyxl` to read Excel files.")  
    except Exception as e:  
        st.error(f"Error loading file: {e}")  