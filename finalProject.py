import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
from sklearn.linear_model import LinearRegression
'''
source: 
Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.
'''
def read_data(fname='wdbc.data'):
    """
    Description: reads the Breast Cancer Wisconsin dataset and maps the diagnosis to binary labels
    Parameters:
        fname: string, file path to the dataset. Default is 'wdbc.data'
    
    Returns:
        df: pandas DataFrame, cleaned dataset with diagnosis converted to binary labels (0 for Benign, 1 for Malignant)
    """
    columns=["ID","Diagnosis","radius_1","texture_1","perimeter_1","area_1","smoothness_1", "compactness_1","concavity_1","concave_points_1","symmetry_1","fractal_dimension_1","radius_2","texture_2","perimeter_2","area_2","smoothness_2","compactness_2","concavity_2","concave_points_2","symmetry_2","fractal_dimension_2","radius_3","texture_3","perimeter_3","area_3","smoothness_3","compactness_3","concavity_3","concave_points_3","symmetry_3","fractal_dimension_3"]
    df=pd.read_csv(fname, names=columns)
    label_map = {"M": 1, "B": 0}
    df['label'] = df['Diagnosis'].map(label_map)
    return df

def clean_data(df):
    """
    Description: cleans the dataset by averaging the feature columns with the same prefix and dropping individual feature columns
    Parameters:
        df: pandas DataFrame, dataset to be cleaned
    Returns:
        df: pandas DataFrame, cleaned dataset with averaged feature columns
    """
    for colB in set(col.rsplit('_',1)[0] for col in df.columns if '_' in col and col!='Diagnosis'):
        cols=[col for col in df.columns if col.startswith(colB)]
        df[colB]=df[cols].mean(axis=1)
        df.drop(columns=cols, inplace=True)
    return df

def scale_data(df, features):
    """
    Description: scales the numeric features in the dataset using StandardScaler
    Parameters:
        df: pandas DataFrame dataset containing the features
        features: list of feature column names to be scaled
    Returns:
        scaled_df: pandas DataFrame, scaled feature columns
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features, index=df.index)
    return scaled_df

def perform_pca(data, n_components):
    """
    Description: performs Principal Component Analysis (PCA) on the data and returns the transformed data and explained variance
    Parameters:
        data: pandas DataFrame of numpy array, data to perform PCA on
        n_components: integer, number of principal components to calculate
    Returns:
        pca (sklearn.decomposition.PCA): fitted PCA model
        principal_components: numpy array, data transformed into principal components
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    return pca, principal_components

def add_pca_columns(df, principal_components):
    """
    Description: adds PCA components to the DataFrame as new columns
    Parameters:
        df: pandas DataFrame, original DataFrame to add PCA columns to
        principal_components: numpy array, principal components to be added as new columns
    Returns:
        None
    """
    for i, pc in enumerate(principal_components.T):
        df[f"PC{i+1}"] = pc

def plot_pca_scatter(principal_components, labels):
    """
    Description: creates a scatter plot of the first two PCA components with class labels
    Parameters:
        principal_components: numpy array, data transformed into principal components
        labels: pandas Series or numpy array, class labels for each data point
    Saves:
        pca.png: image file, scatter plot of PCA components
    Returns:
        None
    """
    df_pca = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
    df_pca['Label'] = labels
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Label', palette='Set2', alpha=0.8)
    plt.title('PCA: First Two Principal Components')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid()
    plt.legend()
    plt.savefig('pca.png')
    plt.show()
  
def plot_regression_line(df, x_feature, y_feature):
    """
    Description: creates a scatter plot with a regression line fitted to the data
    Parameters:
        df: pandas DataFrame, dataset containing the features
        x_feature: string, feature to be plotted on the x-axis
        y_feature: string, feature to be plotted on the y-axis
    Saves:
        {x_feature}_vs_{y_feature}_regression.png: image file,catter plot with regression line
    Returns:
        None
    """
    sns.scatterplot(data=df, x=x_feature, y=y_feature, alpha=0.8)
    
    X = df[x_feature].values.reshape(-1, 1)  
    y = df[y_feature].values  
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    plt.plot(df[x_feature], y_pred, color='red', linewidth=2, label=f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
    r_value, _ = pearsonr(df[x_feature], df[y_feature])
    plt.annotate(f"r = {r_value:.2f}", 
                 xy=(0.05, 0.95), 
                 xycoords='axes fraction', 
                 fontsize=12, 
                 bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
    plt.legend()
    plt.title(f"Regression Line: {x_feature} vs {y_feature}")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.grid()
    plt.savefig(f"{x_feature}_vs_{y_feature}_regression.png")
    plt.show()

def lower_d(df):
    """
    Description: selects a subset of columns to reduce dimensionality
    Parameters:
        df: pandas DataFrame, dataset to reduce
    Returns:
        df2: pandas DataFrame, reduced dataset containing only a subset of features
    """
    df2 = df[['ID', 'label', 'fractal_dimension', 'concave_points']].copy()
    return df2

def supvec(df):
    """
    Description: plots the SVM decision boundary for two features: fractal dimension and concave points
    Parameters:
        df: pandas DataFrame, dataset containing the features and labels
    Saves:
        supvec.png: image file, SVM decision boundary plot
    Returns:
        None
    """
    margin = 0.1
    step = 0.02

    X = df[['fractal_dimension', 'concave_points']]
    y = df['label']

    x_min, x_max = X.iloc[:, 0].min() - margin, X.iloc[:, 0].max() + margin
    y_min, y_max = X.iloc[:, 1].min() - margin, X.iloc[:, 1].max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    model = svm.SVC(kernel='linear')
    model.fit(X, y)

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, shading='auto')
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.xlabel('Fractal Dimension')
    plt.ylabel('Concave Points')
    plt.title('SVM Decision Boundary')
    plt.savefig("supvec.png")
    plt.show()

def euclidean_distance(v1, v2):
    '''
    Description: calculates euclideandistance between 2 vectors
    Parameters:
        v1: numpy array, vector of points
        v2: numpy array, vector of points
    returns:
        np.sqrt(total): float, square root of the differences^2 between the vectors
    '''
    diff = v1 - v2
    total = np.sum(diff ** 2)
    return np.sqrt(total)

def get_distance_series(df, new_vec, features):
    '''
    Description: computes the Euclidean distance between a new vector and all rows in the dataset based on the specified features
    Parameters:
        df: pandas DataFrame, dataset containing feature vectors
        new_vec: pandas Sreies, vector to compare to each row of the dataset
        features: list of feature column names to use for distance calculation
    Returns:
        pandas.Series: pandas Series containing the Euclidean distance between the new vector and each row in the dataset, indexed by the rows of the DataFrame
    '''
    dlist = [euclidean_distance(df.loc[i, features].values, new_vec[features].values) for i in df.index]
    return pd.Series(dlist, index=df.index)

def get_nearest_neighbors(df, new_vec, features, k):
    '''
    Description:
        Returns the k nearest neighbors from the dataset based on Euclidean distance, including the label of each neighbor
    Parameters:
        df: pandas DataFrame, dataset containing feature vectors and labels
        new_vec: pandas Series, new vector to find the nearest neighbors for
        features: list of feature column names to use for distance calculation
        k: integer, number of nearest neighbors to return
    Returns:
        pandas.DataFrame: pandas DataFrame containing the k nearest neighbors, including their labels and features
    '''
    dseries = get_distance_series(df, new_vec, features)
    indices = sorted(df.index, key=lambda i: dseries[i])[:k]
    nearest_neighbors = df.loc[indices, ['label'] + features]
    return nearest_neighbors

def majority_vote(nearest_neighbors):
    '''
    Description: determines the majority label from the k nearest neighbors
    Parameters:
        nearest_neighbors: pandas DataFrame containing the k nearest neighbors, including their labels
    Returns:
        integer: predicted label based on the majority vote
    '''
    label = sorted(nearest_neighbors['label'], key =
          lambda x: sum(nearest_neighbors['label'] ==
          x), reverse = True)[0]
    return label

def tune_k(X, y, k_range=range(1, 21)):
    """
    Description: tunes the value of k for the KNN classifier using cross-validation
    Parameters:
        X: numpy array or DataFrame, feature matrix
        y: numpy array or Series, target labels
        k_range: range of k values to test (default is 1 to 20)
    Returns:
        best_k: integer, value of k with the highest accuracy
        best_score: float, highest cross-validation score achieved
        scores: list of accuracy scores for each k in k_range
    """
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
        scores.append(np.mean(cv_scores))  # Average accuracy for each k
    
    best_k = k_range[np.argmax(scores)]  # Find the k with the highest score
    best_score = np.max(scores)  # Best cross-validation score
    
    return best_k, best_score, scores

def K_NN(df, test_samples, k, features):
    """
    Description: performs K-Nearest Neighbors classification using helper functions
    Parameters:
        df: pandas DataFrame containing training data and labels
        test_samples: pandas DataFrame of sample data to classify
        k: integer, number of nearest neighbors to consider
        features: list of feature column names to use for distance calculation
    Returns:
        predictions: list of predicted labels for each test sample
    """
    predictions=[]
    for _, test_sample in test_samples.iterrows():
        nearest_neighbors = get_nearest_neighbors(df, test_sample, features, k)
        predicted_label = majority_vote(nearest_neighbors)
        predictions.append(predicted_label)
    return predictions

def evaluate_predictions(y_true, y_pred):
    """
    Description: evaluates the performance of KNN predictions by printing the confusion matrix and classification report
    Parameters:
        y_true: numpy array or pandas Series of true labels
        y_pred: numpy array or list of predicted labels
    Returns:
        None
    """
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def knn_predict_point(df, test_sample, k, features):
    """
    Description: classifies a single test sample using KNN and returns the predicted label
    Parameters:
        df: pandas DataFrame, dataset containing feature vectors and labels for training
        test_sample: pandas Series, test sample to classify
        k: integer, number of nearest neighbors to consider
        features: list of feature column names to use for distance calculation
    Returns:
        int: predicted label for the test sample
    """
    nearest_neighbors = get_nearest_neighbors(df, test_sample, features, k)
    return majority_vote(nearest_neighbors)

def plot_knn_decision_boundary(X, y,k):
    """
    Description: plots the decision boundary for the KNN classifier using the first two features of the dataset
    Parameters:
        X: pandas DataFrame, feature matrix (2D)
        y: pandas Series of target labels
        k: integer, number of nearest neighbors to consider
    Saves:
        knn.png: image file of knn decision boundary plot
    Returns:
        None
    """
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y)
 
    x_min, x_max = X.iloc[:, 0].min() - 0.1, X.iloc[:, 0].max() + 0.1
    y_min, y_max = X.iloc[:, 1].min() - 0.1, X.iloc[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    mesh_points=np.c_[xx.ravel(), yy.ravel()]
    Z=knn.predict(mesh_points)
    Z=Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolor='k', cmap='coolwarm', s=100)
    plt.ylabel('Concave Points')
    plt.xlabel('Fractal Dimension')
    plt.title(f'KNN Decision Boundary (k={k})')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Benign', 'Malignant'])
    plt.savefig("knn.png")
    plt.show()
    
def main():
    """
    Description: reads the dataset using `read_data()`, cleans the dataset using `clean_data()`,
        computes and prints the correlation matrix of the cleaned dataset, plots a regression line
         for the specified features using `plot_regression_line()`, encodes the target variable (Diagnosis)
        into numeric labels, plots the tuning curve for K in K-Nearest Neighbors (KNN) classification using
         `plot_k_tune()`., scales the selected features using `scale_data()`, performs Principal Component Analysis
         (PCA) on the scaled data and prints the explained variance ratio, adds PCA components to the original
         dataframe using `add_pca_columns()`, plots a scatter plot of the PCA components with labels using
         `plot_pca_scatter()`, applies the `lower_d()` function to the dataset and then performs support vector
         classification using `supvec()`, splits the dataset into training and testing sets, applies KNN with the best
         K (3), and evaluates the accuracy, evaluates and prints the KNN predictions and accuracy, plots the KNN decision
         boundary using `plot_knn_decision_boundary()`

    Notes:
        - This function assumes that `read_data()`, `clean_data()`, `scale_data()`, `perform_pca()`, 
          `add_pca_columns()`, `plot_pca_scatter()`, `lower_d()`, `supvec()`, `train_test_split()`, 
          `K_NN()`, `evaluate_predictions()`, and `plot_knn_decision_boundary()` are defined elsewhere in the code.
        - It is assumed that the dataset contains columns for "concave_points", "fractal_dimension", and "Diagnosis".
        - The function prints intermediate outputs like correlation matrix, explained variance ratio, and accuracy.
    """
    df = read_data()
    df = clean_data(df)
    correlation_matrix = df.corr()
    print(correlation_matrix)
    plot_regression_line(df, "concave_points", "fractal_dimension")
    label_map = {"M": 1, "B": 0}
    df['label'] = df['Diagnosis'].map(label_map)
    tune_k(df, df['label'])
    #3 is best for k
    features = ['concave_points','fractal_dimension'] 
    df_scaled = scale_data(df, features)
    print(df_scaled)
    pca, principal_components = perform_pca(df_scaled, n_components=2)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])], index=df_scaled.columns)
    print(loadings)

    add_pca_columns(df, principal_components)
    plot_pca_scatter(principal_components, df['label'])

    lower_df = lower_d(df)
    supvec(lower_df)

    ftrs = ['fractal_dimension', 'concave_points']
    X_train, X_test, y_train, y_test= train_test_split(df[features], df['label'],test_size=0.2, random_state=42)
    predictions = K_NN(df, X_test, k=3, features=ftrs)
    actual_labels = y_test.values
    accuracy = np.mean(np.array(predictions) == actual_labels)
    print(f"KNN Accuracy: {accuracy:.2f}")
    evaluate_predictions(actual_labels, predictions)
    plot_knn_decision_boundary(X_train, y_train, 3)

if __name__=="__main__":
    main()    