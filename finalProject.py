import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
    columns=["ID","Diagnosis","radius_1","texture_1","perimeter_1","area_1","smoothness_1", "compactness_1","concavity_1","concave_points_1","symmetry_1","fractal_dimension_1","radius_2","texture_2","perimeter_2","area_2","smoothness_2","compactness_2","concavity_2","concave_points_2","symmetry_2","fractal_dimension_2","radius_3","texture_3","perimeter_3","area_3","smoothness_3","compactness_3","concavity_3","concave_points_3","symmetry_3","fractal_dimension_3"]
    df=pd.read_csv(fname, names=columns)
    label_map = {"M": 1, "B": 0}
    df['label'] = df['Diagnosis'].map(label_map)
    return df

def clean_data(df):
    for colB in set(col.rsplit('_',1)[0] for col in df.columns if '_' in col and col!='Diagnosis'):
        cols=[col for col in df.columns if col.startswith(colB)]
        df[colB]=df[cols].mean(axis=1)
        df.drop(columns=cols, inplace=True)
    return df

def scale_data(df, features):
    """
    Scales the numeric features in the dataset.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features, index=df.index)
    return scaled_df


def perform_pca(data, n_components):
    """
    Performs PCA and returns both the PCA object and transformed data.
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    return pca, principal_components

def add_pca_columns(df, principal_components):
    """
    Adds PCA components to the dataframe for further analysis.
    """
    for i, pc in enumerate(principal_components.T):
        df[f"PC{i+1}"] = pc

def plot_pca_scatter(principal_components, labels):
    """
    Scatter plot of the first two PCA components.
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


def compare(df):
    sns.pairplot(df, hue='label')
    plt.savefig("compare.png")
  
def plot_regression_line(df, x_feature, y_feature):
    sns.scatterplot(data=df, x=x_feature, y=y_feature, alpha=0.8)
    
    X = df[x_feature].values.reshape(-1, 1)  
    y = df[y_feature].values  
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    plt.plot(df[x_feature], y_pred, color='red', linewidth=2, label=f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
    plt.legend()
    plt.title(f"Regression Line: {x_feature} vs {y_feature}")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.grid()
    plt.savefig(f"{x_feature}_vs_{y_feature}_regression.png")
    plt.show()

def lower_d(df):
    df2 = df[['ID', 'label', 'fractal_dimension', 'concave_points']].copy()
    return df2

def supvec(df):
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
    Description: measures distance between points
    Parameters:
        v1: vector of points
        v2: vector of points
    returns:
        np.sqrt(total): integer that is the squre root of the differences^2 between the vectors
    '''
    diff = v1 - v2
    total = np.sum(diff ** 2)
    return np.sqrt(total)

def get_distance_series(df, new_vec, features):
    '''
    Description: measures distance froma new point
    Parameters:
        df: pandas DataFrame
        new_vec: new vector
        features: 
    Returns:
        pd.Series(dlist, index=df.index): pandas Series
    '''
    dlist = [euclidean_distance(df.loc[i, features].values, new_vec[features].values) for i in df.index]
    return pd.Series(dlist, index=df.index)

def get_nearest_neighbors(df, new_vec, features, k):
    '''
    Description: Returns the k nearest neighbors, including the label.
    '''
    dseries = get_distance_series(df, new_vec, features)
    indices = sorted(df.index, key=lambda i: dseries[i])[:k]
    nearest_neighbors = df.loc[indices, ['label'] + features]
    return nearest_neighbors

def majority_vote(nearest_neighbors):
    label = sorted(nearest_neighbors['label'], key =
          lambda x: sum(nearest_neighbors['label'] ==
          x), reverse = True)[0]
    return label

def tune_k(X, y, k_range=range(1, 21)):
    """
    Tunes the value of k for the KNN classifier using cross-validation.
    
    Parameters:
        X: Features (numpy array or DataFrame)
        y: Labels (numpy array or Series)
        k_range: Range of k values to test (default is 1 to 20)
    
    Returns:
        best_k: The value of k with the highest accuracy
        best_score: The highest cross-validation score achieved
        scores: List of accuracy scores for each k in k_range
    """
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
        scores.append(np.mean(cv_scores))  # Average accuracy for each k
    
    best_k = k_range[np.argmax(scores)]  # Find the k with the highest score
    best_score = np.max(scores)  # Best cross-validation score
    
    return best_k, best_score, scores

def plot_k_tune(df):
    ftrs = ['texture', 'concave_points']  # Example feature columns
    X = df[ftrs]  # Features
    y = df['label']  # Labels
    
    best_k, best_score, scores = tune_k(X, y)
    
    print(f"Best k: {best_k}")
    print(f"Best cross-validation score: {best_score:.2f}")
    
    # Plot the accuracy scores for each k
    plt.plot(range(1, 21), scores)
    plt.xlabel('k')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Tuning k for KNN')
    plt.show()

def K_NN(df, test_samples, k, features):
    """
    Description: performs K-Nearest Neighbors classification using helper functions
    Parameters:
        df: DataFrame containing training data and labels
        test_samples: DataFrame of sample data to classify
        k: Number of nearest neighbors to consider
        features: List of feature column names to use for distance calculation
    Returns:
        The predicted label for the test_sample.
    """
    predictions=[]
    for _, test_sample in test_samples.iterrows():
        nearest_neighbors = get_nearest_neighbors(df, test_sample, features, k)
        predicted_label = majority_vote(nearest_neighbors)
        predictions.append(predicted_label)
    return predictions

def evaluate_predictions(y_true, y_pred):
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def knn_predict_point(df, test_sample, k, features):
    nearest_neighbors = get_nearest_neighbors(df, test_sample, features, k)
    return majority_vote(nearest_neighbors)

def plot_knn_decision_boundary(X, y,k):
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
    df = read_data()
   
    df = clean_data(df)
    correlation_matrix = df.corr()
    print(correlation_matrix)
    plot_regression_line(df, "concave_points", "fractal_dimension")
    label_map = {"M": 1, "B": 0}
    df['label'] = df['Diagnosis'].map(label_map)

    #plot_k_tune(df)
    #3 is best for k
    features = ['concave_points','fractal_dimension'] 
    df_scaled = scale_data(df, features)
    print(df_scaled)
    pca, principal_components = perform_pca(df_scaled, n_components=2)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])], index=df_scaled.columns)

    print(loadings)

    # Select important features (those with highest absolute loadings)
    #print("Important features based on PCA loadings:", important_features)
    #add_pca_columns(df, principal_components)
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