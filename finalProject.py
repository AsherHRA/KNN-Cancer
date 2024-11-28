import sklearn
import statsmodels as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import svm, datasets
from sklearn.svm import SVC
import seaborn as sns
from matplotlib.patches import Arc, Rectangle
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix 
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
import networkx as nx

def read_data(fname='wdbc.data'):
    columns=["ID","Diagnosis","radius_1","texture_1","perimeter_1","area_1","smoothness_1", "compactness_1","concavity_1","concave_points_1","symmetry_1","fractal_dimension_1","radius_2","texture_2","perimeter_2","area_2","smoothness_2","compactness_2","concavity_2","concave_points_2","symmetry_2","fractal_dimension_2","radius_3","texture_3","perimeter_3","area_3","smoothness_3","compactness_3","concavity_3","concave_points_3","symmetry_3","fractal_dimension_3"]
    df=pd.read_csv(fname, names=columns)
    return df

def clean_data(df):
    for colB in set(col.rsplit('_',1)[0] for col in df.columns if '_' in col):
        cols=[col for col in df.columns if col.startswith(colB)]
        df[colB]=df[cols].mean(axis=1)
        df.drop(columns=cols, inplace=True)
    return df

def first_look(df):
    sns.pairplot(df, hue='Diagnosis')
    plt.savefig("pca.png")
    #looks like concave points and texture are PCA

def lower_d(df):
    df2 = df[['ID', 'Diagnosis', 'texture', 'concave_points']].copy()
    return df2

def supvec(df):
    margin = 0.5
    step = 0.02
    label_map = {"M": 1, "B": 0}
    df['label'] = df['Diagnosis'].map(label_map)

    X = df[['texture', 'concave_points']]
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
    plt.xlabel('Texture')
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
    Description:
    '''
    dseries = get_distance_series(df, new_vec,
          features)
    indices = sorted(df.index, key = lambda
          i:dseries[i])[:k]
    return df.loc[indices]

def majority_vote(nearest_neighbors):
    label = sorted(nearest_neighbors['label'], key =
          lambda x: sum(nearest_neighbors['label'] ==
          x), reverse = True)[0]
    return label
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

def knn_predict_point(df, test_sample, k, features):
    nearest_neighbors = get_nearest_neighbors(df, test_sample, features, k)
    return majority_vote(nearest_neighbors)

def plot_knn_decision_boundary(X, y,k):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y)
 
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    mesh_points=np.c_[xx.ravel(), yy.ravel()]
    Z=knn.predict(mesh_points)
    Z=Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolor='k', cmap='coolwarm', s=100)
    plt.xlabel('Texture')
    plt.ylabel('Concave Points')
    plt.title(f'KNN Decision Boundary (k={k})')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Benign', 'Malignant'])
    plt.savefig("knn.png")
    plt.show()
    
def main():
    df=read_data()
    df2=clean_data(df)
    first_look(df)
    lowerDF=lower_d(df2)
    print(lowerDF)
    supvec(lowerDF)
    ftrs=['texture','concave_points'] 
    label_map = {"M": 1, "B": 0}
    df2['label'] = df2['Diagnosis'].map(label_map)
    X_train,X_test=train_test_split(df2,test_size=0.2, random_state=42)
    X_train_ftrs=X_train[ftrs]
    y_train=X_train['label']
    plot_knn_decision_boundary(X_train_ftrs, y_train,k=5)

    predictions=K_NN(X_train, X_test,5,ftrs)

    actual_labels = X_test['label'].values
    print(f"Predicted labels: {predictions}")
    print(f"Actual labels: {actual_labels}")

    accuracy = np.mean(np.array(predictions) == actual_labels)
    print(f"Accuracy: {accuracy:.2f}")

if __name__=="__main__":
    main()    