import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from lib import graph
import matplotlib.pyplot as plt
import config
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.optimize import minimize
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def pcawithG(frames,idx_noH,dim,name):
    G = np.zeros((len(frames),int(len(frames[0][np.asarray(idx_noH,dtype=int)])*(len(frames[0][np.asarray(idx_noH,dtype=int)])-1)/2)))
    for i in range(len(frames)):
        G[i]= graph.G_flatten(frames[i][np.asarray(idx_noH,dtype=int)])
    pca = PCA(n_components=dim)
    t = pca.fit_transform(G)
    PCA_components = pd.DataFrame(t)
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    # Find the index where cumulative explained variance is greater or equal to config.explained_variance
    # n_dim = np.argmax(cum_sum_eigenvalues >= config.explained_variance) + 1
    n_dim = config.hard_dim
    fig = plt.figure()
    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(config.data_dir+name+'/output/PCA_variance.png',dpi=450)
    plt.cla()
    eigenvalues = pca.explained_variance_
        
    # Step 3: Find the number of components with eigenvalues > 1
    components_to_retain = sum(eigenvalues > 1)

    print(f"Number of components to retain: {components_to_retain}")

    # Step 4 (optional): Plot the eigenvalues
    plt.plot(eigenvalues, 'o-')
    plt.title('Scree Plot')
    plt.xlabel('Component number')
    plt.ylabel('Eigenvalue')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.savefig(config.data_dir+name+'/output/PCA_eigen.png',dpi=450)
    return PCA_components,n_dim


def find_optimal_bandwidth(data):
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': np.linspace(0.1, 1.0, 30)},
                        cv=5,n_jobs=-1)
    grid.fit(data)
    return grid.best_params_['bandwidth']

def kde_score(point, kde_model):
    return -kde_model.score_samples([point])

def kde_score_vec(points, kde_model):
    return -kde_model.score_samples(points)

def find_closest_point_to_max_kde(cluster_data, kde_model):
    bounds = [(min(cluster_data[:, dim]), max(cluster_data[:, dim])) for dim in range(cluster_data.shape[1])]
    def neg_kde_score(x):
        return kde_score_vec([x], kde_model)[0]
    result = minimize(neg_kde_score, cluster_data.mean(axis=0), bounds=bounds, method='L-BFGS-B')
    return result.x

def best_clustering(n,data):
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm.fit(data)
    labels = gmm.predict(data)
    centroids = gmm.means_
    return labels,centroids


def kde_c(n_clusters,pca_df,selected_columns):
    popp=[]
    pcanp = pca_df[selected_columns].to_numpy()
    kde_centers=[]
    for i in range(n_clusters):
        data = pca_df.loc[pca_df["cluster"] ==str(i)]
        data = data[selected_columns].to_numpy()
        # Scale data
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        bandwidth = find_optimal_bandwidth(data)
        kde_model = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde_model.fit(data)
        closest_point = find_closest_point_to_max_kde(data, kde_model)
        kde_centers.append(closest_point)
    for i in range(n_clusters):
        df0 = pca_df.loc[pca_df["cluster"] ==str(i)]
        o=[]
        # pp=clustering.cluster_centers_[i]
        pp = kde_centers[i]
        for j in range(len(df0.iloc[:,0])):
            o.append([np.linalg.norm(np.asarray(pp[:])-pcanp[j]),df0["i"].iloc[j],df0["cluster"].iloc[j]])
        o.sort()
        popp.append([o[0][1],o[0][2]])
    sizee=[]
    for i in range(len(popp)):
        popp[i].append(100*float(len(pca_df.loc[(pca_df['cluster']==str(i)),['cluster']].iloc[:]['cluster'].to_numpy())/len(pca_df.iloc[:]['cluster'].to_numpy())))
    return popp

def plot_Silhouette(pca_df,name,n_dim):
    selected_columns = [i for i in range(1, n_dim)]
    x = [x for x in range(2,12)]
    y = []
    for i in x:
        y.append(metrics.silhouette_score(pca_df[selected_columns],
                                        best_clustering(i,pca_df[selected_columns])[0]
                ,metric='euclidean'))
    fig = plt.figure()
    plt.plot(x, y)
    plt.scatter(x,y)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters [n_dim : '+str(n_dim)+' ]')
    plt.tight_layout()
    plt.savefig(config.data_dir+name+'/output/Silhouette_Score.png',dpi=450)
    plt.cla()
    n_clus = find_peaks(y)[0]+2
    s_scores = y
    return n_clus,s_scores

def find_peaks(array):
    peaks = []

    if len(array) >= 3:  # Peaks only exist in arrays of 3 or more elements
        # Check elements excluding the first and last ones
        for i in range(1, len(array)-1):
            if array[i] > array[i-1] and array[i] > array[i+1]:
                peaks.append((i, array[i]))

    # Sort by peak value in descending order, then extract indices
    peaks = [i for i, _ in sorted(peaks, key=lambda x: x[1], reverse=True)]

    return peaks