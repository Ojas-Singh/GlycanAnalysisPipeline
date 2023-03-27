import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from lib import graph
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stat
import config
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.optimize import minimize_scalar


def pcawithG(frames,idx_noH,dim,name):
    G = np.zeros((len(frames),int(len(frames[0][np.asarray(idx_noH,dtype=int)])*(len(frames[0][np.asarray(idx_noH,dtype=int)])+1)/2)))
    for i in range(len(frames)):
        G[i]= graph.G_flatten(frames[i][np.asarray(idx_noH,dtype=int)])
    pca = PCA(n_components=dim)
    t = pca.fit_transform(G)
    PCA_components = pd.DataFrame(t)
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    fig = plt.figure()
    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(config.data_dir+name+'/output/PCA_variance.png',dpi=450)
    plt.cla()
    return PCA_components

def findmaxima(f):
    f = -1*f
    a = np.pad(f, (1, 1), mode='constant',
            constant_values=(np.amax(f), np.amax(f)))
    loc_min = []
    rows = a.shape[0]
    cols = a.shape[1]
    for ix in range(0, rows - 1):
        for iy in range(0, cols - 1):
                    if (a[ix, iy] < a[ix, iy + 1]
                        and a[ix, iy] < a[ix, iy - 1]
                        and a[ix, iy] < a[ix + 1, iy]
                        and a[ix, iy] < a[ix + 1, iy - 1]
                        and a[ix, iy] < a[ix + 1, iy + 1]
                        and a[ix, iy] < a[ix - 1, iy]
                        and a[ix, iy] < a[ix - 1, iy - 1]
                        and a[ix, iy] < a[ix - 1, iy + 1]):
                        temp_pos = (ix-1, iy-1)
                        loc_min.append(temp_pos)
    return loc_min

def nux(x,d,xmin):
    # return xmin + d*x/100
    return (x-xmin)*100/d

def nuy(y,d,ymin):
    # return ymin + d*y/100
    return (y-ymin)*100/d

def nx(x,d,xmin):
    return xmin + d*x/100
    # return (x-xmin)*100/d

def ny(y,d,ymin):
    return ymin + d*y/100
    # return (y-ymin)*100/d

def filterlow(data,k):
    x=data["0"].to_numpy()
    y=data["1"].to_numpy()
    s=pd.DataFrame([x,y])
    s=s.transpose()
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stat.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    l=[]
    for i in range(len(x)):
        l.append([f[int(nux(x[i],np.abs(xmax-xmin),xmin))-1][int(nuy(y[i],np.abs(ymax-ymin),ymin))-1],i])
    l.sort()
    idx_top=np.ones(len(x),dtype=bool) 
    idx_bottom = np.zeros(len(x),dtype=bool) 
    for i in range(int(k*len(x))):
        idx_top[l[i][1]] = False
        idx_bottom[l[i][1]] = True
    x= np.asarray(x)
    y= np.asarray(y)
    fig = plt.figure()
    ax = fig.gca()
    cfset = ax.contourf(xx, yy, f, cmap='Blues')
    ax.scatter(x[idx_top],y[idx_top],color="#78517C",s=.2)
    ax.scatter(x[idx_bottom],y[idx_bottom],color="#F65058FF",s=.2)
    ax.set_title("Conformation Filter (>10%)")
    plt.savefig('output/PCA_filter.png',dpi=450)
    # plt.show()
    # plt.clf()
    loc_min = findmaxima(f)
    loc=[]
    for i in loc_min:
        loc.append([f[i[0]][i[1]],(i[0],i[1])])
    loc.sort()
    xw=[]
    yw=[]
    for i in range(len(loc)):
        xw.append(nx(loc[i][1][0],np.abs(xmax-xmin),xmin))
        yw.append(ny(loc[i][1][1],np.abs(ymax-ymin),ymin))
    ini=[]
    numofcluster=len(loc)
    print(numofcluster)
    for i in range(numofcluster):
        ini.append([xw[i],yw[i]])
    popp=[]
    for i in ini:
        o=[]
        for j in range(len(data.iloc[:,0])):
            o.append([np.linalg.norm(np.asarray(i)-[data["0"].iloc[j],data["1"].iloc[j]]),data["i"].iloc[j]])
        o.sort()
        popp.append([i,o[0][1]])
    return idx_top,popp

def cluster(data,numofcluster,idx_top):
    from sklearn.metrics.pairwise import pairwise_distances_argmin
    s = data.loc[:, data.columns!='i'].to_numpy()
    # s = normalizetorsion(data)
    s = s[idx_top]
    # clustering = SpectralClustering(n_clusters=numofcluster).fit(s)
    clustering = MiniBatchKMeans(compute_labels=True,n_clusters=numofcluster,init='k-means++', max_iter=5000,batch_size=4).fit(s)
    # mbk_means_cluster_centers = np.sort(clustering.cluster_centers_, axis = 0)
    # clustering_labels = pairwise_distances_argmin(s, mbk_means_cluster_centers)
    # clustering =  DBSCAN(eps=8, min_samples=10).fit(s)
    # clustering = KMeans(n_clusters=numofcluster).fit(s)
    # clustering =  OPTICS(min_samples=100).fit(s)
    clustering_labels= clustering.labels_
    label = []
    k=0
    for i in idx_top:
        if i:
            label.append(clustering_labels[k])
            k+=1
        else:
            label.append(-1)

    data["cluster"] = label
    return data,label

def plot3dkde(data):
    x=data["0"].to_numpy()
    y=data["1"].to_numpy()
    # for i in range(len(data)):
    #     x.append(data.iloc[i,0])
    #     y.append(data.iloc[i,1])
    s=pd.DataFrame([x,y])
    s=s.transpose()
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stat.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    fig = plt.figure()
    mpl.rcParams['font.family'] = 'Cambria'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 2
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(xx, yy, f, cmap=plt.cm.YlGnBu_r)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title("PCA Space Density")
    ax.view_init(elev=-15, azim=-59)
    ax.contourf(xx, yy, f, zdir='z', offset=-0.8, cmap=plt.cm.YlGnBu_r)
    plt.show()
    # plt.savefig('output/PCA_KDE.png',dpi=450)

def generate_data(n_samples, n_dimensions):
    return np.random.rand(n_samples, n_dimensions)

def find_optimal_bandwidth(data):
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': np.linspace(0.1, 1.0, 30)},
                        cv=5)
    grid.fit(data)
    return grid.best_params_['bandwidth']

def kde_score(point, kde_model):
    return -kde_model.score_samples([point])

def find_closest_point_to_max_kde(data, kde_model):
    max_score = None
    max_score_point = None

    for point in data:
        score = kde_score(point, kde_model)
        if max_score is None or score < max_score:
            max_score = score
            max_score_point = point

    return max_score_point