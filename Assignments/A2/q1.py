# ## Important Library imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import matplotlib.transforms as mtransforms
from sklearn.decomposition import PCA


# ## Data loading
data = pd.read_csv("iris.data",names=["Sepal Length", "Sepal Width","Petal Length" ,"Petal Width", "Class Label"])
data_orig = data
print("Data columns")
print(data.info())
print("\n")


print("Initial rows of data")
print(data.head())
print("\n")



print("Data statistical properties")
print(data.describe())
print("\n")


print("Check for null values")
print(data.isna().sum())
print("\n")


print("Drop duplicates if any............")
data = data.drop_duplicates(keep='first')
print("\n")


print("Data dimension: ",data.shape)
print("\n")


# ##  Exploratory Data Analysis
print("Exploratory Data Analysis of Data..........")

ax = sns.countplot(x = data['Class Label'],label="Count",palette = ['tab:blue', 'tab:green', 'tab:orange'])
plt.show()


y = data["Class Label"] 
x = data.drop("Class Label",axis = 1 )

f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);
plt.show()


sns.set_style("whitegrid");
sns.pairplot(data,hue="Class Label",height=3,palette = ['tab:blue', 'tab:green', 'tab:orange']);
plt.show()


features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
# Separating out the features
x = data.loc[:, features].values
# Separating out the target
y = data.loc[:,['Class Label']].values


print("Principal Component Analysis..............................")
print("Dimension of Data before PCA: ",x.shape)
pca = PCA(n_components=0.95)
principalComponents = pca.fit_transform(x)
print("Dimension of Data after PCA: ",principalComponents.shape)
print("\n")

principalDf = pd.DataFrame(data = principalComponents
              ,columns = ['column_1', 'column_2'])

print("Print first few rows of new lower dimensional data................")
print(principalDf.head())
print("\n")


sns.scatterplot(x = principalDf["column_1"],y = principalDf["column_2"],hue=data["Class Label"],palette=['green','orange','red']);
plt.show()

data.loc[data['Class Label']=="Iris-setosa","Class Label"]=0
data.loc[data['Class Label']=="Iris-versicolor","Class Label"]=1
data.loc[data['Class Label']=="Iris-virginica","Class Label"]=2

np.random.seed(42) #to make the random initialisations deterministic

def computecluster_rep(X, idx, K,cluster_rep):
    N, n = X.shape
    for i in range(K):
        a = []
        a = X[np.where(idx==i)] #find all points who belong to cluster i
        if a.shape[0] != 0:
            cluster_rep[i] = a.sum(axis=0)/a.shape[0] #take the mean of all those datapoints who belong to cluster i and update the location of the cluster
        else:
            cluster_rep[i] = np.zeros(cluster_rep[i].shape)
    return cluster_rep
#the function updates the values of cluster representatives by taking mean of all data points belonging to that cluster



def findClosestcluster_rep(X, cluster_rep):
    N = X.shape[0]
    K = cluster_rep.shape[0]

    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(N):
        dist = np.sum(np.power(X[i] - cluster_rep,2),axis=1)#find distance of this point with all clusters 
        idx[i] = np.argmin(dist)#assign the index of the cluster nearest to the datapoint
    return idx
#the function finds the closest cluster representative for every N data points and assign cluster accordingly.
#idx stores cluster assignment


def fromdatainitialise(X,K):
    rand_idx = np.random.permutation(X.shape[0]) #pick k random datapoints as intial cluster locations
    cluster_rep = X[rand_idx[0:K], :]
    return cluster_rep
#the function randomly picks k data points from sample and initialises cluster representatives using these k data points


def runkMeans(X, cluster_rep, findClosestcluster_rep, computecluster_rep):
    K = cluster_rep.shape[0]
    idx = np.zeros(X.shape[0])
    prev_centroid = np.zeros((cluster_rep.shape[0],cluster_rep.shape[1]))
    curr = cluster_rep
    num_iter=0
    
    for i in range(100) and prev_centroid != curr:
        idx = findClosestcluster_rep(X, cluster_rep)
        cluster_rep_new = computecluster_rep(X, idx, K,curr)
        prev_centroid = curr
        curr = cluster_rep_new
        num_iter=num_iter+1
    return cluster_rep, idx,num_iter


def computeCost(X,cluster_rep,idx):
    sum = 0
    for i in range(X.shape[0]):
        sum += np.linalg.norm(X[i] - cluster_rep[idx[i]])**2
    return sum/X.shape[0]
#the function calculates Jclust 


x = principalDf.values
# Separating out the target
y = data.loc[:,['Class Label']].values


def class_entropy(data_y,num_labels):
    sum_i = 0
    for i in range(1,num_labels+1):
        x = np.sum(data_y==i)
        
        if x!=0:
            sum_i += (x/data_y.shape[0])*np.log2(x/data_y.shape[0])
    return -1*sum_i


def cluster_entropy(cluster,num_cluster):
    sum = 0
    for i in range(num_cluster):
        x = np.sum(cluster == i)
        if x!=0:
            sum += (x/cluster.shape[0])*np.log2(x/cluster.shape[0])
            
    return -1*sum


def mutual(y, cluster, num_cluster, num_label):
    sum_x= 0
    for i in range(num_cluster):
        
        for j in range(1,num_label+1):
            
            sum_l = 0
            for k in range(y.shape[0]):
                if y[k]==j and cluster[k] == i:
                      sum_l += 1
            if sum_l != 0:
                
                sum_x += ((sum_l/np.sum(cluster == i))*np.log2(sum_l/np.sum(cluster == i)))*(np.sum(cluster==i)/cluster.shape[0])
        
    return -1*sum_x


# ## K-Means on dataset after PCA
x = principalDf.values
# Separating out the target
y = data.loc[:,['Class Label']].values



print("Number of data points: ",y.shape[0])
print("\n")
print("K-Means on reduced dataset obtained after PCA")

cluster=[]
id_x = []
nmi = []
for K in range(2,9): 
    initial_cluster_rep = fromdatainitialise(x, K)
    cluster_rep, idx,num = runkMeans(x, initial_cluster_rep,findClosestcluster_rep,computecluster_rep)
    cost = computeCost(x,cluster_rep,idx)
    
    mut = mutual(y,idx,K,3)
    nmi.append(2* (class_entropy(y,3)-mut)/(class_entropy(y,3) + cluster_entropy(idx,K)))
    print("Normalized Mutual Information for k = ", K," ",nmi[K-2])
    
    cluster.append(cluster_rep)
    id_x.append(idx)
    print("Final Cost: {0:.10f}".format(cost))
    print("\n")
#run k means for randomly picking k data points as initial cluster representatives




num_row = 7
num_col = 1
fig, axes = plt.subplots(num_row, num_col, figsize=(8,32))
for i in range(7):
    ax = axes[i%7]
    plt.sca(ax)
    sns.scatterplot(x = principalDf["column_1"],y = principalDf["column_2"],style=id_x[i],hue=id_x[i],palette='viridis');
    ax.set_title("Number of Clusters: " + str(i+2))
    ax.set_xlabel("Column Vector: 1")
    ax.set_ylabel("Column Vector: 2")
plt.show()
#plot the cluster representatives


plt.xlabel("Number of clusters")
plt.ylabel("NMI Score")
plt.title("NMI vs K")
plt.plot(range(2,9),nmi)
plt.show()

# ## PCA with original dataset of 4 features
y = data["Class Label"].values
x = data.drop("Class Label",axis = 1).to_numpy()
print("\nK-Means on original dataset")


cluster=[]
id_x = []
nmi = []
for K in range(2,9): 
    initial_cluster_rep = fromdatainitialise(x, K)
    cluster_rep, idx,num = runkMeans(x, initial_cluster_rep,findClosestcluster_rep,computecluster_rep)
    cost = computeCost(x,cluster_rep,idx)
    
    mut = mutual(y,idx,K,3)
    nmi.append(2* (class_entropy(y,3)-mut)/(class_entropy(y,3) + cluster_entropy(idx,K)))
    print("Normalised Mutual Information for k = ",K," ",nmi[K-2])
    
    cluster.append(cluster_rep)
    id_x.append(idx)
    print("Final Cost: {0:.10f}".format(cost))
    print("\n")
#run k means for randomly picking k data points as initial cluster representatives


plt.xlabel("Number of clusters")
plt.ylabel("NMI Score")
plt.title("NMI vs K")
plt.plot(range(2,9),nmi)
plt.show()
