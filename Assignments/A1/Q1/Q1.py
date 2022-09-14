# # Q1: Regression Decision Tree Construction 
# ### Group Members: Pranav Mehrotra (20CS10085) and Saransh Sharma (20CS30065)

# #### Import Required Libraries. To install Seaborn type in command pip install seaborn in the terminal. 
# #### To run a cell press ctr + enter and press shift + enter to run a cell and move to next cell


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


# #### Read the CSV file in the from of a dataframe
# 

data = pd.read_csv("Train_B_Tree.csv")

# #### Primary Analysis of the data read. 
print("few entries of data: \n",data.head())
print("Shape of data is: ",data.shape)

# #### Check for duplicate data. Duplicate data doesn't help in training and so ideally needs to be analysed.

print("Duplicate data entries: \n",data[data.duplicated()==True])
data = data.drop_duplicates(keep='first')
print("After removing duplicate data, data shape is: ",data.shape)

#basic details of the data
print("\nBasic info about data: \n")
print("\n")
print(data.info())
print(data.describe())
print("Number of null entries:\n",data.isna().sum())


# #### Feature Matrix of the data

X = data.drop(['csMPa'],axis=1)
print("Feature matrix dimensions: ",X.shape)


# #### Output Vector of the data

y = data.csMPa
print("Output Label dimension: ",y.shape)

# #### To visualise the kernel density (peaks in the data) of the individual features and to visualise the distributions' median, 25 percentile and 75 percentile quartiles

data_vis = (X - X.mean()) / (X.std())              # standardization of data so that the data lies in the range [-1,1]
data_vis = pd.concat([y,data_vis],axis=1)
data_vis = pd.melt(data_vis,id_vars="csMPa",var_name="features",value_name='value')

print("Violin Plot of the data: ")
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", data=data_vis, inner="quart")#violinplot represents the distribution pattern of each of the features
plt.xticks(rotation=45);
plt.title("Violin Plot");
plt.savefig('Violin_Plot.jpg',bbox_inches='tight', dpi=300)
plt.show()

print("Box plot of the data: ")
plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", data=data_vis)
plt.xticks(rotation=45);
plt.title("Box Plot");
plt.savefig('Box_Plot.jpg',bbox_inches='tight', dpi=300)
plt.show()

# #### To visualize individual  features' kernel density and pairwise scatter plots to look for correlation between different features.

#print("Kernel density and scatter plot of the data: ")
fig = plt.figure(figsize=(25,25))
sns.set(style="white")
df = X.loc[:,['cement','slag','flyash','water','superplasticizer','coarseaggregate','fineaggregate','age']]
g = sns.PairGrid(df, diag_sharey=False,corner=True);
g.map_lower(sns.regplot,scatter_kws={"color": "red"}, line_kws={"color": "black"});
g.map_diag(sns.kdeplot, lw=3);
plt.title("Kernel and Scatter Plot");
plt.close(fig)
plt.savefig('Kernel_Scatter_Plot.jpg',bbox_inches='tight', dpi=300)


# #### To view how related two features are we plot the heatmap which actually represents the correlation of two features.  Correlation of 1 represents very strong positive correlation and correlation of -1 represents very strong negative correlation.
print("Heatmap of the data: ")
f,ax = plt.subplots(figsize=(11, 11))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);
plt.title("Heat Map");
plt.savefig('HeatMap.jpg',bbox_inches='tight', dpi=300)
plt.show()

# #### Let us individually analyse the feature pair with the highest absolute value of correlation to check if the feature vectors are linearly dependent.

print("Joint plot of water and superplasticizer columns: ")
sns.jointplot(x = df.loc[:,'water'],
              y = df.loc[:,'superplasticizer'],
              kind="reg",
              color="#ce1414",line_kws={"color": "black"});

plt.savefig('JointPlot.jpg',bbox_inches='tight', dpi=300)
plt.show()

# ### Training the model

# #### Dataframe Data contains the data read from csv

# #### The model is basically a tree containing nodes and edges. There exist two types of nodes in the tree. Leaf nodes and decision nodes. Leaf nodes are the nodes which would be helpful in case of predicting (outputting the final value) while decision nodes will represent set of conditions that would help us to make a decision about the predicted value.

class Node():
    def __init__(self, attribute=None, threshold=None, child_left=None, child_right=None, variance_red=None, level = 0, leaf_value=None):
        
        # data members corresponding to decision nodes
        self.attribute = attribute          #splitting attribute
        self.threshold = threshold          #threshold value of the splitting attribute
        self.child_left = child_left        #left child of the node
        self.child_right = child_right      #right child of the node
        self.variance_red = variance_red    #reduction in variance caused by splitting
        self.level = level                  #level of the node from top
        
        #data member corresponding to a leaf node
        self.leaf_value = leaf_value


# ##### Kindly note: We have used the same defination of node for both the types of node. A decision node would have leaf_value = None while a leaf_node would have a numerical leaf_value. This difference would help us to differentiate between a leaf node and a decision node.

# #### Class defination of a regression tree which will encapsulate all the functions and operation needed to construct a regression tree

class RegressionTree():
    def __init__(self, minimum_samples=2, max_depth=2): #constructor that will take two parameters
 
        self.root = None
        self.minimum_samples = minimum_samples #min number of samples that should be available for further splitting
        self.max_depth = max_depth #max- depth the tree is allowed to grow
        #these two parameters act as stopping conditions for the tree
        
    def variance_reduction(self, parent, left_branch, right_branch): #to find the reduction in variance
        
        fraction_left = len(left_branch) / len(parent) #fraction of original data in the left branch
        fraction_right = len(right_branch) / len(parent) #fraction of original data in right branch
        reduction_variance = np.var(parent) - (fraction_left * np.var(left_branch) + fraction_right * np.var(right_branch))
        #variance reduction is defined as variance of original data - weighted sum of variance of branches
        return reduction_variance
    
    def split_left_right(self, dataset, index, threshold): #to split the data in two branches depending upon attribute denoted by index and threshold
        
        left_dataset = np.array([x for x in dataset if x[index]<=threshold]) #left dataset contains all datapoints whose value of the specified attribute is less than or equal to threshold
        right_dataset = np.array([x for x in dataset if x[index]>threshold]) #right dataset contains all datapoints whose value of the specified attribute is more than threshold
        return left_dataset, right_dataset #return the two partitions
    
    def cal_leaf_node(self, y):#to calculate the value of a leaf node simple calculate mean of all the datapoints's y value at that node 
        
        leaf_val = np.mean(y)
        return leaf_val
                
    def get_best_feature(self, dataset, number_datapoints, number_attributes): # to get the feature and threshold with maximum variance reduction
        
        #initialise best_feature dictionary
        best_feature = {}
        best_feature["attribute"] = None
        best_feature["threshold"] = 0
        best_feature["dataset_left"] = None
        best_feature["dataset_right"] = None
        best_feature["variance_reduced"] = 0
        
        maximum_variance_reduction = -float("inf") #initialise the maximum variance reduction varaiable which will be sed to keep track of current maximum
        
        for features in range(number_attributes): #iterate over all features
            values = dataset[:, features] #extract the feature column
            unique_sorted_values = np.unique(values) #find sorted and unique values
            #possible threshold would be decided by taking mean of adjacent entries
            threshold_array = np.array([(unique_sorted_values[i]+unique_sorted_values[i+1])/2 for i in range(0,len(unique_sorted_values)-1)])

            for threshold in threshold_array: #iterate over all possible threshold values
                dataset_left, dataset_right = self.split_left_right(dataset, features, threshold) #split the data according to the feature and threshold
                
                if len(dataset_left)>0 and len(dataset_right)>0: #if two partitions are created
                    
                    dataset_y, dataset_left_y, dataset_right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]#extract target variable columns
                    
                    variance_reduced = self.variance_reduction(dataset_y, dataset_left_y, dataset_right_y)#calculate the reduction in variance caused by this split
                    if variance_reduced > maximum_variance_reduction:#if the variance reduction caused is more than the current maxima
                        #update the feature dictionary and store all relevant details
                        best_feature["attribute"] = features
                        best_feature["threshold"] = threshold
                        best_feature["dataset_left"] = dataset_left
                        best_feature["dataset_right"] = dataset_right
                        best_feature["variance_reduced"] = variance_reduced
                        maximum_variance_reduction = variance_reduced # update the current maxima and continue iterating over all possible combinations 
                        
        return best_feature # return the maximum variance reducing feature dictionary
    
    def construct_tree(self, dataset, current_depth=0): #function to construct tree
        
        X, y = dataset[:,:-1], dataset[:,-1] #extract feature matrix and target variable vector from dataset
        number_datapoints, number_attributes = np.shape(X) 
        current_best_feature = {} #to keep a track of the best splitting attribute for current node 
        
        if number_datapoints >= self.minimum_samples: #if the stopping conditions are not yet reached allow tree to grow as much as possible
            current_best_feature = self.get_best_feature(dataset, number_datapoints, number_attributes) #get the best splitting attribute for the node
            if current_best_feature["variance_reduced"]>0: #if the variance reduction is positive that is the data has been splitted in 2 fractions 
                subtree_left = self.construct_tree(current_best_feature["dataset_left"], current_depth+1) #call construct tree recursively for left subtree
                subtree_right = self.construct_tree(current_best_feature["dataset_right"], current_depth+1)#call construct tree recursively for rigjt subtree
                return Node(current_best_feature["attribute"], current_best_feature["threshold"],subtree_left, subtree_right, current_best_feature["variance_reduced"],current_depth)
                #return a node with left subtree as left child and right subtree as right child
        
        #in case the depth is exhausted or we are left with datapoint less than minimum_samples at a node we make that node a leaf node
        leaf_value = self.cal_leaf_node(y)#calculate the laef value 
        return Node(level = current_depth,leaf_value = leaf_value)#return the leaf node
    
    def construct_tree_depth(self, dataset, current_depth=0,max_depth=20): #function to construct tree with depth constraints
        
        X, y = dataset[:,:-1], dataset[:,-1] #extract feature matrix and target variable vector from dataset
        number_datapoints, number_attributes = np.shape(X) 
        current_best_feature = {} #to keep a track of the best splitting attribute for current node 
        
        if number_datapoints >= self.minimum_samples and current_depth <= self.max_depth: #if the stopping conditions are not yet reached
            current_best_feature = self.get_best_feature(dataset, number_datapoints, number_attributes) #get the best splitting attribute for the node
            if current_best_feature["variance_reduced"]>0: #if the variance reduction is positive that is the data has been splitted in 2 fractions 
                subtree_left = self.construct_tree_depth(current_best_feature["dataset_left"], current_depth+1) #call construct tree recursively for left subtree
                subtree_right = self.construct_tree_depth(current_best_feature["dataset_right"], current_depth+1)#call construct tree recursively for rigjt subtree
                return Node(current_best_feature["attribute"], current_best_feature["threshold"],subtree_left, subtree_right, current_best_feature["variance_reduced"],current_depth)
                #return a node with left subtree as left child and right subtree as right child
        
        #in case the depth is exhausted or we are left with datapoint less than minimum_samples at a node we make that node a leaf node
        leaf_value = self.cal_leaf_node(y)#calculate the laef value 
        return Node(level = current_depth,leaf_value = leaf_value)#return the leaf node
    
    
    def fit_model(self, X, y): #train a model to fit X and y
        
        dataset = np.concatenate((X, y), axis=1)#concatenate X and y to create the dataset
        self.root = self.construct_tree(dataset)#train the tree and store the final returned node in root
            
    def fit_model_depth(self, X, y): #train a model to fit X and y with depth constraint
        
        dataset = np.concatenate((X, y), axis=1)#concatenate X and y to create the dataset
        self.root = self.construct_tree_depth(dataset)
        
    def predict(self, data, decision_tree=None):#to predict target variable for a datapoint x
        
        #basic algo is to traverse the graph depending upon splitting feature and threshold values

        if decision_tree.leaf_value!=None: #if you have reached a leaf node simply return the value of the leaf
            return decision_tree.leaf_value
        
        attribute_value = data[decision_tree.attribute]#else extract the value at splitting attrribute column in x  
        if attribute_value <= decision_tree.threshold: # check if the value is less than or equal to threshold
            return self.predict(data, decision_tree.child_left)#traverse to the left subtree 
        else:
            return self.predict(data, decision_tree.child_right)#else traverse to the right subtree
        

    def post_pruning(self,root,decision_tree,error,depth,original_dataset):
        X_test = original_dataset[:,:-1]
        y_test = original_dataset[:,-1]
        height = find_height(root)
        if decision_tree.leaf_value is not None:
            return root
        
        if decision_tree.leaf_value is None: #if the node is a decision node
            decision_tree.leaf_value = self.cal_leaf_node(y)#assign the corresponding leaf value
            y_pred = [self.predict(x,root) for x in X_test]#make predictions on the new tree
            
            #base condition
            if (mean_error(y_pred,y_test,X_test.shape[0])) <= min(error):#if the tree is succesful in reducing the error
                error.append(mean_error(y_pred,y_test,X_test.shape[0]))
                changenode(root,decision_tree,None,None,self.cal_leaf_node(y))
                num = num_nodes(root)
                y_pred = [self.predict(x,root) for x in X_test]
                error_pred = mean_error(y_pred,y_test,X.shape[0])
                a = {'num':num,'error':error_pred}
                depth.append(a)
                return root#return the root which now has the particular node converted to leaf node
            
            #recursive defination
            else: 
                
                #in case truncating the branch doesn't help
                decision_tree.leaf_value=None 
            
                leftn = decision_tree.child_left
                rightn = decision_tree.child_right
                
                if decision_tree.child_left.leaf_value is None: 
                    leftn = self.post_pruning(root,decision_tree.child_left,error,depth,original_dataset)
                if decision_tree.child_right.leaf_value is None:
                    rightn = self.post_pruning(root,decision_tree.child_right,error,depth,original_dataset)#prune the right subtree recursively
                    
                
                return root#create a node with the pruned left subtree and pruned right subtree as left child and right child respectively 
    


# #### Helper function

def rec_height(decision_tree,height=[]): # to recursively calculate height of tree and store heights in height list
     if decision_tree.leaf_value is not None:
        height.append(decision_tree.level)
        return
     else:
        rec_height(decision_tree.child_left,height)
        rec_height(decision_tree.child_right,height)
        return 

def find_height(decision_tree): #to find height
    level = []
    rec_height(decision_tree,level)#level will contain length of individuall branches
    return max(level)#return the length of longest branch

# changenode will be used to change data members of an intermediate node and it would return the updated tree
def changenode(root,node, child_left,child_right,leaf_value):#to change the data member values of node in root 
     
    if root == node:#if the present root in node then change the value of the root
        root.child_left = child_left
        root.child_right = child_right
        root.leaf_value  = leaf_value
        return root
 
    if root.child_left is not None:#else travel through the tree to find the node whose values need to be changed
        root.child_left = changenode(root.child_left,node, child_left,child_right,leaf_value)
 
    elif root.child_right is not None:
        root.child_right = changenode(root.child_right,node, child_left,child_right,leaf_value)
         
    return root#return root of original tree
 
#to calculate number of nodes recursively
def num_nodes(decision_tree):
    l_nodes = 0
    r_nodes = 0
    if decision_tree.leaf_value is not None:
        return 1
    else:
        l_nodes = num_nodes(decision_tree.child_left)
        r_nodes = num_nodes(decision_tree.child_right)
        return l_nodes+r_nodes+1

def printtree(root):
    h = find_height(root)
    for i in range(1, h + 1):
        printlevel(root, i)
        print("\n")
 
 
def printlevel(root, level):
    if root is None:
        return root
    
    if level == 1:
        if root.leaf_value is not None:
            print("Leaf{",round(root.leaf_value,3),"}",end = '  ')
        else:
            print(columns[root.attribute][:3],'{',round(root.threshold,3),'}',end='  ')
    elif level > 1:
        printlevel(root.child_left, level - 1)
        printlevel(root.child_right, level - 1)



def mean_error(y_pred, y_actual, n): # to calculate root mean square error of the predictions
    
    sum=0
    for i in range(n): #iterate over all n datapoints
        sum = sum+(y_pred[i]-y_actual[i])**2 #add to sum the square of the difference between prediction and actual label
    
    sum = sum/n #take mean of the sum
    sum = np.sqrt(sum) #take square root of the error
    return sum


# #### To select the maximum efficient data split we randomly split the data in 10 sample with 70-30 split and select the distribution that gives us minimum error.

min_error=float("inf") #initialise the minimum error

print("Random Sampling of data: ")
for i in range(10): #repeat for 10 splits
    d = data.sample(frac = 1,random_state=42) #returns a randomly jumbles data
    
    div = int(0.7 * d.shape[0])#calculate 70 percent of the number of input datapoints
    d_train, d_test = d.iloc[:div,:], d.iloc[div:,:]#split the data into test and train
    
    d_train_x = d_train.iloc[:,:-1].values#set training data featutre matrix
    d_train_y = d_train.iloc[:,-1].values.reshape(-1,1)#set training data output label
    d_test_x = d_test.iloc[:,:-1].values#set test data feature matrix
    d_test_y = d_test.iloc[:,-1].values.reshape(-1,1)#set test data output label
    
    regress_tree = RegressionTree(minimum_samples=1)#construct a regression tree allow best tree depending upon training set
    regress_tree.fit_model(d_train_x,d_train_y)
    y_pred_train = [regress_tree.predict(x,regress_tree.root)  for x in d_train_x]#construct the predicted output variable vector
    
    if mean_error(y_pred_train,d_train_y,d_train_x.shape[0])<min_error: #if the error of this tree is less than the current minima
        min_error = mean_error(y_pred_train,d_train_y,d_train_x.shape[0]) #update current minima
        dataset_train = d_train#save the current training dataset
        dataset_test = d_test#save the current test set
        best_tree = regress_tree
        
print("Minimum training Error of the tree after random shuffling: ", min_error)
print("height of the minimum training error tree :",find_height(best_tree.root))
columns = data.iloc[:,:-1].columns #extract the columns of the training data
print("columns of data are: ",columns)


data_train_x = dataset_train.iloc[:,:-1].values #extract training data feature matrix after best splitting found
data_train_y = dataset_train.iloc[:,-1].values.reshape(-1,1) #extract training data target label vector after best splitting found
data_test_x = dataset_test.iloc[:,:-1].values #extract test data feature matrix after best splitting found
data_test_y = dataset_test.iloc[:,-1].values.reshape(-1,1) #extract test data target label vector after best splitting found


train = [] #to store training errors
test = [] #to store test error

train = [] #to store training errors
test = [] #to store test error
nod = []#to store number of nodes

print("Accuracy vs Depth and Number of nodes Analysis: ")
for i in range(1,21):
    regress_tree = RegressionTree(minimum_samples=1, max_depth=i)
    regress_tree.fit_model_depth(data_train_x,data_train_y)#train a tree of heights 1 to 20
    nod.append(num_nodes(regress_tree.root))
    y_pred_train = [regress_tree.predict(x,regress_tree.root) for x in data_train_x] #calculate training error
    train.append(mean_error(y_pred_train,data_train_y,data_train_x.shape[0]))
    
    y_pred_test = [regress_tree.predict(x,regress_tree.root) for x in data_test_x] #calculate test error
    test.append(mean_error(y_pred_test,data_test_y,data_test_x.shape[0]))

    
    
print("Error vs Depth plot: ")
fig = plt.figure(figsize = (8,10))
x = [i for i in range(1,21)] # plot how test and training error vary with depth of the tree
plt.plot(x,train);
plt.plot(x,test);
plt.xlabel("Depth");
plt.ylabel("Root Mean Squared Error");
plt.title("Error vs Depth");
plt.legend(['Train','Test']);

plt.savefig('Error_vs_depth.jpg',bbox_inches='tight', dpi=300)
plt.show()

print("Error vs Number of Nodes plot: ")
fig = plt.figure(figsize = (8,10))
plt.plot(nod,train);
plt.plot(nod,test);
plt.xlabel("Number of Nodes");
plt.ylabel("Root Mean Squared Error");
plt.title("Error vs Number of nodes");
plt.legend(['Train','Test']);

plt.savefig('Error_vs_NumNodes.jpg',bbox_inches='tight', dpi=300)
plt.show()


# #### We can clearly see the optimal depth of the tree should be around 9 but our present tree has depth 20 which leads to overfitting. The train error has reduced significantly but the tree fails to generalize well on unseen data. Thus, Post-pruning is required.

regress_tree = RegressionTree(minimum_samples=1)
regress_tree.fit_model(data_train_x,data_train_y)#train a tree 
        
y_original = [regress_tree.predict(x,regress_tree.root) for x in data_test_x] #calculate test error
err = mean_error(y_original,data_test_y,data_test_x.shape[0])

print('The Regression Decision Tree is now available in decision_tree.txt')

original_stdout = sys.stdout #original standard output

with open('decision_tree.txt', 'w') as f:
    sys.stdout = f # standard output to the file
    printtree(regress_tree.root)
    sys.stdout = original_stdout # Reset the standard output

tree = regress_tree.root
X = dataset_test.iloc[:,:-1].values
y = dataset_test.iloc[:,-1].values.reshape(-1,1)
dataset = np.concatenate((X, y), axis=1)

print("The tree overfits at height: ",find_height(regress_tree.root))
print("Error before pruning: ",mean_error(y_original,data_test_y,data_test_x.shape[0]))
print("Number of nodes before pruning: ",num_nodes(regress_tree.root))

d=[]
err=[err]
pruned = regress_tree.post_pruning(regress_tree.root,tree,err,d,dataset)
y_pred_test = [regress_tree.predict(data = x,decision_tree=pruned) for x in data_test_x] 

print("Error after pruning: ",mean_error(y_pred_test,data_test_y,data_test_x.shape[0]))
print("Number of nodes after pruning: ",num_nodes(pruned))


optimal_tree = RegressionTree(minimum_samples=1, max_depth=9)
optimal_tree.fit_model_depth(data_train_x,data_train_y)
print("Optimal number of nodes: ",num_nodes(optimal_tree.root))


print("Variation of Number of nodes with during pruning: ")
nodes=[d[0]['num']] 
err=[d[0]['error']]
for x in d[1:]:
    num,error = x["num"],x["error"]
    nodes.append(num)
    err.append(error)
plt.plot(nodes,err);
plt.xlabel("Number of Nodes");
plt.ylabel("Error");
plt.title("Variation of Error vs Node during pruning");
plt.savefig('Error_vs_nodes_pruning.jpg',bbox_inches='tight', dpi=300)
plt.show()


print('The Pruned Tree is now available in pruned.txt')
original_stdout = sys.stdout #original standard output
with open('pruned.txt', 'w') as f:
    sys.stdout = f # standard output to the file
    printtree(pruned)
    sys.stdout = original_stdout # Reset the standard output
