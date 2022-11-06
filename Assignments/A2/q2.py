#!/usr/bin/env python
# coding: utf-8

# ## Important imports


#for data handling
import pandas as pd
import numpy as np

#for plotting
import seaborn as sns
import matplotlib.pyplot as plt

#for svm and mlp classifier
import sklearn
from sklearn import svm
from sklearn import neural_network

pd.options.mode.chained_assignment = None  


# ## Dataset processing


#Dataframe reading with appropriate columns
data = pd.read_csv("iris.data",names=["Sepal Length", "Sepal Width","Petal Length" ,"Petal Width", "Class Label"])



#print First Five entries of the dataframe
print("\nShape of data: ",data.shape)
print(data.head())



#drop duplicate entries if any
data = data.drop_duplicates(keep='first')
print("\nShape of data after dropping duplicates: ",data.shape)



#extract x and y
y = data["Class Label"] 
x = data.drop("Class Label",axis = 1 )



#find basic statistical details about x
print(x.describe())


# ## Standard Scalar Normalization


#Standard Scalar Normalization of x
def normalize(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x_n = (x - mean)/std
    return x_n

x_norm = normalize(x)
col = x.columns

x = pd.DataFrame(x_norm, columns = col)
print(x.describe())


print("\nAfter normalization, mean of each column is close to 0 while standard deviation of each column is close to 1")


# ## Random splitting of data into test and training dataset


#re-concatenate x and y to form dataframe
df = pd.concat([x,y],axis=1)

d = df.sample(frac = 1,random_state=2) #returns a randomly jumbles data
div = int(0.8 * d.shape[0])#calculate 80 percent of the number of input datapoints

#categorical encoding
d.loc[data['Class Label']=="Iris-setosa","Class Label"]=0
d.loc[data['Class Label']=="Iris-versicolor","Class Label"]=1
d.loc[data['Class Label']=="Iris-virginica","Class Label"]=2

d_train, d_test = d.iloc[:div,:], d.iloc[div:,:]#split the data into test and train
    
d_train_x = d_train.iloc[:,:-1].values#set training data featutre matrix
d_train_y = d_train.iloc[:,-1].values#set training data output label
d_test_x = d_test.iloc[:,:-1].values#set test data feature matrix
d_test_y = d_test.iloc[:,-1].values#set test data output label
d_train_y = d_train_y.astype("int")
d_test_y = d_test_y.astype("int")

print("\nTraining Dataset shape: ",d_train.shape)

print("\nTesting Dataset shape: ",d_test.shape)


# ## SVM with linear kernel
print("\nSVM Classifier with Linear kernel...........")
clf = svm.SVC(kernel='linear',random_state=42)
clf.fit(d_train_x, d_train_y)
y_pred = clf.predict(d_test_x)
print("\nSVM with Linear kernel:",np.sum(y_pred == d_test_y.astype("int"))/d_test_y.shape)


# ## SVM with quadratic kernel 
print("\nSVM Classifier with Quadratic kernel.........")
clf = svm.SVC(kernel='poly',degree=2,random_state=42)
clf.fit(d_train_x, d_train_y)
y_pred = clf.predict(d_test_x)
print("\nSVM with quadratic kernel: ",np.sum(y_pred == d_test_y.astype("int"))/d_test_y.shape)


# ## SVM with radial basis function kernel
print("\nSVM CLassifer with radial basis function kernel........")
clf = svm.SVC(kernel='rbf',random_state=42)
clf.fit(d_train_x, d_train_y)
y_pred = clf.predict(d_test_x)
print("\nSVM with radial basis function kernel: ",np.sum(y_pred == d_test_y.astype("int"))/d_test_y.shape)


# ## MLP with 16 nodes hidden layers
print("\nMLP Classifer with one hidden layer of 16 nodes.........")
model_1 = neural_network.MLPClassifier(hidden_layer_sizes =(16),solver='sgd',batch_size=32, learning_rate_init = 0.001,max_iter=4000,shuffle=False,random_state=2 )
model_1.fit(d_train_x,d_train_y)
y_pred_one_layer = model_1.predict(d_test_x)
print("\nAccuracy of One 16 node hidden layer MLP Classifier: ",np.sum(y_pred_one_layer == d_test_y.astype("int"))/d_test_y.shape)
accuracy_1 = np.sum(y_pred_one_layer == d_test_y.astype("int"))/d_test_y.shape


# ## MLP with 256,16 nodes hidden layers
print("\nMLP Classifer with two hidden layer of 256 and 16 nodes.........")
model_2 = neural_network.MLPClassifier(hidden_layer_sizes =(256,16),solver='sgd',batch_size=32, learning_rate_init = 0.001,max_iter=4000,shuffle=False,random_state=2 )
model_2.fit(d_train_x,d_train_y)
y_pred_two_layer = model_2.predict(d_test_x)
print("\nAccuracy of 256,16 node hidden layer MLP Classifier: ",np.sum(y_pred_two_layer == d_test_y.astype("int"))/d_test_y.shape)
accuracy_2 = np.sum(y_pred_two_layer == d_test_y.astype("int"))/d_test_y.shape


# ## Best MLP Variant accuracy vs learning rate
if accuracy_1 > accuracy_2:
    print("\nModel 1 with one hidden layer has better accuracy")
else:
     print("\nModel 2 with two hidden layers has better accuracy")


x=[]
y = []
for i in [0.00001,0.0001,0.001,0.01,0.1]:
    if accuracy_1 > accuracy_2:
        model = neural_network.MLPClassifier(hidden_layer_sizes =(16),solver='sgd',batch_size=32, learning_rate_init = i,max_iter=4000 ,shuffle=False,random_state=2)
    else:
        model = neural_network.MLPClassifier(hidden_layer_sizes =(256,16),solver='sgd',batch_size=32, learning_rate_init = i,max_iter=4000,shuffle=False,random_state=2 )
    model.fit(d_train_x,d_train_y)
    y.append([model.n_iter_,model.loss_curve_])
    y_pred = model.predict(d_test_x)
    x.append(np.sum(y_pred == d_test_y.astype("int"))/d_test_y.shape)
    


# ## Best MLP variant Error vs learning rate curve
plt.title("Best MLP classifier Error vs number of iterations curve")
plt.plot(np.arange(1, y[0][0]+1),y[0][1]);
plt.xlabel("Number of iteration");
plt.ylabel("Loss curve learning rate = 0.00001");
plt.savefig("Learning_rate = 0.00001.png",bbox_inches='tight')
plt.show()




plt.title("Best MLP classifier Error vs number of iterations curve")
plt.plot(np.arange(1, y[1][0]+1),y[1][1]);
plt.xlabel("Number of iteration");
plt.ylabel("Loss curve learning rate = 0.0001");
plt.savefig("Learning_rate = 0.0001.png",bbox_inches='tight')
plt.show()



plt.title("Best MLP classifier Error vs number of iterations curve")
plt.plot(np.arange(1, y[2][0]+1),y[2][1]);
plt.xlabel("Number of iteration");
plt.ylabel("Loss curve learning rate = 0.001");
plt.savefig("Learning_rate = 0.001.png",bbox_inches='tight')
plt.show()



plt.title("Best MLP classifier Error vs number of iterations curve")
plt.plot(np.arange(1, y[3][0]+1),y[3][1]);
plt.xlabel("Number of iteration");
plt.ylabel("Loss curve learning rate = 0.01");
plt.savefig("Learning_rate = 0.01.png",bbox_inches='tight')
plt.show()



plt.title("Best MLP classifier Error vs number of iterations curve")
plt.plot(np.arange(1, y[4][0]+1),y[4][1]);
plt.xlabel("Number of iteration");
plt.ylabel("Loss curve learning rate = 0.1");
plt.savefig("Learning_rate = 0.1.png",bbox_inches='tight')
plt.show()



plt.title("Best MLP classifier Accuracy vs learning rate curve")
plt.plot(['0.00001','0.0001','0.001','0.01','0.1'],x);
plt.xlabel("Learning Rate");
plt.ylabel("Accuracy");
plt.savefig("Accuracy vs learning_rate.png",bbox_inches='tight')
plt.show()



#find the best of the two variants of MLP classifier
if accuracy_1> accuracy_2:
    model = model_1
    accuracy = accuracy_1
else:
    model = model_2
    accuracy = accuracy_2



print("\nFollowing is the best variant of MLP classifier: ")
print(model)



# ## Backward elimination
features = d.columns
features = features[:-1]
print(features)
d_train_b = d_train
d_test_b = d_test
print("\nAccuracy before backward elimination: ", accuracy)
print("\nNumber of features before backward elimination: ", features.shape[0])


print("\nBackward Elimination in progress: ")
print("\nFeatures left in each iteration: ")

while features.size >= 2: #do while you have more than 2 features i.e. we have features that can be removed
    max_accuracy = 0    #max accuracy initialised to 0
    f = "null"          #flag to find the index to be dropped 
     
    #find the faeture f of x with maximum E(x-f) where E is accuracy
    for i in features:
        d_train_elim = d_train_b.drop(i,axis=1)
        d_test_elim = d_test_b.drop(i,axis=1)
        
        d_train_x = d_train_elim.iloc[:,:-1].values#set training data featutre matrix
        d_train_y = d_train_elim.iloc[:,-1].values#set training data output label
        d_test_x = d_test_elim.iloc[:,:-1].values#set test data feature matrix
        d_test_y = d_test_elim.iloc[:,-1].values#set test data output label

        d_train_y = d_train_y.astype("int")
        d_test_y = d_test_y.astype("int")
        
        model.fit(d_train_x,d_train_y)
        y_pred = model.predict(d_test_x)
        temp_accuracy = np.sum(y_pred == d_test_y.astype("int"))/d_test_y.shape
        
        #find the feature that brings the maximum accuarcy after it is removed from the data
        if max_accuracy < temp_accuracy:
            f = i
            max_accuracy = temp_accuracy

    #if the accuracy found after removing a feature is greater or equal to the accuracy of the original model
    #update the accuracy and permanently remove the columns
    #continue the process for the new set of features
    
    if max_accuracy >= accuracy:
        d_train_b.drop(f,axis=1,inplace=True)
        d_test_b.drop(f,axis=1,inplace=True)
        features = d_train_b.columns
        features = features[:-1] #remove the class label column
        accuracy = max_accuracy
        
    #in case no such feature exists whose reduction brings improvement in accuracy terminate the process
    elif max_accuracy < accuracy:
        break
    print(features)     #print features after every iteration
    


print("\nAccuracy after backward elimination: ", accuracy)
print("\nNumber of features after backward elimination: ", features.shape[0])
print("\nColumns after backward Elimination: ",d_train_b.columns)



# ## Ensemble Learning
div = int(0.8 * d.shape[0])#calculate 80 percent of the number of input datapoints
d_train, d_test = d.iloc[:div,:], d.iloc[div:,:]#split the data into test and train
    
d_train_x = d_train.iloc[:,:-1].values#set training data featutre matrix
d_train_y = d_train.iloc[:,-1].values#set training data output label
d_test_x = d_test.iloc[:,:-1].values#set test data feature matrix
d_test_y = d_test.iloc[:,-1].values#set test data output label
d_train_y = d_train_y.astype("int")
d_test_y = d_test_y.astype("int")

#chose the best MLP model from part 3
if accuracy_1> accuracy_2:
    model = model_1
    accuracy = accuracy_1
else:
    model = model_2
    accuracy = accuracy_2

# SVM with radial basis function kernel
clf = svm.SVC(kernel='rbf',random_state=42)
clf.fit(d_train_x, d_train_y)
y_pred_svm = clf.predict(d_test_x)
print("\nSVM with quadratic kernel Accuracy: ",np.sum(y_pred_svm == d_test_y.astype("int"))/d_test_y.shape)


#SVM with quadratic kernel
clf = svm.SVC(kernel='poly',degree=2,random_state=42)
clf.fit(d_train_x, d_train_y)
y_pred_svm_q = clf.predict(d_test_x)
print("\nSVM with quadratic kernel Accuracy: ",np.sum(y_pred_svm_q == d_test_y.astype("int"))/d_test_y.shape)

#MLP classifier 
model.fit(d_train_x,d_train_y)
y_mlp = model.predict(d_test_x)
print("\nMLP Classifier Accuracy: ",np.sum(y_mlp == d_test_y.astype("int"))/d_test_y.shape)


c = np.stack((y_pred_svm, y_pred_svm_q,y_mlp),axis=0) #make a 2 dimensional array by stacking all individual predicted list as rows
print("\nShape of array after stacking: ",c.shape)

print("\nPredictions before ensemble learning: ")
print(c)


c_d = pd.DataFrame(c) #convert c to dataframe


print("\nPredictions after ensemble learning: ")
print(c_d.mode().loc[0].astype("int"))

x = c_d.mode().loc[0].astype("int") #find the mode of columns of c and convert datatype to int

print("\nAccuracy after ensemble learning: ",np.sum(x == d_test_y.astype("int"))/d_test_y.shape)
