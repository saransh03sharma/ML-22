# # Q2: Naive Bayes Classifier
# ### Group Members: Pranav Mehrotra (20CS10085) and Saransh Sharma (20CS30065)

# Import Required Libraries. To install Seaborn type in command pip install seaborn in the terminal.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce

# Read the CSV file in the from of a dataframe
# eda_data = pd.read_csv("Q2/df_train_cleaned.csv",index_col=0)
eda_data = pd.read_csv("Q2/Train_B_Bayesian.csv")

# Category plot for the data
eda_catplot = eda_data.copy()
# eda_catplot.loc[eda_catplot['gender']==1,'gender'] ="Male"
# eda_catplot.loc[eda_catplot['gender']==2,'gender'] ="Female"
eda_catplot.loc[eda_catplot['is_patient']==1,'is_patient']="Liver"
eda_catplot.loc[eda_catplot['is_patient']==2,'is_patient']="Non-Liver"
sns.catplot(x ="gender", hue ="is_patient",kind ="count", data = eda_catplot)
plt.savefig('Q2/catplot.jpg',bbox_inches='tight', dpi=300)

eda_data.loc[eda_data['gender']=="Male",'gender'] =1
eda_data.loc[eda_data['gender']=="Female",'gender'] =2
eda_y = eda_data.is_patient
eda_x = eda_data.drop("is_patient",axis=1)

# Violin plot, to get the idea of the distribution of the data across different features
eda_data_dia = eda_y.astype("float")
eda_data = eda_x.astype("float")
eda_data_n_2 = (eda_data - eda_data.mean().astype("float")) / (eda_data.std().astype("float"))
eda_data = pd.concat([eda_y,eda_data_n_2],axis=1)
eda_data = pd.melt(eda_data,id_vars="is_patient",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="is_patient", data=eda_data,split=True, inner="quart")
plt.xticks(rotation=45)
plt.savefig('Q2/Violinplot.jpg')
plt.show()

# Box Plot for the data
eda_data_dia = eda_y
eda_data = eda_x
eda_data_n_2 = (eda_data - eda_data.mean()) / (eda_data.std())
eda_data = pd.concat([eda_y,eda_data_n_2],axis=1)
eda_data = pd.melt(eda_data,id_vars="is_patient",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="is_patient", data=eda_data)
plt.xticks(rotation=45)
plt.show()

f,ax = plt.subplots(figsize=(11, 11))
sns.heatmap(eda_x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
plt.savefig('Q2/heatmap.jpg',bbox_inches='tight', dpi=300)

########################################################################
########################################################################
## Apply the Naive Bayes Classifier on the data and find the accuracy ##
########################################################################
########################################################################

# Read the CSV file in the from of a dataframe
data = pd.read_csv("Train_B_Bayesian.csv")

df = data

# Shuffling data, to get a random distribution of data
df = df.sample(frac=1,random_state=47)

# Reset the index of the dataframe, after shuffling
df = df.reset_index(drop=True)

# Encode the categorical feature using the Ordinal encoder
encoder= ce.OrdinalEncoder(cols=['gender'],return_df=True,
                           mapping=[{'col':'gender',
'mapping':{'Male':1,'Female':2}}])
df = encoder.fit_transform(df)

# Moving the gender column to the last column of the dataframe,for ease of use
genders = df['gender']
df3 = df.drop(columns = ['gender'])
df3['gender'] = genders
df = df3

# Splitting the data into training and testing data
train_ind = int(0.7*df.shape[0])
df_train = df.iloc[:train_ind] 
df_test = df.iloc[train_ind:] 

# Finding Outliers, using mean and standard deviation
outliers = {}
means = list(df_train.mean())
stds = list(df_train.std())
j=0
for i in df_train.columns:
    if(i=='is_patient'):
        break
    # Data values should be less than, (2*mean + 5*std)
    upper_bound = 2*means[j] + 5*stds[j]
    # Finding the outliers
    l_out = list(df_train.index[(df_train[i]>upper_bound)])
    outliers[i] = l_out
    j+=1
# Printing the found outliers
print("\nOutliers found for each feature(indexes based on shuffled data):")
for x in outliers.items():
    print(x)
print("")

# Dropping the outliers from the training data
final_list = []
for i in outliers.keys():
    final_list.extend(outliers[i])
final_list = sorted(set(final_list))
df_train = df_train.drop(final_list)
# Reset the index of the dataframe, after dropping the outliers
df_train = df_train.reset_index(drop=True)


# Function to calculate the probability of a feature, assumming a normal distribution
def normal_distribution(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

# Function to calculate the probability of a given data, to be in a particular category, assuming all the features are independent
def calc_proba(inp,prob_patient,prob_gender,cat_means,cat_stds):
    prob = prob_patient
    prob *= prob_gender[inp['gender']]
    for i in df_train.columns:
        if(i=='is_patient'):
            break
        else:
            prob *= normal_distribution(inp[i],cat_means[i],cat_stds[i])
    return prob*1000 # Returning the probability*1000, to avoid underflow

# Function to train the Naive Bayes Classifier on a given training data, and also do laplace correction(if required)
def initalise(df_p_train,laplace_factor):
    prob_patient = {}
    prob_patient[1] = 0
    prob_patient[2] = 0
    
    # Finding the probability of each category
    prob_patient[1] = df_p_train[df_p_train['is_patient']==1].shape[0]
    prob_patient[2] = df_p_train[df_p_train['is_patient']==2].shape[0]
    prob_patient[1]/=len(df_p_train)
    prob_patient[2]/=len(df_p_train)

    # Distribution of dataset into diffrent categories
    df_train_cat_1 = df_p_train[df_p_train['is_patient']==1]
    df_train_cat_2 = df_p_train[df_p_train['is_patient']==2]

    # Probability of categorical features given the category, and also do laplace correction(if required)
    prob_gender_1 = {}
    prob_gender_1[1] = (df_train_cat_1[df_train_cat_1['gender']==1].shape[0]+1*laplace_factor)
    prob_gender_1[2] = (df_train_cat_1[df_train_cat_1['gender']==2].shape[0]+1*laplace_factor)
    prob_gender_2 = {}
    prob_gender_2[1] = (df_train_cat_2[df_train_cat_2['gender']==1].shape[0]+1*laplace_factor)
    prob_gender_2[2] = (df_train_cat_2[df_train_cat_2['gender']==2].shape[0]+1*laplace_factor)
    prob_gender_1[1]/=(df_train_cat_1.shape[0]+2*laplace_factor)
    prob_gender_1[2]/=(df_train_cat_1.shape[0]+2*laplace_factor)
    prob_gender_2[1]/=(df_train_cat_2.shape[0]+2*laplace_factor)
    prob_gender_2[2]/=(df_train_cat_2.shape[0]+2*laplace_factor)
    # Mean and stds of diffrent features, given the category
    cat_1_means = df_train_cat_1.mean()
    cat_1_stds = df_train_cat_1.std()
    cat_2_means = df_train_cat_2.mean()
    cat_2_stds = df_train_cat_2.std()
    # Returning the required values
    return prob_patient,prob_gender_1,prob_gender_2,cat_1_means,cat_1_stds,cat_2_means,cat_2_stds

# Function to predict the category of a given data, using the trained Naive Bayes Classifier
def classify(inp,prob_patient,prob_gender_1,cat_1_means,cat_1_stds,prob_gender_2,cat_2_means,cat_2_stds):
    if(calc_proba(inp,prob_patient[1],prob_gender_1,cat_1_means,cat_1_stds)>=calc_proba(inp,prob_patient[2],prob_gender_2,cat_2_means,cat_2_stds)):
        return 1
    else:
        return 2

# Function to calculate the accuracy of the classifier on a given test data
def find_accuracy(df_p_train,df_p_test,laplace_factor):
    accuracy = 0
    # Train the classifier on the training data
    prob_patient, prob_gender_1, prob_gender_2, cat_1_means, cat_1_stds, cat_2_means, cat_2_stds = initalise(df_p_train,laplace_factor)
    # Predict the category of each data in the test data and find the accuracy
    for i in range(df_p_test.shape[0]):
        pred = classify(df_p_test.iloc[i,:],prob_patient,prob_gender_1,cat_1_means,cat_1_stds,prob_gender_2,cat_2_means,cat_2_stds)
        if(pred == df_p_test.iloc[i]['is_patient']):
            accuracy+=1
    accuracy/=df_p_test.shape[0]
    return accuracy # Returning the accuracy

# Function to execute 5-fold cross validation on the given training data
def five_fold_cross_validation(df_train,laplace_factor):
    length = df_train.shape[0]
    part_len = length//5
    mean_accuracy = 0
    pred=0
    # Split the training data into 5 parts, and use 4 parts for training and 1 part for testing, and repeat this 5 times, with diffrent test data each time 
    for i in range(5):
        df_p_test = pd.DataFrame(df_train.iloc[part_len*i:part_len*i+part_len])
        df_p_train = pd.DataFrame(df_train.iloc[:part_len*i])
        df_p_train_2 = pd.DataFrame(df_train.iloc[part_len*i+part_len:])
        df_p_train = pd.concat([df_p_train,df_p_train_2])
        # Find the accuracy of the classifier on the test data
        pred = find_accuracy(df_p_train,df_p_test,laplace_factor)
        print(f"Accuracy on fold {i+1} is: {pred}")
        mean_accuracy+=pred
    # Calculating the mean accuracy of the 5-fold cross validation
    mean_accuracy/=5
    print("Five Fold cross validation mean accuracy = ",mean_accuracy) # Output the mean accuracy

# Calling the 5-fold cross validation function
print("Executing 5-fold cross validation on the training data")
five_fold_cross_validation(df_train,0)

# Calling the find_accuracy function to find the accuracy of the classifier on the 30% test data, with no laplace correction
pred = find_accuracy(df_train,df_test,0)
print("Test data accuracy = ",pred)


# Calling the find_accuracy function to find the accuracy of the classifier on the 30% test data, with laplace correction
pred = find_accuracy(df_train,df_test,1)
print("Test data accuracy(with Laplace Correction) = ",pred)

