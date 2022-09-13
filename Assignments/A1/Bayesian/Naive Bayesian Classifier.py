
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce


data = pd.read_csv("/home/pranav/Desktop/SEM 5/ML-22/Assignments/A1/Bayesian/Train_B_Bayesian.csv",encoding="ISO-8859-1")

# df = data.drop(columns = ['is_patient'])
df = data


df1 = df.sample(frac=1)

df1 = df1.reset_index(drop=True)

# df_old holds df of before Shuffling
df_old = df
df = df1

encoder= ce.OrdinalEncoder(cols=['gender'],return_df=True,
                           mapping=[{'col':'gender',
'mapping':{'Male':1,'Female':2}}])
df_transformed = encoder.fit_transform(df)
df_transformed


# df_before_encoding holds df of before Gender encoding
df_before_encoding = df
df = df_transformed


df_before_encoding.to_csv('df_before_encoding.csv')
df_before_outlier = df
df_before_outlier.to_csv('df_before_outlier.csv')


genders = df['gender']
df3 = df.drop(columns = ['gender'])
df3['gender'] = genders
df = df3

train_ind = int(0.7*df.shape[0])
df_train = df.iloc[:train_ind] 
df_test = df.iloc[train_ind:] 


outliers = {}
means = list(df_train.mean())
stds = list(df_train.std())
j=0
for i in df_train.columns:
    if(i=='gender'):
        break
    upper_bound = 2*means[j] + 5*stds[j]
    l_out = list(df_train.index[(df_train[i]>upper_bound)])
    outliers[i] = l_out
    j+=1
print("\nOutliers found for each feature(indexes based on shuffled data):")
for x in outliers.items():
    print(x)
print("")
final_list = []
for i in outliers.keys():
    final_list.extend(outliers[i])
final_list = sorted(set(final_list))
df_train_with_outliers = df_train


df_train = df_train.drop(final_list)
df_train_without_outliers = df_train


df_train = df_train.reset_index(drop=True)



def normal_distribution(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density


def calc_proba(inp,prob_patient,prob_gender,cat_means,cat_stds):
    prob = prob_patient
    prob *= prob_gender[inp['gender']]
    for i in df_train.columns:
        if(i=='is_patient'):
            break
        else:
            prob *= normal_distribution(inp[i],cat_means[i],cat_stds[i])
    return prob*1000


def initalise(df_p_train,laplace_factor):
    prob_patient = {}
    prob_patient[1] = 0
    prob_patient[2] = 0
    
    for i in range(len(df_p_train)):
        if(df_p_train.iloc[i]['is_patient']==1):
            prob_patient[1]+=1
        else:
            prob_patient[2]+=1
    prob_patient[1]/=len(df_p_train)
    prob_patient[2]/=len(df_p_train)

    df_train_cat_1 = df_p_train[df_p_train['is_patient']==1]
    df_train_cat_2 = df_p_train[df_p_train['is_patient']==2]

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
    cat_1_means = df_train_cat_1.mean()
    cat_1_stds = df_train_cat_1.std()
    cat_2_means = df_train_cat_2.mean()
    cat_2_stds = df_train_cat_2.std()
    return prob_patient,prob_gender_1,prob_gender_2,cat_1_means,cat_1_stds,cat_2_means,cat_2_stds


def classify(inp,prob_patient,prob_gender_1,cat_1_means,cat_1_stds,prob_gender_2,cat_2_means,cat_2_stds):
    if(calc_proba(inp,prob_patient[1],prob_gender_1,cat_1_means,cat_1_stds)>=calc_proba(inp,prob_patient[2],prob_gender_2,cat_2_means,cat_2_stds)):
        return 1
    else:
        return 2


def find_accuracy(df_p_train,df_p_test,laplace_factor):
    accuracy = 0
    prob_patient, prob_gender_1, prob_gender_2, cat_1_means, cat_1_stds, cat_2_means, cat_2_stds = initalise(df_p_train,laplace_factor)
    for i in range(df_p_test.shape[0]):
        pred = classify(df_p_test.iloc[i,:],prob_patient,prob_gender_1,cat_1_means,cat_1_stds,prob_gender_2,cat_2_means,cat_2_stds)
        if(pred == df_p_test.iloc[i]['is_patient']):
            accuracy+=1
    accuracy/=df_p_test.shape[0]
    return accuracy


def five_fold_cross_validation(df_train,laplace_factor):
    length = df_train.shape[0]
    part_len = length//5
    mean_accuracy = 0
    pred=0
    for i in range(5):
        df_p_test = pd.DataFrame(df_train.iloc[part_len*i:part_len*i+part_len])
        df_p_train = pd.DataFrame(df_train.iloc[:part_len*i])
        df_p_train_2 = pd.DataFrame(df_train.iloc[part_len*i+part_len:])
        df_p_train = pd.concat([df_p_train,df_p_train_2])
        pred = find_accuracy(df_p_train,df_p_test,laplace_factor)
        mean_accuracy+=pred
    mean_accuracy/=5
    print("Five Fold cross validation mean accuracy = ",mean_accuracy)

five_fold_cross_validation(df_train,0)


pred = find_accuracy(df_train,df_test,0)
print("Test data accuracy = ",pred)


pred = find_accuracy(df_train,df_test,1)
print("Test data accuracy(with Laplace Correction) = ",pred)

