
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



eda_data = pd.read_csv("df_train_cleaned.csv",index_col=0)
eda_data.head()


eda_catplot = eda_data.copy()
eda_catplot.loc[eda_catplot['gender']==1,'gender'] ="Male"
eda_catplot.loc[eda_catplot['gender']==2,'gender'] ="Female"
eda_catplot.loc[eda_catplot['is_patient']==1,'is_patient']="Liver"
eda_catplot.loc[eda_catplot['is_patient']==2,'is_patient']="Non-Liver"
sns.catplot(x ="gender", hue ="is_patient",kind ="count", data = eda_catplot)



eda_data.head()


eda_y = eda_data.is_patient    
eda_y


eda_y.loc[eda_y==1][0] = "Liver"
eda_y.loc[eda_y==2][0] = "Non-liver"
ax = sns.countplot(x = eda_y,label="Count")


eda_data


eda_y = eda_data.is_patient
eda_x = eda_data.drop("is_patient",axis=1)



eda_data_dia = eda_y.astype("float")
eda_data = eda_x.astype("float")
eda_data_n_2 = (eda_data - eda_data.mean().astype("float")) / (eda_data.std().astype("float"))              # standardization
eda_data = pd.concat([eda_y,eda_data_n_2],axis=1)
eda_data = pd.melt(eda_data,id_vars="is_patient",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="is_patient", data=eda_data,split=True, inner="quart")
plt.xticks(rotation=45);




eda_data_dia = eda_y
eda_data = eda_x
eda_data_n_2 = (eda_data - eda_data.mean()) / (eda_data.std())              # standardization
eda_data = pd.concat([eda_y,eda_data_n_2],axis=1)
eda_data = pd.melt(eda_data,id_vars="is_patient",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="is_patient", data=eda_data)
plt.xticks(rotation=45);



plt.figure(figsize=(25,25))
sns.set(style="white")
g = sns.PairGrid(eda_x, diag_sharey=False,corner=True)
g.map_lower(sns.regplot,scatter_kws={"color": "red"}, line_kws={"color": "black"})
g.map_diag(sns.kdeplot, lw=3)



f,ax = plt.subplots(figsize=(11, 11))
sns.heatmap(eda_x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);
plt.savefig('heatmap.jpg',bbox_inches='tight', dpi=300)





