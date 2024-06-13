import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# 1
DF=pd.read_csv("E:/vusialisation_project/medical_examination.csv")
df=pd.DataFrame(DF)
df.head()
# 2
df["bmi"]=(df["weight"]/((df["height"]/100)**2))
df['overweight'] = df['overweight'] = np.where(df['bmi'] > 25, 1, 0)

# # Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

df["normalization"]=np.where((df["cholesterol"]==1)| (df["gluc"]==1),0,1)
# 4
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = df_cat = pd.melt(df, 
                id_vars=['cardio'], 
                value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'],
                var_name='variable',
                value_name='values')
    print(df_cat)


    # # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the collumns for the catplot to work correctly.
    df_cat_cor = df_cat.groupby(['cardio',"variable", "values"])["values"].count().to_frame()
    df_cat_cor = df_cat_cor.rename(columns={"values": "total"})
    df_cat_cor.reset_index(inplace=True)
    print(pd.DataFrame(df_cat_cor))
    

    #Draw the catplot with 'sns.catplot()'
    catplot = sns.catplot(x='variable', y='total', hue='values', col='cardio', kind='bar', data=df_cat_cor)
    fig = catplot.fig
    


    # 8
    plt.show()


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    height025=df["height"].quantile(0.025)
    height975=df["height"].quantile(0.975)
    weight025=df["weight"].quantile(0.025)
    weight975=df["weight"].quantile(0.975)
    df=df[(df["height"]>=df["height"].quantile(0.025))&(df["height"]<=df["height"].quantile(0.975))]
    df=df[(df["weight"]>=df["weight"].quantile(0.025))&(df["weight"]<=df["weight"].quantile(0.975))]
    
    df_heat =df[(df['ap_lo'] <= df['ap_hi'])]

    # 12
    corr=df_heat.corr()
    corr = corr.round(1)

    # 13
    mask = np.triu(np.ones_like(corr))



    # 14
    fig, ax = plt.subplots(figsize=(12, 9))

    # 15
    heat_map = sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", square=True)
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=90)


    # 16
    fig.savefig('heatmap.png')
    return fig