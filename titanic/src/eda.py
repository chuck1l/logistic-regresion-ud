import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns

if __name__ == '__main__':
    train = pd.read_csv('../data/titanic_train.csv')
    print(train.head())
    print(train.info())
    # Take a look at the null values (age and cabin mostly)
    plot1 = sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Take A Look At Null Values For Features')
    plt.tight_layout()
    #plt.savefig('../imgs/feature_nulls.png')
    plt.show();
    # Take a look at the survial balance/count
    sns.set_style('whitegrid')
    plot2 = sns.countplot(x='Survived', data=train, palette='RdBu_r')
    plt.title('Looking At Survival Balance/Count (Survive = 1)')
    #plt.savefig('../imgs/survival_count.png')
    plt.show();
    # Survival divided by gender
    plot3 = sns.countplot(x="Survived", hue='Sex', data=train, palette='RdBu_r')
    plt.title('Look At Survival By Gender (Survive = 1)')
    #plt.savefig('../imgs/survival_gender.png')
    plt.show();
    plot4 = sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
    plt.title('Look At Survival By Ticket Class (Survive = 1)')
    #plt.savefig('../imgs/survival_class.png')
    plt.show;


