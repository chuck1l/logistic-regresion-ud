import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
import seaborn as sns
 
def adjust_columns(data):
    data.columns = data.columns.str.replace(' ', '_')
    data.columns = data.columns.str.lower()

if __name__ == '__main__':
    data = pd.read_csv('../data/advertising.csv')
    adjust_columns(data)
    toggle = True
    if toggle:
        print(data.head())
        print(data.describe())
        print(data.info())
    plot1 = plt.hist(data['age'], bins=30, color='blue')
    plt.title('Histogram of Age')
    plt.xlabel('Age')
    plt.ylabel('Count')
    #plt.savefig('../imgs/age_hist.png')
    plt.show();

    plot2 = sns.pairplot(data, hue='clicked_on_ad', palette='bwr')
    #plt.savefig('../imgs/ad_pairplot.png')
    plt.show();

    plot3 = sns.jointplot(x=data['age'], y=data['area_income'], kind='scatter', color='blue')
    plt.savefig('../imgs/joint_areaincome_age.png')
    plt.show();

    plot4 = sns.jointplot(x=data['age'], y=data['daily_time_spent_on_site'], kind='kde')
    #plt.savefig('../imgs/age_hist.png')
    plt.show();

    plot5 = sns.jointplot(x=data['daily_time_spent_on_site'], y=data['daily_internet_usage'], kind='scatter', color='green')
    #plt.savefig('../imgs/joint_timespent_netusage.png')
    plt.show();

