import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\mba14\OneDrive\Escritorio\GIT\ML\PremierLeague\premier-league-matches.csv')

print(df.shape) #How many rows and columns
print(df.head()) #How does it actually look like?
print(df.dtypes) #What type of data do we have?
print(df.isnull().sum()) #How many missing values do we have? 

#We can see that there are some missing values in the 
# 'home_team_goalkeeper' and 'away_team_goalkeeper' columns. 
# We will need to handle these missing values before we can 
# use the data for machine learning.

#Shape: do you have enough rows? For classification you want at least a few hundred, ideally thousands.

#Head: does this make sense? Can you understand what you're looking at? Are column names clear?

#Dtypes: are columns that should be numbers stored as strings? (More common than you'd think. Will break your model silently if you miss it.)

#Nulls: missing data isn't a dealbreaker, but you need to know where it is before you do anything else.

########################################################################################################################################################


#Step 1: Understand your target variable first

df['result'] = df['FTR'].map({'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'})

print(df['result'].value_counts()) #How many of each class do we have?
print(df['result'].value_counts(normalize=True).round(3)) #What percentage of each class do we have?ç

from sklearn.preprocessing import LabelEncoder
le_temp = LabelEncoder()
df['result_encoded'] = le_temp.fit_transform(df['result'])

df.corr(numeric_only=True)['result_encoded'].sort_values(ascending=False)

df['HomeGoals'].hist(bins=10)
plt.title('Distribution of Home Goals')
plt.show()