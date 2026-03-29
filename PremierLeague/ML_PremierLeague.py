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

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df['home_goals_avg'] = df.groupby('Home')['HomeGoals'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
df['away_goals_avg'] = df.groupby('Away')['AwayGoals'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

df['home_conceded_avg'] = df.groupby('Home')['AwayGoals'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
df['away_conceded_avg'] = df.groupby('Away')['HomeGoals'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

df = df.dropna()

le_home = LabelEncoder()
le_away = LabelEncoder()    
le_result = LabelEncoder()

df['home_encoded'] = le_home.fit_transform(df['Home'])
df['away_encoded'] = le_away.fit_transform(df['Away'])  
df['result_encoded'] = le_result.fit_transform(df['result'])

from sklearn.model_selection import train_test_split

features = ['home_encoded', 'away_encoded', 'home_goals_avg', 'away_goals_avg', 'home_conceded_avg', 'away_conceded_avg']
X = df[features]
y = df['result_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

pred_labels = le_result.inverse_transform(predictions[:10])
print(pred_labels)



