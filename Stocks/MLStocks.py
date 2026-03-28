import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(r'"C:\Users\mba14\OneDrive\Escritorio\GIT\ML\Stocks\stock_details_5_years.csv"')