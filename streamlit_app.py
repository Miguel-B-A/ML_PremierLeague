import os
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Premier League Predictor", page_icon="⚽")

# ── Load and prepare data ──────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    import kagglehub

    os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
    os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]

    path = kagglehub.dataset_download("evangower/premier-league-matches-19922022")
    csv_path = os.path.join(path, 'premier-league-matches.csv')
    df = pd.read_csv(csv_path)

    df['result'] = df['FTR'].map({'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df['home_goals_avg']    = df.groupby('Home')['HomeGoals'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['away_goals_avg']    = df.groupby('Away')['AwayGoals'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['home_conceded_avg'] = df.groupby('Home')['AwayGoals'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['away_conceded_avg'] = df.groupby('Away')['HomeGoals'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

    df = df.dropna()

    le_home   = LabelEncoder()
    le_away   = LabelEncoder()
    le_result = LabelEncoder()

    df['home_encoded']   = le_home.fit_transform(df['Home'])
    df['away_encoded']   = le_away.fit_transform(df['Away'])
    df['result_encoded'] = le_result.fit_transform(df['result'])

    features = ['home_encoded', 'away_encoded', 'home_goals_avg',
                'away_goals_avg', 'home_conceded_avg', 'away_conceded_avg']
    X = df[features]
    y = df['result_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    return df, model, le_home, le_away, le_result

df, model, le_home, le_away, le_result = load_and_train()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("⚽ Premier League Match Predictor")
st.markdown("Predict the result of any Premier League match based on historical data from 1992 to 2022.")

teams = sorted(df['Home'].unique().tolist())

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home team", teams)
with col2:
    away_team = st.selectbox("Away team", teams, index=1)

if st.button("Predict result"):
    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        home_enc = le_home.transform([home_team])[0]
        away_enc = le_away.transform([away_team])[0]

        home_goals_avg    = df[df['Home'] == home_team]['home_goals_avg'].iloc[-1]
        home_conceded_avg = df[df['Home'] == home_team]['home_conceded_avg'].iloc[-1]
        away_goals_avg    = df[df['Away'] == away_team]['away_goals_avg'].iloc[-1]
        away_conceded_avg = df[df['Away'] == away_team]['away_conceded_avg'].iloc[-1]

        partido = pd.DataFrame(
            [[home_enc, away_enc, home_goals_avg, away_goals_avg,
              home_conceded_avg, away_conceded_avg]],
            columns=['home_encoded', 'away_encoded', 'home_goals_avg',
                     'away_goals_avg', 'home_conceded_avg', 'away_conceded_avg']
        )

        prediccion   = model.predict(partido)
        probabilidad = model.predict_proba(partido)[0]
        resultado    = le_result.inverse_transform(prediccion)[0]

        clases = le_result.classes_

        st.markdown("---")
        st.subheader(f"Predicted result: **{resultado}**")

        st.markdown("#### Probabilities")
        for clase, prob in zip(clases, probabilidad):
            st.progress(float(prob), text=f"{clase}: {round(prob * 100, 1)}%")

st.markdown("---")
st.caption("Model: Logistic Regression with class_weight='balanced' · Dataset: Kaggle (1992–2022)")
