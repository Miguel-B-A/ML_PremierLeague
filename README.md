# ML Premier League

A machine learning project to predict Premier League match results (Home Win, Draw, Away Win) using historical match data from 1992 to 2022.

Built following the [QuantifAI](https://quantifai.beehiiv.com) newsletter series as a hands-on introduction to the full ML pipeline.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://premier-league-predictor-mba.streamlit.app/)

---

## About the project

This project applies classification techniques to predict football match outcomes based on features like goals scored, shots, and team statistics. It covers the full ML workflow: data exploration, cleaning, feature engineering, model training, and evaluation.

**Dataset**: [Premier League Matches 1992–2022](https://www.kaggle.com/datasets/evangower/premier-league-matches-19922022) — available on Kaggle.

**Target variable**: Match result — Home Win, Draw, or Away Win.

---

## Installation

### Requirements

- Python 3.7+
- Anaconda (recommended)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/Miguel-B-A/ML_PremierLeague.git
cd ML_PremierLeague
```

2. Install dependencies:

```bash
pip install pandas matplotlib scikit-learn
```

3. Download the dataset from Kaggle:

```python
import kagglehub
path = kagglehub.dataset_download("evangower/premier-league-matches-19922022")
print("Path to dataset files:", path)
```

> You will need a Kaggle account and API key configured. See [Kaggle API docs](https://www.kaggle.com/docs/api) for instructions.

4. Update the CSV path in `ML_PremierLeague.py` to match where the dataset was downloaded on your machine.

---

