
# ATP Tennis Match Outcome Prediction Using PySpark and Machine Learning

A machine learning project that predicts ATP tennis match outcomes using historical ATP data and PySpark MLlib.


## Overview

The objective of this project is to predict ATP tennis match outcomes using historical match data.

The project follows a complete machine learning workflow:

- Data collection
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Match outcome prediction

The implementation was developed using PySpark and focuses on scalable processing of large tennis datasets.

---

## Dataset

The project uses ATP historical match data from:

https://github.com/JeffSackmann/tennis_atp

The dataset contains:

- ATP rankings
- ATP ranking points
- Age
- Height
- Playing surface
- Tournament category
- Match format
- Match results

---

## Data Preprocessing

The original ATP dataset stores information using winner and loser columns.

To avoid data leakage, the dataset was transformed into a neutral:

Player 1 vs Player 2

representation.

Each match was converted into:

- one observation with label = 1
- one observation with label = 0

This produced a balanced binary classification dataset containing approximately 91,000 observations.

---

## Feature Engineering

The following predictive features were created:

### Basic Features

- Rank Difference
- Ranking Points Difference
- Age Difference
- Height Difference
- Surface
- Tournament Level
- Best Of

### Advanced Features

#### Head-to-Head Win Rate

Historical win percentage of Player 1 against Player 2.

#### Fatigue Score

Total minutes played by a player in recent matches.

#### Recent Win Rate

Player performance over the last matches.

#### Surface Win Rate

Historical win percentage on the current playing surface.

---

## Machine Learning Models

The following PySpark MLlib models were trained:

### Logistic Regression

Baseline model used for comparison.

### Random Forest

Ensemble learning approach using multiple decision trees.

### Gradient Boosted Trees (GBT)

Boosting-based model for improved predictive performance.

### Logistic Regression + H2H

Logistic Regression extended with Head-to-Head information.

---

## Evaluation

The models were evaluated using:

- Accuracy
- ROC AUC
- Cross Validation
- Confusion Matrix
- ROC Curve
- Feature Importance Analysis

---

## Results

| Model | AUC | Accuracy |
|---------|---------|---------|
| Logistic Regression | 0.718 | 0.655 |
| Random Forest | 0.717 | 0.655 |
| GBT | 0.717 | 0.652 |
| Logistic Regression + H2H | 0.714 | 0.653 |

### Observations

- Logistic Regression achieved the best overall performance.
- Ranking features were the strongest predictors.
- ATP ranking points were the most important feature.
- H2H information provided limited improvement.
- The models performed significantly better than random guessing.

---

## Match Prediction System

A custom prediction function was implemented to estimate winning probabilities between any two ATP players.

Example:

```python
predict_match(
    "Novak Djokovic",
    "Carlos Alcaraz",
    "Hard",
    tourney_level="G",
    best_of=5
)
```

Example Output:

```text
Novak Djokovic: 87.44%
Carlos Alcaraz: 12.56%

Predicted winner: Novak Djokovic
```

The function automatically:

- retrieves player statistics
- computes feature differences
- generates model input
- estimates winning probabilities
- predicts the winner



## Installation

Clone repository:

```bash
git clone https://github.com/AlexandraMoldovan03/ATP-Tennis-Match-Outcome-Prediction-Using-PySpark-and-Machine-Learning.git
```

Move into project folder:

```bash
cd ATP-Tennis-Match-Outcome-Prediction-Using-PySpark-and-Machine-Learning
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

## Future Improvements

Possible future improvements include:

- Player fatigue modeling
- Tournament location effects
- Home advantage analysis
- Elo ratings
- Recent surface-specific form
- Deep Learning approaches
- Live ATP ranking integration

---

## Author

**Alexandra Moldovan**

Master's Student in Software Engineering

West University of Timisoara

---

## Development Notes

A development discussion used during the project can be found here:
https://chatgpt.com/share/6a25d3d0-1610-83eb-8da3-f4a8050a3029
