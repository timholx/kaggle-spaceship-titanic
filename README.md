# Spaceship Titanic

Predicting which passengers of the *Spaceship Titanic* were transported to an alternate dimension, for the Kaggle competition [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic). It is a binary classification task (`Transported`), and the metric is accuracy.

- [`eda.ipynb`](eda.ipynb): exploratory analysis. It explains every imputation rule and engineered feature.
- [`solution.py`](solution.py): the full pipeline, which writes `submission.csv`.

## Approach

**Imputation from the data's structure.** About 2% of each column is missing, but that adds up to 24% of rows with at least one gap. Instead of dropping those rows, missing values are filled using hard rules found in the training data:

- Passengers in CryoSleep never spend, and children under 13 never spend. So a missing spend is 0 for them, and anyone who spent something was awake.
- Members of a group (`PassengerId` = `gggg_pp`) always share HomePlanet and cabin side, and usually share a deck. Passengers with the same surname share a HomePlanet.
- Decks A, B, C and T hold only passengers from Europa. Deck G holds only passengers from Earth.
- Within a deck, cabin numbers rise linearly with group id, so a missing cabin number is interpolated.

**Features** (parsed and engineered):

- Group size, solo traveller flag, family size (by surname).
- Cabin deck, number and side; a coarse `CabinRegion` band, because the transport rate drifts along the cabin number.
- Total spend, a no-spend flag, and luxury spend (RoomService, Spa, VRDeck) against basic spend (FoodCourt, ShoppingMall), all `log1p`-transformed.
- Age bins and the combinations Deck×Side and HomePlanet×Destination.

**Validation that mirrors the test.** No passenger group appears in both train and test, so cross-validation uses 10-fold `StratifiedGroupKFold` split by group. A plain stratified split would let groupmates leak into validation.

**Models.** XGBoost, scikit-learn's HistGradientBoosting and CatBoost (with native categorical handling). The submission is the plain average of their predicted probabilities, thresholded at 0.5.

## Results (10-fold group CV accuracy)

| Model (59 features) | Accuracy |
|---|---|
| XGBoost | 0.8130 |
| HistGradientBoosting | 0.8165 |
| CatBoost | 0.8161 |
| **Blend (submitted)** | **0.8167** |

## Findings

- **CryoSleep is the strongest single signal:** 82% of sleepers were transported, against 33% of passengers who were awake.
- **Spending matters:** 79% of passengers who spent nothing were transported, against 30% of those who spent something. Luxury spending separates the classes more clearly than basic spending.
- **Solo travellers** were transported less often (45%) than members of groups of 3 to 6 (about 60%).
- **Surname target encoding didn't help:** an out-of-fold surname transport rate lowered blend CV from 0.8167 to 0.8139, so it is off by default (`USE_SURNAME=1` turns it on).

## Running it

1. Download the competition data (`train.csv`, `test.csv`) from Kaggle into this folder. The data files are not included in this repo.
2. Install pandas, numpy, scikit-learn, xgboost and catboost (plus matplotlib, seaborn and jupyter for the notebook).
3. Run `python solution.py`. It takes about 15 minutes on CPU and writes `submission.csv`. Set `GROUP_CV=0` to compare against a plain stratified split.
