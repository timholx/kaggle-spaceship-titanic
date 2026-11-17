"""Spaceship Titanic - feature engineering + gradient boosting ensemble.

Run:  py -3.13 solution.py
Writes submission.csv next to this script.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier

SEED = 42
N_FOLDS = 10
SMOOTH = 10  # pseudo-count pulling small-surname rates toward the overall mean
GROUP_CV = os.environ.get("GROUP_CV", "1") == "1"
# Surname outcome rate: tested, lowered blend CV (0.8167 -> 0.8139), so off by default.
USE_SURNAME = os.environ.get("USE_SURNAME", "0") == "1"
HERE = Path(__file__).parent
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def load():
    train = pd.read_csv(HERE / "train.csv")
    test = pd.read_csv(HERE / "test.csv")
    y = train.pop("Transported").astype(int)
    df = pd.concat([train, test], ignore_index=True)
    return df, y, len(train), test["PassengerId"]


def impute(df):
    """Fill missing values using the strong structure in this dataset."""
    df["Group"] = df["PassengerId"].str.split("_").str[0].astype(int)
    df["Surname"] = df["Name"].str.split().str[-1]
    cab = df["Cabin"].str.split("/", expand=True)
    df["Deck"], df["CabinNum"], df["Side"] = cab[0], pd.to_numeric(cab[1]), cab[2]

    # Group members share HomePlanet, and nearly always Deck/Side.
    for col in ["HomePlanet", "Deck", "Side", "Surname"]:
        mode = df.groupby("Group")[col].transform(
            lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
        df[col] = df[col].fillna(mode)
    # Families (same surname) share HomePlanet.
    fam = df.groupby("Surname")["HomePlanet"].transform(
        lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    df["HomePlanet"] = df["HomePlanet"].fillna(fam)
    # Deck -> HomePlanet: A,B,C,T are Europa only; G is Earth only.
    df.loc[df["HomePlanet"].isna() & df["Deck"].isin(list("ABCT")), "HomePlanet"] = "Europa"
    df.loc[df["HomePlanet"].isna() & (df["Deck"] == "G"), "HomePlanet"] = "Earth"
    df["HomePlanet"] = df["HomePlanet"].fillna("Earth")

    # CryoSleep passengers spend nothing; anyone spending is not asleep.
    total_known = df[SPEND].sum(axis=1, min_count=1)
    df.loc[df["CryoSleep"].isna() & (total_known > 0), "CryoSleep"] = False
    for c in SPEND:
        df.loc[df[c].isna() & (df["CryoSleep"] == True), c] = 0  # noqa: E712
    df["CryoSleep"] = df["CryoSleep"].fillna(False).astype(bool)
    # Kids (<13) never spend.
    for c in SPEND:
        df.loc[df[c].isna() & (df["Age"] < 13), c] = 0
        df[c] = df[c].fillna(df[c].median())

    df["Age"] = df["Age"].fillna(df.groupby("HomePlanet")["Age"].transform("median"))
    df["VIP"] = df["VIP"].fillna(False).astype(bool)
    df["Destination"] = df["Destination"].fillna("TRAPPIST-1e")
    df["Deck"] = df["Deck"].fillna("F")
    df["Side"] = df["Side"].fillna("S")
    # Cabin numbers increase with group id within a deck: interpolate.
    for d in df["Deck"].unique():
        m = df["Deck"] == d
        known = m & df["CabinNum"].notna()
        if known.sum() >= 2:
            fit = np.polyfit(df.loc[known, "Group"], df.loc[known, "CabinNum"], 1)
            miss = m & df["CabinNum"].isna()
            df.loc[miss, "CabinNum"] = np.polyval(fit, df.loc[miss, "Group"]).round()
    df["CabinNum"] = df["CabinNum"].fillna(df["CabinNum"].median())
    return df


def features(df):
    df["GroupSize"] = df.groupby("Group")["Group"].transform("size")
    df["Solo"] = (df["GroupSize"] == 1).astype(int)
    df["FamilySize"] = df.groupby("Surname")["Surname"].transform("size").fillna(1)
    df["TotalSpend"] = df[SPEND].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["LuxurySpend"] = df["Spa"] + df["VRDeck"] + df["RoomService"]
    df["BasicSpend"] = df["FoodCourt"] + df["ShoppingMall"]
    for c in SPEND + ["TotalSpend", "LuxurySpend", "BasicSpend"]:
        df[c + "_log"] = np.log1p(df[c])
    df["AgeBin"] = pd.cut(df["Age"], [-1, 12, 17, 25, 30, 50, 200], labels=False)
    df["CabinRegion"] = (df["CabinNum"] // 300).astype(int)
    df["DeckSide"] = df["Deck"] + "_" + df["Side"]
    df["PlanetDest"] = df["HomePlanet"] + "_" + df["Destination"]
    df["CryoSleep"] = df["CryoSleep"].astype(int)
    df["VIP"] = df["VIP"].astype(int)

    num = ["CryoSleep", "VIP", "Age", "AgeBin", "CabinNum", "CabinRegion", "GroupSize",
           "Solo", "FamilySize", "NoSpend"] + [c for c in df.columns if c.endswith("_log")]

    # One-hot matrix for XGBoost / HistGradientBoosting.
    X = pd.concat([df[num], pd.get_dummies(df[CATS], dtype=int)], axis=1).astype(float)
    # Native categorical matrix for CatBoost (it target-encodes these internally).
    X_cat = pd.concat([df[num].astype(float),
                       df[CATS + ["Surname"]].fillna("NA").astype(str)], axis=1)
    return X, X_cat, df["Group"].values, df["Surname"].fillna("NA").values


CATS = ["HomePlanet", "Destination", "Deck", "Side", "DeckSide", "PlanetDest"]
CAT_NATIVE = CATS + ["Surname"]


def surname_rate(fit_names, fit_y, apply_names, smooth=SMOOTH):
    """Smoothed share of passengers with this surname who were transported."""
    prior = fit_y.mean()
    s = pd.DataFrame({"n": fit_names, "y": fit_y}).groupby("n")["y"].agg(["sum", "count"])
    rate = (s["sum"] + smooth * prior) / (s["count"] + smooth)
    return pd.Series(apply_names).map(rate).fillna(prior).values


def surname_feature(names_tr, y_tr, names_other, groups_tr):
    """Leak-free surname rate: out-of-fold on training rows, full-fit on the rest.

    Inner folds are split by group so a passenger's own groupmates never inform
    their value -- matching test, where no group appears in train.
    """
    inner = np.zeros(len(names_tr))
    for a, b in GroupKFold(5).split(names_tr, groups=groups_tr):
        inner[b] = surname_rate(names_tr[a], y_tr[a], names_tr[b])
    return inner, [surname_rate(names_tr, y_tr, n) for n in names_other]


def models():
    return {
        "xgb": XGBClassifier(n_estimators=600, learning_rate=0.02, max_depth=5,
                             subsample=0.8, colsample_bytree=0.6, min_child_weight=3,
                             reg_lambda=2.0, eval_metric="logloss", random_state=SEED,
                             n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(learning_rate=0.03, max_iter=500,
                                              max_leaf_nodes=24, l2_regularization=1.0,
                                              min_samples_leaf=30, random_state=SEED),
        "cat": CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=6,
                                  l2_leaf_reg=3, random_seed=SEED, verbose=0,
                                  allow_writing_files=False, thread_count=-1),
    }


def main():
    df, y, n_train, test_ids = load()
    X, X_cat, groups, names = features(impute(df))
    X_tr, X_te = X.iloc[:n_train], X.iloc[n_train:]
    C_tr, C_te = X_cat.iloc[:n_train], X_cat.iloc[n_train:]
    g_tr, n_tr, n_te = groups[:n_train], names[:n_train], names[n_train:]
    yv = y.values
    cat_cols = CAT_NATIVE if USE_SURNAME else CATS
    if not USE_SURNAME:
        C_tr, C_te = C_tr.drop(columns="Surname"), C_te.drop(columns="Surname")
    print(f"Features: {X.shape[1]} | CV split by group: {GROUP_CV} | surname: {USE_SURNAME}")

    # Split by passenger group: test shares no groups with train, so CV should not either.
    if GROUP_CV:
        splits = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_tr, y, g_tr)
    else:
        splits = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_tr, y)
    oof = {k: np.zeros(n_train) for k in models()}
    test_pred = {k: np.zeros(len(X_te)) for k in models()}
    for tr, va in splits:
        xa, xv, xt = X_tr.iloc[tr].copy(), X_tr.iloc[va].copy(), X_te.copy()
        if USE_SURNAME:
            xa["SurnameRate"], (xv["SurnameRate"], xt["SurnameRate"]) = surname_feature(
                n_tr[tr], yv[tr], [n_tr[va], n_te], g_tr[tr])
        for name, m in models().items():
            if name == "cat":
                a, v, t = C_tr.iloc[tr], C_tr.iloc[va], C_te
                m.fit(a, yv[tr], cat_features=cat_cols)
            else:
                a, v, t = xa, xv, xt
                m.fit(a, yv[tr])
            oof[name][va] = m.predict_proba(v)[:, 1]
            test_pred[name] += m.predict_proba(t)[:, 1] / N_FOLDS

    for name in oof:
        print(f"{name:>5} CV accuracy: {((oof[name] > 0.5) == y).mean():.4f}")
    blend_oof = np.mean(list(oof.values()), axis=0)
    print(f"blend CV accuracy: {((blend_oof > 0.5) == y).mean():.4f}")

    blend_test = np.mean(list(test_pred.values()), axis=0)
    sub = pd.DataFrame({"PassengerId": test_ids, "Transported": blend_test > 0.5})
    sub.to_csv(HERE / "submission.csv", index=False)
    print(f"Wrote submission.csv ({len(sub)} rows, {sub['Transported'].mean():.1%} True)")


if __name__ == "__main__":
    main()
