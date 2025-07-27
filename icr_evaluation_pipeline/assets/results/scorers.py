import pandas as pd
from imblearn.metrics import specificity_score
from sklearn.metrics import make_scorer

from icr_evaluation_pipeline.evaluation import (
    f1_most_rare_score,
    accuracy_most_rare_score,
    precision_most_rare_score,
    recall_most_rare_score,
    specificity_most_rare_score,
)


def get_scoring(rarity_scores: pd.Series) -> dict[str, make_scorer]:
    return {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_micro": "f1_micro",
        "precision_macro": "precision_macro",
        "precision_micro": "precision_micro",
        "recall_macro": "recall_macro",
        "recall_micro": "recall_micro",
        "specificity_macro": make_scorer(
            specificity_score, greater_is_better=True, average="macro"
        ),
        "specificity_micro": make_scorer(
            specificity_score, greater_is_better=True, average="micro"
        ),
        "accuracy_most_rare": make_scorer(
            accuracy_most_rare_score,
            greater_is_better=True,
            rarity_scores=rarity_scores,
        ),
        "f1_most_rare_macro": make_scorer(
            f1_most_rare_score,
            greater_is_better=True,
            average="macro",
            rarity_scores=rarity_scores,
        ),
        "f1_most_rare_micro": make_scorer(
            f1_most_rare_score,
            greater_is_better=True,
            average="micro",
            rarity_scores=rarity_scores,
        ),
        "precision_most_rare_macro": make_scorer(
            precision_most_rare_score,
            greater_is_better=True,
            average="macro",
            rarity_scores=rarity_scores,
        ),
        "precision_most_rare_micro": make_scorer(
            precision_most_rare_score,
            greater_is_better=True,
            average="micro",
            rarity_scores=rarity_scores,
        ),
        "recall_most_rare_macro": make_scorer(
            recall_most_rare_score,
            greater_is_better=True,
            average="macro",
            rarity_scores=rarity_scores,
        ),
        "recall_most_rare_micro": make_scorer(
            recall_most_rare_score,
            greater_is_better=True,
            average="micro",
            rarity_scores=rarity_scores,
        ),
        "specificity_most_rare_macro": make_scorer(
            specificity_most_rare_score,
            greater_is_better=True,
            average="macro",
            rarity_scores=rarity_scores,
        ),
        "specificity_most_rare_micro": make_scorer(
            specificity_most_rare_score,
            greater_is_better=True,
            average="micro",
            rarity_scores=rarity_scores,
        ),
    }
