"""Model variants compared in the `subject-classifier` experiment.

All variants are sparse linear models over TF-IDF features: cheap to train on a laptop,
fast at inference and interpretable (per-class top features).
"""

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

Analyzer = Literal["word", "char_wb"]
Estimator = Literal["logreg", "linear_svc"]


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    analyzer: Analyzer = "word"
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    max_df: float = 0.9
    max_features: int = 100_000
    sublinear_tf: bool = True
    estimator: Estimator = "logreg"
    C: float = 10.0
    class_weight: str | None = None
    max_iter: int = 2000

    def params(self) -> dict[str, Any]:
        """Flat params for `mlflow.log_params` (None -> "none")."""
        return {key: ("none" if value is None else value) for key, value in asdict(self).items()}


BASELINE = Variant("baseline", "TF-IDF word 1-2 grams + LogisticRegression")
CHAR_NGRAMS = replace(
    BASELINE,
    name="char_ngrams",
    description="TF-IDF char_wb 2-5 grams + LogisticRegression",
    analyzer="char_wb",
    ngram_min=2,
    ngram_max=5,
    max_features=200_000,
)
BALANCED = replace(
    CHAR_NGRAMS,
    name="balanced",
    description="TF-IDF char_wb 2-5 grams + LogisticRegression(class_weight=balanced)",
    class_weight="balanced",
)
WORD_BALANCED = replace(
    BASELINE,
    name="word_balanced",
    description="TF-IDF word 1-2 grams + LogisticRegression(class_weight=balanced)",
    class_weight="balanced",
)
SVM = replace(
    BALANCED,
    name="svm",
    description="TF-IDF char_wb 2-5 grams + sigmoid-calibrated LinearSVC(class_weight=balanced)",
    estimator="linear_svc",
    C=0.5,
)

VARIANTS: dict[str, Variant] = {
    v.name: v for v in (BASELINE, CHAR_NGRAMS, BALANCED, WORD_BALANCED, SVM)
}


def build_pipeline(variant: Variant, seed: int) -> Pipeline:
    """Unfitted `tfidf -> clf` pipeline; `clf` always exposes `predict_proba`."""
    vectorizer = TfidfVectorizer(
        analyzer=variant.analyzer,
        ngram_range=(variant.ngram_min, variant.ngram_max),
        min_df=variant.min_df,
        max_df=variant.max_df,
        max_features=variant.max_features,
        sublinear_tf=variant.sublinear_tf,
        lowercase=True,
    )
    clf: Any
    if variant.estimator == "logreg":
        clf = LogisticRegression(
            C=variant.C,
            class_weight=variant.class_weight,
            max_iter=variant.max_iter,
            random_state=seed,
        )
    else:
        svc = LinearSVC(
            C=variant.C,
            class_weight=variant.class_weight,
            max_iter=variant.max_iter,
            random_state=seed,
        )
        # Sigmoid calibration turns SVM margins into probabilities for `confidence`/`top3`.
        clf = CalibratedClassifierCV(svc, method="sigmoid", cv=3)
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def linear_coefficients(pipeline: Pipeline) -> Any:
    """(n_classes x n_features) weights of the linear model, or None if it has none."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        return clf.coef_
    calibrated = getattr(clf, "calibrated_classifiers_", None)
    if calibrated:
        # Average the SVM weights over the calibration folds.
        return sum(c.estimator.coef_ for c in calibrated) / len(calibrated)
    return None
