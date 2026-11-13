"""Two-way kinematic reweighting of nu_mu and nu_e and the two-sample classifier test."""
from __future__ import annotations

import numpy as np


def _gb_reweight(source, target, source_w, target_w, n_estimators, learning_rate,
                 max_depth, min_samples_leaf, subsample, n_folds, verbose):
    from hep_ml.reweight import GBReweighter, FoldingReweighter
    base = GBReweighter(n_estimators=n_estimators, learning_rate=learning_rate,
                        max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                        gb_args={"subsample": subsample})
    rw = FoldingReweighter(base, n_folds=n_folds, verbose=verbose)
    rw.fit(original=source, target=target, original_weight=source_w, target_weight=target_w)
    return rw.predict_weights(source, original_weight=source_w)


def reweight_two_way(numu_df, nue_df, features=None, weight_col="weights_mc",
                     n_estimators=50, learning_rate=0.05, max_depth=3,
                     min_samples_leaf=300, subsample=0.8, n_folds=3, verbose=False):
    """Reweight nu_mu -> nu_e, then nu_e -> (reweighted) nu_mu, on ``features``.

    Step 1 reweights nu_mu to the nu_e distribution.  Step 2 reweights nu_e to the
    *already reweighted* nu_mu.  Doing both directions means that neither sample is
    left with phase space the other does not cover.  Inputs are not modified;
    returns ``(weights_numu, weights_nue)`` as arrays aligned with the input rows.
    Defaults to the notebook's setup: ``features=[('E','','')]`` and its GBReweighter settings.
    """
    if features is None:
        features = [("E", "", "")]
    kw = dict(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
              min_samples_leaf=min_samples_leaf, subsample=subsample, n_folds=n_folds,
              verbose=verbose)
    x_mu, x_e = numu_df[features].values, nue_df[features].values
    w_mu0, w_e0 = numu_df[weight_col].to_numpy(), nue_df[weight_col].to_numpy()

    w_mu = _gb_reweight(x_mu, x_e, w_mu0, w_e0, **kw)      # numu -> nue
    w_e = _gb_reweight(x_e, x_mu, w_e0, w_mu, **kw)        # nue -> reweighted numu
    return np.asarray(w_mu), np.asarray(w_e)


def test_classifier(numu_df, nue_df, features=None):
    """Two-sample test: can a GBDT tell weighted nu_mu from weighted nu_e? (AUC ~ 0.5 = no)"""
    import matplotlib.pyplot as plt
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split

    if features is None:
        features = [("E", "", ""), ("lepton_genE", "", ""), ("q0", "", ""),
                    ("modq", "", ""), ("Q2", "", "")]
    X_nue, w_nue = nue_df[features].values, nue_df["weights_mc"].values
    X_numu, w_numu = numu_df[features].values, numu_df["weights_mc"].values

    X = np.vstack([X_nue, X_numu])
    y = np.concatenate([np.ones(len(X_nue)), np.zeros(len(X_numu))])
    weights = np.concatenate([w_nue, w_numu])
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.3, random_state=42, stratify=y)

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                     max_depth=3, random_state=42)
    clf.fit(X_train, y_train, sample_weight=w_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba, sample_weight=w_test)
    print("--- Classifier Two-Sample Test Results ---")
    print(f"Multi-Dimensional ROC-AUC Score: {auc_score:.4f}")
    if auc_score < 0.55:
        print("-> AUC close to 0.5: distributions nearly indistinguishable in multi-D space.")
    elif auc_score < 0.70:
        print("-> Decent alignment; some minor differences remain.")
    else:
        print("-> High AUC: the classifier can still tell them apart.")

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba, sample_weight=w_test)
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"Classifier (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2, label="Random Guessing (AUC = 0.5)")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12); plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Multi-Dimensional Two-Sample Classifier Test", fontsize=14)
    plt.legend(loc="lower right", fontsize=11); plt.grid(alpha=0.3); plt.tight_layout()
    plt.show()
    return auc_score


