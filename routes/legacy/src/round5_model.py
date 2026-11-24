"""Train-fold-only scaling and record/class-balanced logistic regression."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from scripts.round2_train import sample_weights


def fit(c, x, meta):
    assert (meta.groupby('record').y.nunique() == 2).all()
    assert x.shape == (len(meta), {'base':125, 'ordinal':20, 'fusion':145}[c['features']])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c['C'], max_iter=2000, random_state=20261104))
    with threadpool_limits(limits=4):
        model.fit(x, meta.y.to_numpy(), logisticregression__sample_weight=sample_weights(meta))
    return model


def predict(model, x):
    with threadpool_limits(limits=4):
        return model.predict_proba(x)[:, 1]
