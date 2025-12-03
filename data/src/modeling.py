"""
Simple panel fixed-effects modeling utilities.

We provide two options:
- If linearmodels is installed, use PanelOLS with entity fixed effects.
- Otherwise, fall back to statsmodels OLS with entity dummies (less efficient).

Example usage:
model = fit_panel_fe(df_panel, depvar='hygiene_index', exog=['inspection_number', 'mean_critical_violations'], entity_id='camis', time_id='inspection_date')
"""
import pandas as pd
import numpy as np

def fit_panel_fe(df, depvar, exog, entity_id="camis", time_id="inspection_date"):
    df = df.copy()
    # drop NA
    df = df.dropna(subset=[depvar] + exog + [entity_id])
    try:
        from linearmodels.panel import PanelOLS
        df[time_id] = pd.to_datetime(df[time_id])
        df = df.set_index([entity_id, time_id])
        y = df[depvar]
        X = df[exog]
        X = X.astype(float)
        model = PanelOLS(y, X, entity_effects=True)
        res = model.fit(cov_type='clustered', cluster_entity=True)
        return res
    except Exception:
        import statsmodels.api as sm
        # create entity dummies
        dummies = pd.get_dummies(df[entity_id], prefix="ent", drop_first=True)
        X = pd.concat([df[exog].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        X = sm.add_constant(X)
        y = df[depvar].reset_index(drop=True)
        ols = sm.OLS(y, X).fit(cov_type='HC3')
        return ols

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    args = parser.parse_args()
    import pandas as pd
    df = pd.read_parquet(args.infile)
    # simple example regression
    res = fit_panel_fe(df, depvar="hygiene_index", exog=["inspection_number", "critical_violations"], entity_id="camis", time_id="inspection_date")
    print(res.summary())
