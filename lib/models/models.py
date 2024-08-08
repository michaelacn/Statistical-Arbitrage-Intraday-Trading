import numpy as np
import pandas as pd

import scipy
import statsmodels.tsa as sm


def is_stationary(timeseries: pd.Series, alpha: float = 0.05) -> bool:
    """
    Perform ADFuller test to check the stationarity of the time series.
    """
    if timeseries.empty:
        print("[INFO]: Empty time series received; skipping stationarity test.")
        return False

    result = sm.stattools.adfuller(timeseries.dropna(), autolag='BIC')  # Ensure NA values are dropped
    return result[1] <= alpha  # Return True if p-value is less than alpha


def fit_var_model(logprices: pd.DataFrame, current_date: str = None) -> tuple:
    """
    Fits a VAR model to the provided data, selecting the optimal lag based on BIC.
    """
    max_lags = min(10, len(logprices) // 4)

    if len(logprices) < max_lags:
        print(f"[INFO] {current_date}: Not enough data to fit VAR model.")
        return None, None

    var = sm.api.VAR(logprices)
    order_selection = var.select_order(maxlags=max_lags)
    optimal_lag = order_selection.selected_orders['bic']
    var_model = var.fit(optimal_lag)
    return var_model, optimal_lag


def count_vecm_params(vecm_results) -> int:
    """
    Calculates the total number of parameters in a VECM results object.
    """
    num_alpha_params = vecm_results.alpha.size # alpha matrix
    num_beta_params = vecm_results.beta.size # beta matrix
    num_gamma_params = vecm_results.gamma.size # gamma matrix
    num_det_coef_params = vecm_results.det_coef.size if vecm_results.det_coef is not None else 0 # deterministic coefficients in the model, if any
    num_det_coint_params = vecm_results.det_coef_coint.size if vecm_results.det_coef_coint is not None else 0 # deterministic cointegration coefficients, if any
    return num_alpha_params + num_beta_params + num_gamma_params + num_det_coef_params + num_det_coint_params


def trace_test(logprices: pd.DataFrame, det_order: int, optimal_lag: int) -> int:
    """
    Performs the Johansen cointegration trace test.
    """
    result = sm.vector_ar.vecm.coint_johansen(logprices, det_order, optimal_lag - 1)
    trace_stat = result.lr1
    critical_values = result.cvt[:, 1]
    rank = sum(trace_stat > critical_values)
    return rank


def likelihood_ratio_test(model_restricted, model_unrestricted) -> float:
    """
    Performs a likelihood ratio test to compare two nested VECM..
    """
    lr_stat = -2 * (np.log(model_restricted.llf) - np.log(model_unrestricted.llf))
    df = count_vecm_params(model_unrestricted) - count_vecm_params(model_restricted)
    p_value = scipy.stats.chi2.sf(lr_stat, df)
    return p_value


def perform_cointegration_analysis(logprices: pd.DataFrame, optimal_lag: int, current_date: str) -> list:
    """
    Performs a two-stage cointegration analysis on time series data.
    """
    eligible_models = []

    # First Stage: Trace Test for Cointegration Rank
    rank_model3 = trace_test(logprices, det_order=1, k_ar_diff=optimal_lag)
    if rank_model3 > 0:
        rank_model2 = trace_test(logprices, det_order=0, k_ar_diff=optimal_lag)
        if rank_model2 > 0:
            rank_model1 = trace_test(logprices, det_order=-1, k_ar_diff=optimal_lag)
            eligible_models.append((logprices, 3 if rank_model1 > 0 else 2))
        else:
            eligible_models.append((logprices, 1))
    else:
        print(f'[INFO] {current_date}: No cointegration relationship, skipping.')

    # Second Stage: Likelihood Ratio Test for Model Selection
    vecm_model = []
    for pair_data, model_type in eligible_models:
        model3 = sm.vector_ar.vecm.VECM(pair_data, k_ar_diff=optimal_lag - 1, deterministic='co').fit()
        if model_type == 3:
            model2 = sm.vector_ar.vecm.VECM(pair_data, k_ar_diff=optimal_lag - 1, deterministic='ci').fit()
            model1 = sm.vector_ar.vecm.VECM(pair_data, k_ar_diff=optimal_lag - 1, deterministic='n').fit()

            p_value = likelihood_ratio_test(model_restricted=model2, model_unrestricted=model3)
            if p_value < 0.05:
                vecm_model.append((model3, 'Model 3 (unrestricted constant)'))
            else:
                p_value = likelihood_ratio_test(model_restricted=model1, model_unrestricted=model2)
                if p_value < 0.05:
                    vecm_model.append((model2, 'Model 2 (restricted constant)'))
                else:
                    vecm_model.append((model1, 'Model 1 (no constant)'))
        elif model_type == 2:
            model2 = sm.vector_ar.vecm.VECM(pair_data, k_ar_diff=optimal_lag - 1, deterministic='ci').fit()
            p_value = likelihood_ratio_test(model_restricted=model2, model_unrestricted=model3)
            if p_value < 0.05:
                vecm_model.append((model3, 'Model 3 (unrestricted constant)'))
            else:
                vecm_model.append((model2, 'Model 2 (restricted constant)'))
        elif model_type == 1:
            vecm_model.append((model3, 'Model 3 (unrestricted constant)'))

    return vecm_model


def get_vecm_spread(logprices: pd.DataFrame, vecm_model, end_train_idx: int) -> pd.Series:
    """
    Calculates the VECM-based spread using the final selected model's parameters.
    """
    beta = vecm_model[0][0].beta[:, 0]
    const = vecm_model[0][0].det_coef_coint[0][0] if vecm_model[0][0].det_coef_coint.size > 0 else 0
    return logprices[:end_train_idx] @ beta + const
