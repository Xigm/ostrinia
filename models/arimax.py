@dataclass
class ARIMAXSpec:
    order: Tuple[int, int, int]
    trend: str = "c"  # constant by default
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False

def fit_arimax(y_tr: pd.Series, X_tr: Optional[np.ndarray], spec: ARIMAXSpec):
    mod = sm.tsa.statespace.SARIMAX(
        y_tr,
        exog=X_tr,
        order=spec.order,
        seasonal_order=(0, 0, 0, 0),
        trend=spec.trend,
        enforce_stationarity=spec.enforce_stationarity,
        enforce_invertibility=spec.enforce_invertibility,
    )
    res = mod.fit(disp=False)
    return res