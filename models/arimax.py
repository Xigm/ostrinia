import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, List
from tsl.nn.models.base_model import BaseModel

class ARIMAX(BaseModel):
    """
    ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables) model
    implemented in PyTorch.
    
    Parameters:
    -----------
    p : int
        Order of autoregression (AR)
    d : int
        Degree of differencing (I)
    q : int
        Order of moving average (MA)
    n_exog : int
        Number of exogenous variables
    include_constant : bool, default=True
        Whether to include a constant term
    """
    
    def __init__(self, p: int, d: int, q: int, exog_size: int = 0, include_constant: bool = True, horizon: int = 12):
        super(ARIMAX, self).__init__()
        
        self.p = p
        self.d = d
        self.q = q
        self.n_exog = exog_size
        self.include_constant = include_constant
        self.horizon = horizon
        
        # AR parameters
        if p > 0:
            self.ar_params = nn.Parameter(torch.randn(p) * 0.1)
        else:
            self.ar_params = None
            
        # MA parameters
        if q > 0:
            self.ma_params = nn.Parameter(torch.randn(q) * 0.1)
        else:
            self.ma_params = None
            
        # Exogenous variables parameters
        if exog_size > 0:
            self.exog_params = nn.Parameter(torch.randn(exog_size) * 0.1)
        else:
            self.exog_params = None
            
        # Constant term
        if include_constant:
            self.constant = nn.Parameter(torch.randn(1) * 0.1)
        else:
            self.constant = None
            
        # Initialize error terms buffer
        self.register_buffer('errors', torch.zeros(64, 13, horizon))  # Large buffer for errors
        self.error_idx = 0
        
    def difference(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """Apply differencing to the time series."""
        if d == 0:
            return x
        
        diff_x = x.clone()
        for _ in range(d):
            diff_x = diff_x[1:] - diff_x[:-1]
        return diff_x
    
    def undifference(self, diff_x: torch.Tensor, original_x: torch.Tensor, d: int) -> torch.Tensor:
        """Reverse differencing operation."""
        if d == 0:
            return diff_x
            
        # Start with the last d values from original series
        result = diff_x.clone()
        
        # Apply inverse differencing d times
        for i in range(d):
            # Get the appropriate initial value
            if i == 0:
                initial_vals = original_x[-(d-i):]
            else:
                initial_vals = result[-(d-i):]
                
            # Cumulative sum starting from the initial value
            undiff = torch.cumsum(result, dim=0)
            if len(initial_vals) > 0:
                undiff = undiff + initial_vals[-1]
            result = undiff
            
        return result
        
def forward(
    self,
    x: torch.Tensor,                 # (B, T, N, F) – F = 1 for the univariate case
    u: Optional[torch.Tensor] = None, # (B, T, N, n_exog)  or (B, T, n_exog)
    enable_mask: Optional[torch.Tensor] = None # (B, T, N) or (B, T)
) -> torch.Tensor:
    """
    Vectorised forward pass for an ARIMAX model that accepts
    a 4-D input tensor (batch, time, nodes, features).

    Returns
    -------
    torch.Tensor
        Predictions with the same shape as `x` (B, T, N, F).

    Notes
    -----
    * The implementation tries to mimic the classical ARIMA(p,d,q)+X
      formulation, but a few shape/logic inconsistencies cause runtime
      errors or silent broadcasting mistakes.  Inline **BUG** comments
      mark the most critical ones.
    """

    # ------------------------------------------------------------------ #
    # 1.  Basic sanity‑check & unpacking                                 #
    # ------------------------------------------------------------------ #
    B, T, N, F = x.shape
    device     = x.device

    # ------------------------------------------------------------------ #
    # 2.  Handle optional exogenous mask                                 #
    # ------------------------------------------------------------------ #
    if enable_mask is not None and u is not None:
        u = torch.cat([u, enable_mask], dim=-1)          # (B,T,N,n_exog+1)

    # ------------------------------------------------------------------ #
    # 3.  Differencing operator ∇^d along the temporal dimension         #
    # ------------------------------------------------------------------ #
    if self.d > 0:
        # Start with the original series
        y_diff   = x
        for _ in range(self.d):
            # Each pass decreases the time dimension by 1
            y_diff = y_diff[:, 1:] - y_diff[:, :-1]          # (B, T‑d, N, F)
        start_idx = self.d                                   # Skip the first d steps
    else:
        y_diff   = x
        start_idx = 0

    # ------------------------------------------------------------------ #
    # 4.  Buffers & book‑keeping                                         #
    # ------------------------------------------------------------------ #
    preds = []                                               # Python list of (B,N,F)

    # if start_idx:
    #     # >>> BUG?  This pushes a *block* of shape (B,start_idx,N,F)
    #     # whereas later we append individual (B,N,F) tensors.  Mixing
    #     # shapes will break torch.cat() at the end.
    #     preds.append(x[:, :start_idx])                       # (B,start_idx,N,F)

    max_lag   = max(self.p, self.q) if self.q else self.p
    start_t   = max(start_idx, max_lag)                      # index of first computable step

    if self.q:
        # Error buffer of shape (q, B_max, N_max, F) created in __init__
        # Reset it for a fresh forward pass
        self.errors.zero_()
        self.error_idx = 0

    # ------------------------------------------------------------------ #
    # 5.  Main forecasting loop                                          #
    # ------------------------------------------------------------------ #
    # >>> BUG?  self.horizon is not defined in this context.  The loop
    # probably intends to iterate over the sequence length `T`.
    horizon = getattr(self, "horizon", T)

    for t in range(start_t, horizon):
        # Running prediction for the entire batch (B,N,F) simultaneously
        pred_t = torch.zeros((B, N, F), device=device)

        # ---- (i) Constant (intercept) term ---------------------------- #
        if getattr(self, "constant", None) is not None:
            pred_t = pred_t + self.constant.view(1, 1, F)    # broadcast to (B,N,F)

        # ---- (ii) Autoregressive (AR) terms --------------------------- #
        if getattr(self, "ar_params", None) is not None:
            for i in range(self.p):
                lag_slice = y_diff[:, t-i-1-start_idx] if self.d else x[:, t-i-1]  # (B,N,F)
                pred_t = pred_t + self.ar_params[i] * lag_slice

        # ---- (iii) Moving‑average (MA) terms -------------------------- #
        if getattr(self, "ma_params", None) is not None and self.q:
            for i in range(self.q):
                idx   = (self.error_idx-i-1) % self.q
                pred_t = pred_t + self.ma_params[i] * self.errors[idx, :B]  # (B,N,F)

        # ---- (iv) Exogenous regressors (X) ---------------------------- #
        if u is not None:
            if u.dim() == 4:                                  # (B,T,N,n_exog)
                # Weighted sum over last dim keeps (B,N,1) then broadcast to F
                pred_t = pred_t + (u[:, t] * self.exog_params).sum(-1, keepdim=True)
            else:                                             # (B,T,n_exog)
                pred_t = pred_t + torch.matmul(u[:, t], self.exog_params).unsqueeze(-1)

        # ---- (v) Local re‑integration for d>0 ------------------------- #
        if self.d:
            # >>> BUG?  preds[t-1] is a **tensor** while `preds` is a
            # Python list.  Indexing with t‑1 may go out of bounds
            # because the list currently holds only (t - start_t) elements.
            previous = preds[-1] if preds else x[:, t-1]      # Fallback
            preds.append(previous + pred_t)                   # (B,N,F)
        else:
            preds.append(pred_t)                              # (B,N,F)

        # ---- (vi) Update error buffer ------------------------------- #
        if self.q:
            # >>> BUG?  preds is a list so preds[:,t] is invalid.  We
            # need the just‑computed pred_t instead.
            err = x[:, t] - pred_t                            # (B,N,F)
            self.errors[self.error_idx, :B] = err
            self.error_idx = (self.error_idx + 1) % self.q

    # ------------------------------------------------------------------ #
    # 6.  Stack predictions into a single tensor                         #
    # ------------------------------------------------------------------ #
    # Remove the problematic pre‑pend block if present
    if preds and preds[0].dim() == 4:
        # (B,start_idx,N,F) -> list of individual steps
        head = torch.unbind(preds.pop(0), dim=1)              # tuple of length start_idx
        preds = list(head) + preds

    # Now preds is a list of (B,N,F) tensors with equal shapes
    preds = torch.stack(preds, dim=1)                         # (B, T_pred, N, F)

    return preds


    
    # def fit(self, y: torch.Tensor, exog: Optional[torch.Tensor] = None, 
    #         epochs: int = 1000, lr: float = 0.01, verbose: bool = True) -> List[float]:
    #     """
    #     Fit the ARIMAX model to the data.
        
    #     Parameters:
    #     -----------
    #     y : torch.Tensor
    #         Time series data
    #     exog : torch.Tensor, optional
    #         Exogenous variables
    #     epochs : int
    #         Number of training epochs
    #     lr : float
    #         Learning rate
    #     verbose : bool
    #         Whether to print training progress
            
    #     Returns:
    #     --------
    #     List[float]
    #         Training losses
    #     """
    #     optimizer = optim.Adam(self.parameters(), lr=lr)
    #     criterion = nn.MSELoss()
        
    #     losses = []
        
    #     for epoch in range(epochs):
    #         optimizer.zero_grad()
            
    #         # Forward pass
    #         predictions = self.forward(y, exog)
            
    #         # Calculate loss (skip initial values that can't be predicted)
    #         max_lag = max(self.p, self.q) if self.q > 0 else self.p
    #         start_idx = max(self.d, max_lag)
            
    #         if start_idx < len(y):
    #             loss = criterion(predictions[start_idx:], y[start_idx:])
    #         else:
    #             loss = criterion(predictions, y)
            
    #         # Backward pass
    #         loss.backward()
    #         optimizer.step()
            
    #         losses.append(loss.item())
            
    #         if verbose and (epoch + 1) % 100 == 0:
    #             print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
                
    #     return losses
    
    # def predict(self, y: torch.Tensor, exog: Optional[torch.Tensor] = None, 
    #             steps: int = 1) -> torch.Tensor:
    #     """
    #     Make predictions for future time steps.
        
    #     Parameters:
    #     -----------
    #     y : torch.Tensor
    #         Historical time series data
    #     exog : torch.Tensor, optional
    #         Future exogenous variables
    #     steps : int
    #         Number of steps to predict
            
    #     Returns:
    #     --------
    #     torch.Tensor
    #         Predicted values
    #     """
    #     self.eval()
        
    #     with torch.no_grad():
    #         # Use the model to get in-sample predictions first
    #         _ = self.forward(y, exog[:-steps] if exog is not None else None)
            
    #         # Now predict future values
    #         predictions = torch.zeros(steps)
            
    #         # Extend the series for prediction
    #         extended_y = y.clone()
            
    #         for step in range(steps):
    #             pred = 0.0
                
    #             # Constant term
    #             if self.constant is not None:
    #                 pred += self.constant
                    
    #             # AR component
    #             if self.ar_params is not None:
    #                 for i in range(self.p):
    #                     if len(extended_y) - i - 1 >= 0:
    #                         if self.d > 0:
    #                             # Use differenced values
    #                             diff_vals = self.difference(extended_y, self.d)
    #                             if len(diff_vals) - i - 1 >= 0:
    #                                 pred += self.ar_params[i] * diff_vals[-i-1]
    #                         else:
    #                             pred += self.ar_params[i] * extended_y[-i-1]
                
    #             # MA component (simplified - uses recent errors)
    #             if self.ma_params is not None:
    #                 for i in range(self.q):
    #                     if self.error_idx - i - 1 >= 0:
    #                         error_idx = (self.error_idx - i - 1) % len(self.errors)
    #                         pred += self.ma_params[i] * self.errors[error_idx]
                
    #             # Exogenous variables
    #             if self.exog_params is not None and exog is not None:
    #                 if len(y) + step < len(exog):
    #                     pred += torch.sum(self.exog_params * exog[len(y) + step])
                
    #             # Handle differencing
    #             if self.d > 0:
    #                 pred = extended_y[-1] + pred
                
    #             predictions[step] = pred
                
    #             # Add prediction to extended series
    #             extended_y = torch.cat([extended_y, pred.unsqueeze(0)])
                
    #     return predictions