from tsl.metrics.torch.metric_base import MaskedMetric
from typing import Any
import torch.nn.functional as F


class MaskedCategoricalCrossEntropy(MaskedMetric):
    """
    Masked Categorical Cross-Entropy metric.
    Computes cross-entropy loss between predictions and targets,
    applying a mask to ignore specified positions.
    """
    
    def __init__(self):
        """
        Initialize the masked categorical cross-entropy metric.
        
        Args:
            reduction (str): Reduction method for outputs: 'none' | 'mean' | 'sum'
            ignore_index (int): Target value to ignore in loss calculation
        """
    
        def __init__(self,
                 mask_nans=False,
                 mask_inf=False,
                 at=None,
                 **kwargs: Any):
            super(MaskedCategoricalCrossEntropy, self).__init__(metric_fn=F.cross_entropy,
                                            mask_nans=mask_nans,
                                            mask_inf=mask_inf,
                                            metric_fn_kwargs={'reduction': 'none'},
                                            at=at,
                                            **kwargs)
