from tsl.engines import Predictor
from typing import Optional, Callable, Mapping, Type, Union, List, Dict
from torchmetrics import Metric
from torch import nn, Tensor


class WrapPredictor(Predictor):
    """
    A class that extends the TSL Predictor to implement a custom prediction method.
    This class can be used to predict future values based on past observations.
    """

    def __init__(self,
                 model: Optional[nn.Module] = None,
                 loss_fn: Optional[Callable] = None,
                 scale_target: bool = False,
                 metrics: Optional[Mapping[str, Metric]] = None,
                 warm_up: int = 5,
                 *,
                 model_class: Optional[Type] = None,
                 model_kwargs: Optional[Mapping] = None,
                 optim_class: Optional[Type] = None,
                 optim_kwargs: Optional[Mapping] = None,
                 scheduler_class: Optional[Type] = None,
                 scheduler_kwargs: Optional[Mapping] = None,
                 sampling: int = 0):
        
        super(WrapPredictor, self).__init__(       model=model,
                                                    model_class=model_class,
                                                    model_kwargs=model_kwargs,
                                                    optim_class=optim_class,
                                                    optim_kwargs=optim_kwargs,
                                                    loss_fn=loss_fn,
                                                    scale_target=scale_target,
                                                    metrics=metrics,
                                                    scheduler_class=scheduler_class,
                                                    scheduler_kwargs=scheduler_kwargs)
        
        self.sampling = sampling

    def predict_batch(self, 
                        batch,
                        preprocess: bool = False, 
                        postprocess: bool = True,
                        return_target: bool = False,
                        maybe_x: Optional[Tensor] = None,
                        maybe_y: Optional[Tensor] = None,
                        **forward_kwargs):

            inputs, targets, mask, transform = self._unpack_batch(batch)

            if maybe_y is not None:
                targets = maybe_y

            if preprocess:
                for key, trans in transform.items():
                    if key in inputs:
                        inputs[key] = trans.transform(inputs[key])

            if forward_kwargs is None:
                forward_kwargs = dict()

            if maybe_x is not None:
                x = maybe_x
                y_hat = self.forward(x, **inputs, **forward_kwargs)
            else:
                y_hat = self.forward(**inputs, **forward_kwargs)

            # Rescale outputs
            if postprocess:
                trans = transform.get('y')
                if trans is not None:
                    y_hat = trans.inverse_transform(y_hat)

            if return_target:
                y = targets.get('y')
                return y, y_hat, mask
            
            return y_hat
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        
        # Unpack batch
        x, y, mask, transform = self._unpack_batch(batch)

        # Make predictions
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)

        output = dict(**y, y_hat=y_hat)
        if mask is not None:
            output['mask'] = mask
        return output

    def training_step(self, batch: Union[Tensor, Dict[str, Tensor]], batch_idx: int = 0) -> Union[Tensor, Dict[str, Tensor]]:

        mask = batch.get('mask')

        # if we sample, preserve only the input and target of the half with the highest std
        if self.sampling != -1:
            if 'x' in batch:
                x = batch['x']
                y = batch['y']
                std = y.std(dim=(1,2,3)) + x.std(dim=(1,2,3))

                if self.sampling == 0:
                    idx = std.nonzero(as_tuple=False).squeeze(1)
                else:
                    idx = std.topk(x.shape[0] // self.sampling).indices

                # drop all zero std samples
                batch['x'] = x[idx, :]
                batch['y'] = y[idx, :]
                
                batch['u'] = batch['u'][idx, :]

                if 'enable_mask' in batch.keys():
                    batch['enable_mask'] = batch['enable_mask'][idx, :]

                mask = mask[idx, :]

            else:
                raise ValueError("Sampling is only supported for batches with 'x'.")

        y = y_loss = batch['y']

        # Compute predictions and compute loss
        y_hat_loss = self.predict_batch(batch, preprocess=False,
                                             postprocess=not self.scale_target)
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Logging
        self.train_metrics.update(y_hat, y, mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)

        return loss
    
    def validation_step(self, batch: Union[Tensor, Dict[str, Tensor]], batch_idx: int = 0) -> Union[Tensor, Dict[str, Tensor]]:
        """"""
        y = y_loss = batch.y
        mask = batch.get('mask')

        # Compute predictions
        y_hat_loss = self.predict_batch(batch, preprocess=False,
                                             postprocess=not self.scale_target)
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        val_loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Logging
        self.val_metrics.update(y_hat, y, mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss, batch_size=batch.batch_size)
        return val_loss
    
    def test_step(self, batch: Union[Tensor, Dict[str, Tensor]], batch_idx: int = 0) -> Union[Tensor, Dict[str, Tensor]]:

        # Compute outputs and rescale
        y_hat = self.predict_batch(batch, preprocess=False,
                                        postprocess=True)

        y, mask = batch.y, batch.get('mask')
        test_loss = self.loss_fn(y_hat, y, mask)

        # Logging
        self.test_metrics.update(y_hat.detach(), y, mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)

        return test_loss
    

class WrapPredictorDoubleTarget(Predictor):
    """
    A class that extends the TSL Predictor to implement a custom prediction method.
    This class can be used to predict future values based on past observations.
    """

    def __init__(self,
                 model: Optional[nn.Module] = None,
                 loss_fn: Optional[Callable] = None,
                 scale_target: bool = False,
                 metrics: Optional[Mapping[str, Metric]] = None,
                 warm_up: int = 5,
                 *,
                 model_class: Optional[Type] = None,
                 model_kwargs: Optional[Mapping] = None,
                 optim_class: Optional[Type] = None,
                 optim_kwargs: Optional[Mapping] = None,
                 scheduler_class: Optional[Type] = None,
                 scheduler_kwargs: Optional[Mapping] = None,
                 sampling: int = 0):
        
        super(WrapPredictorDoubleTarget, self).__init__(       model=model,
                                                    model_class=model_class,
                                                    model_kwargs=model_kwargs,
                                                    optim_class=optim_class,
                                                    optim_kwargs=optim_kwargs,
                                                    loss_fn=loss_fn,
                                                    scale_target=scale_target,
                                                    metrics=metrics,
                                                    scheduler_class=scheduler_class,
                                                    scheduler_kwargs=scheduler_kwargs)
        
        self.sampling = sampling

    def predict_batch(self, 
                        batch,
                        preprocess: bool = False, 
                        postprocess: bool = True,
                        return_target: bool = False,
                        maybe_x: Optional[Tensor] = None,
                        maybe_y: Optional[Tensor] = None,
                        **forward_kwargs):

            inputs, targets, mask, transform = self._unpack_batch(batch)

            if maybe_y is not None:
                targets = maybe_y

            if preprocess:
                for key, trans in transform.items():
                    if key in inputs:
                        inputs[key] = trans.transform(inputs[key])

            if forward_kwargs is None:
                forward_kwargs = dict()

            if maybe_x is not None:
                x = maybe_x
                y_hat = self.forward(x, **inputs, **forward_kwargs)
            else:
                y_hat = self.forward(**inputs, **forward_kwargs)

            # Rescale outputs
            if postprocess:
                trans = transform.get('y')
                if trans is not None:
                    y_hat = trans.inverse_transform(y_hat)

            if return_target:
                y = targets.get('y')
                return y, y_hat, mask
            
            return y_hat
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        
        # Unpack batch
        x, y, mask, transform = self._unpack_batch(batch)

        # Make predictions
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)

        output = dict(**y, y_hat=y_hat)
        if mask is not None:
            output['mask'] = mask
        return output

    def training_step(self, batch: Union[Tensor, Dict[str, Tensor]], batch_idx: int = 0) -> Union[Tensor, Dict[str, Tensor]]:

        mask = batch.get('mask')

        # if we sample, preserve only the input and target of the half with the highest std
        if self.sampling != -1:
            if 'x' in batch:
                x = batch['x']
                y = batch['y']
                std = y.std(dim=(1,2,3)) + x.std(dim=(1,2,3))

                if self.sampling == 0:
                    idx = std.nonzero(as_tuple=False).squeeze(1)
                else:
                    idx = std.topk(x.shape[0] // self.sampling).indices

                # drop all zero std samples
                batch['x'] = x[idx, :]
                batch['y'] = y[idx, :]
                
                batch['u'] = batch['u'][idx, :]

                if 'enable_mask' in batch.keys():
                    batch['enable_mask'] = batch['enable_mask'][idx, :]

                mask = mask[idx, :]

            else:
                raise ValueError("Sampling is only supported for batches with 'x'.")

        y = y_loss = batch['y']

        # Compute predictions and compute loss
        y_hat_loss = self.predict_batch(batch, preprocess=False,
                                             postprocess=not self.scale_target)
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Logging
        self.train_metrics.update(y_hat, y, mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)

        return loss
    
    def validation_step(self, batch: Union[Tensor, Dict[str, Tensor]], batch_idx: int = 0) -> Union[Tensor, Dict[str, Tensor]]:
        """"""
        y = y_loss = batch.y
        mask = batch.get('mask')

        # Compute predictions
        y_hat_loss = self.predict_batch(batch, preprocess=False,
                                             postprocess=not self.scale_target)
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        val_loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Logging
        self.val_metrics.update(y_hat, y, mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss, batch_size=batch.batch_size)
        return val_loss
    
    def test_step(self, batch: Union[Tensor, Dict[str, Tensor]], batch_idx: int = 0) -> Union[Tensor, Dict[str, Tensor]]:

        # Compute outputs and rescale
        y_hat = self.predict_batch(batch, preprocess=False,
                                        postprocess=True)

        y, mask = batch.y, batch.get('mask')
        test_loss = self.loss_fn(y_hat, y, mask)

        # Logging
        self.test_metrics.update(y_hat.detach(), y, mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)

        return test_loss
    