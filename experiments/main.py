import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.metrics import torch_metrics

from datasets.ostrinia import Ostrinia

from models.dcrnn import dcrnn
from models.predictor import Predictor

def get_model(name):
    if name == 'dcrnn':
        return dcrnn
    else:
        raise NotImplementedError(f"Model {name} is not implemented.")
    
def build_cfg(cfg: DictConfig):

    model_name = cfg.model.name  # Get the model name from the config
    model_config_path = f"./models/{model_name}.yaml"  # Construct the model-specific config path

    # Load the model-specific config
    model_cfg = hydra.compose(config_name=model_config_path)
    print(f"Loaded configuration for {model_name}: {model_cfg}")

    # Optimizer configuration
    optimizer_name = cfg.optimizer.name  # Get the optimizer name from the config
    optimizer_config_path = f"./optimizers/{optimizer_name}.yaml"  # Construct the optimizer-specific config path
    optimizer_cfg = hydra.compose(config_name=optimizer_config_path)
    print(f"Loaded configuration for optimizer {optimizer_name}: {optimizer_cfg}")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    
    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg.model, resolve=True),
            name=cfg.wandb.name,
            mode=cfg.wandb.mode
        )

    # Compute covariates
    covariates = None

    #######################################
    # dataset Initialization
    #######################################
    dataset = Ostrinia(root = "datasets", target = cfg.dataset.target)

    torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                          mask=dataset.mask,
                                          covariates=covariates,
                                          horizon=cfg.dataset.horizon,
                                          window=cfg.dataset.window,
                                          stride=cfg.dataset.stride)
    
    input_size = torch_dataset.n_channels
    
    scale_axis = (0,) if cfg.get('scale_axis') == 'node' else (0, 1)
    transform = {
        'target': StandardScaler(axis=scale_axis),
    }

    data_module = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
    )

    data_module.setup()

    ######################################
    # Model Initialization
    ######################################
    model = get_model(cfg.model.name)

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=input_size,
                        exog_size=0,
                        output_size=torch_dataset.n_channels,
                        weighted_graph=torch_dataset.edge_weight is not None,
                        embedding_cfg=cfg.get('embedding'), #### changed from None to embedding_cfg
                        horizon=torch_dataset.horizon)
    
    model = model(**model_kwargs)

    model.filter_model_args_(model_kwargs)

    model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################

    loss_fn = torch_metrics.MaskedMAE(compute_on_step=True)

    log_metrics = {'mae': torch_metrics.MaskedMAE(),
                   "mae_at_3_days": torch_metrics.MaskedMAE(at=2),
                   "mae_at_6_days": torch_metrics.MaskedMAE(at=5),
                   "mae_at_12_days": torch_metrics.MaskedMAE(at=11),
                   'mre': torch_metrics.MaskedMRE(),
                   'mse': torch_metrics.MaskedMSE()}

    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # setup predictor
    predictor = Predictor(
        model_class=model,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        beta=cfg_to_python(cfg.regularization_weight),
        embedding_var=cfg.embedding.get('initial_var', 0.2),
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=cfg.scale_target,
    )

    
    ######################################
    # Optimizer Initialization
    ######################################
    
    

if __name__ == "__main__":
    main()