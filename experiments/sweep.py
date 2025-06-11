import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.metrics import torch_metrics

from datasets.ostrinia import Ostrinia

from models.dcrnn import DCRNNModel
from extras.predictor import WrapPredictor
from extras.metrics_logging import MetricsLogger
from extras.callbacks import Wandb_callback, MetricsHistory
from extras.plots import plot_predictions_test

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import yaml
from colorama import Fore, Style, init
init()


def get_model(name):
    if name == 'dcrnn':
        return DCRNNModel
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

@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):

    wandb.init()

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

    # update config with wandb config: batch_size
    if 'batch_size' in wandb.config.keys():
        cfg.batch_size = wandb.config['batch_size']
        print(Fore.GREEN + f"Updated batch size: {cfg.batch_size}")

    data_module = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        batch_size=cfg.batch_size,
        workers=cfg.optimizer.num_workers,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
    )

    data_module.setup()

    adj = dataset.get_connectivity(**cfg.dataset.connectivity,
                                   train_slice=data_module.train_slice)
    
    data_module.torch_dataset.set_connectivity(adj)

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
    
    model.filter_model_args_(model_kwargs)

    model_kwargs.update(cfg.model.hparams)

    # update with wandb config
    for key in wandb.config.keys():
        if key in model_kwargs.keys():
            model_kwargs[key] = wandb.config[key]
            cfg.model.hparams[key] = wandb.config[key]
            print(Fore.GREEN + f"Updated model kwargs: {key} = {wandb.config[key]}")
        else:
            print(Fore.RED + f"Key {key} not found in model kwargs, skipping.")

    ########################################
    # predictor                            #
    ########################################

    loss_fn = torch_metrics.MaskedMAE(compute_on_step=True)
    
    log_list = cfg.dataset.log_metrics

    log_metrics = MetricsLogger()

    metrics = log_metrics.filter_metrics(log_list)

    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    optimizer_kwargs = dict(cfg.optimizer.hparams)

    # update optimizer kwargs with wandb config
    if 'optimizer' in wandb.config.keys():
        cfg.optimizer.name = wandb.config['optimizer']
        print(Fore.GREEN + f"Updated optimizer: {cfg.optimizer}")

    for key in wandb.config.keys():
        if key in optimizer_kwargs.keys():
            optimizer_kwargs[key] = wandb.config[key]
            cfg.optimizer.hparams[key] = wandb.config[key]
            print(Fore.GREEN + f"Updated optimizer kwargs: {key} = {wandb.config[key]}")
        else:
            print(Fore.RED + f"Key \'{key}\' not found in optimizer kwargs, skipping.")

    # setup predictor
    predictor = WrapPredictor(
        model_class=model,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=optimizer_kwargs,
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
    )

    exp_logger = TensorBoardLogger(save_dir=cfg.run_dir, name=cfg.run_name)
    
    ######################################
    # Training and Setting up
    ######################################

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run_dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )
    

    run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity=cfg.wandb.entity,
                # Set the wandb project where this run will be logged.
                project=cfg.wandb.project,
                # Track hyperparameters and run metadata.
                config={
                    "learning_rate": cfg.optimizer.hparams.lr,
                    "batch_size": cfg.batch_size,
                    "model": cfg.model.name,
                    "optimizer": cfg.optimizer.name,
                    "hidden_size": cfg.model.hparams.hidden_size,
                    "dropout": cfg.model.hparams.dropout,
                    "regularization_weight": cfg.get('regularization_weight', 0.0),
                    "dataset": cfg.dataset.name,
                    "epochs": cfg.epochs,
                    "window": cfg.dataset.window,
                    "horizon": cfg.dataset.horizon
                },
            )
    
    wandb_logger_callback = Wandb_callback(
        log_dir=cfg.run_dir,
        run=run,
        log_metrics=log_list,
    )

    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.train_batches,
                      default_root_dir=cfg.run_dir,
                      logger=exp_logger,  # Disable default logger
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                    #   devices=,
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback, wandb_logger_callback])

    load_model_path = cfg.get('load_model_path')
    if load_model_path is not None:
        predictor.load_model(load_model_path)
    else:
        trainer.fit(predictor, train_dataloaders=data_module.train_dataloader(),
                    val_dataloaders=data_module.val_dataloader())
        predictor.load_model(checkpoint_callback.best_model_path)

    ########################################
    # testing                              #
    ########################################

    predictor.freeze()
    trainer.test(predictor, dataloaders=data_module.test_dataloader())
    
    exp_logger.finalize('success')
    
    plot_predictions_test(
        predictor=predictor,
        data_module=data_module,
        run_dir=cfg.run_dir,
        log_metrics=log_list
    )

def wandb_sweep():
    """
    Run a sweep with wandb.
    """
    wandb_sweep_path = "./config/wandb/sweep.yaml"
    wandb_keys_path = "./config/wandb/keys.yaml"
    with open(wandb_sweep_path, 'r') as f:
        dict_sweep = yaml.safe_load(f)
    with open(wandb_keys_path, 'r') as f:
        dict_keys = yaml.safe_load(f)

    dict_sweep = dict_sweep['sweep']

    wandb.login(key=dict_keys['key'])

    for key in dict_sweep.keys():
        print(f"Setting sweep parameter {key} to {dict_sweep[key]}")

    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep=dict_sweep,
        project=dict_keys['project'],
        entity=dict_keys['entity'],
    )

    # Start the sweep
    wandb.agent(sweep_id, function=main, count=dict_sweep['count'])

if __name__ == "__main__":
    wandb_sweep()