import logging
import os

import dotenv
import hydra
import pandas as pd
import torch

import eval
import training
import util
from adaptation.util import get_adaptation
from datasets import get_dataloader
from model import ModelRegistry

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='default.yaml')
def main(cfg):
    # Setup GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Setup Checkpoints
    ckpt_mngr = util.CheckpointManager(cfg.training)
    
    # Setup Metrics Logging
    if hasattr(cfg, 'logs'):
        experiment_logger = util.get_logger(cfg.logs, cfg)
        experiment_logger.log_config(cfg)
    else:
        experiment_logger = util.DummyLogger()

    # Setup Dataset
    train_loader, test_loader = get_dataloader(cfg.data)

    # Setup Model
    model = ModelRegistry()[cfg.model.name, cfg.model]
    if cfg.model.summary:
        util.summary(model, depth=2)
    model.to(device)

    # Setup Trainer
    trainer = training.get_trainer(cfg.training)(cfg,
                                                 model,
                                                 train_loader,
                                                 ckpt_mngr,
                                                 experiment_logger,
                                                 device)

    # Setup Adaptation
    adaptation = get_adaptation(cfg, 
                                model,
                                test_loader.dataset.transforms.reconstruct)
    if adaptation is not None:
        adaptation.to(device)

    # Setup Evaluation
    metrics = eval.get_metrics(cfg, device)
    evaluator = eval.get_evaluator(cfg, 
                                   model, 
                                   test_loader, 
                                   adaptation, 
                                   metrics, 
                                   device)

    # Run Training
    trainer.run()

    # Run Evaluation
    save = os.environ.get('TRAINING_RUN', default='False') == 'True'
    evaluator.run(save=save)

    if cfg.eval.detailed_results:
        if hasattr(test_loader.dataset, 'format_results'):
            format_df = test_loader.dataset.format_results(evaluator.metrics_df)
        else:
            format_df = util.format_results(evaluator.metrics_df, metrics)

        if format_df is not None:
            with pd.option_context('display.float_format', '{:,.3f}'.format):
                print(format_df)
    
    print(evaluator)
    experiment_logger.log_results(evaluator.result_dict())
    ckpt_mngr.save_results(evaluator.__repr__())


if __name__ == '__main__':
    dotenv.load_dotenv()
    os.environ['MODE'] = 'TRAIN'
    main()
