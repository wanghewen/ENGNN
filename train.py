import datetime
from copy import deepcopy, copy

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.profiler import SimpleProfiler
import os
import torch
import pytorch_lightning as pl

from dataloaders.GNN_dataloader import GNNDataloader
from global_parameters import max_epochs, print_result_for_pl, wrapped_partial
from models.Base import BaseModel

import CommonModules as CM

csv_logger = CSVLogger(".", name='log', version=None, prefix='')
profiler = SimpleProfiler(dirpath=".", filename="profile.log")


def find_gpus(num_of_cards_needed=4, model_parameter_dict={}):
    """
    Find the GPU which uses least memory. Should set CUDA_VISIBLE_DEVICES such that it uses all GPUs.

    :param num_of_cards_needed:
    :return:
    """
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    if (torch.cuda.is_available() and not (model_parameter_dict.get("force_cpu", False)))\
            and (os.environ["CUDA_VISIBLE_DEVICES"] not in ["-1", ""]):
        os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >~/.tmp_free_gpus')
        # If there is no ~ in the path, return the path unchanged
        with open(os.path.expanduser('~/.tmp_free_gpus'), 'r') as lines_txt:
            frees = lines_txt.readlines()
            idx_freeMemory_pair = [(idx, int(x.split()[2]))
                                   for idx, x in enumerate(frees)]
        idx_freeMemory_pair.sort(reverse=True)
        idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
        usingGPUs = [idx_memory_pair[0] for idx_memory_pair in
                     idx_freeMemory_pair[:num_of_cards_needed]]
        # usingGPUs = ','.join(usingGPUs)
        print('using GPUs:', end=' ')
        for pair in idx_freeMemory_pair[:num_of_cards_needed]:
            print(f'{pair[0]} {pair[1] / 1024:.1f}GB')
        accelerator = "gpu"
    else:
        usingGPUs = None
        accelerator = "cpu"
    return usingGPUs, accelerator


def train(dataset_class, dataloader_class, model_class, model_parameter_dict: dict, logger, checkpoint_callback,
          use_pl, training_phase=None, previous_phase_checkpoint=None, initialize_dataset=True):
    model_parameter_dict["training_phase"] = training_phase
    if initialize_dataset:
        dataset = dataset_class()
        train_dataset, val_dataset, test_dataset = copy(dataset), copy(dataset), copy(dataset)
        # Will modify initialized_instance in __init__
        dataset_class(subset="train", initialized_instance=train_dataset)
        dataset_class(subset="val", initialized_instance=val_dataset)
        dataset_class(subset="test", initialized_instance=test_dataset)
        in_nfeats = train_dataset[0]["nfeats"].shape[1]
    else:
        train_dataset, val_dataset, test_dataset = None, None, None  # The model will use dataset class to get instances
        in_nfeats = None
    gpus, accelerator = find_gpus(1, model_parameter_dict)

    if dataloader_class is not None:  # In case you don't use dataloaders in your fitting code
        train_dataloader = dataloader_class(device=gpus).build_dataloader(train_dataset, train_val_test="train")
        val_dataloader = dataloader_class(device=gpus).build_dataloader(val_dataset, train_val_test="val")
        test_dataloader = dataloader_class(device=gpus).build_dataloader(test_dataset, train_val_test="test")
    else:
        train_dataloader, val_dataloader, test_dataloader = None, None, None
    if model_parameter_dict.get("n_classes", None) == "auto":  # Infer n_classes during run time
        if len(train_dataset.train_labels.shape) > 1:
            n_classes = train_dataset.train_labels.shape[1]
        else:
            n_classes = 1
        model_parameter_dict["n_classes"] = n_classes  # Slightly conflict with labelsize. Should fix in the future
    try:  # For edge attributed models
        if previous_phase_checkpoint:
            model = model_class.load_from_checkpoint(previous_phase_checkpoint, in_nfeats=in_nfeats,
                                                     in_efeats=in_nfeats, labelsize=1,
                                                     **model_parameter_dict)
        else:
            model = model_class(in_nfeats=in_nfeats, in_efeats=in_nfeats, labelsize=1, **model_parameter_dict)
    except TypeError:
        if previous_phase_checkpoint:
            model = model_class.load_from_checkpoint(previous_phase_checkpoint, in_nfeats=in_nfeats,
                                                     labelsize=1, **model_parameter_dict)
        else:
            model = model_class(in_nfeats=in_nfeats, labelsize=1, **model_parameter_dict)
    if use_pl:
        if training_phase != "discrimination":
            earlystopping_callback = EarlyStopping(monitor='val/loss_epoch',
                                                   min_delta=0.00,
                                                   # patience=30,
                                                   patience=300,
                                                   verbose=True,
                                                   mode='min')
            callbacks = [checkpoint_callback, earlystopping_callback]
        else:
            callbacks = [checkpoint_callback]
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator, devices=gpus,
                             callbacks=callbacks,
                             logger=[logger, csv_logger], check_val_every_n_epoch=10, profiler=False,
                             move_metrics_to_cpu=True, precision=32)
        # Because there may be another call inside fit, we define these variables to record time
        start_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        trainer.fit(model, train_dataloader, [val_dataloader, test_dataloader])
        end_training_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        if training_phase:
            print(f"Training time elapsed for phase {training_phase}: {end_training_time - start_training_time} sec")
        else:
            print(f"Training time elapsed: {end_training_time - start_training_time} sec")
        print(checkpoint_callback.best_model_path)
        graph_dataloaders = train_dataloader, val_dataloader, test_dataloader
        return trainer, checkpoint_callback, graph_dataloaders, None, model
    else:  # Let the model handle all the training
        graph_features = model.get_static_graph_features([train_dataloader, val_dataloader, test_dataloader])
        # Sometimes the trainer is inside fit. If not, trainer should be None
        trainer = model.fit(graph_features, max_epochs=max_epochs,
                            gpus=gpus, dataset_class=dataset_class)
        return trainer, None, None, graph_features, model


def test(trainer, checkpoint_callback, graph_dataloaders, graph_features, model, use_pl):
    if use_pl:
        train_dataloader, val_dataloader, test_dataloader = graph_dataloaders
        test_results = print_result_for_pl(trainer, checkpoint_callback, train_dataloader, val_dataloader,
                                           test_dataloader)
    else:
        test_results = model.test(trainer, graph_features)
    return test_results


def main(logger_name, dataset_class, dataloader_class, model_class, model_parameter_dict, use_pl=True,
         initialize_dataset=True, as_dynamic=False):
    """

    :param logger_name:
    :param dataset_class:
    :param dataloader_class:
    :param model_class:
    :param model_parameter_dict:
    :param use_pl:
    :param initialize_dataset: Whether automatically initialize dataset or let the model handle the dataset class
    initialization
    :return:
    """
    pl.seed_everything(12)
    logger = TensorBoardLogger("logs_20230502", name=logger_name, version=None)
    # logger = TensorBoardLogger("logs_20220831", name=logger_name, version=None)
    previous_phase_checkpoint = None
    if "training_phases" in model_parameter_dict:
        training_phases = model_parameter_dict["training_phases"]  # e.g. ['discrimination', 'classification']
    else:
        training_phases = [None]
    for training_phase_index, training_phase in enumerate(training_phases):
        if training_phase_index >= 1:
            model_parameter_dict["training_phase"] = training_phase
            previous_phase_checkpoint = checkpoint_callback.best_model_path
        if training_phase == "discrimination":
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join("./models", logger.name, str(datetime.datetime.now())),
                filename='epoch={epoch}-train_loss_epoch={'
                         'train/loss_epoch:.6f}',
                monitor='train/loss_epoch', mode="min", save_last=True,
                auto_insert_metric_name=False)
        else:
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join("./models", logger.name, str(datetime.datetime.now())),
                filename='epoch={epoch}-val_loss_epoch={'
                         'val/loss_epoch:.6f}-val_accuracy_epoch={'
                         'val/accuracy_epoch:.4f}-val_macro_f1_epoch={'
                         'val/macro_f1_epoch:.4f}-val_topk_precision_epoch={'
                         'val/topk_precision_epoch:.4f}-val_f1_anomaly_epoch={'
                         'val/f1_anomaly_epoch:.4f}',
                monitor='val/auc_epoch', mode="max", save_last=True,
                auto_insert_metric_name=False)
        trainer, checkpoint_callback, graph_dataloaders, graph_features, model_class_instance = \
            train(dataset_class, dataloader_class, model_class, model_parameter_dict, logger, checkpoint_callback,
                  use_pl, training_phase=training_phase, previous_phase_checkpoint=previous_phase_checkpoint,
                  initialize_dataset=initialize_dataset)
    # return
    if as_dynamic:
        if graph_features:
            old_graph_features = graph_features
        else:
            _, _, old_test_dataloader = graph_dataloaders
        test_results = test(trainer, checkpoint_callback, graph_dataloaders, graph_features, model_class_instance, use_pl)
        for test_portion_index in range(2, 10):
            print("Current test_portion_index:", test_portion_index)
            # as_dynamic should already be True for dataset_class
            test_dataset = dataset_class(subset="test", test_portion_index=test_portion_index)
            gpus, _ = find_gpus(1, model_parameter_dict)
            new_test_dataloader = dataloader_class(device=gpus).build_dataloader(test_dataset, train_val_test="test")
            if graph_features:
                start_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                new_graph_features = model_class_instance.get_dynamic_graph_features(
                    new_test_dataloader, old_graph_features)
                end_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                print(f"get_dynamic_graph_features time elapsed: {end_time - start_time} sec")
                graph_features |= new_graph_features
                # import ipdb; ipdb.set_trace()
                start_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                test_results = test(trainer, checkpoint_callback, graph_dataloaders, graph_features, model_class_instance, use_pl)
                end_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                print(f"test_dynamic_graph time elapsed: {end_time - start_time} sec")
                old_graph_features = graph_features
            else:
                graph_dataloaders = None, None, new_test_dataloader
                start_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                test_results = test(trainer, checkpoint_callback, graph_dataloaders, graph_features,
                                    model_class_instance, use_pl)
                end_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
                print(f"test_dynamic_graph time elapsed: {end_time - start_time} sec")
    else:
        test_results = test(trainer, checkpoint_callback, graph_dataloaders, graph_features, model_class_instance, use_pl)

    return test_results
