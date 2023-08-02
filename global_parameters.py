import os
from functools import partial, update_wrapper

import numpy as np
import scipy.sparse

import inspect
import torch
os.environ["MKL_INTERFACE_LAYER"] = "ILP64"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import datetime
import pytorch_lightning as pl
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
# else:
#     device = torch.device("cpu")
pl.seed_everything(12)
log_file = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + ".log"
import CommonModules as CM

def convert_matrix_to_numpy(m):
    if isinstance(m, scipy.sparse.spmatrix):
        m = m.A
    elif isinstance(m, np.matrix):
        m = m.A
    elif isinstance(m, torch.Tensor):
        m = m.cpu().numpy()
    return m


def wrapped_partial(func, *args, **kwargs):
    if isinstance(func, partial):
        for kwarg in kwargs:
            func.keywords[kwarg] = kwargs[kwarg]
        return func
    else:
        partial_func = partial(func, *args, **kwargs)
        update_wrapper(partial_func, func)
        # Set class function to partial_func
        for method_name in dir(func):
            if callable(getattr(func, method_name)) and not method_name.startswith("__"):
                setattr(partial_func, method_name, getattr(func, method_name))
        return partial_func


def print_result_for_pl(trainer: pl.Trainer, checkpoint_callback, train_dataloader, val_dataloader, test_dataloader,
                        log_file=log_file):
    last_checkpoint = checkpoint_callback.last_model_path
    print(last_checkpoint)
    if train_dataloader:
        print("train_dataloader, last_checkpoint", trainer.test(dataloaders=train_dataloader, ckpt_path=last_checkpoint,
                                                                verbose=False), flush=True)
    if val_dataloader:
        print("val_dataloader, last_checkpoint", trainer.test(dataloaders=val_dataloader, ckpt_path=last_checkpoint,
                                                              verbose=False), flush=True)
    if test_dataloader:
        print("test_dataloader, last_checkpoint", trainer.test(dataloaders=test_dataloader, ckpt_path=last_checkpoint,
                                                               verbose=False), flush=True)
    print(checkpoint_callback.best_model_path)
    best_model_path = checkpoint_callback.best_model_path
    print(best_model_path)
    train_result = {}
    if train_dataloader:
        train_result = trainer.test(dataloaders=train_dataloader, ckpt_path=best_model_path,
                                    verbose=False)[0]
    if val_dataloader:
        val_result = trainer.test(dataloaders=val_dataloader, ckpt_path=best_model_path,
                                  verbose=False)[0]
    if test_dataloader:
        start_testing_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        test_result = trainer.test(dataloaders=test_dataloader, ckpt_path=best_model_path,
                                   verbose=False)[0]
        end_testing_time = CM.Utilities.TimeElapsed(Unit=False, LastTime=False)
        print(f"Testing time elapsed: {end_testing_time - start_testing_time} sec")
    if train_dataloader:
        print("train_dataloader, best_model", train_result)
    if val_dataloader:
        print("val_dataloader, best_model", val_result)
    if test_dataloader:
        print("test_dataloader, best_model", test_result)
        print(f"{best_model_path},{test_result['test/accuracy_epoch']},"
              f"{test_result['test/ap_epoch']},{test_result['test/topk_precision_epoch']},"
              f"{test_result['test/macro_f1_epoch']},{test_result['test/f1_anomaly_epoch']},"
              f"{test_result['test/auc_epoch']},{test_result.get('test/best_anomaly_f1_epoch', test_result['test/f1_anomaly_epoch'])},"
              f"{test_result.get('test/best_macro_f1_epoch', test_result['test/macro_f1_epoch'])},{test_result['test/loss_epoch']},"
              f"{train_result.get('test/loss_epoch', None)}", flush=True)
        with open(log_file, "a+") as fd:
            print(f"{best_model_path},{test_result['test/accuracy_epoch']},"
                  f"{test_result['test/ap_epoch']},"
                  f"{test_result['test/topk_precision_epoch']},"
                  f"{test_result['test/macro_f1_epoch']}," 
                  f"{test_result['test/f1_anomaly_epoch']},"
                  f"{test_result['test/auc_epoch']},"
                  f"{test_result.get('test/best_anomaly_f1_epoch', test_result['test/f1_anomaly_epoch'])},"
                  f"{test_result.get('test/best_macro_f1_epoch', test_result['test/macro_f1_epoch'])},"
                  f"{test_result['test/loss_epoch']},"
                  f"{train_result.get('test/loss_epoch', None)}", file=fd, flush=True)
        return test_result.get('test/best_anomaly_f1_epoch', test_result['test/f1_anomaly_epoch']), \
            test_result['test/auc_epoch'], test_result['test/ap_epoch'], \
            test_result.get('test/best_macro_f1_epoch', test_result['test/macro_f1_epoch'])
    else:
        return None, None, None, None, None, None

data_folder = "./data"
YelpchiEdgeToNodeDataFolder = os.path.join(data_folder, "Yelpchi/edge_to_node_data")


max_epochs = 200
learning_rate = 0.001
print("max_epochs2:", max_epochs)