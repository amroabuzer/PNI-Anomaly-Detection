import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml
import contextlib
import logging
import os
import numpy as np
import torch
import model.patchcore.backbones
import model.patchcore.common
import model.patchcore.metrics
import model.patchcore.patchcore
import model.patchcore.sampler
import model.patchcore.utils
import pickle

from data_loader import TrainDataModule, get_all_test_dataloaders, get_test_dataloader
from model.patchcore.run_patchcore import dataset, run, patch_core, sampler
from model.patchcore.patchcore import PatchCore
from data_loader import TrainDataModule, get_all_test_dataloaders

with open('./configs/autoencoder_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Reproducibility
pl.seed_everything(config['seed'])

train_data_module = TrainDataModule(
    split_dir=config['split_dir'],
    target_size=config['target_size'],
    batch_size=config['batch_size'])

test_dataloaders = get_all_test_dataloaders(config['split_dir'], config['target_size'], config['batch_size'])

# Plot some images
batch = next(iter(train_data_module.train_dataloader()))

# Print statistics
print(f"Batch shape: {batch.shape}")
print(f"Batch min: {batch.min()}")
print(f"Batch max: {batch.max()}")

img_num = min(5, batch.shape[0])
print(img_num)

plots = False
if plots:
    fig, ax = plt.subplots(1, img_num, figsize=(15, img_num))
    for i in range(img_num):
        ax[i].imshow(batch[i].squeeze().permute(1,2,0)) if img_num>1 else batch[i].permute(1,2,0)
        ax[i].axis('off') if img_num>1 else ax.axis('off')
    # plt.show()
    plt.imshow(batch[i].permute(1,2,0))
    plt.show()

# we run the patchcore model

methods ={}
methods["get_dataloaders"] = {"training": train_data_module,
            "validation": train_data_module,
            "testing": test_dataloaders,
            "names": [
        'absent_septum',
        'artefacts',
        'craniatomy',
        'dural',
        'ea_mass',
        'edema',
        'encephalomalacia',
        'enlarged_ventricles',
        'intraventricular',
        'lesions',
        'mass',
        'posttreatment',
        'resection',
        'sinus',
        'wml',
        'other'
    ]}

n1, f1 = sampler('identity', 0.01)
methods[n1] =f1
# we also pick greedy_coreset because that is what PNI paper mentions
# 0.1 is the default value according to documentation
n2, f2 =patch_core(["wideresnet50"], ['layer2', 'layer3'], 1024, 1024, "mean", "mean", 5, 3, "max", 0.0, [], False, 8)
methods[n2] = f2
# rest of values are just default


results_path = "results"
# make sure gpus is correctly used
gpu = [0]
seed = config["seed"]
log_group = "group"
log_project = "project"
save_segmentation_images = True
save_patchcore_model = True

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mri_images": ["PNI_Medical_Anomaly.data_loader", "TrainDataset"], }


run_save_path = model.patchcore.utils.create_storage_folder(
    results_path, log_project, log_group, mode="iterate"
)

dataloaders = methods["get_dataloaders"]

device = model.patchcore.utils.set_torch_device(gpu)
# Device context here is specifically set and used later
# because there was GPU memory-bleeding which I could only fix with
# context managers.
device_context = (
    torch.cuda.device("cuda:{}".format(device.index))
    if "cuda" in device.type.lower()
    else contextlib.suppress()
)

result_collect = []

model.patchcore.utils.fix_seeds(seed, device)

dataset_name = dataloaders["training"].name
print("sanity check")
print("name: " + dataset_name)

train_patch = False
train_PNI = True
file_path = os.path.join('saved_PatchCore', '2_Dummy_PatchCore')


if train_patch:
    # with device_context:
    torch.cuda.empty_cache()
    imagesize = dataloaders["training"].input_size
    sampler = methods["get_sampler"](
        device,
    )
    PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
    if len(PatchCore_list) > 1:
        LOGGER.info(
            "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
        )
    for i, PatchCore in enumerate(PatchCore_list):
        torch.cuda.empty_cache()
        if PatchCore.backbone.seed is not None:
            model.patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
        LOGGER.info(
            "Training models ({}/{})".format(i + 1, len(PatchCore_list))
        )
        torch.cuda.empty_cache()
        PatchCore.fit(dataloaders["training"].train_dataloader())
        
    os.makedirs(file_path, exist_ok=True)
    for i, PatchCore in enumerate(PatchCore_list):
        prepend = (
            "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
            if len(PatchCore_list) > 1
            else ""
        )
        PatchCore.save_to_path(file_path, prepend)
elif train_PNI:
    nn_method = model.patchcore.common.FaissNN(False, 8)
    patch_core = PatchCore(device)
    patch_core.load_from_path(file_path, device, nn_method)
    patch_core.train_PNI(dataloader=dataloaders["training"], path_to_PNI=os.path.join("MLP_histograms", "PNI_1"))
    # patch_core.generate_PNI_dataset(dataloader=dataloaders["training"], csv_file=os.path.join(file_path, "PNI_dataset.csv"))
else:
    nn_method = model.patchcore.common.FaissNN(False, 8)
    patch_core = PatchCore(device)
    patch_core.load_from_path(file_path, device, nn_method)
    patch_core.load_PNI(path_to_model=os.path.join("MLP_histograms", "PNI_1"))
    patch_core.PNI_predict("data/fastMRI/brain_mid_png/file_brain_AXT1_201_6002688.png")