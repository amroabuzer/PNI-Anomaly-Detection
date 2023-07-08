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

from data_loader import TrainDataModule, get_all_test_dataloaders, get_test_dataloader
from model.patchcore.run_patchcore import dataset, run, patch_core, sampler
from model.patchcore.patchcore import PatchCore
from data_loader import TrainDataModule, get_all_test_dataloaders
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from tqdm import tqdm
from model.cutpaste import CutPasteNormal,CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn
from model.refinement_model import Refinement
# from model.eval import eval_model

with open('./configs/autoencoder_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

dummy_array = np.ones((1,1,224,224))

after_cutpaste_transform = transforms.Compose([])
after_cutpaste_transform.transforms.append(transforms.ToTensor())
after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]))

cutpaste_type = CutPasteNormal

test_epochs = 10

train_transform = transforms.Compose([])
#train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
# train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
train_transform.transforms.append(transforms.Resize(config['target_size']))
train_transform.transforms.append(cutpaste_type(transform = after_cutpaste_transform))
# train_transform.transforms.append(transforms.ToTensor())


# Reproducibility
pl.seed_everything(config['seed'])
train_data_module = TrainDataModule(
    split_dir=config['split_dir'],
    target_size=config['target_size'],
    batch_size=config['batch_size'],
    train_transform=train_transform, 
    cutpaste = True)

test_dataloaders = get_all_test_dataloaders(config['split_dir'], config['target_size'], config['batch_size'])

# Plot some images
batch, gt = next(iter(train_data_module.train_dataloader()))

# Print statistics
print(f"Batch shape: {batch.shape}")
print(f"Batch min: {batch.min()}")
print(f"Batch max: {batch.max()}")

img_num = min(5, batch.shape[0])

# fig, ax = plt.subplots(1, img_num, figsize=(15, img_num))
# if (img_num)>1:
#     for i in range(img_num):
#         ax[i].imshow(batch[i].squeeze().permute(1,2,0))
#         ax[i].axis('off')
# else:
#     ax.imshow(batch.squeeze().permute(1,2,0))
#     ax.axis('off')
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

n1, f1 = sampler('approx_greedy_coreset', 0.1 )
methods[n1] =f1
# we also pick greedy_coreset because that is what PNI paper mentions
# 0.1 is the default value according to documentation
n2, f2 =patch_core(["resnet50"], ['layer2', 'layer3'], 1024  , 1024  , "mean", "mean", 5, 3, "max", 0.0, [], True, 8)
methods[n2] = f2
# rest of values are just default 


results_path = r"D:\D_Coding\Coding\PNI_Medical_Anomaly\results"
# make sure gpus is correctly used
gpu = [0]
seed = config["seed"]
log_group = "group"
log_project = "project"
save_segmentation_images = True
save_patchcore_model = False

neighborhood_info = False
position_info = False
refinement = False

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

num_classes = 2

model = Refinement(config)

model.to(torch.device('cuda'))

def get_data_inf():
    while True:
        for out in enumerate(train_data_module.train_dataloader()):
            yield out
def get_val_inf():
    while True:
        for out in enumerate(train_data_module.val_dataloader()):
            yield out
            

dataloader_inf =  get_data_inf()
optimizer = model.configure_optimizers()
loss_fn = lambda logits, y: model.loss_fn(logits, y)



for step in tqdm(range(config['num_epochs'])):
    epoch = int(step / 1)

    batch_embeds = []
    batch_idx, [data, gt] = next(dataloader_inf)
    xs = [x.to(device) for x in data]
    gts = [y.to(device) for y in gt]


    # zero the parameter gradients
    optimizer.zero_grad()
    for i,x in enumerate(xs):
      
      if len(x.shape) == 3:
        x = x[None,:,:,:]
        
      xc = dummy_array
    #   xc = PatchCore.predict(x)
      xc = torch.FloatTensor(xc).to(device)

      if len(xc.shape) == 3:
        xc = xc[None,:,:,:]

      logits = model(x, xc)

      y = gt[i]
      loss = loss_fn(logits, y.cuda())

      loss.backward()
      optimizer.step()

      if test_epochs > 0 and epoch % test_epochs == 0:
          # run auc calculation
          #TODO: create dataset only once.
          #TODO: train predictor here or in the model class itself. Should not be in the eval part
          #TODO: we might not want to use the training datat because of droupout etc. but it should give a indecation of the model performance???
          # batch_embeds = torch.cat(batch_embeds)
          # print(batch_embeds.shape)
          model.eval()
          _, [val_data, val_gt] = next(get_val_inf())
          x_val = [x.to(device) for x in val_data]
          val_gts = [y.to(device) for y in val_gt]
          
          tps, fps, tns, fns= 0, [], [], 0
          precs, recs, f1s = [], [], []
          for i, x in enumerate(x_val):
            #   val_ano_map = PatchCore.predict(x)
              val_ano_map = torch.FloatTensor(dummy_array).to(device)
              if len(x.shape) == 3:
                x = x[None,:,:,:]
              
              if len(val_ano_map.shape) == 3:
                val_ano_map = val_ano_map[None,:,:,:]
              
              val_out = model(x, val_ano_map)
              y = val_gts[i] 
              neg_y = torch.where(y==1, 0, 1)
              
              x_pos = val_out * y
              x_neg = val_out * neg_y
              
              x_pos = x_pos.detach().cpu().numpy()
              x_neg = x_neg.detach().cpu().numpy()
              
              res_anomaly = np.sum(x_pos)
              res_healthy = np.sum(x_neg)
              
              amount_anomaly = np.count_nonzero(x_pos)
              amount_mask = np.count_nonzero(y.detach().cpu().numpy())
              
              tp = 1 if amount_anomaly > 0.1 * amount_mask else 0 ## 10% overlap due to large bboxes e.g. for enlarged ventricles
              tps+= tp
              fn = 1 if tp == 0 else 0
              fns += fn
  
              fp = int(res_healthy / max(res_anomaly, 1))
              fps.append(fp)
              
              precision = tp / max((tp+fp), 1)
              f1 = 2 * (precision * tp) / (precision + tp + 1e-8)
              recall = tp / (tp+fn)
              
              precs.append(precision)
              recs.append(recall)
              f1s.append(f1)
              
          print("************** Precision **************")
          print(np.nanmean(np.array(precs)))
          print()
          print("**************  Recall  **************")
          print(np.nanmean(np.array(recs)))
          print()
          print("**************    F1    **************")
          print(np.nanmean(np.array(f1s)))
          print()
          model.train()