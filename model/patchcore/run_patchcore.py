import contextlib
import logging
import os
import sys

import click
import numpy as np
import torch

import model.patchcore.backbones
import model.patchcore.common
import model.patchcore.metrics
import model.patchcore.patchcore
import model.patchcore.sampler
import model.patchcore.utils

from data_loader import TrainDataModule, TrainDataset, TestDataset, get_all_test_dataloaders, get_test_dataloader

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mri_images": ["PNI_Medical_Anomaly.data_loader", "TrainDataset"], }

# @click.group(chain=True)
# @click.argument("results_path", type=str)
# @click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
# @click.option("--seed", type=int, default=0, show_default=True)
# @click.option("--log_group", type=str, default="group")
# @click.option("--log_project", type=str, default="project")
# @click.option("--save_segmentation_images", is_flag=True)
# @click.option("--save_patchcore_model", is_flag=True)
# def main(**kwargs):
#     pass


# @main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
):
    # methods = {key: item for (key, item) in methods}

    run_save_path = model.patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)

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

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        model.patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name
        print("sanity check")
        print("name: " + dataset_name)
        
        with device_context:
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

            torch.cuda.empty_cache()
            aggregator = {"pathologies": [], "scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                for name in methods["names"]:
                    scores, segmentations, labels_gt, masks_gt = PatchCore._predict_dataloader(
                        dataloaders["testing"][name]
                    )
                    aggregator["pathologies"].append(name)
                    aggregator["scores"].append(scores)
                    aggregator["segmentations"].append(segmentations)
                    
            # (Optional) Plot example images.
            if save_segmentation_images:
                for name in methods["names"]:
                    image_paths = [
                        x for x in dataloaders["testing"][name].img_paths
                    ]
                    mask_paths = [
                        x[3] for x in dataloaders["testing"][name].pos_mask_paths
                    ]

                    image_save_path = os.path.join(
                        run_save_path, name, "segmentation_images", 
                    )
                    os.makedirs(image_save_path, exist_ok=True)
                    
                    model.patchcore.utils.plot_segmentation_images(
                        image_save_path,
                        image_paths,
                        segmentations,
                        scores,
                        mask_paths
                    )
            for name in methods["names"]:
                LOGGER.info("Computing evaluation metrics for pathology "+ name)
                auroc = model.patchcore.metrics.compute_imagewise_retrieval_metrics(
                    scores, [1 for x in range(len(scores))]
                )["auroc"]

                # Compute PRO score & PW Auroc for all images
                pixel_scores = model.patchcore.metrics.compute_pixelwise_retrieval_metrics(
                    segmentations, masks_gt
                )
                full_pixel_auroc = pixel_scores["auroc"]

                # Compute PRO score & PW Auroc only images with anomalies
                sel_idxs = []
                for i in range(len(masks_gt)):
                    if np.sum(masks_gt[i]) > 0:
                        sel_idxs.append(i)
                pixel_scores = model.patchcore.metrics.compute_pixelwise_retrieval_metrics(
                    [segmentations[i] for i in sel_idxs],
                    [masks_gt[i] for i in sel_idxs],
                )
                anomaly_pixel_auroc = pixel_scores["auroc"]

                result_collect.append(
                    {
                        "pathology_name": name,
                        "instance_auroc": auroc,
                        "full_pixel_auroc": full_pixel_auroc,
                        "anomaly_pixel_auroc": anomaly_pixel_auroc,
                    }
                )

                for key, item in result_collect[-1].items():
                    if key != "dataset_name":
                        LOGGER.info("{0}: {1:3.3f}".format(key, item))

                # (Optional) Store PatchCore model for later re-use.
                # SAVE all patchcores only if mean_threshold is passed?
                if save_patchcore_model:
                    patchcore_save_path = os.path.join(
                        run_save_path, "models", dataset_name
                    )
                    os.makedirs(patchcore_save_path, exist_ok=True)
                    for i, PatchCore in enumerate(PatchCore_list):
                        prepend = (
                            "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                            if len(PatchCore_list) > 1
                            else ""
                        )
                        PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["pathology_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    model.patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


# @main.command("patch_core")
# # Pretraining-specific parameters.
# @click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
# @click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# # Parameters for Glue-code (to merge different parts of the pipeline.
# @click.option("--pretrain_embed_dimension", type=int, default=1024)
# @click.option("--target_embed_dimension", type=int, default=1024)
# @click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
# @click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# # Nearest-Neighbour Anomaly Scorer parameters.
# @click.option("--anomaly_scorer_num_nn", type=int, default=5)
# # Patch-parameters.
# @click.option("--patchsize", type=int, default=3)
# @click.option("--patchscore", type=str, default="max")
# @click.option("--patchoverlap", type=float, default=0.0)
# @click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# # NN on GPU.
# @click.option("--faiss_on_gpu", is_flag=True)
# @click.option("--faiss_num_workers", type=int, default=8)
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = model.patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = model.patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            patchcore_instance = model.patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


# @main.command("sampler")
# @click.argument("name", type=str)
# @click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return model.patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return model.patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return model.patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)


# @main.command("dataset")
# @click.argument("name", type=str)
# @click.argument("data_path", type=click.Path(exists=True, file_okay=False))
# @click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
# @click.option("--train_val_split", type=float, default=1, show_default=True)
# @click.option("--batch_size", default=2, type=int, show_default=True)
# @click.option("--num_workers", default=8, type=int, show_default=True)
# @click.option("--resize", default=256, type=int, show_default=True)
# @click.option("--imagesize", default=224, type=int, show_default=True)
# @click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)



# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
#     main()