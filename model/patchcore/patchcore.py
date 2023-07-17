"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import math

import model.patchcore
import model.patchcore.backbones
import model.patchcore.common
import model.patchcore.sampler

import torch.nn as nn

LOGGER = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.layers.append( nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),  # Batch normalization
                nn.ReLU(inplace=True)  # ReLU activation
            ))
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        output = self.output_layer(x)
        return output
    


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device
        # self.model = MLP()

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=model.patchcore.sampler.IdentitySampler(),
        nn_method=model.patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        
        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = model.patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = model.patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = model.patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator


        self.anomaly_scorer = model.patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        
        self.dist_scorer = model.patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        
        self.anomaly_segmentor = model.patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )
        

        
        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        
        features = self.featuresampler.run(features)
        
        self.cdist_sampler = model.patchcore.sampler.ApproximateGreedyCoresetSampler(
            percentage = (2048.0 / features.shape[0]),
            device = self.device
        )

        dist_features = self.cdist_sampler.run(features)
        
        self.dist_to_ano_mapper = self.featuresampler._compute_greedy_coreset_indices(features)
        
        self.anomaly_scorer.fit(detection_features=[features])
        self.dist_scorer.fit(detection_features=[dist_features])

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def generate_PNI_dataset(self, dataloader, csv_file):
        
        train_dataloader = dataloader.train_dataloader()
        num_embedded_features = len(self.dist_scorer.detection_features)
        
        csv_file = "PNI_dataset.csv"
        
        with tqdm.tqdm(train_dataloader, desc="Generating Dataset...", leave=False) as data_iterator:
            
                for image in data_iterator:
                    
                    image = image.to(torch.float).to(self.device)
                    _ = self.forward_modules.eval()
                    batchsize = image.shape[0]
                    
                    # for batch in range(batchsize):
                    features, patch_shapes = self._embed(image, provide_patch_shapes=True)
                    features = np.asarray(features)
                        
                    side_dim = patch_shapes[0][0]
                    
                    # should get c_dist closest to each feature in shape of side_patch**2
                    _, _ ,dist_idx = self.dist_scorer.predict([features])
                    dist_idx = dist_idx.reshape(batchsize, side_dim, side_dim)
                    
                    c_one_hot = self.mat_to_onehot(dist_idx, num_embedded_features)
                    
                    for row in range(1, side_dim -1):
                        for col in range(1, side_dim-1):
                            
                            curr_c_one_hot = c_one_hot[:,row,col,:]
                            neighbors = features[:, row-1:row+2, col-1:col+2, :]
                            cat_features = torch.cat((neighbors[:,0,...], 
                                                    neighbors[:,2,...],
                                                    neighbors[:,1,1,:][:,None,:],
                                                    neighbors[:,1,2,:][:,None,:]), dim = 1)

                            np.vstack()
                            np.vstack()
                    
                    np.savetxt()
                    np.savetxt()
                            

    def train_PNI(self, dataloader):
        
        train_dataloader = dataloader.train_dataloader()
        
        #for debugging only:
        self.dist_scorer.detection_features = self.dist_scorer.detection_features[:10] 
        
        num_embedded_features = len(self.dist_scorer.detection_features)
        
        self.histogram = [[0 for x in range(self.dist_scorer.detection_features.shape[0])] for x in range(784)]
        # manually putting in 28^2 since 28^2 is the number of embeddings per pic (probably changes based on size of image)
        
        self.model = MLP(input_size=self.dist_scorer.detection_features.shape[1] * 8,
                         hidden_size=2048,
                         num_layers=10,
                         output_size=self.dist_scorer.detection_features.shape[0])
        
        self.model.train()
        self.model.to(device=self.device)
        optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        epochs = 15
        loss_fn = nn.CrossEntropyLoss()
        val_epochs = 51
        
        with tqdm.tqdm(train_dataloader, desc="Inferring...", leave=False) as data_iterator:
            
            for epoch in range(epochs):
                losses = []
                accs = []
                for image in data_iterator:
                    
                    image = image.to(torch.float).to(self.device)
                    _ = self.forward_modules.eval()
                    batchsize = image.shape[0]
                    
                    features, patch_shapes = self._embed(image, provide_patch_shapes=True)
                    features = np.asarray(features)
                    
                    side_dim = patch_shapes[0][0]
                    embed_dim = features.shape[1]
                    
                    _, _ ,dist_idx = self.dist_scorer.predict([features])
                    
                    #reshape to get batch x features
                    dist_idx = dist_idx.reshape(batchsize, side_dim**2)
                    
                    ## I'm supposed to get a [b, 28, 28, 1024] here
                    features = features.reshape(batchsize, side_dim, side_dim, embed_dim)
                    features = torch.tensor(features).to(self.device)
                    
                    #i goes from 1 -> 784 
                    #idx goes from 0->num_coresets
                    #building our histogram based on closest coreset to each patch
                    for batch in range(batchsize):
                        for i, idx in enumerate(dist_idx[batch,:]):  
                            #for debugging!!!!!!
                            if idx > 9:
                                idx = 9
                            self.histogram[i][idx] += 1 
                    
                    dist_idx = dist_idx.reshape(batchsize, side_dim, side_dim)
                    
                    c_one_hot = self.mat_to_onehot(dist_idx, num_embedded_features)
                    
                    for row in range(1, side_dim -1):
                        for col in range(1, side_dim-1):
                            curr_c_one_hot = c_one_hot[:,row,col,:]
                            
                            neighbors = features[:, row-1:row+2, col-1:col+2, :]
                            
                            # should contain all the features of neighbours minus the current row, col features
                            # concatinating all neighboring features on axis 1 => dims: b, 8, 1024
                            cat_features = torch.cat((neighbors[:,0,...], 
                                                        neighbors[:,2,...],
                                                        neighbors[:,1,1,:][:,None,:],
                                                        neighbors[:,1,2,:][:,None,:]), dim = 1)
                            
                            cat_features = cat_features.reshape(cat_features.shape[0],cat_features.shape[-2] * cat_features.shape[-1])
                            
                            c_pred = self.model(cat_features)
                            
                            loss = loss_fn(c_pred, curr_c_one_hot)
                            
                            loss.backward()
                            optim.step()
                            
                            accuracy = (torch.argmax(c_pred, axis = 1)) - torch.argmax(curr_c_one_hot, axis = 1)
                            
                            losses.append(loss)
                            accs.append(accuracy)
                            
                print("loss: ", sum(loss)/len(loss))
                print("accuracy:", sum(accs)/len(accs))
        
        #normalizing histogram before exiting training:) 
        for i in range(len(self.histogram)):
            for val in self.histogram[i]:
                val /= sum(self.histogram[i])
                
                
    def PNI_predict(self, image):
        self.model.eval()
        features, patch_shapes = self._embed(image, provide_patch_shapes=True)
        
        _, _ ,dist_idx = self.anomaly_scorer.predict([features])[0] # gives me the c_emb idx for each patch 
        c_emb = np.take(self.anomaly_scorer.detection_features, dist_idx) # c_emb based on idx found for each patch 
        side_dim = math.sqrt(features[-1][0])
        embed_dim = features[-1][1]
        
        features = features.reshape(image[0], side_dim, side_dim, embed_dim)
        
        for row in side_dim:
            for col in side_dim:
                
                neighbors = features[:, row-1:row+2, col-1:col+2, :]
                            
                # should contain all the features of neighbours minus the current row, col features
                cat_features = torch.cat(neighbors[0,:], neighbors[2,:], neighbors[1,1], neighbors[1,2], dim = 0)
                
                # make sure we're outputting logits here
                p_cN = self.model(cat_features)
                
                p_cx = self.histogram[row*side_dim + col]
                
                p_cOmega = p_cN * p_cx / 2
                
                # we have to map back to nn of c_dist i.e. to c_emb
                # self.dist_to_ano_mapper should be the size of c_dist and contains 
                # nn indices to c_emb for each c_dist
                
                self.dist_to_ano_mapper
                
                p_cOmega_transformed = np.zeros(self.anomaly_scorer.detection_features.shape[0])
                
                p_cOmega_transformed = np.take(p_cOmega_transformed, self.dist_to_ano_mapper)
                
                p_Phi_c = math.exp(-np.linalg.norm(features - c_emb))
                
                patch_score = p_Phi_c * p_cOmega_transformed
                
    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["dict"]
                image, pos_mask, neg_mask = image
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def mat_to_onehot(self, data_idx, num_classes):
        b, h, w = data_idx.shape
        c_one_hot = torch.zeros(b, h, w, num_classes).to(self.device)
        for batch in range(b):
            for row in range(h):
                for col in range(w):
                    if data_idx[batch][row][col] > 9:
                        data_idx[batch][row][col] = 9
                    c_one_hot[batch, row, col] = torch.eye(num_classes)[data_idx[batch][row][col]]
        
        return c_one_hot
    
    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=True, prepend= "orig_nn"
        )
        self.dist_scorer.save(
            save_path, save_features_separately=True, prepend= "dist_nn"
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: model.patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = model.patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, "orig_nn")
        self.dist_scorer.load(load_path, "dist_nn")


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
