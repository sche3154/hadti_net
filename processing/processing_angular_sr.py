#==============================================================================#
#  Author:       Dominik Müller, Sheng                                         #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
import nibabel as nib
import os
import numpy as np
from processing.abstract_subfunction import Abstract_Subfunction
from utils.patch_operations import find_bounding_box, slice_matrix, concat_matrices


#-----------------------------------------------------#
#          Subfunction class: Normalization           #
#-----------------------------------------------------#
""" A Normalization Subfunction class which normalizes the intensity pixel values of an image using
    the Z-Score technique (default setting), through scaling to [0,1] or to grayscale [0,255].

Args:
    mode (string):          Mode which normalization approach should be performed.
                            Possible modi: "z-score", "minmax" or "grayscale"

Methods:
    __init__                Object creation function
    preprocessing:          Pixel intensity value normalization the imaging data
    postprocessing:         Do nothing
"""
class ProcessingAngularSR(Abstract_Subfunction):        
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, norm ='z_scoreV2', bounding = False,
             crop = False, patch_shape = 64, patch_overlap = 32, 
             padding = False, padding_size = 256, isTrain = True, statistics_path = None):
        Abstract_Subfunction.__init__(self)
        self.norm = norm
        self.bounding = bounding
        self.crop = crop
        self.padding = padding
        self.isTrain = isTrain
        self.patch_shape = (patch_shape, patch_shape, patch_shape)
        self.overlap = (patch_overlap, patch_overlap, patch_overlap)
        self.padding_size = padding_size
        self.patchwise_skip_blanks = True

        if self.isTrain is False:
            self.all_gt_means = np.load(os.path.join(statistics_path, 'z_scoreV2_gt_mean.npy'), allow_pickle=True)
            self.all_gt_stds =  np.load(os.path.join(statistics_path, 'z_scoreV2_gt_std.npy'), allow_pickle=True)
    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample):

        # Access input and mask
        dwi = sample.dwi.squeeze()
        # t1 = np.transpose(sample.t1.squeeze(), (0,2,1))
        t1 = sample.t1.squeeze()
        gt_dti = sample.gt_dti.squeeze()
        brain_mask = sample.brain_mask.squeeze()

        print('Normalize DWI+T1 (z_score) the {:}'.format(sample.index))
        normalized_dwi = self.z_score_normlization(dwi, brain_mask)
        normalized_t1  = self.z_score_normlization(t1, brain_mask)
        # Update the sample with the normalized image
        sample.dwi = normalized_dwi
        sample.t1 = normalized_t1
        
        # Perform z-score normalization
        if self.norm == 'z_score':
            if self.isTrain:
                print('Normalize DTI (z_score) the {:}'.format(sample.index))
                normalized_gt_dti = self.z_score_normlization(gt_dti, brain_mask)
                sample.gt_dti = normalized_gt_dti

        # Perform z-scoreV2 normalization
        if self.norm == 'z_scoreV2': 
            if self.isTrain:
                print('Normalize DTI (z_scoreV2) the {:}'.format(sample.index))
                normalized_gt_dti, means, stds = self.z_score_normlizationV2(gt_dti, brain_mask)
                sample.gt_dti = normalized_gt_dti
                sample.add_gt_dti_mean(means)
                sample.add_gt_dti_std(stds)
  
        if self.bounding:
            print('Bounding the {:}'.format(sample.index))
            # print('Before bounding dwi_patch: {:}, gt_dti: {:}, t1: {:}, wm_mask : {:}'.format(sample.dwi.shape
        # ,sample.gt_dti.shape, sample.t1.shape, sample.wm_mask.shape))
            x_l1, x_l2, y_l1, y_l2, z_l1, z_l2 = find_bounding_box(brain_mask)
            bb = [x_l1, x_l2, y_l1, y_l2, z_l1, z_l2]
            
            sample.dwi = sample.dwi[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
            sample.t1 = sample.t1[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
            if self.isTrain:
                sample.gt_dti = sample.gt_dti[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
                sample.wm_mask = sample.wm_mask[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
            sample.add_bb(bounding_box = bb)
        #     print('After bounding dwi_patch: {:}, gt_dti: {:}, t1: {:}, wm_mask : {:}'.format(sample.dwi.shape
        # ,sample.gt_dti.shape, sample.t1.shape, sample.wm_mask.shape))

        if self.crop:
            print('Cropping the {:}'.format(sample.index))
            coords_img_data = self.analysis_patchwise_grid(sample.dwi, index = sample.index)
            sample.add_coords_data(coords_img_data)


    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, sample, preds):

        gt_dti = sample.gt_dti.squeeze()
        brain_mask = sample.brain_mask.squeeze()
        if self.crop:
            print('Un-Cropping the {:}'.format(sample.index))
            prediction = concat_matrices(patches=preds,
                                    image_size = (sample.dwi.shape[0],sample.dwi.shape[1],sample.dwi.shape[2],6),
                                    window= self.patch_shape,
                                    overlap= self.overlap,
                                    three_dim= True,
                                    coords=sample.coords_data)
            # print('After uncropping prediction {:}'.format(prediction.shape))

        if self.bounding:
            print('Un-bounding the {:}'.format(sample.index))
            tmp = np.zeros(gt_dti.shape)
            bb = sample.bb
            tmp[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]] = prediction
            prediction = tmp

        if self.norm == 'z_score':
            # Scaling all voxels back to original range
            print('Un-Normalization pred (z_score) the {:}'.format(sample.index))
            prediction[brain_mask ==1] = prediction[brain_mask ==1] * np.std(gt_dti[brain_mask ==1]) 
            prediction[brain_mask ==1] = prediction[brain_mask ==1] + np.mean(gt_dti[brain_mask ==1]) 
            prediction[brain_mask != 1] = 0
        #     print('After unnormlization prediction: {:}, gt_dti: {:}'.format(prediction.shape
        # ,gt_dti.shape))

        if self.norm == 'z_scoreV2':
            # Scaling all voxels back to original range
            print('Un-Normalization pred (z_scoreV2) the {:}'.format(sample.index))

            for c in range(prediction.shape[3]):
                mean = self.all_gt_means[c]
                std = self.all_gt_stds[c]
                prediction[:, :, :, c][brain_mask == 1] = prediction[:, :, :, c][brain_mask == 1] * std + mean

            prediction[brain_mask != 1] = 0
        # if self.norm == 'z_scoreV2':
        #     prediction = prediction * np.repeat(np.expand_dims(np.std(gt_dti, axis = 3), axis =3), repeats = 6, axis = 3)
        #     prediction = prediction + np.repeat(np.expand_dims(np.mean(gt_dti, axis = 3), axis =3),  repeats = 6, axis = 3)
        #     prediction[brain_mask != 1] = 0

        return prediction


    def z_score_normlization(self, image, mask):
        image_mean = np.mean(image[mask == 1])
        image_std = np.std(image[mask == 1])
        # Scaling
        image[mask == 1] = (image[mask == 1] - image_mean) / image_std
        image[mask != 1] = 0

        return image


    def z_score_normlizationV2(self, image, mask):
        # mean, std 
        means  = np.mean(image[mask==1], axis = (0)) # (6, )
        stds = np.std(image[mask==1], axis = (0))  # (6,)

        for c in range(image.shape[3]):
            image[:,:,:,c][mask==1] =  (image[:,:,:,c][mask == 1] - means[c])/stds[c]
  
        image[mask != 1] = 0

        return image, means, stds

    # def z_score_normlizationV3(self, image, mask, axis = None):
    #     # mean, std of 6 coefficients in each voxel 
    #     image_mean_coefficient_wise = np.repeat(np.expand_dims(np.mean(image, axis = axis), axis), repeats = image.shape[3], axis = axis)
    #     image_std_coefficient_wise = np.repeat(np.expand_dims(np.std(image, axis = axis), axis), repeats = image.shape[3], axis = axis)
    #     # Scaling
    #     image[mask == 1] = (image[mask == 1] - image_mean_coefficient_wise[mask == 1]) / image_std_coefficient_wise[mask == 1]
    #     image[mask != 1] = 0
    #     return image


    def analysis_patchwise_grid(self, image, index =None):
        # Slice image into patches
        patch_imgs, coord_dicts = slice_matrix(image, self.patch_shape, self.overlap,
                                   three_dim = True, index = index , save_coords=True)
        # Skip blank patches (only background)
        if self.isTrain and self.patchwise_skip_blanks:
            # Iterate over each patch
            for i in reversed(range(0, len(patch_imgs))):
                # IF patch DON'T contain anything -> remove it
                if np.sum(patch_imgs[i]) == 0:
                    del patch_imgs[i]
                    del coord_dicts[i]
        # Concatenate a list of patches coordinates into a single numpy array
        coords_img_data = np.stack(coord_dicts, axis=0)
        
        # Return preprocessed data tuple
        return coords_img_data


    def padding_image(self, image):
        padx1  = int((self.padding_size - image.shape[0])/2)
        padx2  = self.padding_size - (padx1 + image.shape[0])
        pady1  = int((self.padding_size - image.shape[1])/2)
        pady2  = self.padding_size - (pady1 + image.shape[1])
        padz1  = int((self.padding_size - image.shape[2])/2)
        padz2  = self.padding_size - (padz1 + image.shape[2])

        if len(image.shape) > 3:
            image_padded = np.pad(image, ((padx1, padx2), (pady1, pady2)
            , (padz1, padz2), (0,0)), 'constant')
        else:
            image_padded = np.pad(image, ((padx1, padx2), (pady1, pady2)
            , (padz1, padz2)), 'constant')

        return image_padded