import os
from data_loading.interfaces.angular_sr_nifti_io import AngularSRNiftiIO
from data_loading.angular_sr_sample import AnuglarSRSample


class RandomSampleLoader:

    def __init__(self, root, isTrain = True, opt = None):
        self.root = root
        self.interface = AngularSRNiftiIO()
        self.isTrain = isTrain
        self.suffix = opt.random_suffix


    #---------------------------------------------#
    #                Load a sample                #
    #---------------------------------------------#
    # Load a sample from the data set
    def load_sample(self, uni):
        # load data of a sample
        if self.isTrain:
            dwi = self.load_dwi(dir_name = 'dwi', uni = uni,  suffix = '_DWI_processed_b1000.mif.gz', random = True)
        else:
            dwi = self.load_dwi(dir_name = 'self/10_random_dwi_mif', uni = uni,  suffix = self.suffix, random = False)
        gt_dti, affine = self.load_gt(dir_name = 'self/dti_nii_tensor_only', uni = uni,  suffix = '_DTI.nii.gz')
        t1 =  self.load_t1(dir_name = 'self/T1_registered_RAS', uni = uni,  suffix = '_t1.nii.gz')
        brain_mask = self.load_brain_mask(dir_name = 'self/mask_RAS', uni = uni,  suffix = '_DWI_brainmask.mif.gz')
        wm_mask = self.load_wm_mask(dir_name = 'self/wm_mask_RAS', uni = uni,  suffix = '_wm.mif.gz')

        sample = AnuglarSRSample(uni, dwi, affine)
        sample.add_gt_dti(gt_dti)
        sample.add_t1(t1)
        sample.add_brain_mask(brain_mask)
        sample.add_wm_mask(wm_mask)

        return sample

    #---------------------------------------------#
    #                Load DWI                     #
    #---------------------------------------------#
    # Load the data of a sample from the data set
    def load_dwi(self, dir_name , uni , suffix, random = True):
        if random:
            dwi_path = os.path.join(self.root, dir_name,  uni + suffix)
            dwi = self.interface.load_downsampled_mif_dwi(dwi_path, index = uni)
        else:
            dwi_path = os.path.join(self.root, dir_name, uni, uni + suffix)
            dwi = self.interface.load_data(dwi_path)

        return dwi

    #---------------------------------------------#
    #                Load T1                      #
    #---------------------------------------------#
    # Load the data of a sample from the data set
    def load_t1(self, dir_name , uni , suffix):
        t1_path = os.path.join(self.root, dir_name,  uni + suffix)
        t1 = self.interface.load_data(t1_path)

        return t1

    #---------------------------------------------#
    #                Load GT                      #
    #---------------------------------------------#
    # Load the data of a sample from the data set
    def load_gt(self, dir_name , uni , suffix):
        gt_path = os.path.join(self.root, dir_name,  uni + suffix)
        gt, affine = self.interface.load_data(gt_path, needs_affine = True)

        return gt, affine

    #---------------------------------------------#
    #                Load Brain  mask             #
    #---------------------------------------------#
    def load_brain_mask(self, dir_name , uni , suffix):
        brain_mask_path = os.path.join(self.root, dir_name,  uni + suffix)
        brain_mask = self.interface.load_data(brain_mask_path)

        return brain_mask

    #---------------------------------------------#
    #                Load WM mask                 #
    #---------------------------------------------#
    def load_wm_mask(self, dir_name , uni , suffix):
        wm_mask_path = os.path.join(self.root, dir_name,  uni + suffix)
        wm_mask = self.interface.load_data(wm_mask_path)

        return wm_mask

    def load_roi(self, dir_name , uni , suffix):
        roi_path = os.path.join(self.root, dir_name, uni, uni + suffix)
        roi = self.interface.load_data(roi_path)

        return roi

    def load_fa(self, dir_name , uni , suffix):
        fa_path = os.path.join(self.root, dir_name, uni + '_dti', uni + suffix)
        fa = self.interface.load_data(fa_path)

        return fa

    def load_md(self, dir_name , uni , suffix):
        md_path = os.path.join(self.root, dir_name, uni + '_dti', uni + suffix)
        md = self.interface.load_data(md_path)

        return md
        
    def save_prediction(self, prediction, affine, output_name):
        self.interface.save_prediction(prediction, affine, output_name)

    





        



