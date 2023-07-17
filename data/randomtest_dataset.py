import os
import torch
import nibabel as nib
import pickle
from data.base_dataset import BaseDataset
from data_loading.random_sample_loader import  RandomSampleLoader
from processing.processing_angular_sr import ProcessingAngularSR


class RandomTestDataset(BaseDataset):  # Can only run sequentially

    def __init__(self, opt):
        """
        Initialize this dataset class.
        """
        BaseDataset.__init__(self, opt)

        hcp_split_path = os.path.join(self.root, 'HCP_list_split_80_20.pickle')
        if not os.path.exists(hcp_split_path):
            raise IOError(
                "hcp splited list path, {}, could not be resolved".format(hcp_split_path)
            )
            exit(0)
        
        with open(hcp_split_path, 'rb') as handle:
            sub_list = pickle.load(handle)
            if opt.isTrain:
                self.sample_list = sub_list['train']
            else:
                self.sample_list = sub_list['test']

        # debugging uncomment it
        # self.sample_list = [self.sample_list[0]]

        self.sample_loader = RandomSampleLoader(root = self.root, isTrain = opt.isTrain, opt = opt)
        self.processing = ProcessingAngularSR(norm = opt.data_norm, bounding = opt.bounding
                        , crop = True, isTrain = opt.isTrain, patch_shape = opt.patch_shape , patch_overlap = opt.patch_overlap
                        , statistics_path  = os.path.join(opt.checkpoints_dir, opt.name))


    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--random_suffix', type=str, default='_0.mif.gz', help='suffix of ten random dwi')
        parser.add_argument('--shuffle_direction', action='store_true', help='whether to shuffle 6 directions')
        parser.add_argument('--shuffle_b0', action='store_true', help='whether to shuffle b0')

        return parser


    def __len__(self):
        """Return the total number of images in the dataset."""

        return len(self.sample_list)


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        uni = self.sample_list[index]
        # debug uncomment it
        # uni = '111312'
        sample = self.sample_loader.load_sample(uni)
        self.preprocessing(sample)
        
        dwi_patches = []
        t1_patches = []
        gt_dti = torch.from_numpy(sample.gt_dti)
        wm_mask = torch.from_numpy(sample.wm_mask)

        for patch_coord in sample.coords_data:
            x_start = patch_coord['x_start']
            x_end = patch_coord['x_end']
            y_start = patch_coord['y_start']
            y_end = patch_coord['y_end']
            z_start = patch_coord['z_start']
            z_end = patch_coord['z_end']

            dwi_patch = torch.from_numpy(sample.dwi[x_start:x_end, y_start:y_end, z_start:z_end])
            t1_patch = torch.from_numpy(sample.t1[x_start:x_end, y_start:y_end, z_start:z_end])
            dwi_patch = dwi_patch.permute(3,0,1,2)
            if self.opt.shuffle_direction:
                # dwi_patch = dwi_patch[torch.randperm(7),:,:,:]
                if self.opt.shuffle_b0:
                    isb0 = True
                    while isb0:
                        randperm7 = torch.randperm(7)
                        if randperm7[0] != 0:
                            isb0 = False
                            break
                    dwi_patch = dwi_patch[randperm7,:,:,:]
                else:
                    random6 = dwi_patch[1:7,:,:,:]
                    dwi_patch = torch.cat((dwi_patch[0,:,:,:].unsqueeze(0)
                                                , random6[torch.randperm(6),:,:,:]), dim = 0)
            t1_patch = t1_patch.unsqueeze(0)
            dwi_patches.append(dwi_patch)
            t1_patches.append(t1_patch)

        dwi_patches = torch.stack(dwi_patches, dim = 0)
        t1_patches = torch.stack(t1_patches, dim = 0)
   
        return {'index': sample.index, 'dwi' : dwi_patches, 't1': t1_patches, 'gt_dti': gt_dti, 'wm_mask': wm_mask}


    def preprocessing(self, sample):
        self.processing.preprocessing(sample)
        self.preprocessed_sample = sample
        # print(sample.wm_mask.shape)
        

    def postprocessing(self, preds):
        prediction = self.processing.postprocessing(self.preprocessed_sample, preds)
        print('shape of post-processed prediction: {:}'.format(prediction.shape))

        return prediction


    def save_prediction(self, prediction, suffix):
        output_name = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.results_dir
                    , self.preprocessed_sample.index + suffix + '.nii.gz')

        self.sample_loader.save_prediction(prediction, self.preprocessed_sample.aff, output_name = output_name)
