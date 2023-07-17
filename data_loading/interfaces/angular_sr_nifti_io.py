
import nibabel as nib
import numpy as np
import os 
from data_loading.interfaces.abstract_io import Abstract_IO
from utils.mrtrix import *


class AngularSRNiftiIO(Abstract_IO):


    def __init__(self):
        pass

    #---------------------------------------------#
    #                Load data                    #
    #---------------------------------------------#
    # Load the data of a sample from the data set
    def load_data(self, path, index = None, needs_affine = False):
  
        if not os.path.exists(path):
            raise ValueError(
                "Data could not be found \"{}\"".format(path)
            )
            exit(0)

        if path.endswith('.mif.gz') or path.endswith('.mif'):
            vol = load_mrtrix(path)
            data_copied = vol.data.copy()
            affine_copied = vol.transform.copy()
        elif path.endswith('.nii.gz') or path.endswith('.nii'):
            vol = nib.load(path)
            data_copied = vol.get_fdata(dtype = np.float32).copy()
            ## debug
            # print(data_copied.dtype)
            affine_copied = vol.affine.copy()
        else:
            raise IOError('file extension not supported: ' + str(path))
            exit(0)

        # Return volume
        if needs_affine:
            return data_copied, affine_copied
        else:
            return data_copied

    #---------------------------------------------#
    #                Load downsampled DWI         #
    #---------------------------------------------#
    def load_downsampled_mif_dwi(self, path, index = None, needs_affine = False):

        if not os.path.exists(path):
            raise ValueError(
                "Data could not be found \"{}\"".format(path)
            )
            exit(0)

        if path.endswith('.mif.gz') or path.endswith('.mif'):
            mif = load_mrtrix(path)
            affine_copied = mif.transform.copy()
            old_grad = mif.grad
            bvals = old_grad[:, -1]
            bvecs = old_grad[:, :-1]
            print('Random downsampling {:}'.format(index))

            lr_bvecs, lr_bvals, lr_index, b0_index = extract_single_shell(
                bvals, bvecs, directions = 6
            )
            lr_index = np.array(lr_index.tolist())

            mif_b0 = np.mean(mif.data[..., b0_index], axis=-1, keepdims=True)
            downsampled_data_copied = np.concatenate([mif_b0, mif.data[..., lr_index]], axis=-1).copy()

            # print('Finish random downsampling {:}'.format(index))

            if needs_affine:
                return downsampled_data_copied, affine_copied
            else:
                return downsampled_data_copied

        else:
            raise IOError('file extension is not supported for downsampling must mif: ' + str(path))
            exit(0)

    #---------------------------------------------#
    # Load totally random-downsampled DWI         #
    #---------------------------------------------#
    def load_totallyrandom_downsampled_mif_dwi(self, path, index = None, needs_affine = False):
        
        if not os.path.exists(path):
            raise ValueError(
                "Data could not be found \"{}\"".format(path)
            )
            exit(0)

        if path.endswith('.mif.gz') or path.endswith('.mif'):
            mif = load_mrtrix(path)
            affine_copied = mif.transform.copy()
            print('Totally Random downsampling :) {:}'.format(index))

            b0_index = np.array([0])
            lr_index = np.random.permutation([i for i in range(1,91)])[0:6]
            mif_b0 = np.mean(mif.data[..., b0_index], axis=-1, keepdims=True)
            downsampled_data_copied = np.concatenate([mif_b0, mif.data[..., lr_index]], axis=-1).copy()
        
            if needs_affine:
                return downsampled_data_copied, affine_copied
            else:
                return downsampled_data_copied

        else:
            raise IOError('file extension is not supported for downsampling must mif: ' + str(path))
            exit(0)

    def save_prediction(self, prediction, affine, output_name):
        nifti = nib.Nifti1Image(prediction, affine=affine)
        nib.save(nifti, output_name)
        print('Save image to the path {:}'.format(output_name))


