import os
import math
import os
import numpy as np
import torch

IMG_EXTENSIONS = [
    '.nii.gz', 'mif.gz'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def psnr2(img1, img2, valid):
    mae = np.sum(np.abs(img1 - img2)) / valid
    mse = (np.sum((img1 - img2) ** 2)) / valid
    # if mse < 1.0e-10:
    #   return 100
    PIXEL_MAX = 1
    # print(mse)
    return mae, mse, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# # Copy from  https://github.com/marianocabezas/quad22/utils.py
def unique_to_matrix(tensor):
    """
    Function to convert a FSL-like tensor image (with the unique values) to
    a true DTI matrix.
    :param tensor: FSL-like tensor image.
    :return:
    """
    # Vector representation of the matrix for each unique value.
    # This vectorization is based on how FSL defines the DTI image (tensor).
    tensor_vec = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ])
    tensor = np.expand_dims(tensor, -1)
    tensor_vec = np.expand_dims(tensor_vec, (0, 1, 2))
    tensor_vectorized = np.sum(tensor * tensor_vec, -2)
    matrix_shape = tensor_vectorized.shape[:-1] + (3, 3)
    return np.reshape(tensor_vectorized, matrix_shape)

def get_dti_metrics(tensor):
    """
    Function to get the QUaD22 DTI metrics based on an FSL-like tensor
    image.
    :param tensor: FSL-like tensor image.
    :return:
    """
    # Initial preprocessing to obtain the eignevalues of the tensor.
    # It seems that float16 is the prefered type to replicate FA as closely
    # as possible to the FSL image. The small cascaded errors seem to lead to
    # a large MSE difference.
    dti_matrix = unique_to_matrix(tensor.astype(np.float16))
    v = np.real(np.linalg.eigvals(dti_matrix))

    num_sq12 = (v[..., 0] - v[..., 1]) ** 2
    num_sq13 = (v[..., 0] - v[..., 2]) ** 2
    num_sq23 = (v[..., 1] - v[..., 2]) ** 2
    fa_num = num_sq12 + num_sq13 + num_sq23
    fa_den = np.sum(v ** 2, axis=-1)
    fa_den[fa_den == 0] = 1e-5

    fa = np.sqrt(0.5 * fa_num / fa_den)
    md = np.sum(v, axis=-1) / 3
    ad = np.sort(v, axis=-1)[..., -1]

    return fa, md, ad


def get_dti_metricsV2(input):
    input = input.permute(0,2,3,4,1) # (b,c,w,h,d) => (b,w,h,d,c)
    # print(input.shape)
    line1 = torch.stack((input[:,:,:,:,0],input[:,:,:,:,1],input[:,:,:,:,2])
                    , dim = -1) # (b,w,h,d,3)
    line1 = torch.unsqueeze(line1, dim = 4) # (b,w,h,d,1,3)
    line2 = torch.stack((input[:,:,:,:,1],input[:,:,:,:,3],input[:,:,:,:,4])
                    , dim = -1) 
    line2 = torch.unsqueeze(line2, dim = 4) 
    line3 = torch.stack((input[:,:,:,:,2],input[:,:,:,:,4],input[:,:,:,:,5])
                    , dim = -1)
    line3 = torch.unsqueeze(line3, dim = 4)
    real_dti = torch.cat((line1,line2,line3), dim = -2) # (b,w,h,d,3,3)
    eigvals = torch.real(torch.linalg.eigvals(real_dti)) # (b,w,h,d,3)

    num_sq12 = (eigvals[:,:,:,:,0] - eigvals[:,:,:,:, 1]) ** 2  # (b,w,h,d)
    num_sq13 = (eigvals[:,:,:,:,0] - eigvals[:,:,:,:, 2]) ** 2
    num_sq23 = (eigvals[:,:,:,:,1] - eigvals[:,:,:,:, 2]) ** 2
    fa_num = num_sq12 + num_sq13 + num_sq23 
    fa_den = torch.sum(eigvals ** 2,dim=-1) 
    fa_den[fa_den == 0] = 1e-5
    fa = torch.sqrt(0.5 * fa_num / fa_den)
    md = torch.sum(eigvals, dim=-1) / 3
    # ad = torch.sort(eigvals, dim=-1)[0][:,:,:,:,-1]
    fa = torch.unsqueeze(fa, dim = 1)
    md = torch.unsqueeze(md, dim = 1)

    return fa, md


def get_eigvals(input):

    input = input.permute(0,2,3,4,1) # (b,c,w,h,d) => (b,w,h,d,c)
    # print(input.shape)
    line1 = torch.stack((input[:,:,:,:,0],input[:,:,:,:,1],input[:,:,:,:,2])
                    , dim = -1) # (b,w,h,d,3)
    line1 = torch.unsqueeze(line1, dim = 4) # (b,w,h,d,1,3)
    line2 = torch.stack((input[:,:,:,:,1],input[:,:,:,:,3],input[:,:,:,:,4])
                    , dim = -1) 
    line2 = torch.unsqueeze(line2, dim = 4) 
    line3 = torch.stack((input[:,:,:,:,2],input[:,:,:,:,4],input[:,:,:,:,5])
                    , dim = -1)
    line3 = torch.unsqueeze(line3, dim = 4)
    real_dti = torch.cat((line1,line2,line3), dim = -2) # (b,w,h,d,3,3)
    eigvals, _ = torch.linalg.eigh(real_dti) # (b,w,h,d,3)

    eigvals = eigvals.permute(0,4,1,2,3)

    e1 = eigvals[:,0,:,:,:].unsqueeze(1)
    e2 = eigvals[:,1,:,:,:].unsqueeze(1)
    e3 = eigvals[:,2,:,:,:].unsqueeze(1)

    return e1, e2, e3


def get_eigs(input):
    input = input.permute(0,2,3,4,1) # (b,c,w,h,d) => (b,w,h,d,c)
    # print(input.shape)
    line1 = torch.stack((input[:,:,:,:,0],input[:,:,:,:,1],input[:,:,:,:,2])
                    , dim = -1) # (b,w,h,d,3)
    line1 = torch.unsqueeze(line1, dim = 4) # (b,w,h,d,1,3)
    line2 = torch.stack((input[:,:,:,:,1],input[:,:,:,:,3],input[:,:,:,:,4])
                    , dim = -1) 
    line2 = torch.unsqueeze(line2, dim = 4) 
    line3 = torch.stack((input[:,:,:,:,2],input[:,:,:,:,4],input[:,:,:,:,5])
                    , dim = -1)
    line3 = torch.unsqueeze(line3, dim = 4)
    real_dti = torch.cat((line1,line2,line3), dim = -2) # (b,w,h,d,3,3)

    eig_vals, eig_vecs = torch.linalg.eig(real_dti)  

    eig_vals = torch.real(eig_vals)  # (b,w,h,d,3)
    eig_vecs = torch.real(eig_vecs) #  (b,w,h,d,3,3)

    return eig_vals, eig_vecs


