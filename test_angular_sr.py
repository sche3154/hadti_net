import torch
import numpy as np
import random
import os 
import time
import pickle
from utils.utils import psnr2, get_dti_metrics, mkdirs
from options.test_options import TestOptions
from data import create_dataset
from models import create_model


torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.cuda.manual_seed_all(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    torch.cuda.empty_cache()
    opt = TestOptions().parse()  # get test options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of testing images = %d' % dataset_size)
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    results_dir = os.path.join(opt.results_dir, opt.name)
    mkdirs(results_dir)
    summary_file = open(os.path.join(results_dir, 'summary.txt'), 'w')
    entry = ('Load checkpoint from: ' + opt.checkpoints_dir + opt.name + '\n')

    mse_list = []
    mse_fa_list = []
    mse_md_list = []
    mse_ad_list = []

    mae_list = []
    mae_fa_list = []
    mae_md_list = []
    mae_ad_list = []

    psnr_list = []
    psnr_fa_list = []
    psnr_md_list = []
    psnr_ad_list = []

    time_spent = []

    mae_fa_dict = {}

    # testing
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        start_time = time.time()
        data['index'] = data['index'][0]
        data['dwi'] = data['dwi'].squeeze(0)
        data['t1'] = data['t1'].squeeze(0)
        data['gt_dti'] = data['gt_dti'].squeeze(0).numpy()
        data['wm_mask'] = data['wm_mask'].squeeze(0).numpy()

        patch_nums =  len(data['dwi'])
        print('Shape of dwi_patch: {:}, gt_dti: {:}, t1_patch: {:}, wm_mask : {:}'.format(data['dwi'].shape
        ,data['gt_dti'].shape, data['t1'].shape, data['wm_mask'].shape))
       
        pred_list = []
        
        for j in range(0, patch_nums, opt.input_batch_sizes):
        
            inputs = {}
            inputs['dwi'] = data['dwi'][j:min(j+opt.input_batch_sizes, patch_nums),:,:,:,:]
            inputs['t1'] = data['t1'][j:min(j+opt.input_batch_sizes, patch_nums),:,:,:,:]

            model.set_input(inputs)
            output = model.test()  # run inference

            output = output.detach().cpu().numpy()
            # b c w h d -> b w h d c
            output = np.transpose(output, (0, 2, 3, 4, 1))
            pred_list.append(output)

        # Postprocess predicted patches
        preds = np.concatenate(pred_list, axis=0)
        # print('Shape of preds: {:}'.format(preds.shape))
        resulted_img = dataset.dataset.postprocessing(preds)
        resulted_fa, resulted_md, resulted_ad = get_dti_metrics(resulted_img)
        gt_fa, gt_md, gt_ad = get_dti_metrics(data['gt_dti'])

        # compute metrics 
        gt_wm = data['gt_dti'] * np.expand_dims(data['wm_mask'], 3)
        resulted_wm = resulted_img * np.expand_dims(data['wm_mask'], 3)
        gt_fa_wm = gt_fa * data['wm_mask']
        gt_md_wm = gt_md * data['wm_mask']
        gt_ad_wm = gt_ad * data['wm_mask']
        resulted_fa_wm = resulted_fa * data['wm_mask']
        resulted_md_wm = resulted_md * data['wm_mask']
        resulted_ad_wm = resulted_ad * data['wm_mask']


        if (i+1) % opt.save_prediction == 0:
            dataset.dataset.save_prediction(resulted_wm, suffix = '_dti_wm_pred')
            dataset.dataset.save_prediction(resulted_fa_wm, suffix = '_fa_wm_pred')
            dataset.dataset.save_prediction(gt_wm, suffix = '_dti_wm_gt')
            dataset.dataset.save_prediction(gt_fa_wm, suffix = '_fa_wm_gt')
         
        # metrics in wm area
        valid_no = np.sum(data['wm_mask'])  # valid pixels
        sub_mae, sub_mse, sub_psnr = psnr2(gt_wm, resulted_wm, valid_no)
        sub_fa_mae, sub_fa_mse, sub_fa_psnr = psnr2(gt_fa_wm, resulted_fa_wm, valid_no)
        sub_md_mae, sub_md_mse, sub_md_psnr = psnr2(gt_md_wm, resulted_md_wm, valid_no)
        sub_ad_mae, sub_ad_mse, sub_ad_psnr = psnr2(gt_ad_wm, resulted_ad_wm, valid_no)
   
        time_spent.append(time.time() - start_time)

        sample_info = 'Sample index: ' + data['index']
        sample_info_cv = 'MAE:' + str(sub_mae)[:6] + ' MSE: ' + str(sub_mse)[:6] + ' PSNR: ' + str(sub_psnr)[:6]
        sample_info_fa = 'MAE FA:' + str(sub_fa_mae)[:6] +  ' MSE FA: ' + str(sub_fa_mse)[:6] + ' PSNR FA: ' + str(sub_fa_psnr)[:6]
        sample_info_md = 'MAE MD:' + str(sub_md_mae)[:6] +  ' MSE MD: ' + str(sub_md_mse)[:6] + ' PSNR MD: ' + str(sub_md_psnr)[:6]
        sample_info_ad = 'MAE AD:' + str(sub_ad_mae)[:6] +  ' MSE AD: ' + str(sub_ad_mse)[:6] + ' PSNR AD: ' + str(sub_ad_psnr)[:6]
       
       
        entry += (sample_info+ '\n')
        entry += (sample_info_cv + '\n')
        entry += (sample_info_fa + '\n')
        entry += (sample_info_md + '\n')
        entry += (sample_info_ad + '\n')
        entry += ('\n')

        mae_list.append(sub_mae)
        mae_fa_list.append(sub_fa_mae)
        mae_md_list.append(sub_md_mae)
        mae_ad_list.append(sub_ad_mae)
        # dict of mae fa
        mae_fa_dict[data['index']] = sub_fa_mae

        mse_list.append(sub_mse)
        mse_fa_list.append(sub_fa_mse)
        mse_md_list.append(sub_md_mse)
        mse_ad_list.append(sub_ad_mse)

        psnr_list.append(sub_psnr)
        psnr_fa_list.append(sub_fa_psnr)
        psnr_md_list.append(sub_md_psnr)
        psnr_ad_list.append(sub_ad_psnr)

        print(sample_info)
        print(sample_info_cv)
        print(sample_info_fa)
        print(sample_info_md)
        print(sample_info_ad)
        print()

 

    print('***************************End inference******************************')
    mean_mae_info = 'mean wm mae: {:.6f}+/-{:.6f}'.format(np.mean(mae_list), np.std(mae_list))
    mean_mse_info = 'mean wm mse: {:.6f}+/-{:.6f}'.format(np.mean(mse_list), np.std(mse_list))
    mean_psnr_info = 'mean wm psnr: {:.4f}+/-{:.4f}'.format(np.mean(psnr_list), np.std(psnr_list))

    mean_mae_info_fa = 'mean wm mae fa: {:.6f}+/-{:.6f}'.format(np.mean(mae_fa_list), np.std(mae_fa_list))
    mean_mse_info_fa = 'mean wm mse fa: {:.6f}+/-{:.6f}'.format(np.mean(mse_fa_list), np.std(mse_fa_list))
    mean_psnr_info_fa = 'mean wm psnr fa: {:.4f}+/-{:.4f}'.format(np.mean(psnr_fa_list), np.std(psnr_fa_list))

    mean_mae_info_md = 'mean wm mae md: {:.6f}+/-{:.6f}'.format(np.mean(mae_md_list), np.std(mae_md_list))
    mean_mse_info_md = 'mean wm mse md: {:.6f}+/-{:.6f}'.format(np.mean(mse_md_list), np.std(mse_md_list))
    mean_psnr_info_md = 'mean wm psnr md: {:.4f}+/-{:.4f}'.format(np.mean(psnr_md_list), np.std(psnr_md_list))

    mean_mae_info_ad = 'mean wm mae ad: {:.6f}+/-{:.6f}'.format(np.mean(mae_ad_list), np.std(mae_ad_list))
    mean_mse_info_ad = 'mean wm mse ad: {:.6f}+/-{:.6f}'.format(np.mean(mse_ad_list), np.std(mse_ad_list))
    mean_psnr_info_ad = 'mean wm psnr ad: {:.4f}+/-{:.4f}'.format(np.mean(psnr_ad_list), np.std(psnr_ad_list))

    mean_time = 'average case time: {:.4f} seconds'.format(np.mean(time_spent))
    
    entry += (mean_mae_info + '\n')
    entry += (mean_mae_info_fa + '\n')
    entry += (mean_mae_info_md + '\n')
    entry += (mean_mae_info_ad + '\n')
    entry += '\n'

    entry += (mean_mse_info + '\n')
    entry += (mean_mse_info_fa + '\n')
    entry += (mean_mse_info_md + '\n')
    entry += (mean_mse_info_ad + '\n')
    entry += '\n'
    
    entry += (mean_psnr_info + '\n')    
    entry += (mean_psnr_info_fa + '\n')
    entry += (mean_psnr_info_md + '\n')
    entry += (mean_psnr_info_ad + '\n')
    entry += '\n'
    
    print(mean_time)
    summary_file.write(entry)
    summary_file.close()

    maefa_path = os.path.join(opt.results_dir, opt.name, 'maefa.pickle')
    with open(maefa_path, 'wb') as handle:
        pickle.dump(mae_fa_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)