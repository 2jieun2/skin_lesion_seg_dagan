import numpy as np
import torch
from torch.autograd import Variable
from data import save_nii


# def save_feature_map(fmap, path):
#     # fmap = fmap.cpu().detach().numpy().squeeze()
#     if len(fmap.shape) == 5:
#         fmap = fmap[0, 0, :, :, :]
#     elif len(fmap.shape) == 4:
#         fmap = fmap[0, :, :, :]
#     fmap = fmap.cpu().detach().numpy()
#     save_nii(fmap, path)
#
#
# def visualize_feature_maps(encoder, data, save_fmap_path):
#     enc_list = encoder(data)
#     for idx, fmap in enc_list:
#         save_feature_map(fmap, f'{save_fmap_path}_{idx+1}')


def save_feature_maps(enc_list, path_fname):
    for idx, fmap in enumerate(enc_list):
        try:
            np.save(f'{path_fname}_{idx+1}', fmap.cpu().detach().numpy())

            if len(fmap.shape) == 5:
                fmap = fmap[0, :, :, :, :]
                fmap = torch.mean(fmap, dim=0).cpu().detach().numpy()
            elif len(fmap.shape) == 4:
                fmap = torch.mean(fmap, dim=0).cpu().detach().numpy()
            save_nii(fmap, f'{path_fname}_mean_{idx+1}')
        except:
            pass


def visualize_feature_maps(dataloader, t_encoder, g_encoder, t_decoder, device, save_fmap_path):
    t_encoder.eval()
    g_encoder.eval()
    t_decoder.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):

            patient_id = str(data['patient_id'][0])

            real_x = Variable(data['x']).to(device)
            real_y = Variable(data['y']).to(device)

            # pred_y = t_decoder(g_encoder(real_x))

            enc_list1, enc_list2 = t_encoder(real_y)
            save_feature_maps(enc_list1, f'{save_fmap_path}{patient_id}_TEncoder_recons_y')
            save_feature_maps(enc_list2, f'{save_fmap_path}{patient_id}_TEncoder_recover_y')

            # save_feature_maps(t_encoder(real_y), f'{save_fmap_path}{patient_id}_TEncoder_real_y')
            # save_feature_maps(t_encoder(real_y), f'{save_fmap_path}{patient_id}_TEncoder_real_y')
            # save_feature_maps(t_encoder(pred_y), f'{save_fmap_path}{patient_id}_TEncoder_pred_y')
            save_feature_maps(g_encoder(real_x), f'{save_fmap_path}{patient_id}_GEncoder_real_x')


def visualize_feature_maps_(dataloader, t_encoder, f_extractor, g_encoder, t_decoder, device, save_fmap_path):
    t_encoder.eval()
    f_extractor.eval()
    g_encoder.eval()
    t_decoder.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):

            patient_id = str(data['patient_id'][0])

            real_x = Variable(data['x']).to(device)
            real_y = Variable(data['y']).to(device)

            pred_y = t_decoder(g_encoder(f_extractor(real_x)))

            t_enc_real_y = t_encoder(real_y)
            t_enc_pred_y = t_encoder(pred_y)
            g_enc_real_x = g_encoder(f_extractor(real_x))
            g_enc_real_y = g_encoder(t_enc_real_y[0])

            save_feature_maps(t_enc_real_y, f'{save_fmap_path}{patient_id}_TEncoder_real_y')
            save_feature_maps(t_enc_pred_y, f'{save_fmap_path}{patient_id}_TEncoder_pred_y')
            save_feature_maps(g_enc_real_x, f'{save_fmap_path}{patient_id}_GEncoder_real_x')
            save_feature_maps(g_enc_real_y, f'{save_fmap_path}{patient_id}_GEncoder_real_y')


def visualize_feature_maps_self(dataloader, t_encoder, device, save_fmap_path):
    t_encoder.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):

            patient_id = str(data['patient_id'][0])

            real_y = Variable(data['y']).to(device)

            enc_list1, enc_list2 = t_encoder(real_y)

            # save_feature_maps(t_encoder(real_y), f'{save_fmap_path}{patient_id}_TEncoder_real_y')
            save_feature_maps(enc_list1, f'{save_fmap_path}{patient_id}_TEncoder_recons_y')
            save_feature_maps(enc_list2, f'{save_fmap_path}{patient_id}_TEncoder_recover_y')

