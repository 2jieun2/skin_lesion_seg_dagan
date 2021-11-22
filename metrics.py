import numpy as np
import cv2
from torch.autograd import Variable

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, jaccard_score, f1_score



def cal_metrics(true, pred):
    confm = confusion_matrix(true.reshape(-1,1), pred.reshape(-1,1))

    tn = confm[0][0]
    fp = confm[0][1]
    fn = confm[1][0]
    tp = confm[1][1]

    # accuracy = (tn + tp) / np.sum(confm)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # jaccard_index = jaccard_score(true, pred)
    jaccard_index = tp / (tp + fp + fn)
    dice_coeff = 2*tp / (2*tp + fn + fp)

    return sensitivity, specificity, jaccard_index, dice_coeff


def evaluate_val_test(dataloader, model, device, save_fake_path=False):
    patient_id_list = []
    acc_list = []
    sen_list = []
    spec_list = []
    ji_list = []
    dc_list = []

    # for idx, batch in enumerate(dataloader):
    for batch in dataloader:
        real_x = Variable(batch['image']).to(device)
        real_y = Variable(batch['mask']).to(device)

        fake_y = model(real_x)

        real_y = real_y.cpu().detach().numpy()
        fake_y = fake_y.cpu().detach().numpy()

        for i in range(real_x.shape[0]):
            real_y_ = np.squeeze(real_y[i])
            fake_y_ = np.squeeze(fake_y[i])

            origin_h = int(batch['origin_h'][i])
            origin_w = int(batch['origin_w'][i])

            real_y_ = cv2.resize(real_y_, (origin_w, origin_h), interpolation=cv2.INTER_LINEAR)
            fake_y_ = cv2.resize(fake_y_, (origin_w, origin_h), interpolation=cv2.INTER_LINEAR)

            real_y_ = np.where(real_y_ > 0.5, 1, 0)
            fake_y_ = np.where(fake_y_ > 0.5, 1, 0)

            acc = accuracy_score(real_y_.reshape(-1,1), fake_y_.reshape(-1,1))
            acc_list.append(acc)

            if save_fake_path:
                sen, spec, ji, dc = cal_metrics(real_y_, fake_y_)

                sen_list.append(sen)
                spec_list.append(spec)
                ji_list.append(ji)
                dc_list.append(dc)

                fake_y_ = fake_y_ * 255
                patient_id = str(batch['patient_id'][i])
                patient_id_list.append(patient_id)
                cv2.imwrite(f'{save_fake_path}/{patient_id}.png', fake_y_)

    acc_mean = sum(acc_list) / len(acc_list)

    if save_fake_path:
        sen_mean = sum(sen_list) / len(sen_list)
        spec_mean = sum(spec_list) / len(spec_list)
        ji_mean = sum(ji_list) / len(ji_list)
        dc_mean = sum(dc_list) / len(dc_list)

        eval_mean_list = [acc_mean, sen_mean, spec_mean, ji_mean, dc_mean]

        return patient_id_list, eval_mean_list, acc_list, sen_list, spec_list, ji_list, dc_list
    else:
        return acc_mean
