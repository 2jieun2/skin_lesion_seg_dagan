import os
from glob import glob
import sys
import argparse
import numpy as np
import GPUtil
from tqdm import tqdm
from datetime import datetime
import logging
import warnings

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from data import SkinDataset
from metrics import *


torch.cuda.empty_cache()
########## GPU Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()

# parser.add_argument('--mode', type=str, default='train')

parser.add_argument('--path_train', type=str, default='/home/ubuntu/jelee/dataset/skin_ISIC/ISIC-2017_Training')
parser.add_argument('--path_valid', type=str, default='/home/ubuntu/jelee/dataset/skin_ISIC/ISIC-2017_Validation')
parser.add_argument('--path_test', type=str, default='/home/ubuntu/jelee/dataset/skin_ISIC/ISIC-2017_Test_v2')

parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--re_size', type=int, nargs='+', default=[512, 512, 3])

parser.add_argument('--nf', type=int, default=32)
parser.add_argument('--lr_g', type=float, default=0.0002)
parser.add_argument('--lr_d', type=float, default=0.0002)

parser.add_argument('--lambda_1', type=int, default=100)

parser.add_argument('--checkpoint_sample', type=int, default=10)
parser.add_argument('--stop_patience', type=int, default=50)

args = parser.parse_args()

warnings.filterwarnings(action='ignore')


def logger_setting(save_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=f'{save_path}/log.log')

    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # stream_handler.setFormatter(formatter)
    # file_handler.setFormatter(formatter)

    stream_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        # nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity="leaky_relu")
    elif classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity="leaky_relu")
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def train():

    ##### Directory
    datetime_train = datetime.today().strftime('%Y%m%d_%H%M%S')
    dir_log = f'./log/{datetime_train}'
    dir_model = f'./log/{datetime_train}/model'
    dir_tboard = f'./log/{datetime_train}/tboard'
    dir_result = f'./log/{datetime_train}/result_valid'

    directory = [dir_log, dir_model, dir_tboard, dir_result]
    for dir in directory:
            os.makedirs(dir, exist_ok=True)


    ##### Training Log
    logger = logger_setting(dir_log)
    logger.debug('============================================')
    logger.debug('Batch Size: %d' % args.batch_size)
    logger.debug('Epoch: %d' % args.epochs)

    writer = SummaryWriter(dir_tboard)

    ##### Initialize
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    loss_L1 = nn.L1Loss()
    loss_BCE = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    ##### Model (Generator, Discriminator)
    generator = nn.DataParallel(GeneratorUNet(args)).to(device)
    discriminator1 = nn.DataParallel(Discriminator1(args)).to(device)
    discriminator2 = nn.DataParallel(Discriminator2(args)).to(device)

    generator.apply(weights_init_normal)
    discriminator1.apply(weights_init_normal)
    discriminator2.apply(weights_init_normal)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.007, betas=(0.9, 0.999))
    optimizer_D1 = torch.optim.Adam(discriminator1.parameters(), lr=args.lr_d, betas=(0.9, 0.999))
    optimizer_D2 = torch.optim.Adam(discriminator2.parameters(), lr=args.lr_d, betas=(0.9, 0.999))

    # scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=1, gamma=0.837719)
    # scheduler_D1 = torch.optim.lr_scheduler.StepLR(optimizer_D1, step_size=1, gamma=0.837719)
    # scheduler_D2 = torch.optim.lr_scheduler.StepLR(optimizer_D2, step_size=1, gamma=0.837719)


    ##### Dataset Load
    train_dataset = SkinDataset(args.path_train, args.re_size)
    valid_dataset = SkinDataset(args.path_valid, args.re_size, valid_test=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)


    ##### Training
    logger.debug('============================================')

    valid_best = {'epoch': 0, 'acc': 0, 'ji': 0, 'loss': np.inf}
    patience = 0
    generator.train()
    discriminator1.train()
    discriminator2.train()

    for epoch in tqdm(range(1, args.epochs + 1), desc='Epoch'):
        valid_update = False
        train_loss = {'loss_D1': 0, 'loss_D2': 0, 'loss_G_total': 0, 'loss_G_pixel': 0, 'loss_G_adv1': 0, 'loss_G_adv2': 0}

        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Batch'):

            real_image = Variable(batch['image']).to(device)
            real_mask = Variable(batch['mask']).to(device)

            fake_mask = generator(real_image)

            # ------------------------------
            # Train discriminator1
            # ------------------------------
            # Real loss
            pred_real = discriminator1(real_mask, real_image)
            valid = Variable(Tensor(np.ones(pred_real.size())), requires_grad=False)
            loss_D1_real = loss_BCE(sigmoid(pred_real), valid)

            # Fake loss
            pred_fake = discriminator1(fake_mask.detach(), real_image)
            fake = Variable(Tensor(np.zeros(pred_fake.size())), requires_grad=False)
            loss_D1_fake = loss_BCE(sigmoid(pred_fake), fake)

            # Total loss
            loss_D1 = loss_D1_real + loss_D1_fake

            optimizer_D1.zero_grad()
            loss_D1.backward()
            optimizer_D1.step()


            # ------------------------------
            # Train discriminator2
            # ------------------------------
            # Real loss
            pred_real = discriminator2(real_mask)
            valid = Variable(Tensor(np.ones(pred_real.size())), requires_grad=False)
            loss_D2_real = loss_BCE(sigmoid(pred_real), valid)

            # Fake loss
            pred_fake = discriminator2(fake_mask.detach())
            fake = Variable(Tensor(np.zeros(pred_fake.size())), requires_grad=False)
            loss_D2_fake = loss_BCE(sigmoid(pred_fake), fake)

            # Total loss
            loss_D2 = loss_D2_real + loss_D2_fake

            optimizer_D2.zero_grad()
            loss_D2.backward()
            optimizer_D2.step()

            # ------------------------------
            # Train generator
            # ------------------------------

            fake_mask = generator(real_image)

            # GAN loss
            pred_fake1 = discriminator1(fake_mask, real_image)
            pred_fake2 = discriminator2(fake_mask)

            loss_G_adv1 = loss_BCE(sigmoid(pred_fake1), valid)
            loss_G_adv2 = loss_BCE(sigmoid(pred_fake2), valid)
            loss_G_pixel = args.lambda_1 * loss_L1(fake_mask, real_mask)

            # Total loss
            loss_G = loss_G_adv1 + loss_G_adv2 + loss_G_pixel

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            train_loss['loss_D1'] += loss_D1.item()
            train_loss['loss_D2'] += loss_D2.item()
            train_loss['loss_G_total'] += loss_G.item()
            train_loss['loss_G_adv1'] += loss_G_adv1.item()
            train_loss['loss_G_adv2'] += loss_G_adv2.item()
            train_loss['loss_G_pixel'] += loss_G_pixel.item()

        val_acc = evaluate_val_test(val_dataloader, generator, device)

        loss_D1_ = train_loss['loss_D1'] / len(train_dataloader)
        loss_D2_ = train_loss['loss_D2'] / len(train_dataloader)
        loss_G_ = train_loss['loss_G_total'] / len(train_dataloader)
        loss_G_adv1_ = train_loss['loss_G_adv1'] / len(train_dataloader)
        loss_G_adv2_ = train_loss['loss_G_adv2'] / len(train_dataloader)
        loss_G_pixel_ = train_loss['loss_G_pixel'] / len(train_dataloader)

        # if valid_best['acc'] < val_acc and valid_best['ji'] < val_ji:
        if valid_best['loss'] > loss_G_:
            torch.save(generator.state_dict(), f'{dir_model}/generator.pth')
            # torch.save(discriminator.state_dict(), f'{dir_model}/discriminator.pth')
            valid_best['epoch'] = epoch
            valid_best['loss'] = loss_G_
            valid_update = True
            patience = 0
        else:
            patience += 1

        # logger.debug('--------------------------------------------')
        logger.info(f'[Epoch: {epoch}/{args.epochs}]')
        logger.info(
            f'D1 loss: {round(loss_D1_, 4)} | D1 loss: {round(loss_D2_, 4)} | G loss: {round(loss_G_, 4)} | G loss_adv1: {round(loss_G_adv1_, 4)} | G loss_adv2: {round(loss_G_adv2_, 4)} | G loss_pixel: {round(loss_G_pixel_, 4)} | val_acc: {round(val_acc, 4)}  | valid_update: {str(valid_update)}({patience})'
            )

        writer.add_scalar('D1 loss', loss_D1_, epoch)
        writer.add_scalar('D2 loss', loss_D2_, epoch)
        writer.add_scalar('G loss', loss_G_, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)

        # scheduler_G.step()
        # scheduler_D1.step()
        # scheduler_D2.step()

        if patience > args.stop_patience:
            logger.info(f'-------------------------------------------- Early Stopping !')
            break

    writer.close()

    logger.info('============================================')
    logger.info(f'[Best Performance of Validation]')
    logger.info(f'''Epoch: {valid_best['epoch']} | Acc: {valid_best['acc']}''')

    logger.debug(f'[Prediction for Test Data]')
    pred_test(generator_path=f'{dir_model}/generator.pth')
    logger.info('============================================')


def pred_test(generator_path):
    generator = nn.DataParallel(GeneratorUNet(args)).to(device)

    generator.load_state_dict(torch.load(generator_path))
    model_dtime = generator_path.split('/')[2]

    generator = generator.to(device)

    dir_test = f'./log/{model_dtime}/result_test2/'
    os.makedirs(dir_test, exist_ok=True)

    logger_test = logger_setting(dir_test)
    logger_test.info(f'Generator: {generator_path}')

    test_dataset = SkinDataset(args.path_test, args.re_size, valid_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    patient_id_list, eval_mean_list, acc_list, sen_list, spec_list, ji_list, dc_list = evaluate_val_test(test_dataloader, generator, device, save_fake_path=dir_test)

    np.save(f'{dir_test}total_acc', np.array(acc_list))
    np.save(f'{dir_test}total_sen', np.array(sen_list))
    np.save(f'{dir_test}total_spec', np.array(spec_list))
    np.save(f'{dir_test}total_ji', np.array(ji_list))
    np.save(f'{dir_test}total_dc', np.array(dc_list))

    logger_test.debug('[Patient ID | Acc | Sen | Spec | JI | DC]')
    for idx in range(len(patient_id_list)):
        logger_test.debug(f'{patient_id_list[idx]} | {acc_list[idx]} | {sen_list[idx]} | {spec_list[idx]} | {ji_list[idx]} | {dc_list[idx]}')

    logger_test.info('[Prediction for Test Data]')
    logger_test.info(f'Acc: {eval_mean_list[0]}')
    logger_test.info(f'Sen: {eval_mean_list[1]}')
    logger_test.info(f'Spec: {eval_mean_list[2]}')
    logger_test.info(f'JI: {eval_mean_list[3]}')
    logger_test.info(f'DC: {eval_mean_list[4]}')


if __name__ == "__main__":
    train()
