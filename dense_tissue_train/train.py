import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from dice_loss import DiceLoss, BinaryFocalLoss, DiceSensitivityLoss, DiceSensitivityLoss2


# data path and checkpoint path
dir_img_train = '/home/sh542/workspace/maciej/UNet_0505/dense_tissue_train/new_png_dense/dense_train/imgs_norm/'
dir_img_test = '/home/sh542/workspace/maciej/UNet_0505/dense_tissue_train/new_png_dense/dense_test/imgs_norm/'
dir_mask_train = '/home/sh542/workspace/maciej/UNet_0505/dense_tissue_train/new_png_dense/dense_train/masks/'
dir_mask_test = '/home/sh542/workspace/maciej/UNet_0505/dense_tissue_train/new_png_dense/dense_test/masks/'

dir_checkpoint = 'checkpoint_dense_tissue/' # path where save the model
dir_test_pred = '/home/sh542/workspace/maciej/UNet_0505/dense_tissue_train/pred_dense_tissue/'

# image is normalized before transform
img_Transforms = transforms.Compose([
    transforms.Resize(512),  
    transforms.ToTensor()
])

mask_Transforms = transforms.Compose([
    transforms.Resize(512), 
    transforms.ToTensor()
])

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              img_scale=1):

    dataset_train = BasicDataset(dir_img_train, dir_mask_train, img_scale, img_Transforms, mask_Transforms)
    dataset_test = BasicDataset(dir_img_test, dir_mask_test, img_scale, img_Transforms, mask_Transforms)
    n_test = 84
    n_train = 300
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Test size:       {n_test}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    gamma = 0.8
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=40, gamma=gamma)

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
        # criterion = DiceLoss()
        # criterion = BinaryFocalLoss()
        # criterion = DiceSensitivityLoss2()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                true_masks = true_masks[:, 0, :, :].unsqueeze(1)
                true_masks = (true_masks > 0.5).float()
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch + 1)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch + 1)
        val_score = eval_net(net, test_loader, device, epoch, dir_test_pred)
        scheduler.step()
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch + 1) 

        if net.n_classes > 1:
            logging.info('Test cross entropy: {}'.format(val_score))
            writer.add_scalar('Loss/test', val_score, epoch + 1)
        else:
            logging.info('Test Dice Coeff: {}'.format(val_score))
            writer.add_scalar('Dice/test', val_score, epoch + 1)

        writer.add_images('images', imgs, epoch + 1)
        if net.n_classes == 1:
            writer.add_images('masks/true', true_masks > 0.5, epoch + 1)  # change by siping
            writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, epoch + 1)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=1, n_classes=1, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
