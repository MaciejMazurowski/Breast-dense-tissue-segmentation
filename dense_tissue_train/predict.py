import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from dice_loss import dice_coeff
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net_breast,
                net_dense,
                full_img,
                true_mask,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net_breast.eval()
    net_dense.eval()

    img = BasicDataset.preprocess(full_img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    true_mask = BasicDataset.preprocess(true_mask)
    true_mask = true_mask.unsqueeze(0)
    true_mask = true_mask.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output_breast = net_breast(img)
        output_dense = net_dense(img)

        if net_breast.n_classes > 1:
            probs_breast = F.softmax(output_breast, dim=1)
            probs_dense = F.softmax(output_dense, dim=1)
        else:
            probs_breast = torch.sigmoid(output_breast)
            probs_dense = torch.sigmoid(output_dense)
        pred_breast = (probs_breast > 0.5).float()
        pred_dense = (probs_dense > 0.5).float()
        pred_dense_combine = pred_breast + pred_dense
        pred_dense_combine = (pred_dense_combine > 1.5).float() 
        true_mask = (true_mask > 0.5).float() 
        dice = dice_coeff(pred_dense_combine, true_mask, num=1).item()

        pred_dense_combine = pred_dense_combine.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        pred_dense_combine = tf(pred_dense_combine.cpu())
        full_mask = pred_dense_combine.squeeze().cpu().numpy()
    return full_mask, dice

    # with torch.no_grad():
    #     output = net(img)

    #     if net.n_classes > 1:
    #         probs = F.softmax(output, dim=1)
    #     else:
    #         probs = torch.sigmoid(output)
    #     pred = (probs > 0.5).float()
    #     true_mask = (true_mask > 0.5).float() 
    #     dice = dice_coeff(pred, true_mask, num=1).item()

    #     probs = probs.squeeze(0)

    #     tf = transforms.Compose(
    #         [
    #             transforms.ToPILImage(),
    #             transforms.Resize(full_img.size[1]),
    #             transforms.ToTensor()
    #         ]
    #     )

    #     probs = tf(probs.cpu())
    #     full_mask = probs.squeeze().cpu().numpy()

    # return full_mask > out_threshold, dice


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_breast', '-m_b', metavar='FILE',
                        help="Specify the file in which the breast model is stored")

    parser.add_argument('--model_dense', '-m_d', metavar='FILE',
                        help="Specify the file in which the dense tissue model is stored")
    
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--mask', '-true_m', metavar='INPUT', nargs='+',
                        help='filenames of true masks', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    mask_files = args.mask
    out_files = get_output_filenames(args)
    if not os.path.exists(out_files[0]):
        os.makedirs(out_files[0])
    imgList = os.listdir(in_files[0])
    net_breast = UNet(n_channels=1, n_classes=1)
    net_dense = UNet(n_channels=1, n_classes=1)

    logging.info("Loading model {}".format(args.model_breast))
    logging.info("Loading model {}".format(args.model_dense))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net_breast.to(device=device)
    net_dense.to(device=device)
    net_breast.load_state_dict(torch.load(args.model_breast, map_location=device))
    net_dense.load_state_dict(torch.load(args.model_dense, map_location=device))

    logging.info("Model loaded !")

    Image_order = []
    Dice = []
    for i, fn in enumerate(imgList):
        Image_order.append(fn)
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(in_files[0]+fn)
        true_mask = Image.open(mask_files[0]+fn)
        mask, dice = predict_img(net_breast=net_breast,
                            net_dense=net_dense,
                           full_img=img,
                           true_mask = true_mask,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        Dice.append(dice)
        if not args.no_save:
            result = mask_to_image(mask)
            result.save(out_files[0]+fn)

            logging.info("Mask saved to {}".format(out_files[0]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)

    print('Image List: ', Image_order)
    print('Dice Coeff', Dice)
