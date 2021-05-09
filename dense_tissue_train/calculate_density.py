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
                device,
                scale_factor=1,
                out_threshold=0.5):
    net_breast.eval()
    net_dense.eval()

    img = BasicDataset.preprocess(full_img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

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

        pred_breast = pred_breast.squeeze(0)
        pred_dense_combine = pred_dense_combine.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        pred_breast = tf(pred_breast.cpu())
        pred_dense_combine = tf(pred_dense_combine.cpu())
        breast_mask = pred_breast.squeeze().cpu().numpy()
        dense_mask = pred_dense_combine.squeeze().cpu().numpy()
    return breast_mask, dense_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_breast', '-m_b', metavar='FILE',
                        help="Specify the file in which the breast model is stored")

    parser.add_argument('--model_dense', '-m_d', metavar='FILE',
                        help="Specify the file in which the dense tissue model is stored")
    
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output_breast', '-o_b', metavar='INPUT', nargs='+',
                        help='Filenames of ouput breast')

    parser.add_argument('--output_dense', '-o_d', metavar='INPUT', nargs='+',
                        help='Filenames of ouput dense')

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


# def get_output_filenames(args):
#     in_files = args.input
#     out_files = []

#     if not args.output:
#         for f in in_files:
#             pathsplit = os.path.splitext(f)
#             out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
#     elif len(in_files) != len(args.output):
#         logging.error("Input files and output files are not of the same length")
#         raise SystemExit()
#     else:
#         out_files = args.output

#     return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files_b = args.output_breast
    # if not os.path.exists(out_files_b[0]):
    #     os.makedirs(out_files_b[0])
    # out_files_d = args.output_dense
    # if not os.path.exists(out_files_d[0]):
    #     os.makedirs(out_files_d[0])
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

    volume_breast = 0
    volume_dense_tissue = 0

    # imgList.remove('_OUT')
    for i, fn in enumerate(imgList):
        logging.info("\nCalculating density {} ...".format(fn))
        img = Image.open(in_files[0]+fn)
        breast_mask, dense_mask = predict_img(net_breast=net_breast,
                            net_dense=net_dense,
                            full_img=img,
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            device=device)

        volume_breast += np.sum(breast_mask)
        volume_dense_tissue += np.sum(dense_mask)
        
        # if not args.no_save:
        #     result_b = mask_to_image(breast_mask)
        #     result_b.save(out_files_b[0]+fn)

        #     logging.info("Mask saved to {}".format(out_files_b[0]))

        #     result_d = mask_to_image(dense_mask)
        #     result_d.save(out_files_d[0]+fn)

        #     logging.info("Mask saved to {}".format(out_files_d[0]))

        # if args.viz:
        #     logging.info("Visualizing results for image {}, close to continue ...".format(fn))
        #     plot_img_and_mask(img, mask)

    density = volume_dense_tissue/volume_breast
    # if density < 0.25:
    #     category = 'a'
    # elif density < 0.5:
    #     category = 'b'
    # elif density < 0.75:
    #     category = 'c'
    # else:
    #     category = 'd'

    print('density = ', density)
    # print('category = ', category)
