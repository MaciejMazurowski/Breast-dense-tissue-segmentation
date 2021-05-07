import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from torchvision import transforms
from dice_loss import dice_coeff

unloader = transforms.ToPILImage() 
def save_image(tensor, dir, epoch, file_name, img_num):
    image = tensor.cpu().clone()  
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    save_dir = dir + file_name + '/' + str(epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image.save(save_dir + '/val_' + str(img_num) + '.png')

def eval_net(net, loader, device, epoch, dir):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    num = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            true_masks = true_masks[:, 0, :, :].unsqueeze(1)
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                true_masks = (true_masks > 0.5).float() 
                # tot += dice_coeff(pred2, pred, true_masks, num).item()
                dice = dice_coeff(pred, true_masks, num) 
                tot += dice.item()
            pbar.update()

            if (epoch == 0):
                save_image(imgs, dir, epoch, 'mri', num)
                save_image(true_masks, dir, epoch, 'true_mask', num)
            save_image(pred, dir, epoch, 'pred', num)
            num += 1

    net.train()
    return tot / n_val
