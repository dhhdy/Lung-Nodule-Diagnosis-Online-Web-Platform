import torch
import numpy as np
import SimpleITK as sitk
from utils.indicator import Dice
import torchvision.transforms as transforms
from utils.copyImage import copy_geometry
from utils.maxC import maxConnectArea

def TCDNetPred(arr_img, ori_img, model):
    # D W H
    model.cuda()
    model.eval()
    img = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.369],
                             [0.251])
    ])
    with torch.no_grad():
        for i in range(arr_img.shape[0]):
            # print('step: ', i)
            image = arr_img[i]
            #pred
            image = np.array(image).astype(np.float32)
            image = transform(image).unsqueeze(0) # B C H W
            image = image.cuda()
            _, __, image = model(image)
            image = image.sigmoid().data.cpu().numpy().squeeze()
            image = 1 * (image > 0.5)
            image = image.astype(np.uint8)
            img.append(image)
        pred = np.stack(img, axis=0)
        pred = sitk.GetImageFromArray(pred.astype(np.int32))
        out = copy_geometry(pred, ori_img)
        out, vol, siz, ori, spe = maxConnectArea(out)
        sitk.WriteImage(out, 'F:/test/weight/gg.nii.gz')
        out = sitk.GetArrayFromImage(out)
        y_arr = sitk.GetArrayFromImage(ori_img)
        for i in range(y_arr.shape[0]):
            for j in range(y_arr.shape[1]):
                for k in range(y_arr.shape[2]):
                    # if y_arr[i][j][k]>=255:
                    #     print(y_arr[i][j][k])
                    if out[i][j][k] > 0:
                        y_arr[i][j][k] = 5000
        # y_arr = y_arr.transpose([0, 2, 1])
        # print(y_arr.shape)
        y_nii = sitk.GetImageFromArray(y_arr)
        y_nii = copy_geometry(y_nii, ori_img)
        sitk.WriteImage(y_nii, 'save_zip/data.nii.gz')
        return vol, siz, spe