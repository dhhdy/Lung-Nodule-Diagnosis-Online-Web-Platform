import torch
import SimpleITK as sitk
from lib.ab_bmcy import TCDNet
from utils.TCDNet_predict import TCDNetPred
from utils.resizeImage import resize_image_itk
def nii2np():
    img_path = "load_zip/data.nii.gz"
    image = sitk.ReadImage(img_path)
    # print('size:', image.GetSize())
    siz = image.GetSize()
    image = resize_image_itk(image, (224, 224, image.GetSize()[2]), sitk.sitkNearestNeighbor)
    arr_img = sitk.GetArrayFromImage(image)
    # print('arr_img:', arr_img.shape)
    return arr_img, image, siz
def nodule_predict(option):
    arr_img, ori_img, siz = nii2np()
    if option == "TCDNet":
        model = TCDNet(in_channels=1, out_classes=1)
        model.load_state_dict(torch.load('F:/bmcy/bmcy_JM/late/2024221/bmcy_dice.pth'))
        vol, new_siz, spacing = TCDNetPred(arr_img, ori_img, model)
        return siz, vol, new_siz, spacing  # ori_siz, vol, now_siz, spacing

