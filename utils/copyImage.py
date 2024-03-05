import SimpleITK as sitk

def copy_geometry(image: sitk.Image, ref: sitk.Image):  #取出图像
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image