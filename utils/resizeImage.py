import SimpleITK as sitk
import numpy as np

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize() # 获取原图size
    originSpacing = itkimage.GetSpacing() # 获取原图spacing
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.uint8) # spacing格式转换
    resampler.SetReferenceImage(itkimage) # 指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  #得到重新采样后的图像
    return itkimgResampled