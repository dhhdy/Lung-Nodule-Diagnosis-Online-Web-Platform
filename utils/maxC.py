import SimpleITK as sitk
import numpy as np

def maxConnectArea(itk_image_, ):
    """ 获取最大连通域
    return: itk image"""
    # 获取图像的大小、原点和间距
    size = itk_image_.GetSize()
    origin = itk_image_.GetOrigin()
    spacing = itk_image_.GetSpacing()
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    output_connected = cc_filter.Execute(itk_image_)
    # -> 0,1,2,....一系列的连通区域编号, 0表示背景
    output_connected_array = sitk.GetArrayFromImage(output_connected)
    # print(np.unique(output_connected_array))
    num_connected_label = cc_filter.GetObjectCount()

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_connected)

    max_area = 0
    max_label_idx = 0
    # -> 找出最大的area
    # 连通域label从1开始, 0表示背景
    for i in range(1, num_connected_label + 1):
        cur_area = lss_filter.GetNumberOfPixels(i)
        # print(cur_area, i)
        if cur_area > max_area:
            max_area = cur_area
            max_label_idx = i
    # print(max_label_idx, max_area)
    re_mask = np.zeros_like(output_connected_array, dtype='uint8')
    re_mask[output_connected_array == max_label_idx] = 1

    re_image = sitk.GetImageFromArray(re_mask)
    # 计算非零像素的数量
    non_zero_voxels = (re_mask > 0).sum()
    # 计算像素的体积（以立方毫米为单位）
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

    # 计算体积（以 mm³ 为单位）
    volume_cm3 = non_zero_voxels * voxel_volume_mm3 / 1000.

    re_image.SetDirection(itk_image_.GetDirection())
    re_image.SetSpacing(itk_image_.GetSpacing())
    re_image.SetOrigin(itk_image_.GetOrigin())
    return re_image, volume_cm3, size, origin, spacing