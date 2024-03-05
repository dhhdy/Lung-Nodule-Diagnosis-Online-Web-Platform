# def nodule_predict(option):
#     step = 0
#     if option == "3D_UNet":
#         img_path = "load_zip/data.nii.gz"
#         test_loader = test_dataset(img_path)
#         model = torch.load('F:/test/weight/UNet3D.pth')  # 载入训练权重路径
#         model.eval()
#         tot = 0.
#         num = test_loader.size
#         img, y_t = test_loader.test_one(0)
#         img = img.transpose([2, 1, 0]) # W H D
#         img_list = patchify(img, (64, 64, 64), step=64)
#         wid = img_list.shape[0]
#         high = img_list.shape[1]
#         depth = img_list.shape[2]
#         print(depth * high * wid)
#         for i in range(img_list.shape[0]):
#             for j in range(img_list.shape[1]):
#                 for k in range(img_list.shape[2]):
#                     img = img_list[i][j][k][:][:][:]
#                     # img = img.astype(np.float32)
#                     img = transforms.ToTensor()(img)  # D W H
#                     img = torch.transpose(img, 2, 1) # D H W
#                     img = torch.unsqueeze(img, 0)
#                     img = torch.unsqueeze(img, 0)
#                     img = img.type(torch.FloatTensor)
#                     img = img.cuda()
#                     model.cuda()
#                     with torch.no_grad():
#                         res = model(img)
#                         res = res.sigmoid().data.cpu().numpy()
#                         res = res[0, 0, :, :, :]
#                         res = 1 * (res > 0.5)
#                         res = res.transpose([2, 1, 0]) # W H D
#                         img_list[i][j][k][:][:][:] = res
#                         step += 1
#                         print(step)
#         nii_file = unpatchify(img_list, [64 * wid, 64 * high, 64 * depth]) # W H D
#         y_arr = sitk.GetArrayFromImage(y_t)  # D H W
#         nii_file = nii_file.transpose([2, 1, 0])  # D H W
#
#         out = sitk.GetImageFromArray(nii_file.astype(np.int32))
#         out = copy_geometry(out, y_t)
#         out, vol, siz, ori, spe = maxConnectArea(out)
#         sitk.WriteImage(out, 'F:/test/weight/gg.nii.gz')
#         out = sitk.GetArrayFromImage(out)
#         print(out.shape, y_arr.shape)
#         for i in range(y_arr.shape[0]):
#             for j in range(y_arr.shape[1]):
#                 for k in range(y_arr.shape[2]):
#                     if out[i][j][k] > 0:
#                         y_arr[i][j][k] = 255
#         y_arr = y_arr.transpose([0,2,1])
#         print(y_arr.shape)
#         out = sitk.GetImageFromArray(y_arr)
#         sitk.WriteImage(out, 'save_zip/data.nii.gz')
#         return vol, siz, ori, spe