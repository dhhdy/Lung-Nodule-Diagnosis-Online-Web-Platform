# **Lung-Nodule-Diagnosis-Online-Web-Platform**
![fig50](https://github.com/dhhdy/Lung-Nodule-Diagnosis-Online-Web-Platform/assets/122719285/880063a8-788e-4de5-b921-744e4036e39c)

## **run**
Terminal Command: streamlit run Hello.py

## **Change Model**
1. File path: utils/predict.py. Line: 18,19.

  Function: nodule_predict

  Change model = Your Model

Change model.load_state_dict(torch.load('Your pretrained_weight path')

2. File path: utils/TCDNet_predict.py. Line: 27.

  Function: TCDNet_predict

  Line Change _, __, image = model(image). Modify the Return of Your Model.


