# **Lung-Nodule-Diagnosis-Online-Web-Platform**
![fig50](https://github.com/dhhdy/Lung-Nodule-Diagnosis-Online-Web-Platform/assets/122719285/880063a8-788e-4de5-b921-744e4036e39c)

## **run**
Terminal Command: streamlit run Hello.py

## **Change Model**
#### File path: utils/predict.py. Line: 18,19.


model = Your Model

model.load_state_dict(torch.load('Your pretrained_weight path')


#### File path: utils/TCDNet_predict.py. Line: 27.

_, __, image = model(image). Modify the Return of Your Model.


