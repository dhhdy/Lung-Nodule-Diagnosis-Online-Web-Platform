import numpy
import streamlit as st
from streamlit.elements.image import image_to_url
import numpy as np
from utils.predict import nodule_predict
import pandas as pd
import time
import SimpleITK as sitk
import streamlit as st
import streamlit.components.v1 as components
from ipywidgets import embed
import vtk
from itkwidgets import view
from streamlit_lottie import st_lottie
import requests
from glob import glob
import os
import io
import tempfile
import cv2
from pathlib import Path
from utils.StoreData import store_data

colors = [(0, 0, 255), '红色',
          (0, 255, 0), '绿色',
          (255, 0, 255), '紫色',
          (255, 255, 0), '青色',
          (0, 255, 255), '黄色',
          (255, 0, 0), '蓝色']
params = []


st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title="在线肺结节病灶检测-功能页", page_icon="💻", layout="wide")
# st.sidebar.header("在线肺结节病灶检测-:blue[功能页]")
st.sidebar.header("Online Lung Nodule Lesions Diagnosis-:blue[Function Page]")
# img_url = image_to_url("background.jpg", width=-3, clamp=False, channels='RGB', output_format='auto', image_id='')
# st.markdown('''
# <style>.css-fg4pbf {background-image: url(''' + img_url + ''');
#     width:100%;
#     height:100%;
#     background-size: cover;
#     background-position: center;}</style>
# ''', unsafe_allow_html=True)
hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
hide_streamlit_style = """
    <style>
    #footer {
                position: fixed;
                bottom: 0;
                text-align: center;
                color: black;
                font-family: Arial;
                font-size: 12px;
                letter-spacing: 1px;
            }
            
    </style>

    """
# <div id="footer">©2024 桂林电子科技大学. All Right Reserved</div>
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 上传文件
upload_file = st.file_uploader("**请选择nii.gz文件上传**", type=['nii.gz'],
                               help=":red[可以点击按钮上传，也可以拖拽文件上传，注意只支持.gz格式]")

if upload_file is not None:
    # 存储选择文件
    store_data(upload_file)

    # # 选择分割算法
    option = st.selectbox("**选择一种算法模型**", ('TCDNet', ))
    st.info('点击按钮即可进行病灶检测👇', icon="ℹ️")
    with st.columns(3)[1]:
        predict = st.button(":green[👉开始检测👈]", use_container_width=True) #检测目标
    if predict:
        with st.spinner('Waiting...'):
            time.sleep(10)
            ori_siz, vol, siz, spe = nodule_predict(option=option)

        st.markdown("**检测完成！结果如下**")
        st.info('检测结果👇', icon="ℹ️")

    # # # 展示
        file_path = 'save_zip/data.nii.gz'
        if file_path:
            with st.container():
                reader = vtk.vtkNIFTIImageReader()
                reader.SetFileName(file_path)
                reader.Update()
                view_width = 1800
                view_height = 800
                snippet = embed.embed_snippet(views=view(reader.GetOutput()))
                html = embed.html_template.format(title="", snippet=snippet)
                components.html(html, width=view_width, height=view_height)

            siz = list(siz)
            spe = list(spe)
            ori_siz = list(ori_siz)
            for i in range(len(siz)):
                siz[i] = round(siz[i], 2)
                spe[i] = round(spe[i], 2)
                ori_siz[i] = round(ori_siz[i])
            vol = round(vol, 2)

            df = pd.DataFrame(columns=["输入尺寸/(x,y,z)", "输出尺寸/(x,y,z)", "体素间距/(x,y,z)", "病灶体积/cm3"])
            params.append('('+str(ori_siz[0])+','+str(ori_siz[1])+','+str(ori_siz[2])+')')
            params.append('('+str(siz[0])+','+str(siz[1])+','+str(siz[2])+')')
            params.append('('+str(spe[0])+','+str(spe[1])+','+str(spe[2])+')')
            params.append(str(vol))
            new_row = [params[0], params[1], params[2], params[3]]
            df.loc[len(df)] = new_row

        st.info(f'View the Generated Report👇', icon="ℹ️")
        tab1 = st.tabs(["1️⃣Nodule 1"])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="Input Size/(x,y,z)", value=f"{params[0]}")
        col2.metric(label="Output Size/(x,y,z)", value=f"{params[1]}")
        col3.metric(label="Voxel Spacing/(x,y,z)", value=f"{params[2]}")
        col4.metric(label="Lesion Volume/cm3", value=f"{params[3]}")
            # else:
            #     st.write("该CT图片中未检测到病灶")

        # st.dataframe(df, use_container_width=True)
        st.write("❕Attention：AI Results are for Reference Only，Please Refer to the Doctor's Actual Diagnosis！")
        report = df.to_csv()
        with st.columns(3)[1]:
            st.download_button(":green[Save Report to Excel]", report,
                               file_name="检测报告.csv",
                               use_container_width=True)

