import streamlit as st

def store_data(file):
    temporary_location = 'load_zip/data.nii.gz'
    with open(temporary_location, 'wb') as out:
        out.write(file.getbuffer())
    st.success('文件上传成功!')
