import streamlit as st
from streamlit.elements.image import image_to_url
import time

upload_flag = False
predict_flag = False
gray_img_flag = False


# @st.cache_data
# def upload(upload_file):
#     global upload_flag, predict_flag, gray_img_flag
#     if upload_file is not None:
#         upload_flag = True
#         image = np.array(bytearray(upload_file.read()))
#         image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#         gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         cv2.imwrite("tmp/original.png", gray_img)
#         img_resize = cv2.resize(gray_img, (256, 256))
#         st.markdown("### 上传CT图片成功！显示如下：")
#         with st.columns(3)[1]:
#             st.image(cv2.resize(gray_img, (2048, 2048)))
#         input_image = img_resize.reshape(1, 1, img_resize.shape[0], img_resize.shape[1])
#         st.markdown("**点击按钮开始检测**")
#         predict = st.button("病灶检测")
#         if predict:
#             predict_flag = True
#             latest_iteration = st.empty()
#             my_bar = st.progress(0)
#             for percent_complete in range(100):
#                 time.sleep(0.1)
#                 my_bar.progress(percent_complete + 1)
#                 latest_iteration.text(f"AI检测中，请耐心等待。当前进度：{percent_complete + 1}%")
#             result = nodule_predict(input_image=input_image)
#             cv2.imwrite("result/result.png", result)


# def model_predict():
#     global upload_global, gray_img_global, predict_bool_global
#     nodule = cv2.imread("result/result.png", 0)
#     gray_img = cv2.imread("tmp/original.png", 0)
#     st.markdown("**检测完成！结果如下**")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.image(gray_img, caption="原CT")
#     with col2:
#         st.image(nodule, caption="病灶检测结果")
#     with col3:
#         # nodule_ = cv2.cvtColor(nodule, cv2.COLOR_GRAY2RGB)
#         gray_img = cv2.cvtColor(gray_img_global, cv2.COLOR_GRAY2RGB)
#         # for i in range(nodule_.shape[0]):
#         #     for j in range(nodule_.shape[1]):
#         #         for k in range(nodule_.shape[2]):
#         #             if nodule_[i, j, k] == 255:
#         #                 nodule_[i, j, 0] = 255
#         #                 nodule_[i, j, 1] = 0
#         #                 nodule_[i, j, 2] = 0
#         nodule_params = contours_norm_compute(nodule)
#         norm = cv2.imread("result/rect.png")
#         # roi = cv2.addWeighted(gray_img, 0.8, norm, 1, 0)
#         roi = cv2.add(norm, gray_img)
#         roi = cv2.resize(roi, (2048, 2048))
#         st.image(roi, caption="结果标注", channels='BGR')
#     col4, col5, col6 = st.columns(3)
#     col7, col8, col9 = st.columns(3)
#     for i in range(len(nodule_params)):
#         col4.metric("结节区域", str(nodule_params[i][0]))
#         col5.metric("结节面积", str(nodule_params[i][1]))
#         col6.metric("中心坐标", "(" + str(nodule_params[i][2]) + "," + str(nodule_params[i][3]) + ")")
#         col7.metric("最大直径", str(nodule_params[i][4]))


def test1(st):
    st.title("test1")


def test2(st):
    st.title("test2")


if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_page_config(
        page_title="在线肺腺癌病灶检测-欢迎页",
        page_icon="👋",
        layout="wide"
    )
    img_url = image_to_url("./pages/background.jpg", width=-3, clamp=False, channels='RGB', output_format='auto', image_id='')

    st.markdown('''
    <style>.css-fg4pbf {background-image: url(''' + img_url + ''');
    width:100%;
    height:100%;
    background-size: cover;
    background-position: center;}</style>
    ''', unsafe_allow_html=True)
    st.balloons()
    st.sidebar.header("在线肺腺癌病灶检测-:blue[欢迎页]")
    # st.sidebar.success("👆👆")
    st.title("**:violet[早期肺腺癌CT影像病灶检测系统]**", anchor=False)
    st.info('欢迎来到本系统👋请先阅读下面的使用须知👇')
    # icon = "🌏"
    with st.expander("**使用须知**"):
        st.markdown("### 1. 系统简介")
        st.write("本系统基于AI神经网络算法，实现早期肺腺癌CT影像的在线检测，并提供查看检测结果、生成检测报告和下载报告的功能。")
        st.write("适用人群：医院影像科医生或实习医生、影像学专业学生、从事或热爱医学图像处理方向的计算机专业学生以及所有对AI辅助"
                 "医疗诊断感兴趣的人。")
        st.markdown("### 2. 页面引导")
        st.write("本系统共有*欢迎页* 和*功能页* 两个页面，当前您正处在**欢迎页**，点击页面左侧侧边栏的相关页面链接即可实现页面跳转。")
        st.markdown("* 欢迎页👇")
        st.markdown("> 👈对应页面左侧侧边栏的**Hello**链接")
        st.markdown("* 功能页👇")
        st.markdown("> 👈对应页面左侧侧边栏的**Function**链接")
        st.markdown("### 3. 检测步骤")
        st.markdown("> (1)在:blue[**功能页**]，选择或拖拽一张CT图片到上传区域")
        st.markdown("> (2)点击:blue[**开始预测**]按钮")
        st.markdown("> (3)等待AI自动检测完成后，下滑页面即可查看检测结果和检测报告")
        st.markdown("> (4)点击:blue[**保存报告至Excel**]按钮，即可下载检测报告")
        st.markdown("### 4. 注意事项")
        st.markdown("⚠*切换页面或者点击下载按钮后页面将刷新，若想继续查看原先的检测结果，可以重新点击:blue[**开始预测**]按钮*")
        st.markdown("⚠*本系统解释权最终归桂林电子科技大学所有*")

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
        <div id="footer">©2023 桂林电子科技大学. All Right Reserved</div>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
