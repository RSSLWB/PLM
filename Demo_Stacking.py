import streamlit as st
import pandas as pd
from joblib import load
import plotly.express as px

# 从文件读取Base64编码的字符串
def load_base64_encoded_image(image_path):
    with open(image_path, "r") as image_file:
        return image_file.read()

# 假设您的Base64文本文件在同一目录中名为 'bg_image_base64.txt'
bg_image_base64 = load_base64_encoded_image(r"C:\Users\LSSR\Desktop\image_base64.txt")

# 设置背景图片和亮度调整的函数
def set_bg_and_adjust_brightness(bg_base64, brightness_level=0.85):
    brightness_level = max(0, min(1, brightness_level))
    overlay_color = f'rgba(0, 0, 0, {1 - brightness_level})'
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{bg_base64}");
            background-size: cover;
            background-position: center;
        }}
        .stApp::after {{
            content: "";
            display: block;
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: {overlay_color};
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 调用函数设置背景图片并调整亮度
set_bg_and_adjust_brightness(bg_image_base64, brightness_level=0.5)

# 加载模型
model = load("D:/毕设/2023-11/stacking_model.joblib")

# 页面标题
# 使用CSS来自定义样式
import streamlit as st

# 使用CSS来自定义样式
# 使用CSS来自定义样式
st.markdown("""
    <style>
    .title {
        font-size: 45px;  # 设置字体大小
        font-weight: bold;  # 字体加粗
        color: #5F91F4;  # 设置字体颜色
        text-align: center;  # 设置文本居中显示
        white-space: nowrap;  # 阻止换行
    }
    </style>
    """, unsafe_allow_html=True)

# 设置页面标题，并使用自定义的CSS样式
st.markdown('<p class="title">Predict the occurrence of lung metastases</p>', unsafe_allow_html=True)


# 在侧边栏添加用户输入特征的部分
with st.sidebar:
    st.write('Select the corresponding eigenvalue based on the drop-down list for prediction：')
    age = st.number_input('Age', min_value=0, max_value=120, value=30)  # 设置合理的默认值和范围
    sex = st.selectbox('Sex', options=[('Female', 0), ('Male', 1)], format_func=lambda x: x[0])
    race = st.selectbox('Race', options=[('Black', 0), ('Other', 1), ('White', 2)], format_func=lambda x: x[0])
    tumor_number = st.selectbox('Tumor_number', options=[('Multiple', 0), ('Single', 1)], format_func=lambda x: x[0])
    sequence = st.selectbox('Sequence', options=[('One or first', 0), ('Second or last', 1)],format_func=lambda x: x[0])
    grade = st.selectbox('Grade', options=[('I-II', 0), ('III-IV', 1), ('Unknown', 2)], format_func=lambda x: x[0])
    marital = st.selectbox('Marital', options=[('Married', 0), ('Other', 1)], format_func=lambda x: x[0])
    surgery = st.selectbox('Surgery', options=[('Other_surgery', 0), ('Total_thyroidectomy', 1)],format_func=lambda x: x[0])
    radiation = st.selectbox('Radiation', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])
    survival_months = st.number_input('Survivalmonths', min_value=0, value=12)  # 设置合理的默认值
    t_stage = st.selectbox('T_stage', options=[('T0', 0), ('T1', 1), ('T2', 2), ('T3', 3), ('T4a', 4), ('T4b', 5), ('TX', 6)],format_func=lambda x: x[0])
    n_stage = st.selectbox('N_stage', options=[('N0', 0), ('N1a', 1), ('N1b', 2), ('NX', 3)],format_func=lambda x: x[0])
    m_stage = st.selectbox('M_stage', options=[('M0', 0), ('M1', 1), ('MX', 2)], format_func=lambda x: x[0])

# 预测按钮
if st.button('Predict'):
    # 将用户输入整理为模型的输入格式
    input_data = pd.DataFrame([[age, sex[1], race[1], tumor_number[1], sequence[1], grade[1], marital[1], surgery[1],
                                radiation[1], survival_months, t_stage[1], n_stage[1], m_stage[1]]],
                              columns=['Age', 'Sex', 'Race', 'Tumor_number', 'Sequence', 'Grade', 'Marital', 'Surgery',
                                       'Radiation', 'Survivalmonths', 'T_stage', 'N_stage', 'M_stage'])

    # 使用模型进行预测
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # 显示预测结果，使用自定义的字体样式
    result = "Alive" if prediction == 0 else "Dead"
    st.markdown(f'<p class="big-font">Predict result is: {result}</p>', unsafe_allow_html=True)

    # 创建概率条形图并设置透明背景
    fig = px.bar(x=['Alive', 'Dead'], y=probabilities, labels={'x': 'Outcome', 'y': 'Probability'}, title="Prediction Confidence")
    fig.update_traces(marker_color='#5F91F4')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    st.plotly_chart(fig)