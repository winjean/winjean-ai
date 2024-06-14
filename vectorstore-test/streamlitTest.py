import streamlit as st
import pandas as pd

# 使用方法
# streamlit run streamlitTest.py
#

# 加载数据
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

# 写入标题
st.title('My First Streamlit App')

# 显示数据框
st.write(df)

# 添加一个滑块并根据滑块值显示信息
value = st.slider('Select a value')
st.write(f'Selected value: {value}')

name = st.text_input('请输入你的名字：')

if name:
    st.write(f'你好，{name}！')
else:
    st.write('还在等你输入名字呢！')

st.button('点击我！')

if __name__ == '__main__':
    pass
