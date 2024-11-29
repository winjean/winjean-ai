import streamlit
import pandas

# 使用方法
# streamlit run vectorstore_test/streamlit-feed_forward.py
#

streamlit.set_page_config(
    "Langchain-Chatchat WebUI",
    # get_img_base64("chatchat_icon_blue_square_v2.png"),
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/chatchat-space/Langchain-Chatchat",
        "Report a bug": "https://github.com/chatchat-space/Langchain-Chatchat/issues",
        "About": f"""欢迎使用 Langchain-Chatchat WebUI {1}！""",
    },
    layout="centered",
)

# 写入标题
streamlit.title('My First Streamlit App')

sidebar_selection = streamlit.sidebar.selectbox('Select an option', ['Option 1', 'Option 2'])

streamlit.radio('Choose a number:', [1, 2, 3, 4, 5])
streamlit.checkbox('I agree')



streamlit.markdown(
    """
    <style>
    [data-testid="stSidebarUserContent"] {
        padding-top: 20px;
    }
    .block-container {
        padding-top: 25px;
    }
    [data-testid="stBottomBlockContainer"] {
        padding-bottom: 20px;
    }
    """,
    unsafe_allow_html=True,
)


# 加载数据
df = pandas.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

# 显示数据框
streamlit.write(df)

# 添加一个滑块并根据滑块值显示信息
value = streamlit.slider('Select a value')
streamlit.write(f'Selected value: {value}')

name = streamlit.text_input('请输入你的名字：')

if name:
    streamlit.write(f'你好，{name}！')
else:
    streamlit.write('还在等你输入名字呢！')

streamlit.button('点击我！')

if streamlit.button('Click me'):
    streamlit.write('Button clicked!')

if __name__ == '__main__':
    pass
