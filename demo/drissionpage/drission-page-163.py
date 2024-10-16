from DrissionPage import Chromium
import traceback

# 检查是否已登录
def check_login() -> bool:
    account_box=tab.ele('#account-box')
    password_text=tab.ele('#pwdtext')
    return False if account_box and password_text else True

# 登录
def long_in(account: str, password: str) -> bool:
    try:
        if not check_login():
            # 定位到账号文本框，获取文本框元素
            ele = tab.ele('#account-box')
            ele.clear()
            # 输入对文本框输入账号
            ele.input(account)

            # 定位到密码文本框并输入密码
            tab.ele('#pwdtext').clear().input(password)

            # 点击登录按钮
            tab.ele('#dologin').click()

            # 点击登录按钮
            error_message = tab.ele('.ferrorhead')
            if error_message:
                print("账号或密码错误")
                return False

            # 安全验证
            yidun=tab.ele('.yidun_modal__wrap')
            if yidun:
                print("请手动通过安全验证")
                return False

            return True
    except Exception as e:
        print("login fail{}", e)
        traceback.print_exc()
        return False

# 切换到首页
def switch_to_first_page():
    tab.ele('text=首页').click()

# 点击写信
def click_write_email():
    switch_to_first_page()
    tab.ele('text=写 信').click()

# 填入收件人
def fill_recipient_name(address: str):
    recipient_name=tab.ele('.nui-editableAddr-ipt')
    recipient_name.input(address)

# 填入主题
def fill_subject(subject: str):
    e=tab.ele('#$subjectInput').input("ssss")
    e.input(subject)

# 填入邮件内容
def fill_content(content: str):
    tab.ele('.nui-scroll').input(content)

# 上传附件
def upload_file(files: str):
    upload = tab('tag:input@type=file')
    upload.input(files.split(','))

# 发送邮件
def send_email():
    tab.ele('text=发送').click()

def write_email(address: str, subject: str, content: str,files:str = None):
    try:
        click_write_email()
        fill_recipient_name(address)
        fill_subject(subject)
        fill_content(content)
        if files:
            upload_file(files)
    except Exception as e:
        print("write email fail{}", e)
        traceback.print_exc()

if __name__ == '__main__':
    # 启动或接管浏览器，并创建标签页对象
    tab = Chromium().latest_tab

    # 跳转到登录页面
    tab.get('https://mail.163.com/')

    success=long_in("13851941860", 'Winjean1979')

    if success:
        # write_email("19548901@qq.com", "邮件主题", "邮件内容")
        write_email("19548901@qq.com", "邮件主题", "邮件内容","D:/test/1.txt,D:/test/2.txt")
        send_email()