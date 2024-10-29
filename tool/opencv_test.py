import cv2

def read_image(file_path):
    # 读取图像
    image = cv2.imread(file_path)

    # 转换为灰度图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 缩放图像
    image = cv2.resize(image, (400, 300))

    # 显示图像
    cv2.imshow('Image', image)

    # 等待用户按键后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def detect_faces(file_path):
    # 加载预训练的 Haar 级联分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 读取图像
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    # 绘制矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 显示图像
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    file_path = 'output.jpg'
    # read_image(file_path)
    detect_faces(file_path)