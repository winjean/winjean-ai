import matplotlib.pyplot as plt
import numpy as np


def plot_line():
    # 生成数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # 创建线图
    plt.plot(x, y, label='sin(x)')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title('Sine Wave')
    plt.legend()
    plt.savefig("sine_wave.png")
    plt.show()


def scatter():
    # 生成数据
    x = np.random.rand(50)
    y = np.random.rand(50)
    colors = np.random.rand(50)
    sizes = 1000 * np.random.rand(50)

    # 创建散点图
    plt.scatter(x, y, c=colors, s=sizes, alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot')
    plt.show()


def subplots():
    # 生成数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 第一个子图
    ax1.plot(x, y1, label='sin(x)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('sin(x)')
    ax1.set_title('Sine Wave')
    ax1.legend()

    # 第二个子图
    ax2.plot(x, y2, label='cos(x)', color='orange')
    ax2.set_xlabel('x')
    ax2.set_ylabel('cos(x)')
    ax2.set_title('Cosine Wave')
    ax2.legend()

    plt.show()


def bar_chart():
    # 生成数据
    categories = ['A', 'B', 'C', 'D']
    values = [23, 45, 56, 78]

    # 创建条形图
    plt.bar(categories, values, color=['red', 'green', 'blue', 'orange'])
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Chart')
    plt.show()


def mpimg():
    import matplotlib.image as mpimg

    # 读取图像
    img = mpimg.imread('image.jpg')

    # 显示图像
    plt.imshow(img)
    plt.axis('off')  # 关闭坐标轴
    plt.title('Image Display')
    plt.show()


def custom_line():
    # 生成数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # 创建线图
    plt.plot(x, y, label='sin(x)', linestyle='--', color='red', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title('Sine Wave')
    plt.legend()

    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.5)

    # 设置背景颜色
    plt.gca().set_facecolor('#f0f0f0')

    plt.show()


def slider_plots():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.widgets import Slider

    # 生成数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # 创建图表
    fig, ax = plt.subplots()
    line, = ax.plot(x, y, label='sin(x)')
    ax.set_xlabel('x')
    ax.set_ylabel('sin(x)')
    ax.set_title('Interactive Sine Wave')
    ax.legend()

    # 创建滑块
    ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03])
    slider = Slider(ax_slider, 'Frequency', valmin=0.1, valmax=2.0, valinit=1.0)

    # 更新函数
    def update(val):
        freq = slider.val
        line.set_ydata(np.sin(freq * x))
        fig.canvas.draw_idle()

    # 绑定更新函数
    slider.on_changed(update)

    plt.show()


if __name__ == '__main__':
    # plot_line()
    # scatter()
    # bar_chart()
    # subplots()
    # mpimg()
    # custom_line()
    slider_plots()
