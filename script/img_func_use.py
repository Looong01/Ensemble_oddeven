# import libraries
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import PIL
from PIL import Image
from sklearn.model_selection import train_test_split
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# import dataset (comes with colab!)
data_train = np.loadtxt(open('../data/small_mnist/mnist_train_small.csv','rb'),delimiter=',')
labels_train = data_train[:,0]
data_train = data_train[:,1:]

test_tmp = np.loadtxt(open('../data/small_mnist/mnist_test.csv','rb'),delimiter=',')
labels_tmp = test_tmp[:,0]
data_tmp = test_tmp[:,1:]

data_val, data_test, labels_val, labels_test = train_test_split(data_tmp, labels_tmp, test_size=0.3, random_state=42)

# %%
def generate_imgs(data, labels, noise, img_shape):
    even_num = 0
    label_imgs = []
    merged_images = np.zeros((2 * img_shape[0], 2 * img_shape[1]))  # 创建一个空的2x2的图像矩阵
    for i in range(4):
        idx = np.random.randint(0, high=data.shape[0])  # 修改此处为 data.shape[0]
        label = labels[idx]
        label_imgs.append(label)
        img = np.reshape(data[idx, :], img_shape)

        if (label % 2) == 0:
            even_num += 1

        row = i // 2
        col = i % 2
        merged_images[row * img_shape[0]:(row + 1) * img_shape[0], col * img_shape[1]:(col + 1) * img_shape[1]] = img
    if even_num > 2:
        final_label = "even"
    elif even_num == 2:
        final_label = "same"
    else:
        final_label = "odd"

    gaussian_noise = np.random.normal(0, noise, merged_images.shape)
    final_img = merged_images + gaussian_noise
    return final_img, final_label, even_num



# %%

def generate_dataset_images(data, labels, noise, img_shape, output_path, num_images):
    for i in range(num_images):
        final_img, final_label, even_num = generate_imgs(data, labels, noise, img_shape)
        im = Image.fromarray(final_img.astype(np.uint8))
        if im.mode == "F":
            im = im.convert("RGB")
        
        if final_label == "even":
            tmp_path = output_path + "/even"
        elif final_label == "odd":
            tmp_path = output_path + "/odd"
        else:
            tmp_path = output_path + "/same"
        
        im_name = str(even_num) + '_' + str(noise) + '_' + str(i) + '.jpg'
        im.save(tmp_path + "/" + im_name)

# %%
# 使用函数生成训练数据集图像
train_path = "../data/train"
generate_dataset_images(data_train, labels_train, noise=0, img_shape=(28, 28), output_path=train_path, num_images=9000)

# 使用函数生成验证数据集图像
val_path = "../data/validation"
generate_dataset_images(data_val, labels_val, noise=0, img_shape=(28, 28), output_path=val_path, num_images=3000)

# 使用函数生成测试数据集图像
test_path = "../data/test"
generate_dataset_images(data_test, labels_test, noise=0, img_shape=(28, 28), output_path=test_path, num_images=1500)

# %%
import os
import shutil
test_path = "../data/test"
# 创建subtest文件夹
subtest_path = "../data/subtest"
if not os.path.exists(subtest_path):
    os.makedirs(subtest_path)

# 从test文件夹中抽取不同类别的图片各80张加上随机噪音，放到subtest文件夹里面
categories = ['even', 'odd', 'same']
num_images_per_category = 80
noise_levels = [2, 1, 0]  # 不同的噪音水平

for category in categories:
    category_path = os.path.join(test_path, category)
    subcategory_path = os.path.join(subtest_path, category)
    images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    selected_images = np.random.choice(images, num_images_per_category, replace=False)
    print(selected_images)
    for image_name in selected_images:
        # 读取原始图片
        image_path = os.path.join(category_path, image_name)
        original_img = Image.open(image_path)
        #original_img.show()
        
        # 将图片转换为 NumPy 数组
        np_image = np.array(original_img)
        
        # 为每张图片添加三个不同水平的噪音
        for noise_level in noise_levels:
            noisy_np_image = np_image + np.random.normal(0, noise_level, np_image.shape)
            
            # 将带有噪音的图像转换回 PIL Image 对象
            noisy_img = Image.fromarray(noisy_np_image.astype(np.uint8))
            
            # 添加噪音后，将图像转换为 "RGB" 模式
            noisy_img = noisy_img.convert("RGB")
            #原文件"even_num_噪音值_循环编号.jpg"
            # 保存带有噪音的图片到subtest文件夹中，文件名为 "原文件信息+新噪音值.jpg"
            im_name = image_name.split('.')[0] + '_' + str(noise_level) + '.jpg'
            sub_image_path = os.path.join(subcategory_path, im_name)
            noisy_img.save(sub_image_path)



# %%
