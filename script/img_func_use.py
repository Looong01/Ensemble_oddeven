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

def generate_dataset_images(target_label, data, labels, noise, img_shape, output_path, num_images):
    target_count = 0
    while (target_count<num_images):
        final_img, final_label, even_num = generate_imgs(data, labels, noise=noise, img_shape=img_shape)
        im = Image.fromarray(final_img.astype(np.uint8))
        if im.mode == "F":
                im = im.convert("RGB")
        if final_label == target_label:
            tmp_path = output_path + "/" +target_label
            target_count += 1
            im_name = str(even_num) + '_' + str(noise) + '_' + str(target_count) + '.jpg'
            im.save(tmp_path + "/" + im_name)     

# %%
# 使用函数生成训练数据集图像
train_path = "../data/train"
generate_dataset_images("same", data_train, labels_train, noise=0, img_shape=(28, 28), output_path=train_path, num_images=4000)
generate_dataset_images("odd",data_train, labels_train, noise=0, img_shape=(28, 28), output_path=train_path, num_images=2000)
generate_dataset_images("even",data_train, labels_train, noise=0, img_shape=(28, 28), output_path=train_path, num_images=2000)

# 使用函数生成验证数据集图像
val_path = "../data/validation"

generate_dataset_images("same", data_val, labels_val, noise=0, img_shape=(28, 28), output_path=val_path, num_images=1200)
generate_dataset_images("odd",data_val, labels_val, noise=0, img_shape=(28, 28), output_path=val_path, num_images=600)
generate_dataset_images("even",data_val, labels_val, noise=0, img_shape=(28, 28), output_path=val_path, num_images=600)

# 使用函数生成测试数据集图像
test_path = "../data/test"
generate_dataset_images("same", data_test, labels_test, noise=0, img_shape=(28, 28), output_path=test_path, num_images=600)
generate_dataset_images("odd",data_test, labels_test, noise=0, img_shape=(28, 28), output_path=test_path, num_images=300)
generate_dataset_images("even",data_test, labels_test, noise=0, img_shape=(28, 28), output_path=test_path, num_images=300)


# %%
import os
import shutil
import numpy as np
from PIL import Image

test_path = "../data/test"
subtest_path = "../data/subtest"

# 创建subtest文件夹和各个类别的子文件夹
if not os.path.exists(subtest_path):
    os.makedirs(subtest_path)
for category in ['even', 'odd', 'same']:
    category_subpath = os.path.join(subtest_path, category)
    if not os.path.exists(category_subpath):
        os.makedirs(category_subpath)

num_images_per_category = 50
noise_levels = [1, 0.5, 0]

for category in os.listdir(test_path):
    category_path = os.path.join(test_path, category)
    if not os.path.isdir(category_path):
        continue
    if category == 'even':
        list_num = ['3', '4']
    elif category == 'odd':
        list_num = ['0', '1']

    else:
        list_num = ['2']
    for prefix in list_num:
        images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and f.startswith(prefix)]
        print(os.listdir(category_path))
        if category == 'same':
            selected_images = np.random.choice(images, num_images_per_category*4, replace=False)
        else:
            selected_images = np.random.choice(images, num_images_per_category, replace=False)
        
        for image_name in selected_images:
            image_path = os.path.join(category_path, image_name)
            original_img = Image.open(image_path)
            np_image = np.array(original_img)
            
            for noise_level in noise_levels:
                noisy_np_image =  np_image + np.random.normal(0, noise_level, np_image.shape)
                noisy_img = Image.fromarray(noisy_np_image.astype(np.uint8))
                noisy_img = noisy_img.convert("RGB")
                
                im_name = f"{image_name.split('.')[0]}_{noise_level}.jpg"
                sub_image_path = os.path.join(subtest_path, category, im_name)
                noisy_img.save(sub_image_path)

# %%
