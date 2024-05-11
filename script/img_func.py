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
    odd_num = 0
    label_imgs = []
    merged_images = np.zeros((2 * img_shape[0], 2 * img_shape[1]))  # 创建一个空的2x2的图像矩阵
    for i in range(4):
        idx = np.random.randint(0, high=data.shape[0])  # 修改此处为 data.shape[0]
        label = labels[idx]
        label_imgs.append(label)
        img = np.reshape(data[idx, :], img_shape)

        if (label % 2) == 0:
            odd_num += 1

        row = i // 2
        col = i % 2
        merged_images[row * img_shape[0]:(row + 1) * img_shape[0], col * img_shape[1]:(col + 1) * img_shape[1]] = img
    if odd_num > 2:
        final_label = "odd"
    elif odd_num == 2:
        final_label = "same"
    else:
        final_label = "even"

    gaussian_noise = np.random.normal(0, noise, merged_images.shape)
    final_img = merged_images + gaussian_noise
    return final_img, final_label

# %%

train_path = "../data/train"
noise=3
for i in range(6000):
    final_img, final_label = generate_imgs(data_train, labels_train, noise, img_shape=(28, 28))
    im = Image.fromarray(final_img.astype(np.uint8))
    if im.mode == "F":
        im = im.convert("RGB")
    if final_label == "even":
        tmp_path = train_path + "/even"
        im_name = str(noise) + '_'+ str(i) + '.jpg'
        im.save(tmp_path + "/" + im_name)
    elif final_label == "odd":
        tmp_path = train_path + "/odd"
        im_name = str(noise) + '_'+ str(i) + '.jpg'
        im.save(tmp_path + "/" + im_name)
    else:
        tmp_path = train_path + "/same"
        im_name = str(noise) + '_'+ str(i) + '.jpg'
        im.save(tmp_path + "/" + im_name)



val_path = "../data/validation"
noise_val=2
for i in range(4500):
    final_img, final_label = generate_imgs(data_val, labels_val, noise_val, img_shape=(28, 28))
    im = Image.fromarray(final_img.astype(np.uint8))
    if im.mode == "F":
        im = im.convert("RGB")
    if final_label == "even":
        tmp_path = val_path + "/even"
        im_name = str(noise_val) + '_'+ str(i) + '.jpg'
        im.save(tmp_path + "/" + im_name)
    elif final_label == "odd":
        tmp_path = val_path + "/odd"
        im_name = str(noise_val) + '_'+ str(i) + '.jpg'
        im.save(tmp_path + "/" + im_name)
    else:
        pass
        tmp_path = val_path + "/same"
        im_name = str(noise_val) + '_'+ str(i) + '.jpg'
        im.save(tmp_path + "/" + im_name)



test_path = "../data/test"
noise_test=1.5
for i in range(3000):
    final_img, final_label = generate_imgs(data_test, labels_test, noise_test, img_shape=(28, 28))
    im = Image.fromarray(final_img.astype(np.uint8))
    if im.mode == "F":
        im = im.convert("RGB")
    if final_label == "even":
        tmp_path = test_path + "/even"
        im_name = str(noise_test) + '_'+ str(i) + '.jpg'
        im.save(tmp_path + "/" + im_name)
    elif final_label == "odd":
        tmp_path = test_path + "/odd"
        im_name = str(noise_test) + '_'+ str(i) + '.jpg'
        im.save(tmp_path + "/" + im_name)
    else:
        tmp_path = test_path + "/same"
        im_name = str(noise_test) + '_'+ str(i) + '.jpg'
        im.save(tmp_path + "/" + im_name)
# %%

