import os
import shutil

# label_dir = '../data/biology/labels_no_chinese/'
# image_dir = '../dataset/biology/biology_formula_images_grey/'
# output_dir ='../data/biology/formula_images/'

label_dir = '../data/maths/labels_no_chinese/'
image_dir = '../dataset/maths/math_formula_images_grey/'
output_dir = '../data/maths/formula_images/'

label_name_list = os.listdir(label_dir)

for i in range(len(label_name_list)):
    label_name_list[i] = label_name_list[i][:-4]

# print(label_list)

image_name_list = os.listdir(image_dir)

# 将图片对齐latex（input->labels）
for image_name in image_name_list:
    if image_name[:-4] in label_name_list:
        print(image_name)
        # 复制文件内容（不包含元数据）从src到dst
        shutil.copy(image_dir + image_name, output_dir + image_name)