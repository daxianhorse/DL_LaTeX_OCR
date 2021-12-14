# Created: 210313 14:02
# Last edited: 210421 14:02 

import os
import shutil

# input_label_dir = './data/math_210421/formula_labels/'
# output_label_dir = './data/math_210421/formula_labels_210421/'

# biology
# input_label_dir = '../data/biology/labels_no_error1'
# output_label_dir = '../data/biology/labels_no_error'


# maths
input_label_dir = '../data/maths/labels_no_error1'
output_label_dir = '../data/maths/labels_no_error'


label_name_list = os.listdir(input_label_dir)

# # 筛除多行的label.txt


# def no_multi_lines():
#     for label_name in label_name_list:
#         label_file_name = input_label_dir + label_name
#         with open(label_file_name, 'r', encoding='utf-8') as f1:
#             lines = f1.readlines()
#         # print(lines)
#         if len(lines) > 1:
#             # print(lines[1])
#             # shutil.copy(label_file_name, './error_data/biology/mult-line_label/' + label_name)
#             continue
#         shutil.copy(label_file_name, output_label_dir + label_name)

# # 筛除多行的label.txt end

# 筛除error mathpix


# label_name_list = os.listdir(input_label_dir)
def no_error_data():
    # label_file_name = input_label_dir + '/805_1.txt'
    # with open(label_file_name, 'r',encoding='utf-8') as f:
    #     lines=f.readlines()
    #     print(lines, len(lines))
    for label_name in label_name_list:
        # print(label_name)
        label_file_name = input_label_dir + '/' + label_name
        # label_file_name = input_label_dir + '/805_1.txt'
        # 先执行read，再执行readlines（或反之），readlines返回结果为空
        # 因此要分开执行
        with open(label_file_name, 'r', encoding='utf-8') as f1:
            # content = f1.read()
            lines = f1.readlines()
        # 筛除error mathpix
        # if 'error mathpix' in content:
        #     print(label_name, content)
        #     continue
        # 筛除多行的label.txt
        if len(lines) > 1:
            print(label_name, 'multi_lines')
            continue
        shutil.copy(label_file_name, output_label_dir + '/' + label_name)


# 筛除error mathpix end
if __name__ == '__main__':
    no_error_data()
