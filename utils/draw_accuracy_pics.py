# -*- coding:utf-8 -*-

import os
import re

import numpy as np
from matplotlib import pyplot as plt

mod = ['rnn_efficient', 'rnn_efficient_attention', 'transformer', 'rnn_mobilenet']
math_path = ''
bio_path = ''


def draw_acc_pics():
    for m in mod:
        math_path = f'../train_history/{m}_math.txt'
        bio_path = f'../train_history/{m}_biology.txt'

        math_acc_list = []
        math_loss_list = []

        biology_acc_list = []
        biology_loss_list = []

        with open(math_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            pat_1 = re.compile(r'.*val_loss: (0.\d{4}).*val_accuracy: (0.\d{4})$')
            for line in lines:
                # line = f.readline()
                if re.findall(pat_1, line):
                    val_loss, val_acc = re.search(pat_1, line).groups()
                    math_acc_list.append(float(val_acc))
                    math_loss_list.append(float(val_loss))

            with open(bio_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                pat_1 = re.compile(r'.*val_loss: (0.\d{4}).*val_accuracy: (0.\d{4})$')
                for line in lines:
                    # line = f.readline()
                    if re.findall(pat_1, line):
                        val_loss, val_acc = re.search(pat_1, line).groups()
                        biology_acc_list.append(float(val_acc))
                        biology_loss_list.append(float(val_loss))

            epochs = len(math_acc_list)
            x = np.linspace(1, epochs, epochs)
            # plt.plot(x, train_acc_list,label='train accuracy')
            plt.plot(x, math_acc_list, label='accuracy of maths')
            plt.plot(x, biology_acc_list, label='accuracy of biology')
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            plt.title(f'{m}')
            plt.legend()
            plt.savefig(f'../train_history/pngs/{m}_acc.png', dpi=500)
            plt.show()

            plt.plot(x, math_loss_list, label='loss of maths')
            plt.plot(x, biology_loss_list, label='loss of biology')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.title(f'{m}')
            plt.legend(loc=1)
            plt.savefig(f'../train_history/pngs/{m}_loss.png', dpi=500)
            plt.show()


if __name__ == '__main__':
    draw_acc_pics()
