def edit_distance(label, dest):
    """
    计算字符串 str1 和 str2 的编辑距离
    :param label
    :param dest
    :return:
    """
    matrix = [[i + j for j in range(len(dest) + 1)] for i in range(len(label) + 1)]

    for i in range(1, len(label) + 1):
        for j in range(1, len(dest) + 1):
            if (label[i - 1] == dest[j - 1]):
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    # return matrix[len(label)][len(dest)]
    return 1. - matrix[len(label)][len(dest)]/len(label)