import numpy as np


# 计算精度和召回率
def compute_f1_score(y, y_hat):
    """
    计算精度和召回率
    :param y:
    :param y_hat:
    :return:
    """
    tp = np.sum((y == 1) & (y_hat == 1))  # 真阳性
    fp = np.sum((y == 0) & (y_hat == 1))  # 假阳性
    fn = np.sum((y == 1) & (y_hat == 0))  # 假阴性

    precision = tp / (tp + fp)  # 精度
    recall = tp / (tp + fn)  # 召回率

    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score


if __name__ == '__main__':
    # 数据集
    y = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # y有20个元素，10个1和10个0
    y_hat = np.array(
        [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0])  # y_hat有20个元素

    print(f"shape of y: {y.shape}; type of y: {type(y)}; \nval of y: {y}")
    print(f"shape of y_hat: {y_hat.shape}; type of y_hat: {type(y_hat)}; \nval of y_hat: {y_hat}")

    score = compute_f1_score(y, y_hat)
    print(f"F1分数是：{score}")
