import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def render_img(arr):
    arr = (arr * 0.5) + 0.5
    arr = np.uint8(arr * 255)
    # img = Image.fromarray(arr, mode='L').resize((1, 224))
    img = Image.fromarray(arr, mode='L')
    # plt.imshow(img,cmap='gray')
    # plt.show()
    img.show()

if __name__ == '__main__':

    dataset_path = "database/IITDdata.npy"
    raw_data = np.load(dataset_path, allow_pickle=True).copy()
    # print("raw_data.shape:", raw_data.shape)
    # print("raw_data[1]:", type(raw_data[1]))
    # print("raw_data[1]大小:", len(raw_data[1]))
    # print("raw_data[1][0].shape:", raw_data[1][0].shape)
    print("raw_data[1][1]:",np.array(raw_data[0][1]).shape)

    # num_classes = raw_data.shape[0]
    # x2_data = list(raw_data[3])
    # np.random.shuffle(x2_data)

    raw_inp = np.array(raw_data[101][2])
    inp = render_img(raw_inp)
