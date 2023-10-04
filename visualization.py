import numpy as np
from tqdm import tqdm
from skimage import io

# 可视化 data 中的数据
def visualization(net, data, save_path, device, color_map, size):
    """
    net: 训练好的网络
    data: 需要可视化的数据集
    save_path: 图片保存的位置，包含图片名
    color_map: 可视化用到的颜色，白色可能被用来可视化没有标签的数据，请勿使用
    size: 图片的尺寸 Huston: (349, 1905) Trento: (166, 600) 
    """
    net.eval()
    h, w = size[:]
    pred = -np.ones((h, w))
    for hsi, sar, i, j in tqdm(data):
        hsi = hsi.to(device)
        sar = sar.to(device)
        output = net(hsi, sar)
        output = np.argmax(output.detach().cpu().numpy(), axis=1)
        idx = 0
        for x, y in zip(i, j):
            pred[x, y] = output[idx]
            idx += 1
    res = np.zeros((h, w, 3), dtype=np.uint8)
    pos = pred > -1
    for i in range(h):
        for j in range(w):
            if pos[i, j]:
                res[i, j] = color_map[int(pred[i, j])]
            else:
                res[i, j] = [0, 0, 0]
    io.imsave(save_path, res)

# 可视化 Houston2013 数据集
def visHouston2013(net, data, save_path, device):
    """
    net: 训练好的网络
    data: 需要可视化的数据集
    save_path: 图片保存的位置，包含图片名
    """
    # Houston2013 color map
    houston2013_color_map = [[0, 0, 131], [0, 0, 203], [0, 19, 255], [0, 91, 255], [0, 167, 255], [0, 239, 255], [55, 255, 199], [131, 255, 123], [203, 255, 51], [255, 235, 0], [255, 163, 0], [255, 87, 0], [255, 15, 0], [199, 0, 0], [127, 0, 0]]
    # Houston2013 尺寸
    houston2013_size = [349, 1905]
    print("Houston2013 Start!")
    visualization(net, data, save_path, device, houston2013_color_map, houston2013_size)
    print("Visualization Success!")

# 可视化 Houston2018 数据集
def visHouston2018(net, data, save_path, device):
    # Houston2018 color map
    houston2018_color_map = [[50, 205, 51], [173, 255, 48], [0, 128, 129], [34, 139, 34], [46, 79, 78], [139, 69, 18], [0, 255, 255], [255, 255, 255], [211, 211, 211], [254, 0, 0], [169, 169, 169], [105, 105, 105], [139, 0, 1], [200, 100, 0], [254, 165, 0], [255, 255, 0], [218, 165, 33], [255, 0, 254], [0, 0, 254], [63, 224, 208]]
    # Houston2018 尺寸
    houston2018_size = [1202, 4768]
    print("Houston2018 Start!")
    visualization(net, data, save_path, device, houston2018_color_map, houston2018_size)
    print("Visualization Success!")

# 可视化 Trento 数据集
def visTrento(net, data, save_path, device):
    # Trento color map
    trento_color_map = [[0, 47, 255], [0, 223, 255], [143, 255, 111], [255, 207, 0], [255, 31, 0], [127, 0, 0]]
    # Trento 尺寸
    trento_size = [166, 600]
    print("Trento Start!")
    visualization(net, data, save_path, device, trento_color_map, trento_size)
    print("Visualization Success!")

# 可视化 Berlin 数据集
def visBerlin(net, data, save_path, device):
    # Berlin color map
    berlin_color_map = [[26, 163, 25], [216, 216, 216], [216, 89, 89], [0, 204, 51], [204, 153, 52], [244, 231, 1], [204, 102, 204], [0, 53, 255]]
    # Berlin 尺寸
    berlin_size = [1723, 476]
    print("Berlin Start!")
    visualization(net, data, save_path, device, berlin_color_map, berlin_size)
    print("Visualization Success!")

# 可视化 Augsburg 数据集
def visAugsburg(net, data, save_path, device):
    # Augsburg color map
    augsburg_color_map = [[26, 163, 25], [216, 216, 216], [216, 89, 89], [0, 204, 51], [244, 231, 1], [204, 102, 204], [0, 53, 255]]
    # Augsburg 尺寸
    augsburg_size = [332, 485]
    print("Augsburg Start!")
    visualization(net, data, save_path, device, augsburg_color_map, augsburg_size)
    print("Visualization Success!")

def getMyVisualization(datasetType, net, data, save_path, device):
    if(datasetType == 0):
        visHouston2013(net, data, save_path, device)
    elif(datasetType == 1):
        visHouston2018(net, data, save_path, device)
    elif(datasetType == 2):    
        visTrento(net, data, save_path, device)
    elif(datasetType == 3):    
        visBerlin(net, data, save_path, device)
    elif(datasetType == 4):    
        visAugsburg(net, data, save_path, device)