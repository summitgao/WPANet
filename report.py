import numpy as np
import time
from operator import truediv
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
import parameter
import warnings

parameter._init()
warnings.filterwarnings("ignore")

# 计算 average accuracy 和 每个类别的准确率
def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

# 生成报告
def getReport(net, test_loader, report_path, class_names, device):
    '''
    net: 训练好的网络
    test_loader: 测试集
    report_path: 报告保存的位置，包含文件名
    class_names: 类别名
    '''
    net.eval()
    count = 0
    for hsi, sar, test_labels in test_loader:
        hsi = hsi.to(device)
        sar = sar.to(device)
        outputs, hsi_feature, hsi_compensation, sar_feature, sar_compensation = net(hsi, sar)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred = outputs
            y_true = test_labels
            count = 1
        else:
            y_pred = np.concatenate((y_pred, outputs))
            y_true = np.concatenate((y_true, test_labels))
    classification = classification_report(
        y_true, y_pred, target_names=class_names, digits=4)
    confusion = confusion_matrix(y_true, y_pred)
    oa = accuracy_score(y_true, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_true, y_pred)
    classification = str(classification)
    confusion = str(confusion)
    oa = oa * 100
    each_acc = each_acc * 100
    aa = aa * 100
    kappa = kappa * 100
    print('{} Overall accuracy (%)'.format(oa))
    print('{} Average accuracy (%)'.format(aa))
    print('{} Kappa accuracy (%)'.format(kappa))
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    current_time_log = 'Report time: {}\n'.format(current_time)
    with open(report_path, 'a+') as report:
        report.write('{}'.format(parameter.get_taskInfo()))
        report.write('\n')
        report.write('{}'.format(current_time_log))
        report.write('\n')
        report.write('{} Overall accuracy (%)'.format(oa))
        report.write('\n')
        report.write('{} Average accuracy (%)'.format(aa))
        report.write('\n')
        report.write('{} Kappa accuracy (%)'.format(kappa))
        report.write('\n\n')
        report.write('{}'.format(classification))
        report.write('\n')
        report.write('{}'.format(confusion))
        report.write('\n')

# 生成 Houston2013 数据集的报告
def getHouston2013Report(net, test_loader, report_path, device):
    '''
    net: 训练好的网络
    test_loader: 测试集
    report_path: 报告保存的位置，包含文件名
    '''
    # Houston2013 数据集的类别名
    houston2013_class_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees', 'Soil', 'Water', 'Residential',
                               'Commercial', 'Road', 'Highway', 'Railway', 'Parking lot 1', 'parking lot 2', 'Tennis court', 'Running track']
    print("Houston2013 Start!")
    getReport(net, test_loader, report_path, houston2013_class_names, device)
    print("Report Success!")

# 生成 Houston2018 数据集的报告
def getHouston2018Report(net, test_loader, report_path, device):
    # Houston2018 数据集的类别名
    houston2018_class_names = ['Healthy grass', 'Stressed grass', 'Artificial turf', 'Evergreen trees', 'Deciduous trees', 'Bare earth', 'Water', 'Residential buildings', 'Non-residential buildings',
                               'Roads', 'Sidewalks', 'Crosswalks', 'Major thoroughfares', 'Highways', 'Railways', 'Paved parking lots', 'Unpaved parking lots', 'Cars', 'Trains', 'Stadium seats']
    print("Houston2018 Start!")
    getReport(net, test_loader, report_path, houston2018_class_names, device)
    print("Report Success!")

# 生成 Trento 数据集的报告
def getTrentoReport(net, test_loader, report_path, device):
    # Trento 数据集的类别名
    trento_class_names = ['Apple trees', 'Buildings', 'Ground', 'Woods', 'Vineyard', 'Roads']
    print("Trento Start!")
    getReport(net, test_loader, report_path, trento_class_names, device)
    print("Report Success!")

# 生成 Berlin 数据集的报告
def getBerlinReport(net, test_loader, report_path, device):
    # Berlin 数据集的类别名
    berlin_class_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Soil', 'Allotment', 'Commercial Area', 'Water']
    print("Berlin Start!")
    getReport(net, test_loader, report_path, berlin_class_names, device)
    print("Report Success!")

# 生成 Augsburg 数据集的报告
def getAugsburgReport(net, test_loader, report_path, device):
    # Augsburg 数据集的类别名
    augsburg_class_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Allotment', 'Commercial Area', 'Water']
    print("Augsburg Start!")
    getReport(net, test_loader, report_path, augsburg_class_names, device)
    print("Report Success!")

def getMyReport(datasetType, net, test_loader, report_path, device):
    if(datasetType == 0):
        getHouston2013Report(net, test_loader, report_path, device)
    elif(datasetType == 1):
        getHouston2018Report(net, test_loader, report_path, device)
    elif(datasetType == 2):    
        getTrentoReport(net, test_loader, report_path, device)
    elif(datasetType == 3):    
        getBerlinReport(net, test_loader, report_path, device)
    elif(datasetType == 4):    
        getAugsburgReport(net, test_loader, report_path, device)
