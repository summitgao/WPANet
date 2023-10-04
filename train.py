from net import *
from dataset import *

import time
from sklearn.metrics import accuracy_score

import parameter

parameter._init()

def train(epochs, lr, model, cuda, train_loader, test_loader, out_features, model_savepath, log_path):
    if cuda == 'cuda0':
        device = torch.device("cuda:0")
    if cuda == 'cuda1':
        device = torch.device("cuda:1")
    if model == 'WPANet':
        net = WPANet(out_features).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    max_acc = 0
    sum_time = 0
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    current_time_log = 'start time: {}'.format(current_time)
    getLog(log_path, parameter.get_taskInfo())
    getLog(log_path, '-------------------Started Training-------------------')
    getLog(log_path, current_time_log)
    for epoch in range(epochs):
        since = time.time()
        net.train()
        for i, (hsi, sar, tr_labels) in enumerate(train_loader):
            hsi = hsi.to(device)
            sar = sar.to(device)
            tr_labels = tr_labels.to(device)
            optimizer.zero_grad()
            if model == 'WPANet':
                outputs = net(hsi, sar)
                loss = criterion(outputs, tr_labels)
            loss.backward()
            optimizer.step()
        net.eval()
        count = 0
        for hsi, sar, gtlabels in test_loader:
            hsi = hsi.to(device)
            sar = sar.to(device)
            if model == 'WPANet':
                outputs = net(hsi, sar)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test =  outputs
                gty = gtlabels
                count = 1
            else:
                y_pred_test = np.concatenate( (y_pred_test, outputs) )
                gty = np.concatenate( (gty, gtlabels) )
        acc1 = accuracy_score(gty, y_pred_test)
        if acc1 > max_acc:
            torch.save(net, model_savepath)
            max_acc = acc1
        time_elapsed = time.time() - since
        sum_time += time_elapsed
        rest_time = (sum_time / (epoch + 1)) * (epochs - epoch - 1)
        currentTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        log = currentTime + ' [Epoch: %d] [%.0fs, %.0fh %.0fm %.0fs] [current loss: %.4f] acc: %.4f' %(epoch + 1, time_elapsed, (rest_time // 60) // 60, (rest_time // 60) % 60, rest_time % 60, loss.item(), acc1)
        print(log)
        getLog(log_path, log)
    print('max_acc: %.4f' %(max_acc))  
    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    finish_time_log = 'finish time: {} '.format(finish_time)
    mac_acc_log = 'max_acc: {} '.format(max_acc)
    getLog(log_path, mac_acc_log)
    getLog(log_path, finish_time_log)
    getLog(log_path, '-------------------Finished Training-------------------')

def getLog(log_path, str):
    with open(log_path, 'a+') as log:
        log.write('{}'.format(str))
        log.write('\n')

def myTrain(datasetType, model):
    channels = parameter.get_value('channels')
    windowSize = parameter.get_value('windowSize')
    out_features = parameter.get_value('out_features')
    cuda = parameter.get_value('cuda')
    lr = parameter.get_value('lr')
    epoch_nums = parameter.get_value('epoch_nums')
    batch_size = parameter.get_value('batch_size')
    num_workers = parameter.get_value('num_workers')
    random_seed = parameter.get_value('random_seed')
    model_savepath = parameter.get_value('model_savepath')
    log_path = parameter.get_value('log_path')
    train_loader, test_loader, trntst_loader, all_loader = getMyData(datasetType, channels, windowSize, batch_size, num_workers)
    set_random_seed(random_seed)
    train(epoch_nums, lr, model, cuda, train_loader, test_loader, out_features[datasetType], model_savepath[datasetType], log_path[datasetType])