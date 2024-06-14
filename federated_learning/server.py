import os
import h5py
import socket
import struct
import pickle
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import csv
import copy
from threading import Thread
from threading import Lock
from torch.utils.data import Dataset, DataLoader
import time
import math
from tqdm import tqdm
import numpy as np
import random
from torch.autograd import Variable
import torch.nn.init as init   
from args import args_parser
import sys
sys.argv=['']
del sys
args = args_parser()     
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
users = args.num_users
num_digits=args.num_digits
rounds = args.rounds
local_epochs = args.local_ep
port = args.port
root_path = '../../models/cifar10_data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu" 
print(device)


class SL_Baseblock(nn.Module):
    expansion = 1
    def __init__(self, input_planes, planes, stride = 1, downsample = None):
        super(SL_Baseblock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride =  stride, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride = 1, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        
    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        
        if self.downsample is not None:
            res =self.downsample(res)
            
        output += res
        output = F.relu(output)
        
        return output       

class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 16
        super(ResNet18, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=3,
                               bias=False),
                               nn.BatchNorm2d(16),
                               nn.ReLU(inplace=True),
                               nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer2 = nn.Sequential  (
                nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(16),
                nn.ReLU (inplace = True),
                nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(16),              
            )
        self.layer3 = nn.Sequential (
                nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(16),
                nn.ReLU (inplace = True),
                nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(16),       
                )   
        # self.layer2, self.layer3 = self._make_layer(block, 64, layers[0])
        self.layer4, self.layer5 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer6, self.layer7 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer8, self.layer9  = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d): # 如果是卷积层
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(layers[0]),nn.Sequential(layers[1])
        # return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.layer1(x))

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18():
    return ResNet18(SL_Baseblock, [2, 2, 2, 2], 10)

def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    l_send = len(msg)
    msg = struct.pack('>I', l_send) + msg
    sock.sendall(msg)
    return l_send

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg =  recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg, msglen

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def average_weights(w, datasize):
    """
    Returns the average of the weights.
    """
        
    for i, data in enumerate(datasize):
        for key in w[i].keys():
            w[i][key] *= (data)
    
    w_avg = copy.deepcopy(w[0])

# when client use only one kinds of device

    # for key in w_avg.keys():
    #     for i in range(1, len(w)):
    #         w_avg[key] += w[i][key]
    #     w_avg[key] = torch.div(w_avg[key], float(sum(datasize)))

# when client use various devices (cpu, gpu) you need to use it instead
#
    for key, val in w_avg.items():
        common_device = val.device
        break
    for key in w_avg.keys():
        for i in range(1, len(w)):
            if common_device == 'cpu':
                w_avg[key] += w[i][key].cpu()
            else:
                w_avg[key] += w[i][key].cuda()
        w_avg[key] = torch.div(w_avg[key], float(sum(datasize)))

    return w_avg

def run_thread(func, num_user):
    global clientsoclist
    global start_time
    thrs = []
    for i in range(num_user):
        conn, addr = s.accept()
        print('Conntected with', addr)
        # append client socket on list
        clientsoclist[i] = conn
        args = (i, num_user, conn)
        thread = Thread(target=func, args=args)
        thrs.append(thread)
        thread.start()
    print("timmer start!")
    start_time = time.time()    # store start time
    for thread in thrs:
        thread.join()
    end_time = time.time()  # store end time
    print("TrainingTime: {} sec".format(end_time - start_time))

def receive(userid, num_users, conn): #thread for receive clients
    global weight_count
    
    global datasetsize

    msg = {
        'rounds': rounds,
        'client_id': userid,
        'local_epoch': local_epochs
    }

    datasize = send_msg(conn, msg)    #send epoch
    total_sendsize_list.append(datasize)
    client_sendsize_list[userid].append(datasize)

    train_dataset_size, datasize = recv_msg(conn)    # get total_batch of train dataset
    total_receivesize_list.append(datasize)
    client_receivesize_list[userid].append(datasize)
    
    
    with lock:
        datasetsize[userid] = train_dataset_size
        weight_count += 1
    train(userid, train_dataset_size, num_users, conn)

def train(userid, train_dataset_size, num_users, client_conn):
    global weights_list
    global global_weights
    global weight_count
    global train_loss_list
    global train_acc_list 
    global test_acc_list
    global train_loss_round
    global train_acc_round
    
    for r in range(rounds):
        with lock:
            if weight_count == num_users:
                global_weights_cpu = {key:global_weights[key].cpu() for key in global_weights} 
                for i, conn in enumerate(clientsoclist):
                    datasize = send_msg(conn, global_weights_cpu)
                    total_sendsize_list.append(datasize)
                    client_sendsize_list[i].append(datasize)
                    train_sendsize_list.append(datasize)
                    train_loss_round=[]
                    train_acc_round=[]
                    weight_count = 0
        
        msg, datasize = recv_msg(client_conn)
        client_weights = msg['weights']
        train_loss_round.append(msg['train_loss'])
        train_acc_round.append(msg["train_acc"])
        total_receivesize_list.append(datasize)
        client_receivesize_list[userid].append(datasize)
        train_receivesize_list.append(datasize)
        client_model_dict = {key:client_weights[key].cuda() for key in client_weights}
        weights_list[userid] = client_model_dict
        print("User" + str(userid) + "'s Round " + str(r + 1) +  " is done")
        with lock:
            weight_count += 1
            if weight_count == num_users:
                #average
                global_weights = average_weights(weights_list, datasetsize)
                res_net.load_state_dict(global_weights)
                with torch.no_grad():
                    corr_num = 0
                    total_num = 0
                    val_loss = 0.0
                    for j, val in enumerate(testloader):
                        val_x, val_label = val
                        val_x = val_x.to(device)
                        val_label = val_label.clone().detach().long().to(device)
                        val_output = res_net(val_x)
                        loss = criterion(val_output, val_label)
                        val_loss += loss.item()
                        model_label = val_output.argmax(dim=1)
                        corr = val_label[val_label == model_label].size(0)
                        corr_num += corr
                        total_num += val_label.size(0)
                        accuracy = corr_num / total_num * 100
                        test_loss = val_loss / len(testloader)
                    test_acc_list.append(accuracy)
                    print("test_acc: {:.2f}%, test_loss: {:.4f}".format(accuracy, test_loss))
                    train_loss = sum(train_loss_round)/len(train_loss_round)
                    train_loss_list.append(train_loss)
                    train_acc = sum(train_acc_round)/len(train_acc_round)
                    train_acc_list.append(train_acc)
                    propose = [train_loss_list, train_acc_list, test_acc_list]
                    df = pd.DataFrame(propose)
                    df.to_csv(f'./federatedlearning_{args.lr}_{args.bs}_{args.local_ep}_{args.num_users}_{args.choosen_number}_{args.niid}_{args.rounds}.csv')  

def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))  
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

res_net = resnet18()
res_net.to(device)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10 (root=root_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)
clientsoclist = [0]*users
start_time = 0
weight_count = 0
global_weights = copy.deepcopy(res_net.state_dict())
datasetsize = [0]*users
weights_list = [0]*users
lock = Lock()
total_sendsize_list = []
total_receivesize_list = []
client_sendsize_list = [[] for i in range(users)]
client_receivesize_list = [[] for i in range(users)]
train_sendsize_list = [] 
train_receivesize_list = []
lr = args.lr
criterion = nn.CrossEntropyLoss()
test_acc_list = []
train_loss_list = []
train_acc_list = []
global_weights = copy.deepcopy(res_net.state_dict())
global_round = []
# get server ip
host = get_host_ip()
print(host)
s = socket.socket()
print(port)
s.bind((host, port))
s.listen(5)

run_thread(receive, users)

end_time = time.time()  # store end time
print("TrainingTime: {} sec".format(end_time - start_time))

size_list = []
print('\n')
print('---total_sendsize_list---')
total_size = 0
for size in total_sendsize_list:
#     print(size)
    total_size += size
size_list.append(total_size)
print("total_sendsize size: {} bytes".format(total_size))
print("number of total_send: ", len(total_sendsize_list))
print('\n')

print('---total_receivesize_list---')
total_size = 0
for size in total_receivesize_list:
#     print(size)
    total_size += size
size_list.append(total_size)
print("total receive sizes: {} bytes".format(total_size) )
print("number of total receive: ", len(total_receivesize_list) )
print('\n')

send_size = []
recv_size = []
for i in range(users):
    print('---client_sendsize_list(user{})---'.format(i))
    total_size = 0
    for size in client_sendsize_list[i]:
#         print(size)
        total_size += size
    send_size.append(total_size)
    print("total client_sendsizes(user{}): {} bytes".format(i, total_size))
    print("number of client_send(user{}): ".format(i), len(client_sendsize_list[i]))
    print('\n')

    print('---client_receivesize_list(user{})---'.format(i))
    total_size = 0
    for size in client_receivesize_list[i]:
#         print(size)
        total_size += size
    recv_size.append(total_size)
    print("total client_receive sizes(user{}): {} bytes".format(i, total_size))
    print("number of client_send(user{}): ".format(i), len(client_receivesize_list[i]))
    print('\n')
size_list.append(send_size)
size_list.append(recv_size)
print('---train_sendsize_list---')
total_size = 0
for size in train_sendsize_list:
#     print(size)
    total_size += size
size_list.append(total_size)
print("total train_sendsizes: {} bytes".format(total_size))
print("number of train_send: ", len(train_sendsize_list) )
print('\n')

print('---train_receivesize_list---')
total_size = 0
for size in train_receivesize_list:
#     print(size)
    total_size += size
size_list.append(total_size)
print("total train_receivesizes: {} bytes".format(total_size))
print("number of train_receive: ", len(train_receivesize_list) )
print('\n')
print(size_list)




