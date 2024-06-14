import math
import torch.nn.functional as F
import socket
import struct
import pickle
import sys
import pandas as pd
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter1d  
from threading import Thread
from threading import Lock
from torch.utils.data import Dataset, DataLoader
import time
import copy
from tqdm import tqdm
import numpy as np
import random
from args import args_parser
import sys
import matplotlib.pyplot as plt
sys.argv=['']
del sys
args = args_parser()    
print(args) 
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
lr = args.lr
bs = args.bs
choosen_number =  args.choosen_number
niid = args.niid
root_path = '../../models/cifar10_data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu ="cpu"
client_ip = ["**"]*users
client_port = ["***"]*users
client_state = ["preparing"]*users
table = [[i, "**.**", "**", "preparing"] for i in range(users)]
df = pd.DataFrame(table)
# print(table)
df.to_csv(f'./data/table.csv',header=["Vehicle","ip","port","status"],index=False)

plt.close("all")
plt.figure()
plt.plot(range(1,1+len([])), [],  linestyle='-', label=f'train acc', linewidth=1.5,marker='.',markevery=15,markersize=8)
plt.legend()
plt.grid(True)
plt.xlabel('Number of Global Iterations')  # 设置x,y轴标记
plt.ylabel('Training Acc')
plt.title('Training Acc')
plt.savefig(f'./data/train_acc.png')
plt.close()

plt.close("all")
plt.figure()
plt.plot(range(1,1+len([])), [],  linestyle='-', label=f'test acc', linewidth=1.5,marker='.',markevery=15,markersize=8)
plt.legend()
plt.grid(True)
plt.xlabel('Number of Global Iterations')  # 设置x,y轴标记
plt.ylabel('Testing Acc')
plt.title('Testing Acc')
plt.savefig(f'./data/test_acc.png')
plt.close()

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

    
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

class ResNet18_client_side(nn.Module):
    def __init__(self, cut_layer, block, layers):
        super(ResNet18_client_side, self).__init__()
        self.inplanes = 16
        self.layer1 = nn.Sequential (
                nn.Conv2d(3, 16, kernel_size = 3, stride = 2, padding = 3, bias = False),
                nn.BatchNorm2d(16),
                nn.ReLU (inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding =1),
            )
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

        if cut_layer < 9:
            self.layer9 = None
        if cut_layer < 8:
            self.layer8 = None
        if cut_layer < 7:
            self.layer7 = None
        if cut_layer < 6:
            self.layer6 = None
        if cut_layer < 5:
            self.layer5 = None
        if cut_layer < 4:
            self.layer4 = None
        if cut_layer < 3:
            self.layer3 = None
        if cut_layer < 2:
            self.layer2 = None
   

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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

         
    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        # out1 = self.layer2(resudial1)
        # out1 = out1 + resudial1 # adding the resudial inputs -- downsampling not required in this layer
        # resudial2 = F.relu(out1)

        if self.layer2 == None:
            x2 = x1
        else:
            x2 = self.layer2(x1)

        if self.layer3 == None:
            x3 = x2
        else:
            x3 = self.layer3(x2)

        if self.layer4 == None:
            x4 = x3
        else:
            x4 = self.layer4(x3)
        
        if self.layer5 == None:
            x5 = x4
        else:
            x5 = self.layer5(x4)

        if self.layer6 == None:
            x6 = x5
        else:
            x6 = self.layer6(x5)
        
        if self.layer7 == None:
            x7 = x6
        else:
            x7 = self.layer7(x6)
        
        if self.layer8 == None:
            x8 = x7
        else:
            x8 = self.layer8(x7)

        return x8

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

class ResNet18_server_side(nn.Module):
    def __init__(self, cut_layer, block, num_layers, classes):
        super(ResNet18_server_side, self).__init__()
        self.inplanes = 16
        # self.layer2, self.layer3 = self._make_layer(block, 64, num_layers[0])
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
        self.layer4, self.layer5 = self._make_layer(block, 32, num_layers[1], stride=2)
        self.layer6, self.layer7 = self._make_layer(block, 64, num_layers[2], stride=2)
        self.layer8, self.layer9  = self._make_layer(block, 128, num_layers[3], stride=2)
        self. averagePool = nn.AvgPool2d(2)
        self.fc = nn.Linear(128 * block.expansion, classes)
        
        if cut_layer >= 8:
            self.layer8 = None
        if cut_layer >= 7:
            self.layer7 = None
        if cut_layer >= 6:
            self.layer6 = None
        if cut_layer >= 5:
            self.layer5 = None
        if cut_layer >= 4:
            self.layer4 = None
        if cut_layer >= 3:
            self.layer3 = None
        if cut_layer >= 2:
            self.layer2 = None
        if cut_layer >= 1:
            self.layer1 = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or planes != self.inplanes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride = stride, downsample = downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(layers[0]),nn.Sequential(layers[1])

    def forward(self, x):

        if self.layer2 == None:
            x2 = x
        else:
            x2 = self.layer2(x)

        if self.layer3 == None:   
            x3 = x
        else:
            x3 = self.layer3(x2)

        if self.layer4 == None:
            x4 = x3
        else:
            x4 = self.layer4(x3)


        if self.layer5 == None: 
            x5 = x
        else:
            x5 = self.layer5(x4)

        if self.layer6 == None:
            x6 = x
        else:
            x6 = self.layer6(x5)
        
        if self.layer7 == None:
            x7 = x
        else:
            x7 = self.layer7(x6)
       
        if self.layer8 == None:
            x8 = x
        else:
            x8 = self.layer8(x7)
      
        x9= self.layer9(x8)
      
        
        x10 = F.avg_pool2d(x9, 2)
        x11 = x10.view(x10.size(0), -1) 
        y_hat =self.fc(x11)

        return y_hat

def resnet18():
    return ResNet18(SL_Baseblock, [2, 2, 2, 2], 10)
def resnet18_client(cut_layer):
    return ResNet18_client_side(cut_layer, SL_Baseblock, [2, 2, 2, 2])
def resnet18_server(cut_layer):
    return ResNet18_server_side(cut_layer, SL_Baseblock,[2,2,2,2],10)

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
    # agg_model = copy.deepcopy(w[0])    
    # agg_state_dict = agg_model.state_dict()
    # for key in agg_state_dict:
    #     agg_state_dict[key].zero_()

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
    global table
    thrs = []
    for i in range(num_user):
        conn, addr = s.accept()
        # print('Conntected with', addr)
        # append client socket on list
        clientsoclist[i] = conn
        args = (i, num_user, conn)
        thread = Thread(target=func, args=args)
        thrs.append(thread)
        thread.start()
        table[i][1] = addr[0]
        table[i][2] = addr[1]
        # print(table)
        df = pd.DataFrame(table)
        df.to_csv(f'./data/table.csv',header=["Vehicle","ip","port","status"],index=False)
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
    # print("*******************conn-userid",conn,userid)
    datasize = send_msg(conn, msg)    #send epoch
    total_sendsize_list.append(datasize)
    client_sendsize_list[userid].append(datasize)

    ini_msg, datasize = recv_msg(conn)    # get total_batch of train dataset
    total_receivesize_list.append(datasize)
    client_receivesize_list[userid].append(datasize)
    
    train_dataset_size = int(ini_msg['train_dataset_size'])
    with lock:
        datasetsize[userid] = train_dataset_size
        uploadspeed = float(ini_msg['speed'])
        weight_count += 1
        if uploadspeed > 15:
            cutlayers[userid] = 2
        elif uploadspeed > 10:
            cutlayers[userid] = 4
        elif uploadspeed > 5:
            cutlayers[userid] = 6
    train(userid, train_dataset_size, num_users, conn)

def train(userid, train_dataset_size, num_users, client_conn):
    global weights_list
    global whole_model_weights
    global weight_count
    global res_net
    global val_acc
    global res_net_client_list
    global res_net_server_list 
    global cutlayers
    global train_loss_list
    global test_acc_list
    global train_acc_list
    global train_loss_round
    global train_acc_round
    global table

    for r in range(rounds):
        random.seed(random.randint(3333,66666))
        with lock:  
            if weight_count == num_users:
                for i, conn in enumerate(clientsoclist):
                    table[i][3]="distribution"
                    df = pd.DataFrame(table)
                    df.to_csv(f'./data/table.csv',header=["Vehicle","ip","port","status"],index=False)
                    choosen_cut_layer = random.randint(2,8) #random selection
                    res_net_whole.load_state_dict(whole_model_weights)
                    res_net_client_new = resnet18_client(choosen_cut_layer).to(cpu)
                    res_net_server_new = resnet18_server(choosen_cut_layer).to(device)
                    client_dict = res_net_client_new.state_dict()
                    server_dict = res_net_server_new.state_dict()
                    # Filter the pre-trained weights to match the keys in the client model
                    pretrained_client_dict = {k: v for k, v in res_net_whole.state_dict().items() if k in client_dict}
                    pretrained_server_dict = {k: v for k, v in res_net_whole.state_dict().items() if k in server_dict}
                    client_dict.update(pretrained_client_dict)
                    server_dict.update(pretrained_server_dict)
                    # Load the updated state dictionary into the client model
                    res_net_client_new.load_state_dict(client_dict)
                    res_net_client_new.to(cpu)
                    res_net_server_new.load_state_dict(server_dict)
                    res_net_server_list[i]=res_net_server_new
                    global_weights_cpu = {k: v.cpu() for k, v in res_net_client_new.state_dict().items()}
                    model_distribute_msg = {
                        'cut_layer' : choosen_cut_layer,
                        'global_weights' : global_weights_cpu#这里要根据cutlayer改globalweights    
                    }
                    datasize = send_msg(conn, model_distribute_msg)
                    total_sendsize_list.append(datasize)
                    client_sendsize_list[i].append(datasize)
                    train_sendsize_list.append(datasize)
                    weight_count = 0
                    train_loss_round = []
                    train_acc_round = []
        table[userid][3]="training"
        # print(table)
        df = pd.DataFrame(table)
        df.to_csv(f'./data/table.csv',header=["Vehicle","ip","port","status"],index=False)
        optimizer = optim.SGD(res_net_server_list[userid].parameters(), lr=lr, momentum=0.9)
        for local_epoch in range(local_epochs):
            corr_num = 0
            total_num = 0
            train_loss = 0.0
            for i in tqdm(range(train_dataset_size), ncols=100, desc='Round {} Epoch {} Client{} '.format(r+1, local_epoch, userid)):  
                table[userid][3]="training"
                df = pd.DataFrame(table)
                df.to_csv(f'./data/table.csv',header=["Vehicle","ip","port","status"],index=False)
                optimizer.zero_grad()  # initialize all gradients to zero
                msg, datasize = recv_msg(client_conn)  # receive client message from socket
                total_receivesize_list.append(datasize)
                client_receivesize_list[userid].append(datasize)
                train_receivesize_list.append(datasize)
                # print("smashed data size",datasize/(1024*1024),"MB")
                client_output_cpu = msg['client_output']  # client output tensor
                label = msg['label']  # label
                client_output = client_output_cpu.to(device)
                label = label.clone().detach().long().to(device)
                output = res_net_server_list[userid](client_output)  # forward propagation
                loss = criterion(output, label)  # calculates cross-entropy loss
                train_loss += loss.item()
                model_label = output.argmax(dim=1)
                corr = label[label == model_label].size(0)
                corr_num += corr
                total_num += label.size(0)
                loss.backward()  # backward propagation
                msg = client_output_cpu.grad.clone().detach()
                datasize = send_msg(client_conn, msg)
                # print("smashed data 梯度大小",datasize/(1024*1024),"MB")
                total_sendsize_list.append(datasize)
                client_sendsize_list[userid].append(datasize)
                train_sendsize_list.append(datasize)
                optimizer.step()
            train_loss_temp =  train_loss / train_dataset_size
            train_acc_temp = corr_num / total_num * 100
            train_acc_round.append(train_acc_temp)
            train_loss_round.append(train_loss_temp)
            print(f"epochs_{local_epoch}:trainacc{train_acc_temp},trainloss{train_loss_temp}")
          
        msg, datasize = recv_msg(client_conn)
        client_weights = msg['client_weights']
        print("模型大小size",datasize/(1024*1024),"MB")
        uploadspeed = msg['uploadspeed']
        table[userid][3]="aggregation"
        # print(table)
        df = pd.DataFrame(table)
        df.to_csv(f'./data/table.csv',header=["Vehicle","ip","port","status"],index=False)
        total_receivesize_list.append(datasize)
        client_receivesize_list[userid].append(datasize)
        train_receivesize_list.append(datasize)   
        client_model_dict = {key:client_weights[key].cuda() for key in client_weights} #将client的cpu转为cuda类型的
        server_model_dict = copy.deepcopy(res_net_server_list[userid]).state_dict()
        ###############################
        whole_model_temp = copy.deepcopy(client_model_dict)
        whole_model_temp.update(server_model_dict)  #
        weights_list[userid] = whole_model_temp
        # print(userid,cutlayers[userid],whole_model_temp.keys())
        print("User" + str(userid) + "'s Epoch " + str(r + 1) +  " is done")
        with lock:
            weight_count += 1
            if weight_count == num_users:
                #average
                print("aggregation")
                whole_model_weights = average_weights(weights_list, datasetsize)
                train_loss = sum(train_loss_round)/len(train_loss_round)
                train_acc = sum(train_acc_round)/len(train_acc_round)
                print("rounds{} trainloss{} trainacc".format(r,train_loss,train_acc))
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                train_loss_round = []
                train_acc_round = []
                # test acc
                res_net_whole.load_state_dict(whole_model_weights)
                with torch.no_grad():
                    corr_num = 0
                    total_num = 0
                    val_loss = 0.0
                    for j, val in enumerate(testloader):
                        val_x, val_label = val
                        val_x = val_x.to(device)
                        val_label = val_label.clone().detach().long().to(device)
    
                        val_output = res_net_whole(val_x)
                        loss = criterion(val_output, val_label)
                        val_loss += loss.item()
                        model_label = val_output.argmax(dim=1)
                        corr = val_label[val_label == model_label].size(0)
                        corr_num += corr
                        total_num += val_label.size(0)
                        accuracy = corr_num / total_num * 100
                        test_loss = val_loss / len(testloader)
                    test_acc_list.append(accuracy)
                    # wandb.log({'train acc': train_acc,'train loss': train_loss, 'test accuracy': accuracy})
                    print("test_acc: {:.2f}%, test_loss: {:.4f}".format(accuracy, test_loss))
                    propose = [train_loss_list, train_acc_list, test_acc_list]
                    df = pd.DataFrame(propose)
                    df.to_csv(f'./res/propose_{lr}_{bs}_{local_epochs}_{num_users}_{choosen_number}_{niid}_{rounds}.csv')
                    

                    plt.close("all")
                    plt.figure()
                    plt.plot(range(1,1+len(train_acc_list)), gaussian_filter1d(train_acc_list, sigma=1),  linestyle='-', label=f'train acc', linewidth=1.5,marker='.',markevery=15,markersize=8)
                    plt.legend()
                    plt.grid(True)
                    plt.xlabel('Number of Global Iterations')  # 设置x,y轴标记
                    plt.ylabel('Training Acc')
                    plt.title('Training Acc')
                    plt.savefig(f'./data/train_acc.png')
                    plt.close()

                    plt.close("all")
                    plt.figure()
                    plt.plot(range(1,1+len(test_acc_list)), gaussian_filter1d(test_acc_list, sigma=1),  linestyle='-', label=f'test acc', linewidth=1.5,marker='.',markevery=15,markersize=8)
                    plt.legend()
                    plt.grid(True)
                    plt.xlabel('Number of Global Iterations')  # 设置x,y轴标记
                    plt.ylabel('Testing Acc')
                    plt.title('Testing Acc')
                    plt.savefig(f'./data/test_acc.png')
                    plt.close()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10 (root=root_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)
clientsoclist = [0]*users
start_time = 0
weight_count = 0
res_net_whole = resnet18().to(device)
whole_model_weights = copy.deepcopy(res_net_whole.state_dict())
datasetsize = [0]*users
weights_list = [0]*users
lock = Lock()

total_sendsize_list = []
total_receivesize_list = []

client_sendsize_list = [[] for i in range(users)]
client_receivesize_list = [[] for i in range(users)]

train_sendsize_list = [] 
train_receivesize_list = []
criterion = nn.CrossEntropyLoss()
cutlayers = [2]*users
train_loss_list = []
test_acc_list =[]
train_acc_list = []

res_net_client_new = resnet18_client(2).to(cpu)
res_net_server_new = resnet18_server(2).to(device)
client_dict = res_net_client_new.state_dict()
server_dict = res_net_server_new.state_dict()
res_net_server_list = [res_net_server_new for i in range(users)]

s = socket.socket()
host = args.host_ip
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
    total_size += size
size_list.append(total_size)
print("total_sendsize size: {} bytes".format(total_size))
print("number of total_send: ", len(total_sendsize_list))
print('\n')

print('---total_receivesize_list---')
total_size = 0
for size in total_receivesize_list:
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
        total_size += size
    send_size.append(total_size)
    print("total client_sendsizes(user{}): {} bytes".format(i, total_size))
    print("number of client_send(user{}): ".format(i), len(client_sendsize_list[i]))
    print('\n')

    print('---client_receivesize_list(user{})---'.format(i))
    total_size = 0
    for size in client_receivesize_list[i]:
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
    total_size += size
size_list.append(total_size)
print("total train_sendsizes: {} bytes".format(total_size))
print("number of train_send: ", len(train_sendsize_list) )
print('\n')

print('---train_receivesize_list---')
total_size = 0
for size in train_receivesize_list:
    total_size += size
size_list.append(total_size)
print("total train_receivesizes: {} bytes".format(total_size))
print("number of train_receive: ", len(train_receivesize_list) )
print('\n')

print(size_list)

df = pd.DataFrame({'total_sendsize': size_list[0],"total_receivesize":size_list[1],"client_sendsize_list":size_list[2],"client_receivesize_list":size_list[3],"train_send":size_list[4],"train_recv":size_list[5],"time":end_time - start_time})     
df.to_csv(f'./res/asfvcommunication_{args.lr}_{args.bs}_{args.local_ep}_{args.num_users}_{args.choosen_number}_{args.niid}_{args.rounds}.csv')  