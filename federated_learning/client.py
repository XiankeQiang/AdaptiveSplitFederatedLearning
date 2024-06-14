import os
import h5py
from torch.autograd import Variable
import torch.nn.init as init
import socket
import struct
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import random
from torch.utils.data import Subset
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
lr = args.lr
host = args.host_ip
port = args.port
max_recv = 100000
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
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg =  recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# train_dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # converting images to tensor
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)) 
    # if the image dataset is black and white image, there can be just one number. 
])
train_idxs =[[] for i in range(10)]
trainset = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True, transform=transform_train)

#resnet model
res_net = resnet18()
res_net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(res_net.parameters(), lr=lr, momentum=0.9)

# start connection!
s = socket.socket()
s.connect((host, port))
start_time = time.time()   
print("timmer start!")
# receive parameters from server
msg = recv_msg(s)
rounds = msg['rounds'] 
client_id = msg['client_id']
local_epochs = msg['local_epoch']

if args.niid == 1:
    print("niid")

elif args.niid == 0:
    print("iid")
    num_traindata = 50000 // users
    indices = list(range(50000))
    part_tr = indices[num_traindata * client_id : num_traindata * (client_id + 1)]
trainset_sub = Subset(trainset,part_tr)
train_loader = torch.utils.data.DataLoader(trainset_sub, batch_size=args.bs, shuffle=True, num_workers=2)
train_total_batch = len(train_loader)
send_msg(s, len(train_loader))

for r in range(rounds):  # loop over the dataset multiple times
    weights = recv_msg(s)
    res_net.load_state_dict(weights)
    res_net.eval()
    train_loss_list = []
    train_acc_list = []
    for local_epoch in range(local_epochs):
        corr_num = 0
        total_num = 0
        train_loss = 0.0
        for i, data in enumerate(tqdm(train_loader, ncols=100, desc='Round '+str(r+1)+'_'+str(local_epoch+1))):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.clone().detach().long().to(device)
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = res_net(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            model_label = outputs.argmax(dim=1)
            corr = labels[labels == model_label].size(0)
            corr_num += corr
            total_num += labels.size(0)
            loss.backward()
            optimizer.step()
        train_loss_temp =  train_loss / len(train_loader)
        train_acc_temp = corr_num / total_num * 100
        train_acc_list.append(train_acc_temp)
        train_loss_list.append(train_loss_temp)
        print(f"epochs_{local_epoch}:trainacc{train_acc_temp},trainloss{train_loss_temp}")
    
    msg = {   
        'train_loss': sum(train_loss_list)/len(train_loss_list),
        "train_acc": sum(train_acc_list)/len(train_acc_list),
        'weights': res_net.state_dict()
    }
    print(type(msg))
    send_msg(s, msg)

print('Finished Training')

end_time = time.time()  #store end time
print("Training Time: {} sec".format(end_time - start_time))




