```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
```


```python
from google.colab import files
uploaded = files.upload()
print(uploaded.keys())

for uploaded_file in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=uploaded_file, length=len(uploaded[uploaded_file])))
```



     <input type="file" id="files-c8d59cc4-ee57-41e0-93b5-edb801543c8a" name="files[]" multiple disabled
        style="border:none" />
     <output id="result-c8d59cc4-ee57-41e0-93b5-edb801543c8a">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script>// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


##**Do NOT touch this cell**


```python
# load csv
data = np.genfromtxt(uploaded_file, delimiter=',')
data = data[1:,1:]
assert data.shape == (9879, 39), "dataset이 다릅니다. 조교에게 문의하세요."

# split train / test
learn_data = data[:8000]  # len: 8000
test_data = data[8000:]   # len: 1879


### Dataset information & Data sample visualization
label = 'blueWins'
input_feature_names = ['blueWardsPlaced','blueWardsDestroyed','blueFirstBlood','blueKills','blueDeaths','blueAssists','blueEliteMonsters','blueDragons',
  'blueHeralds','blueTowersDestroyed','blueTotalGold','blueAvgLevel','blueTotalExperience','blueTotalMinionsKilled','blueTotalJungleMinionsKilled',
  'blueGoldDiff','blueExperienceDiff','blueCSPerMin','blueGoldPerMin','redWardsPlaced','redWardsDestroyed','redFirstBlood','redKills','redDeaths',
  'redAssists','redEliteMonsters','redDragons','redHeralds','redTowersDestroyed','redTotalGold','redAvgLevel','redTotalExperience','redTotalMinionsKilled',
  'redTotalJungleMinionsKilled','redGoldDiff','redExperienceDiff','redCSPerMin','redGoldPerMin']

# sample
sample_idx = np.random.randint(0,8000)
print("Data Sample Visualization")
print("Data index: %d"%sample_idx)
print("label: %d  (%s wins)"%(data[sample_idx,0], 'blue' if data[sample_idx,0] else 'red'))
print("inputs:", end='')
for i in range(len(input_feature_names)):
  print("\t  %s: %.1f"%(input_feature_names[i], data[sample_idx,1+i]), end='')
  if (i % 5 == 0) and (i != 0):
    print("")
```

    Data Sample Visualization
    Data index: 2439
    label: 0  (red wins)
    inputs:	  blueWardsPlaced: 18.0	  blueWardsDestroyed: 6.0	  blueFirstBlood: 0.0	  blueKills: 4.0	  blueDeaths: 6.0	  blueAssists: 3.0
    	  blueEliteMonsters: 0.0	  blueDragons: 0.0	  blueHeralds: 0.0	  blueTowersDestroyed: 0.0	  blueTotalGold: 15967.0
    	  blueAvgLevel: 7.0	  blueTotalExperience: 18406.0	  blueTotalMinionsKilled: 227.0	  blueTotalJungleMinionsKilled: 56.0	  blueGoldDiff: -1413.0
    	  blueExperienceDiff: -1302.0	  blueCSPerMin: 22.7	  blueGoldPerMin: 1596.7	  redWardsPlaced: 18.0	  redWardsDestroyed: 4.0
    	  redFirstBlood: 1.0	  redKills: 6.0	  redDeaths: 4.0	  redAssists: 9.0	  redEliteMonsters: 2.0
    	  redDragons: 1.0	  redHeralds: 1.0	  redTowersDestroyed: 0.0	  redTotalGold: 17380.0	  redAvgLevel: 7.2
    	  redTotalExperience: 19708.0	  redTotalMinionsKilled: 256.0	  redTotalJungleMinionsKilled: 72.0	  redGoldDiff: 1413.0	  redExperienceDiff: 1302.0
    	  redCSPerMin: 25.6	  redGoldPerMin: 1738.0

##**You can change the list of training and validation set, and batch size**


```python
# split train / val 
train_data = learn_data[:7000]
val_data = learn_data[7000:]
```


```python
# Hyperparameters
batch_size = 128
```

##**You can modify** `__getitem__` **function.** 
###Do NOT make changes outside the indicated range.


```python
class custom_dataset(Dataset):
    def __init__(self, data_list = None):
        super(custom_dataset, self).__init__()        
        self.input_list = data_list[:,1:]        
        self.target_list = data_list[:,0]  
        
    def __getitem__(self, i):
        input = self.input_list[i]
        target = self.target_list[i]
        ###############################################################################################

        ## 중복된 정보라고 판단되는 정보값들은 제외
        ## red팀과 blue팀 각각의 수치보다는 상대적인 수치 차이가 중요하므로 퍼스트블러드 외의 다른 값들은 서로 빼줌.

        input_list_trans = self.input_list.T.reshape(38, 1, -1) # 필요한 정보값들만 추려내어 concatenate하기 위해 
                                                                # [38, 1 ,7000]으로 사이즈를 수정하여 인덱싱하기 편하게 만듦
        redidx = 19                                             # red team의 시작 인덱스

        modified_input_list = np.concatenate((input_list_trans[0] - input_list_trans[0 + redidx],     # 와드수
                                              input_list_trans[1] - input_list_trans[1 + redidx],     # 와드 파괴수
                                              input_list_trans[2],                                    # 퍼스트블러드
                                              input_list_trans[3] - input_list_trans[3 + redidx],     # 킬수
                                              input_list_trans[7] - input_list_trans[7 + redidx],     # 용 처치수
                                              input_list_trans[8] - input_list_trans[8 + redidx],     # 전령 처치수
                                              input_list_trans[9] - input_list_trans[9 + redidx],     # 타워 파괴수
                                              input_list_trans[10] - input_list_trans[10 + redidx],   # 총 골드량
                                              input_list_trans[12] - input_list_trans[12 + redidx],   # 총 경험치
                                              input_list_trans[13] - input_list_trans[13 + redidx],   # 총 CS
                                              input_list_trans[14] - input_list_trans[14 + redidx]),  # 정글몹 처치수

                                              axis = 0
                        ).T #[7000, 11]으로 transpose하여 인덱싱 편하게 만들어줌.
                            # 필요한 feature들만 추려내어, concatenate함. concatenate한 결과의 사이즈는 [7000, 11] 
        
        ## [0,1]로 normalize하기 위해 최솟값을 뺀후 최대값으로 나누어줌
        min_features        = np.min(modified_input_list, axis = 0)
        modified_input_list = modified_input_list - min_features    # range: [0, max-min]
        
        max_features        = np.max(modified_input_list, axis = 0)
        normalized_input = modified_input_list / max_features       # range: [0, 1]
        
        ## 데이터 확인해 본 결과, 와드수와 와드 파괴수는 상대적으로 중요도가 떨어지기 때문에 별도로 weight를 줄여줌.
        ## 퍼스트 블러드의 경우 0 혹은 1의 binary 값을 가지는데, 
        ## 1이라는 값은 feature range [0,1]내에서 최대값을 의미하기 때문에 이를 낮춰줌. 
        normalized_input[:, 0] = normalized_input[:, 0] * 0.4
        normalized_input[:, 1] = normalized_input[:, 1] * 0.4
        normalized_input[:, 2] = normalized_input[:, 2] * 0.5
        input = normalized_input[i] # i번째 sample의 input feature를 반환함

        ################################################################################################
        ## 이 아래로 수정하지 말 것
        return input, target

    def __len__(self):
        return self.input_list.shape[0]
```

##**Do NOT touch this cell**


```python
# DataLoader
train_dataset = custom_dataset(data_list = train_data)
val_dataset = custom_dataset(data_list = val_data)
test_dataset = custom_dataset(data_list = test_data)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                            batch_size = batch_size, 
                                            shuffle = True, num_workers = 2)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset, 
                                          batch_size = batch_size, 
                                          shuffle = False, num_workers = 2)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, 
                                          batch_size = 1, 
                                          shuffle = False, num_workers = 2)
```

## **You have to make your own network.**


```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # 11 -> 30 -> 30 -> 1의 3-layer structure를 가지고, activation function으로 ReLU를 사용한 일반적인 모델임.
        # binary classification model이기 때문에 마지막 layer에 sigmoid를 추가하여 [0, 1]사이의 확률값을 추출함. 

        self.layer1 = nn.Linear(11, 30, bias = True)     # 11->30
        self.relu = nn.ReLU(inplace=False)               # Activation function
        self.layer2 = nn.Linear(30,  30, bias = True)    # 30 -> 30
        self.layer3 = nn.Linear(30,  1, bias = True)     # 30 -> 1
        self.sigmoid = nn.Sigmoid()                      # 1  -> output

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(x)
      
        x = self.layer2(x)
        x = self.relu(x)
      
        x = self.layer3(x)
        x = self.sigmoid(x)

        return x

net = MLP()

##
# sigmoid 함수를 모델내에 포함시켰기 때문에, BCEWithLogitsLoss()가 아닌, BCELoss()함수를 사용
criterion = nn.BCELoss()                                                         
# learning rate = 0.1로 init하고, momentum factor=0.9로 update 관성을, weight decay = 5e-4로 L2 penalty를 주어 weight를 학습하도록 설정하였다.
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) 
# weight값이 overshoot하지 않고 최적값으로 수렴하도록 20 epoch마다 0.5씩 learning rate가 감소하도록 설정하였다.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  
      
epochs = 50
```

##**Training / Validation** (Can be modified, but not recommended)


```python
# Training
def train(epoch):
    print('\nTrain:')
    net.train()
    train_loss = 0
    correct, total = 0, 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = inputs.float(), targets.unsqueeze(1).float()#.long()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = outputs > 0.5 
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if ((batch_idx+1) %30 ==1) or batch_idx+ 1 == len(train_loader):
            print('[%3d/%3d] | Loss: %.5f | Acc: %.3f%% (%d/%d)'%(batch_idx+1, len(train_loader), 
                                                                  train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def val(epoch):
    print('\nValidation:')
    net.eval()
    val_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):            
            optimizer.zero_grad()
            inputs, targets = inputs.float(), targets.unsqueeze(1).float()
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            val_loss += loss.item()
            predicted = outputs > 0.5
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if ((batch_idx+1) %30 == 1) or batch_idx+ 1 == len(val_loader):
              print('[%3d/%3d] | Loss: %.5f | Acc: %.3f%% (%d/%d)'%(batch_idx+1, len(train_loader), 
                                                                  val_loss/(batch_idx+1), 100.*correct/total, correct, total))
    scheduler.step()

for epoch in range(epochs):
    print('\nEpoch %d'%(epoch))
    train(epoch)
    val(epoch)
```

    
    Epoch 0
    
    Train:
    [  1/ 55] | Loss: 0.69893 | Acc: 42.969% (55/128)
    [ 31/ 55] | Loss: 0.68746 | Acc: 56.779% (2253/3968)
    [ 55/ 55] | Loss: 0.67102 | Acc: 60.243% (4217/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.60287 | Acc: 67.188% (86/128)
    [  8/ 55] | Loss: 0.61010 | Acc: 67.300% (673/1000)
    
    Epoch 1
    
    Train:
    [  1/ 55] | Loss: 0.59523 | Acc: 70.312% (90/128)
    [ 31/ 55] | Loss: 0.58310 | Acc: 69.078% (2741/3968)
    [ 55/ 55] | Loss: 0.58157 | Acc: 68.957% (4827/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.57483 | Acc: 68.750% (88/128)
    [  8/ 55] | Loss: 0.62725 | Acc: 64.700% (647/1000)
    
    Epoch 2
    
    Train:
    [  1/ 55] | Loss: 0.53600 | Acc: 73.438% (94/128)
    [ 31/ 55] | Loss: 0.54637 | Acc: 71.547% (2839/3968)
    [ 55/ 55] | Loss: 0.54596 | Acc: 71.557% (5009/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.49603 | Acc: 75.781% (97/128)
    [  8/ 55] | Loss: 0.52507 | Acc: 72.100% (721/1000)
    
    Epoch 3
    
    Train:
    [  1/ 55] | Loss: 0.53764 | Acc: 75.000% (96/128)
    [ 31/ 55] | Loss: 0.53614 | Acc: 72.354% (2871/3968)
    [ 55/ 55] | Loss: 0.53716 | Acc: 72.229% (5056/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.53029 | Acc: 67.969% (87/128)
    [  8/ 55] | Loss: 0.58538 | Acc: 68.200% (682/1000)
    
    Epoch 4
    
    Train:
    [  1/ 55] | Loss: 0.58907 | Acc: 66.406% (85/128)
    [ 31/ 55] | Loss: 0.55149 | Acc: 71.447% (2835/3968)
    [ 55/ 55] | Loss: 0.54819 | Acc: 71.586% (5011/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.53041 | Acc: 70.312% (90/128)
    [  8/ 55] | Loss: 0.57291 | Acc: 68.500% (685/1000)
    
    Epoch 5
    
    Train:
    [  1/ 55] | Loss: 0.55291 | Acc: 71.875% (92/128)
    [ 31/ 55] | Loss: 0.53712 | Acc: 72.555% (2879/3968)
    [ 55/ 55] | Loss: 0.54437 | Acc: 72.000% (5040/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.62075 | Acc: 67.969% (87/128)
    [  8/ 55] | Loss: 0.68451 | Acc: 64.400% (644/1000)
    
    Epoch 6
    
    Train:
    [  1/ 55] | Loss: 0.58198 | Acc: 73.438% (94/128)
    [ 31/ 55] | Loss: 0.54234 | Acc: 72.581% (2880/3968)
    [ 55/ 55] | Loss: 0.53519 | Acc: 72.686% (5088/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.52322 | Acc: 69.531% (89/128)
    [  8/ 55] | Loss: 0.57310 | Acc: 69.100% (691/1000)
    
    Epoch 7
    
    Train:
    [  1/ 55] | Loss: 0.43731 | Acc: 80.469% (103/128)
    [ 31/ 55] | Loss: 0.53401 | Acc: 72.152% (2863/3968)
    [ 55/ 55] | Loss: 0.54106 | Acc: 71.957% (5037/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.49641 | Acc: 74.219% (95/128)
    [  8/ 55] | Loss: 0.51658 | Acc: 72.100% (721/1000)
    
    Epoch 8
    
    Train:
    [  1/ 55] | Loss: 0.59055 | Acc: 65.625% (84/128)
    [ 31/ 55] | Loss: 0.55096 | Acc: 71.976% (2856/3968)
    [ 55/ 55] | Loss: 0.55349 | Acc: 71.414% (4999/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.50660 | Acc: 72.656% (93/128)
    [  8/ 55] | Loss: 0.53341 | Acc: 72.200% (722/1000)
    
    Epoch 9
    
    Train:
    [  1/ 55] | Loss: 0.54495 | Acc: 71.875% (92/128)
    [ 31/ 55] | Loss: 0.54075 | Acc: 72.026% (2858/3968)
    [ 55/ 55] | Loss: 0.53961 | Acc: 72.457% (5072/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.59551 | Acc: 68.750% (88/128)
    [  8/ 55] | Loss: 0.65699 | Acc: 65.600% (656/1000)
    
    Epoch 10
    
    Train:
    [  1/ 55] | Loss: 0.53478 | Acc: 73.438% (94/128)
    [ 31/ 55] | Loss: 0.54239 | Acc: 71.976% (2856/3968)
    [ 55/ 55] | Loss: 0.54397 | Acc: 71.686% (5018/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.60993 | Acc: 67.969% (87/128)
    [  8/ 55] | Loss: 0.67401 | Acc: 63.800% (638/1000)
    
    Epoch 11
    
    Train:
    [  1/ 55] | Loss: 0.52347 | Acc: 71.875% (92/128)
    [ 31/ 55] | Loss: 0.53812 | Acc: 72.581% (2880/3968)
    [ 55/ 55] | Loss: 0.53478 | Acc: 72.743% (5092/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.52099 | Acc: 71.094% (91/128)
    [  8/ 55] | Loss: 0.55698 | Acc: 70.900% (709/1000)
    
    Epoch 12
    
    Train:
    [  1/ 55] | Loss: 0.57066 | Acc: 71.094% (91/128)
    [ 31/ 55] | Loss: 0.53651 | Acc: 72.455% (2875/3968)
    [ 55/ 55] | Loss: 0.54365 | Acc: 71.814% (5027/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.50696 | Acc: 73.438% (94/128)
    [  8/ 55] | Loss: 0.51666 | Acc: 73.700% (737/1000)
    
    Epoch 13
    
    Train:
    [  1/ 55] | Loss: 0.60562 | Acc: 64.062% (82/128)
    [ 31/ 55] | Loss: 0.54846 | Acc: 72.077% (2860/3968)
    [ 55/ 55] | Loss: 0.54918 | Acc: 71.757% (5023/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.50462 | Acc: 72.656% (93/128)
    [  8/ 55] | Loss: 0.53145 | Acc: 72.400% (724/1000)
    
    Epoch 14
    
    Train:
    [  1/ 55] | Loss: 0.63213 | Acc: 68.750% (88/128)
    [ 31/ 55] | Loss: 0.54501 | Acc: 71.673% (2844/3968)
    [ 55/ 55] | Loss: 0.53875 | Acc: 72.257% (5058/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.54063 | Acc: 68.750% (88/128)
    [  8/ 55] | Loss: 0.59193 | Acc: 67.900% (679/1000)
    
    Epoch 15
    
    Train:
    [  1/ 55] | Loss: 0.53967 | Acc: 71.094% (91/128)
    [ 31/ 55] | Loss: 0.53431 | Acc: 72.555% (2879/3968)
    [ 55/ 55] | Loss: 0.53616 | Acc: 72.429% (5070/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.51357 | Acc: 69.531% (89/128)
    [  8/ 55] | Loss: 0.55808 | Acc: 69.500% (695/1000)
    
    Epoch 16
    
    Train:
    [  1/ 55] | Loss: 0.54165 | Acc: 75.000% (96/128)
    [ 31/ 55] | Loss: 0.53448 | Acc: 73.059% (2899/3968)
    [ 55/ 55] | Loss: 0.53946 | Acc: 72.800% (5096/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.52436 | Acc: 69.531% (89/128)
    [  8/ 55] | Loss: 0.56844 | Acc: 68.900% (689/1000)
    
    Epoch 17
    
    Train:
    [  1/ 55] | Loss: 0.56997 | Acc: 71.875% (92/128)
    [ 31/ 55] | Loss: 0.53655 | Acc: 72.404% (2873/3968)
    [ 55/ 55] | Loss: 0.53889 | Acc: 72.571% (5080/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.52906 | Acc: 68.750% (88/128)
    [  8/ 55] | Loss: 0.57076 | Acc: 68.800% (688/1000)
    
    Epoch 18
    
    Train:
    [  1/ 55] | Loss: 0.52352 | Acc: 74.219% (95/128)
    [ 31/ 55] | Loss: 0.53955 | Acc: 72.429% (2874/3968)
    [ 55/ 55] | Loss: 0.53915 | Acc: 72.414% (5069/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.51703 | Acc: 68.750% (88/128)
    [  8/ 55] | Loss: 0.56372 | Acc: 69.100% (691/1000)
    
    Epoch 19
    
    Train:
    [  1/ 55] | Loss: 0.46562 | Acc: 78.906% (101/128)
    [ 31/ 55] | Loss: 0.54755 | Acc: 71.472% (2836/3968)
    [ 55/ 55] | Loss: 0.54510 | Acc: 71.743% (5022/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.52519 | Acc: 70.312% (90/128)
    [  8/ 55] | Loss: 0.56799 | Acc: 69.500% (695/1000)
    
    Epoch 20
    
    Train:
    [  1/ 55] | Loss: 0.58651 | Acc: 70.312% (90/128)
    [ 31/ 55] | Loss: 0.54412 | Acc: 71.825% (2850/3968)
    [ 55/ 55] | Loss: 0.53508 | Acc: 72.843% (5099/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.54667 | Acc: 67.969% (87/128)
    [  8/ 55] | Loss: 0.60112 | Acc: 67.800% (678/1000)
    
    Epoch 21
    
    Train:
    [  1/ 55] | Loss: 0.49021 | Acc: 77.344% (99/128)
    [ 31/ 55] | Loss: 0.53547 | Acc: 72.933% (2894/3968)
    [ 55/ 55] | Loss: 0.53382 | Acc: 72.986% (5109/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.60240 | Acc: 67.969% (87/128)
    [  8/ 55] | Loss: 0.66834 | Acc: 65.300% (653/1000)
    
    Epoch 22
    
    Train:
    [  1/ 55] | Loss: 0.52281 | Acc: 75.781% (97/128)
    [ 31/ 55] | Loss: 0.53044 | Acc: 73.110% (2901/3968)
    [ 55/ 55] | Loss: 0.53806 | Acc: 72.586% (5081/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.50938 | Acc: 70.312% (90/128)
    [  8/ 55] | Loss: 0.54824 | Acc: 70.400% (704/1000)
    
    Epoch 23
    
    Train:
    [  1/ 55] | Loss: 0.49832 | Acc: 71.094% (91/128)
    [ 31/ 55] | Loss: 0.52829 | Acc: 72.555% (2879/3968)
    [ 55/ 55] | Loss: 0.53276 | Acc: 72.557% (5079/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.52415 | Acc: 69.531% (89/128)
    [  8/ 55] | Loss: 0.57259 | Acc: 68.800% (688/1000)
    
    Epoch 24
    
    Train:
    [  1/ 55] | Loss: 0.50844 | Acc: 72.656% (93/128)
    [ 31/ 55] | Loss: 0.52067 | Acc: 73.564% (2919/3968)
    [ 55/ 55] | Loss: 0.53120 | Acc: 72.743% (5092/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.51363 | Acc: 70.312% (90/128)
    [  8/ 55] | Loss: 0.55475 | Acc: 69.900% (699/1000)
    
    Epoch 25
    
    Train:
    [  1/ 55] | Loss: 0.46849 | Acc: 78.125% (100/128)
    [ 31/ 55] | Loss: 0.53539 | Acc: 72.908% (2893/3968)
    [ 55/ 55] | Loss: 0.53200 | Acc: 73.143% (5120/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.51461 | Acc: 70.312% (90/128)
    [  8/ 55] | Loss: 0.55856 | Acc: 69.800% (698/1000)
    
    Epoch 26
    
    Train:
    [  1/ 55] | Loss: 0.50982 | Acc: 78.906% (101/128)
    [ 31/ 55] | Loss: 0.53814 | Acc: 72.732% (2886/3968)
    [ 55/ 55] | Loss: 0.53366 | Acc: 72.914% (5104/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.55071 | Acc: 67.969% (87/128)
    [  8/ 55] | Loss: 0.60197 | Acc: 68.100% (681/1000)
    
    Epoch 27
    
    Train:
    [  1/ 55] | Loss: 0.56114 | Acc: 72.656% (93/128)
    [ 31/ 55] | Loss: 0.53808 | Acc: 72.253% (2867/3968)
    [ 55/ 55] | Loss: 0.53399 | Acc: 72.671% (5087/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.52451 | Acc: 69.531% (89/128)
    [  8/ 55] | Loss: 0.56855 | Acc: 69.300% (693/1000)
    
    Epoch 28
    
    Train:
    [  1/ 55] | Loss: 0.51355 | Acc: 74.219% (95/128)
    [ 31/ 55] | Loss: 0.52967 | Acc: 72.933% (2894/3968)
    [ 55/ 55] | Loss: 0.53206 | Acc: 72.871% (5101/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.53538 | Acc: 67.969% (87/128)
    [  8/ 55] | Loss: 0.58291 | Acc: 68.100% (681/1000)
    
    Epoch 29
    
    Train:
    [  1/ 55] | Loss: 0.57169 | Acc: 71.094% (91/128)
    [ 31/ 55] | Loss: 0.53524 | Acc: 72.228% (2866/3968)
    [ 55/ 55] | Loss: 0.53576 | Acc: 72.614% (5083/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.56951 | Acc: 67.969% (87/128)
    [  8/ 55] | Loss: 0.62618 | Acc: 66.800% (668/1000)
    
    Epoch 30
    
    Train:
    [  1/ 55] | Loss: 0.48186 | Acc: 75.000% (96/128)
    [ 31/ 55] | Loss: 0.53908 | Acc: 72.908% (2893/3968)
    [ 55/ 55] | Loss: 0.53790 | Acc: 72.857% (5100/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.55516 | Acc: 68.750% (88/128)
    [  8/ 55] | Loss: 0.60863 | Acc: 67.600% (676/1000)
    
    Epoch 31
    
    Train:
    [  1/ 55] | Loss: 0.48176 | Acc: 79.688% (102/128)
    [ 31/ 55] | Loss: 0.54436 | Acc: 72.001% (2857/3968)
    [ 55/ 55] | Loss: 0.54439 | Acc: 72.057% (5044/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.53316 | Acc: 67.969% (87/128)
    [  8/ 55] | Loss: 0.57741 | Acc: 69.000% (690/1000)
    
    Epoch 32
    
    Train:
    [  1/ 55] | Loss: 0.57992 | Acc: 74.219% (95/128)
    [ 31/ 55] | Loss: 0.53741 | Acc: 72.908% (2893/3968)
    [ 55/ 55] | Loss: 0.53571 | Acc: 72.729% (5091/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.56521 | Acc: 67.969% (87/128)
    [  8/ 55] | Loss: 0.61848 | Acc: 67.300% (673/1000)
    
    Epoch 33
    
    Train:
    [  1/ 55] | Loss: 0.47651 | Acc: 76.562% (98/128)
    [ 31/ 55] | Loss: 0.53279 | Acc: 72.833% (2890/3968)
    [ 55/ 55] | Loss: 0.53241 | Acc: 72.757% (5093/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.57454 | Acc: 67.969% (87/128)
    [  8/ 55] | Loss: 0.63237 | Acc: 66.200% (662/1000)
    
    Epoch 34
    
    Train:
    [  1/ 55] | Loss: 0.58875 | Acc: 67.969% (87/128)
    [ 31/ 55] | Loss: 0.54007 | Acc: 72.152% (2863/3968)
    [ 55/ 55] | Loss: 0.53629 | Acc: 72.814% (5097/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.58766 | Acc: 67.969% (87/128)
    [  8/ 55] | Loss: 0.64871 | Acc: 66.100% (661/1000)
    
    Epoch 35
    
    Train:
    [  1/ 55] | Loss: 0.60291 | Acc: 63.281% (81/128)
    [ 31/ 55] | Loss: 0.53356 | Acc: 72.858% (2891/3968)
    [ 55/ 55] | Loss: 0.54048 | Acc: 72.400% (5068/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.52503 | Acc: 70.312% (90/128)
    [  8/ 55] | Loss: 0.56456 | Acc: 69.300% (693/1000)
    
    Epoch 36
    
    Train:
    [  1/ 55] | Loss: 0.52539 | Acc: 74.219% (95/128)
    [ 31/ 55] | Loss: 0.53343 | Acc: 72.757% (2887/3968)
    [ 55/ 55] | Loss: 0.53544 | Acc: 72.786% (5095/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.50105 | Acc: 71.094% (91/128)
    [  8/ 55] | Loss: 0.53692 | Acc: 71.800% (718/1000)
    
    Epoch 37
    
    Train:
    [  1/ 55] | Loss: 0.53303 | Acc: 76.562% (98/128)
    [ 31/ 55] | Loss: 0.52908 | Acc: 73.110% (2901/3968)
    [ 55/ 55] | Loss: 0.53368 | Acc: 72.814% (5097/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.51064 | Acc: 70.312% (90/128)
    [  8/ 55] | Loss: 0.55147 | Acc: 70.100% (701/1000)
    
    Epoch 38
    
    Train:
    [  1/ 55] | Loss: 0.49019 | Acc: 76.562% (98/128)
    [ 31/ 55] | Loss: 0.53861 | Acc: 72.278% (2868/3968)
    [ 55/ 55] | Loss: 0.53815 | Acc: 72.414% (5069/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.49988 | Acc: 72.656% (93/128)
    [  8/ 55] | Loss: 0.53177 | Acc: 72.300% (723/1000)
    
    Epoch 39
    
    Train:
    [  1/ 55] | Loss: 0.50416 | Acc: 77.344% (99/128)
    [ 31/ 55] | Loss: 0.53899 | Acc: 72.933% (2894/3968)
    [ 55/ 55] | Loss: 0.53613 | Acc: 72.957% (5107/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.53325 | Acc: 69.531% (89/128)
    [  8/ 55] | Loss: 0.58018 | Acc: 68.800% (688/1000)
    
    Epoch 40
    
    Train:
    [  1/ 55] | Loss: 0.57540 | Acc: 66.406% (85/128)
    [ 31/ 55] | Loss: 0.54150 | Acc: 72.429% (2874/3968)
    [ 55/ 55] | Loss: 0.53129 | Acc: 72.986% (5109/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.52465 | Acc: 69.531% (89/128)
    [  8/ 55] | Loss: 0.57016 | Acc: 69.200% (692/1000)
    
    Epoch 41
    
    Train:
    [  1/ 55] | Loss: 0.56385 | Acc: 64.844% (83/128)
    [ 31/ 55] | Loss: 0.53429 | Acc: 72.379% (2872/3968)
    [ 55/ 55] | Loss: 0.53267 | Acc: 72.671% (5087/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.61537 | Acc: 69.531% (89/128)
    [  8/ 55] | Loss: 0.68163 | Acc: 64.600% (646/1000)
    
    Epoch 42
    
    Train:
    [  1/ 55] | Loss: 0.47191 | Acc: 74.219% (95/128)
    [ 31/ 55] | Loss: 0.53363 | Acc: 72.530% (2878/3968)
    [ 55/ 55] | Loss: 0.53065 | Acc: 72.800% (5096/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.54798 | Acc: 67.188% (86/128)
    [  8/ 55] | Loss: 0.60124 | Acc: 68.300% (683/1000)
    
    Epoch 43
    
    Train:
    [  1/ 55] | Loss: 0.61029 | Acc: 67.969% (87/128)
    [ 31/ 55] | Loss: 0.52864 | Acc: 73.589% (2920/3968)
    [ 55/ 55] | Loss: 0.53137 | Acc: 73.157% (5121/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.54849 | Acc: 68.750% (88/128)
    [  8/ 55] | Loss: 0.60046 | Acc: 68.500% (685/1000)
    
    Epoch 44
    
    Train:
    [  1/ 55] | Loss: 0.50352 | Acc: 75.000% (96/128)
    [ 31/ 55] | Loss: 0.52857 | Acc: 73.387% (2912/3968)
    [ 55/ 55] | Loss: 0.53104 | Acc: 73.043% (5113/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.56080 | Acc: 68.750% (88/128)
    [  8/ 55] | Loss: 0.61789 | Acc: 67.100% (671/1000)
    
    Epoch 45
    
    Train:
    [  1/ 55] | Loss: 0.46987 | Acc: 76.562% (98/128)
    [ 31/ 55] | Loss: 0.53484 | Acc: 72.681% (2884/3968)
    [ 55/ 55] | Loss: 0.53498 | Acc: 72.757% (5093/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.58677 | Acc: 67.969% (87/128)
    [  8/ 55] | Loss: 0.64719 | Acc: 65.600% (656/1000)
    
    Epoch 46
    
    Train:
    [  1/ 55] | Loss: 0.52086 | Acc: 71.875% (92/128)
    [ 31/ 55] | Loss: 0.53133 | Acc: 72.908% (2893/3968)
    [ 55/ 55] | Loss: 0.53128 | Acc: 73.014% (5111/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.51454 | Acc: 70.312% (90/128)
    [  8/ 55] | Loss: 0.55649 | Acc: 70.100% (701/1000)
    
    Epoch 47
    
    Train:
    [  1/ 55] | Loss: 0.57383 | Acc: 68.750% (88/128)
    [ 31/ 55] | Loss: 0.53422 | Acc: 72.833% (2890/3968)
    [ 55/ 55] | Loss: 0.53063 | Acc: 72.957% (5107/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.56505 | Acc: 68.750% (88/128)
    [  8/ 55] | Loss: 0.62168 | Acc: 67.400% (674/1000)
    
    Epoch 48
    
    Train:
    [  1/ 55] | Loss: 0.63185 | Acc: 67.188% (86/128)
    [ 31/ 55] | Loss: 0.52619 | Acc: 73.463% (2915/3968)
    [ 55/ 55] | Loss: 0.53042 | Acc: 73.014% (5111/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.54348 | Acc: 67.969% (87/128)
    [  8/ 55] | Loss: 0.59519 | Acc: 68.400% (684/1000)
    
    Epoch 49
    
    Train:
    [  1/ 55] | Loss: 0.52913 | Acc: 76.562% (98/128)
    [ 31/ 55] | Loss: 0.52711 | Acc: 73.841% (2930/3968)
    [ 55/ 55] | Loss: 0.53171 | Acc: 72.986% (5109/7000)
    
    Validation:
    [  1/ 55] | Loss: 0.54524 | Acc: 67.188% (86/128)
    [  8/ 55] | Loss: 0.59777 | Acc: 68.300% (683/1000)


##**Do NOT touch this cell (Evaluation)**


```python
import sklearn.metrics as skl
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


net.eval()
ylabel = []
yhatlabel = []

for batch_idx, (inputs, targets) in enumerate(test_loader):
    inputs, targets = inputs.float(), targets.float()
    outputs = net(inputs)
    predicted = outputs.detach().squeeze(1) > 0.5
    ylabel = np.concatenate((ylabel, targets.numpy()))
    yhatlabel = np.concatenate((yhatlabel, predicted.numpy()))

# Compute confusion matrix
cnf_matrix = skl.confusion_matrix(ylabel, yhatlabel)
np.set_printoptions(precision=2)
is_correct = (ylabel == yhatlabel)
acc = np.sum(is_correct * 1) / len(is_correct)
print('accuracy: %.3f%%' %(acc*100))

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,1],
                  title='Confusion matrix')
plt.show()
```

    accuracy: 70.303%



![png](Project1_game_prediction_for_students_files/Project1_game_prediction_for_students_16_1.png)


##채점:
### 70% 이상: A 
### 60% 이상: B
### 51% 이상: C
### 51% 미만 및 실행불가: D
