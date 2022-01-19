from distutils.command.build import build
import random
import os
import sys
import time
import numpy as np
from glob import glob

from PIL import Image
from sklearn.metrics import f1_score
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

random_seed = 8138

torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_db_dir = "./downsample_data"

label_list = sorted(os.listdir(base_db_dir))
num_valid = 10
num_class = 25
train_epoch = 30

train_image_list = []
valid_image_list = []
train_label_list = []
valid_label_list = []


# IMAGE PATH
for i in label_list:
    train_image_list.extend(glob(os.path.join(base_db_dir, i) + "/*.jpg")[:-num_valid])
    valid_image_list.extend(glob(os.path.join(base_db_dir, i) + "/*.jpg")[-num_valid:])


# LABEL PATH
for i in train_image_list:
    train_label_list.append(i.split("/")[2])

for i in valid_image_list:
    valid_label_list.append(i.split("/")[2])



# making label
train_label_list = np.array(train_label_list)
valid_label_list = np.array(valid_label_list)

label_unique = sorted(np.unique(train_label_list))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

for key in label_unique:
    train_label_list[train_label_list == key] = label_unique[key]
    valid_label_list[valid_label_list == key] = label_unique[key]

train_label_list = train_label_list.astype(int)
valid_label_list = train_label_list.astype(int)

class custom_dataset(Dataset):
    def __init__(self, img_path, labels):
        self.img_path = img_path
        self.labels = labels
        pass

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, index):

        img = self.img_path[index]
        label = self.labels[index]

        img = Image.open(img)
        img = transforms.Resize((512, 512))(img)
        img = transforms.ToTensor()(img)


        return img, label


def build_model():
    model = models.efficientnet_b7(pretrained=True)

    # checking required grad
    # for param in model.parameters():
    #     print(param)

    # New layer
    num_in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_in_features, num_class)

    return model

def accuracy_function(real, pred):    
    score = f1_score(real, pred, average='macro')
    return score

train_dataset = custom_dataset(train_image_list, train_label_list)
valid_dataset = custom_dataset(valid_image_list, valid_label_list)

train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=2, pin_memory=True, num_workers=8)
valid_loader = DataLoader(valid_dataset, shuffle=False, drop_last=False, batch_size=1, pin_memory=True, num_workers=8)


model = build_model()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(0, train_epoch + 1):
    start=time.time()
    print('Epoch {}/{}'.format(epoch, train_epoch))
    print('-' * 10)

    train_loss = 0
    train_acc = 0
    train_pred = []
    train_y = []
    model.train()
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(train_loader)
        train_pred += outputs.argmax(1).detach().cpu().numpy().tolist()
        train_y += labels.detach().cpu().numpy().tolist()


        if idx % 100 == 0:
            print(f'iter : {idx} : loss : {loss.item()}')

    
    train_f1 = accuracy_function(train_y, train_pred)    


    model.eval()
    valid_loss = 0
    valid_pred=[]
    valid_y=[]
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            valid_loss += loss.item()/len(valid_loader)
            valid_pred += outputs.argmax(1).detach().cpu().numpy().tolist()
            valid_y += labels.detach().cpu().numpy().tolist()

        valid_f1 = accuracy_function(valid_y, valid_pred)
        
    
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"./checkpoints/epoch_{epoch}.pth")

    TIME = time.time() - start
    print(f'epoch : {epoch+1}/{train_epoch}    time : {TIME:.0f}s/{TIME*(train_epoch-epoch-1):.0f}s')
    print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')
    print(f'VALID    loss : {valid_loss:.5f}    f1 : {valid_f1:.5f}    best : {best:.5f}')

