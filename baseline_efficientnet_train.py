import os
import torch
import numpy as np
import random
import json
import time
import sys
from PIL import Image
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

random_seed = 8138
num_class = 25
batch_size = 128
epochs = 30
checkpoints = "./b0_checkpoints"

torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU


def accuracy_function(real, pred):    
    score = f1_score(real, pred, average='macro')
    return score

def model_save(model, path):
    os.makedirs(checkpoints, exist_ok=True)
    torch.save(model.state_dict(), path)


# DATA PATH
# train_csv = sorted(glob('../data/train/*/*.csv'))
# train_jpg = sorted(glob('../data/train/*/*.jpg'))
# train_json = sorted(glob('../data/train/*/*.json'))

base_dir = "../data/train"
train_data = sorted(os.listdir(base_dir))

train_csv = [os.path.join(base_dir, i, i + ".csv") for i in train_data]
train_path = [os.path.join(base_dir, i, i + ".jpg") for i in train_data]
train_json = [os.path.join(base_dir, i, i + ".json") for i in train_data]

crops = []
diseases = []
risks = []
labels = []

# TRAINING DATA LABELING
for i in range(len(train_json)):
    with open(train_json[i], 'r') as f:
        sample = json.load(f)
        crop = sample['annotations']['crop']
        disease = sample['annotations']['disease']
        risk = sample['annotations']['risk']
        label=f"{crop}_{disease}_{risk}"
    
        crops.append(crop)
        diseases.append(disease)
        risks.append(risk)
        labels.append(label)


label_unique = sorted(np.unique(labels))

# label dictionary
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

# make integer label
labels = [label_unique[k] for k in labels]


# GET IMAGES -> 추후 수정하자
# def img_load(path):
#     img = cv2.imread(path)[:,:,::-1]
#     img = cv2.resize(img, (384, 512))
#     return img


# IMAGE ARRAY -> 추후 수정
# imgs = [img_load(k) for k in tqdm(train_jpg)]



def init_network():
    model = models.efficientnet_b0(pretrained=True)
    num_in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_in_features, num_class)

    return model

class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode=mode
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = self.img_paths[idx]
        img = Image.open(img)

        # if self.mode=='train':
        #     augmentation = random.randint(0,2)
        #     if augmentation==1:
        #         img = img[::-1].copy()
        #     elif augmentation==2:
        #         img = img[:,::-1].copy()
        img = transforms.Resize((224, 224))(img)
        img = transforms.ToTensor()(img)
        label = self.labels[idx]
        return img,label


model = init_network()


folds = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
# for train_idx, valid_idx in skf.split(train_path, labels):
#     folds.append((train_idx, valid_idx))

# fold=0 # 이거 수정해야 함.

# train_idx, valid_idx = folds[fold]




# # Train
# train_dataset = Custom_dataset(np.array(train_path)[train_idx], np.array(labels)[train_idx], mode='train')
# train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=8)

# # Validation 
# valid_dataset = Custom_dataset(np.array(train_path)[valid_idx], np.array(labels)[valid_idx], mode='valid')
# valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=8)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler() 



best=0
fold_f1 = 0
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_path, labels)):
    # Train
    train_dataset = Custom_dataset(np.array(train_path)[train_idx], np.array(labels)[train_idx], mode='train')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=8)

    # Validation 
    valid_dataset = Custom_dataset(np.array(train_path)[valid_idx], np.array(labels)[valid_idx], mode='valid')
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=8)
    
    print(f"FOLD START ===================>   : {fold}")
    for epoch in range(epochs):
        start=time.time()
        train_loss = 0
        train_pred=[]
        train_y=[]
        model.train()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x = torch.tensor(batch[0], dtype=torch.float32, device=device)
            y = torch.tensor(batch[1], dtype=torch.long, device=device)
            with torch.cuda.amp.autocast():
                pred = model(x)
            loss = criterion(pred, y)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()/len(train_loader)
            train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
            train_y += y.detach().cpu().numpy().tolist()

            if iter % 10 == 0:
                print(f'iter : {iter} : loss : {loss.item()}')
            
        
        train_f1 = accuracy_function(train_y, train_pred)
        
        model.eval()
        valid_loss = 0
        valid_pred=[]
        valid_y=[]
        with torch.no_grad():
            for batch in (valid_loader):
                x = torch.tensor(batch[0], dtype=torch.float32, device=device)
                y = torch.tensor(batch[1], dtype=torch.long, device=device)
                with torch.cuda.amp.autocast():
                    pred = model(x)
                loss = criterion(pred, y)
                valid_loss += loss.item()/len(valid_loader)
                valid_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                valid_y += y.detach().cpu().numpy().tolist()
            valid_f1 = accuracy_function(valid_y, valid_pred)
        if valid_f1>=best:
            best=valid_f1
            model_save(model, f'{checkpoints}/best.pth')
        model_save(model, f'{checkpoints}/epoch_{epoch}.pth')
        TIME = time.time() - start
        
        print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')
        print(f'VALID    loss : {valid_loss:.5f}    f1 : {valid_f1:.5f}    best : {best:.5f}')
        print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')

    fold_f1 += best # score 갱신
    
    
    print("END_FOLD===========================")
print("mean F1 : ", fold_f1 / 5)
    
