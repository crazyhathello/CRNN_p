import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from functions_test import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
from itertools import chain
from collections import OrderedDict

# set path
data_path = "/input"    # define UCF-101 RGB data path
if os.environ["VIDEO_PATH"] is not None:
    data_path = os.environ["VIDEO_PATH"]

# output_path = "./result/220504/dy3/F4_20211214"
output_path = "/postures"
if os.environ["OUTPUT_PATH"] is not None:
    output_path = os.environ["OUTPUT_PATH"]

action_name_path = './action.txt'

cnn_choose_epoch = "cnn_encoder_epoch.pth"  
if os.environ["CNN"] is not None:
    cnn_choose_epoch = os.environ["CNN"]

rnn_choose_epoch = "rnn_decoder_epoch.pth"
if os.environ["RNN"] is not None:
    rnn_choose_epoch = os.environ["RNN"]

# EncoderCNN architecture
CNN_fc_hidden1 = 1024
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.2

# DecoderRNN architecture
RNN_hidden_layers = 1
RNN_hidden_nodes = 512
RNN_FC_dim = 256

num_classes = 7             # number of target category
batch_size = 64
# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 5, 115, 5 

try:
    os.makedirs(output_path)
except FileExistsError:
    print(output_path + "  exist")


with open(action_name_path, 'r') as f:
    action_names = f.readline().split(",")

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

# # example
# y = ['HorseRace', 'YoYo', 'WalkingWithDog']
# y_onehot = labels2onehot(enc, le, y)
# y2 = onehot2labels(le, y_onehot)

actions = []
fnames = os.listdir(data_path)

actions = []
all_names = []

for img_dir in os.listdir(data_path):
    # actions.append(action)
    actions.append('Sitting')
    # all_names.append(img_dir.split("/")[-1])
    all_names.append(img_dir)

# list all data files
all_X_list = all_names                  # all video file names
# print(all_names)
all_y_list = labels2cat(le, actions)    # all video labels

# data loading parameters
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU


transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

# reset data loader
all_data_params = {'batch_size': batch_size, 'shuffle': False} if use_cuda else {}
all_data_loader = data.DataLoader(Dataset_CRNN(data_path, all_X_list, all_y_list, selected_frames, action_names, transform=transform), **all_data_params)


# reload CRNN model


cnn_encoder = EfficientNetCNNEncoder(fc_hidden1=CNN_fc_hidden1, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=num_classes).to(device)
# print model
# print(cnn_encoder)
# summary(cnn_encoder, input_size = (batch_size, 110, 3, res_size, res_size))
# print(rnn_decoder)
# summary(rnn_decoder, input_size = (batch_size, 110, CNN_embed_dim))

cnn_state_dict = torch.load(cnn_choose_epoch)
new_state_dict = OrderedDict()
for k, v in cnn_state_dict.items():
    name = k[7:] # remove module.
    new_state_dict[name] = v
cnn_state_dict = new_state_dict


rnn_state_dict = torch.load(rnn_choose_epoch)
new_state_dict = OrderedDict()
for k, v in rnn_state_dict.items():
    name = k[7:] # remove module.
    new_state_dict[name] = v
rnn_state_dict = new_state_dict

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)
    # cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, cnn_choose_epoch)))
    # rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, rnn_choose_epoch)))
    cnn_encoder.load_state_dict(cnn_state_dict)
    rnn_decoder.load_state_dict(rnn_state_dict)
    cnn_encoder = cnn_encoder.module
    rnn_decoder = rnn_decoder

elif torch.cuda.device_count() == 1:
    # cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, cnn_choose_epoch)))
    # rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, rnn_choose_epoch)))
    cnn_encoder.load_state_dict(cnn_state_dict)
    rnn_decoder.load_state_dict(rnn_state_dict)

print('Model reloaded!')


# make all video predictions by reloaded model
print('Predicting all {} videos:'.format(len(all_data_loader.dataset)))
all_y_pred = CRNN_final_prediction([cnn_encoder, rnn_decoder], device, all_data_loader)

try:
    all_y_pred = list(chain(*all_y_pred))
    all_y_pred = np.array(all_y_pred)
except:
    all_y_pred = np.array(all_y_pred)

# write in pandas dataframe
df = pd.DataFrame(data={'filename': all_names, 'PD': cat2labels(le, all_y_pred)})

# file name -------------------------------------------------------------------------------------------------------------------------------------------------------
df.to_pickle(os.path.join(output_path, "prediction.pkl"))  # save pandas dataframe
# pd.read_pickle("./all_videos_prediction.pkl")
print('video prediction finished!')