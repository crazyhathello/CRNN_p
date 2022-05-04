# %%
from functions_test import *
import sys
import numpy as np
import pickle

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix
import csv
import os.path

# %% [markdown]
# ### 讀檔、校正

# %%
action_names = ['Feeding','LactatingLeft','LactatingRight','NonLactating','Sitting','Standing','VRecumbency']
# count testdata (no groung truth)
 
def read_cor(prediction_path, cor = True, so = True):
    date = prediction_path[-9:-1]

    with open(os.path.join(prediction_path, "prediction.pkl"), 'rb') as f:
        data_o = np.array(pickle.load(f))
        np.set_printoptions(threshold=sys.maxsize)
        data_o = data_o.tolist()
        if so == True:
            data = sorted(data_o, key = lambda s: s[0])
        else:
            data = data_o

        data = np.array(data)
        data_num = len(data)
        data_date = data[:,0]
        data_pd = data[:,1]

    le = LabelEncoder()
    le.fit(action_names) 
    list(le.classes_)
    # print(le.classes_[0])
    pd_list = labels2cat(le, data_pd)

    # 校正
    if cor == True:
        for i in range(len(pd_list)):
            if i <= len(pd_list)-6 and i > 0:
                pd_p = pd_list[i-1]
                pd = pd_list[i]
                pd_1 = pd_list[i+1]
                pd_2 = pd_list[i+2]
                pd_3 = pd_list[i+3]
                pd_4 = pd_list[i+4]
                pd_5 = pd_list[i+5]

                # Lactating 校正 (23)
                if pd == 1 or pd == 2 or pd == 3:
                    if pd != pd_p and pd != pd_1:                
                        pd_list[i] = pd_p
                        pd = pd_p
                    elif pd != pd_p and pd != pd_2:                
                        pd_list[i] = pd_p
                        pd = pd_p

                # Lactating 校正 (時長)
                if pd == 1 or pd == 2:
                    if pd != pd_p and (pd != pd_1 or pd != pd_2 or pd != pd_3 or pd != pd_4):                
                        pd_list[i] = pd_p

                # VRecumbency 校正
                if pd == 6:
                    if pd != pd_p and pd != pd_1:                
                        pd_list[i] = pd_p
                    elif pd != pd_p and pd != pd_2:                
                        pd_list[i] = pd_p

    pd_final = cat2labels(le, pd_list)
    return  date, data_num, pd_list, pd_final, data_date

# %% [markdown]
# ### posture change

# %%
# posture change (times) 

# np.reshape(pd_list,(1,data_num))
def posture_change(data_num, pd_list):
    pc_list = np.zeros(data_num)

    for i in range(data_num):
        if pd_list[i]==1 or pd_list[i]==2 or pd_list[i]==3:
            pc_list[i]=1
        elif pd_list[i]==6:
            pc_list[i]=2
        elif pd_list[i]==4:
            pc_list[i]=3
        elif pd_list[i]==5 or pd_list[i]==0:
            pc_list[i]=4

    a=np.zeros(data_num)
    for i in range(data_num):
        if i+2<=data_num:
            a[i] = bool(abs(pd_list[i+1]-pd_list[i]))

    pc = sum(a)
    return pc

# %% [markdown]
# ### 計算各姿態數量

# %%
# 計算各姿態數量  ['Feeding','LactatingLeft','LactatingRight','NonLactating','Sitting','Standing','VRecumbency']

def posture_cal(data_num, pd_list):

    mylist = list(pd_list)
    myset = set(mylist)
    y = np.zeros(7)
    for item in myset:
        y[item] = mylist.count(item)
    #     print("the %d has found %d" %(item,mylist.count(item)))     

    # 計算母豬吃料時長
    F = y[0]/2
    Fh = y[0]/60
    Fd = y[0]/(data_num)*100    

    # 計算母豬哺乳時長
    LT = (y[1] + y[2])/2
    LTh = LT/60
    LTd = (y[1] + y[2])/(data_num)*100    

    # 計算母豬趴臥時長
    VR = y[6]/2
    VRh = VR/60
    VRd = y[6]/(data_num)*100    

    # 計算母豬活動力
    AR = (y[3] + y[6])/(data_num)*100
    
    return F, Fh, Fd, LT, LTh, LTd, VR, VRh, VRd, AR, y

# %% [markdown]
# ### pie chart

# %%
# pie chart
def pie_chart(y, date, png_path):
    plt.figure(figsize=(15,6))
    ax = plt.axes()

    labels = ['Feeding','Standing','Sitting','Recumbency','Lying','LactatingLeft','LactatingRight']
    color = np.array(["green","lightgreen","royalblue","gold","brown","lightcoral","pink"])

    y_plt = [y[0], y[5], y[4], y[6], y[3], y[1], y[2]]

    x=plt.pie(y_plt, colors=color)
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', lw=2, label='Feeding %1.1f%%'%(y_plt[0]/data_num*100)),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', label='Standing %1.1f%%'%(y_plt[1]/data_num*100)),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', lw=2, label='Sitting %1.1f%%'%(y_plt[2]/data_num*100)),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', lw=2, label='Sternal or ventral recumbency %1.1f%%'%(y_plt[3]/data_num*100)),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='brown', lw=2, label='Lying %1.1f%%'%(y_plt[4]/data_num*100)),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', lw=2, label='Lactating on the left side %1.1f%%'%(y_plt[5]/data_num*100)),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', lw=2, label='Lactating on the right side %1.1f%%'%(y_plt[6]/data_num*100))]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1, 0.7, 0.3, 0.2), loc='upper left', fontsize=12)
    plt.show() 
#     plt.savefig(png_path, bbox_inches='tight',pad_inches = 0)

# %% [markdown]
# ### ALL

# %%
#讀檔、校正
prediction_path = './result/220504/dy3/F4_20211214/'
png_path = 'figure/' + prediction_path[-16:-13] + '_' + prediction_path[-12:-10] + '.png'

date, data_num, pd_list, pd_final, data_date = read_cor(prediction_path, cor = True, so = True)
for i in range(len(pd_list)):   
    print("{}: {}".format(data_date[i], pd_final[i]))

# 計算姿態變化頻率
pc = posture_change(data_num, pd_list)
print("posture change: {} times\n".format(pc))

# 計算其他姿態
F, Fh, Fd, LT, LTh, LTd, VR, VRh, VRd, AR, y = posture_cal(data_num, pd_list)
print("Time : %s ~ %s" %(data_date[0], data_date[-1]))
print("sow feeding time\t%d min\t %1.1f hr\t%1.2f %%\t平均一天(8:00-18:00): %1.2f hr" %(F, Fh, Fd, Fd/10)) 
print("sow Lactating time\t%d min\t %1.1f hr\t%1.2f %%\t平均一天(8:00-18:00): %1.2f hr" %(LT, LTh, LTd, LTd/10)) 
print("sow Recumbency time\t%d min\t %1.1f hr\t%1.2f %%\t平均一天(8:00-18:00): %1.2f hr" %(VR, VRh, VRd, VRd/10)) 
print("sow inactive ratio\t%d %%\t (VRecumbency+Lying)/ALL*100%%" %(AR))

# test資料夾內各姿態站比圖
# pie_chart(y, date, png_path)

# %%



