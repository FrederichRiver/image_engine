#!/usr/bin/python3
# filename: Solar/model_01/model_01.py

from sqlalchemy.orm import Session
from finance_model.stock_model import get_stock_data
from libsql_utils.head import root
from libsql_utils.engine import engine_init
from pandas import NA
from torch import Tensor
import numpy as np
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset

eng = engine_init(*root)
session = Session(eng)
stock_id = 'SZ300015'

df = get_stock_data(session, stock_id)
df = df.loc[:,['close', 'open', 'high', 'low', 'amplitude', 'volume']]
df['label']  = df.apply(lambda d:  1 if (d['close'] > d['open']) else 0, axis=1)
df.loc[df['open'] == 0] = NA
df.dropna(axis=0, how='any', inplace=True)
df['close'] = df['close'] / df['open'] 
df['high'] = df['high'] / df['open'] 
df['low'] = df['low'] / df['open'] 
df['open'] = 1.0

# print(df.head(15))

d = df.loc[:,['open', 'close', 'high', 'low', 'label']]

# label = d['label'].tolist() 
# labels = []
# for item in label:
#     s = [1 if item==i else 0 for i in range(4)]
#     labels.append(s) 

class TradeDataSet(Dataset):
    """
    二分类数据集，涨为1,跌为0
    """
    def __init__(self, df: DataFrame) -> None:
        super().__init__()
        train_data = torch.tensor(data=df.iloc[:,:-1].astype(float).values)
        label_data = torch.tensor(data=df['label'].astype(float).values).view(-1, 1)
        # label_data = torch.tensor(data=labels)
        print(label_data.shape)
        train_data = train_data.view(train_data.shape[0], 1, train_data.shape[1])
        # label_data = label_data.view(label_data.shape[0], label_data.shape[1], 1)
        self.data = []
        for d, l in zip(train_data, label_data):
            item = { "data": d, "label": l}
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].get("data"), self.data[index].get("label")

trade_dataset = TradeDataSet(d)

loader = DataLoader(trade_dataset, batch_size=50, shuffle=True)

from torch import nn

class Model001(nn.Module):
    """
    单层一维数据的卷积分类器。只有一个卷积层。输出通道根据分类数量进行区分。
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        torch.nn.init.xavier_normal_(self.conv1.weight)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.conv1(x)
        # x = x.view(1, 4)
        return x

model = Model001()

# # 训练过程
criterion = nn.CrossEntropyLoss()
run_loss = 0
optimzer = torch.optim.Adam(model.parameters(), lr=1e-4)
model.train()
i = 0
from visdom import Visdom
viz = Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
# viz.line([[0.], [0.]], [0.], win='test', opts=dict(title='test loss & acc'), legend=['loss', 'acc'])
for epoch in range(500):
    for step, (batch_x, batch_y) in enumerate(loader):
        i += 1
        # print(batch_x.shape)
        # print(batch_x)
        optimzer.zero_grad()
        outputs = model.forward(batch_x)
        loss = criterion(outputs, batch_y.long())
        loss.backward()
        optimzer.step()
        viz.line([loss.item()], [i], win='train_loss', update='append')
        if i % 1000 == 0:
            torch.save(model, '/home/fred/Documents/dev/Solar/model_01/model_01.pkl')
        run_loss += loss.item()