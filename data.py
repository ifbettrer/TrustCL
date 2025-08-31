import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# 数据处理

# 将数据集XY分离为特征矩阵X和标签矩阵Y
def seperate_XY(XY, num):
    data_X = XY.iloc[:, num:].values
    data_Y = XY.iloc[:, :num].values
    data_Y = data_Y.T
    return data_X, data_Y

def CT(before = False):
    org = pd.read_csv('data/dcdb.csv', encoding="utf-8")
    Y_list = ['chemotherapy_id_user_27', 'chemotherapy_id_user_28', 'chemotherapy_id_user_29',
              'chemotherapy_id_user_30', 'chemotherapy_id_user_31', 'chemotherapy_id_user_34']
    usr = len(Y_list) + 1
    data_X, data_Y = seperate_XY(org, usr)

    # MultiViewDataset("CT", data_X, data_Y, XY)

    return MultiViewDataset("CT", data_X, data_Y, org)

class MultiViewDataset(Dataset):
    def __init__(self, data_name, data_X, data_Y, XY):
        super(MultiViewDataset, self).__init__()
        self.data_name = data_name

        self.X = dict()
        self.Y = dict()  # !!
        self.num_views = len(data_Y) - 1
        for v in range(self.num_views):
            self.X[v] = self.normalize(data_X)
        self.X['index'] = XY.index.values
        for v in range(self.num_views):
            y_labels = data_Y[v]
            y_labels = y_labels.astype(dtype=np.int64)
            self.Y[v] = y_labels.copy()

        for v in range(self.num_views - 1):
            print(self.X[v] == self.X[v+1])

        self.Y['syn'] = data_Y[-1].astype(dtype=np.int64).copy()

        self.num_classes = (data_Y[0].max().astype(dtype=np.int64)) + 1
        self.dims = self.get_dims()  # 每个模态维度

    def __getitem__(self, index):
        data = dict()
        target = dict()  # !!
        for v_num in range(self.num_views):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
            target[v_num] = self.Y[v_num][index]
        data['index'] = self.X['index'][index]
        target['syn'] = self.Y['syn'][index]
        return data, target, index

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])
        return np.array(dims)

    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def postprocessing(self, index, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False,
                       ratio_conflicts=[0,0,0,0,0,0,0],view=None):
        if addNoise:
            self.addNoise(index, ratio_noise, sigma=sigma,view=view)
        if addConflict:
            for view, ratio_conflict in enumerate(ratio_conflicts):
                self.addConflictView(index, ratio_conflict, view)
        pass

    def addNoise(self, index, ratio, sigma, view=None):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            views = np.random.choice(np.array(self.num_views), size=np.random.randint(self.num_views), replace=False)
            if view==None:
                for v in views:
                    self.X[v][i] = np.random.normal(self.X[v][i], sigma)
            else:
                self.X[view][i] = np.random.normal(self.X[view][i], sigma)
        pass

    def addConflictView(self, index, ratio, view):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            self.Y[view][i] = np.random.randint(self.num_classes)

        pass
