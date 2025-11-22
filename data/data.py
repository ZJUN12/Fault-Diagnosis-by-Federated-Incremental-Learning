import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class SCADA_data(Dataset):
    def __init__(self,windfarm):
        self.windfarm = windfarm
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        self.read_data()
        
    def read_data(self):
        Data = pd.read_csv(f'./2. Federated_Class_Incremental_for_Diagnosis/dataset/{self.windfarm}/Single_WT/{self.windfarm}_Sampled_Data_Features_delete_faulttype1349.csv')
        scaler = MinMaxScaler()
        Data.iloc[:, :-1] = scaler.fit_transform(Data.iloc[:, :-1])
        Data = np.array(Data)
        np.random.shuffle(Data)
        self.features = Data[:,:-1]
        self.targets = Data[:,-1]
        
    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label
    
    def getTestData(self, classes):
        print("Classes:", classes)
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.features[np.array(self.targets) == label]
            datas.append(data)  
            labels.append(np.full((data.shape[0]), label))
        self.TestData, self.TestLabels = self.concatenate(datas, labels)


    def getTrainData(self, classes, exemplar_set, exemplar_label_set):
        datas,labels=[],[]
        if len(exemplar_set)!=0 and len(exemplar_label_set)!=0:
            datas=[exemplar for exemplar in exemplar_set]
            length=len(datas[0])
            labels=[np.full((length), label) for label in exemplar_label_set]
        for label in classes:
            data=self.features[np.array(self.targets)==label]
            datas.append(data)
            labels.append(np.full((data.shape[0]),label))
        self.TrainData, self.TrainLabels=self.concatenate(datas,labels)

    def getSampleData(self, classes, exemplar_set, exemplar_label_set, group):
        datas,labels=[],[]
        if len(exemplar_set)!=0 and len(exemplar_label_set)!=0:
            datas=[exemplar for exemplar in exemplar_set]
            length=len(datas[0])
            labels=[np.full((length), label) for label in exemplar_label_set]

        if group == 0:
            for label in classes:
                data=self.data[np.array(self.targets)==label]
                datas.append(data)
                labels.append(np.full((data.shape[0]),label))
        self.TrainData, self.TrainLabels=self.concatenate(datas,labels)

    def getTrainItem(self,index):
        features, target = self.TrainData[index], self.TrainLabels[index]

        return index,features,target
    
    def getTestItem(self,index):
        features, target = self.TestData[index], self.TestLabels[index]

        return index,features,target

    def __getitem__(self, index):
        if self.TrainData!=[]:
            return self.getTrainItem(index)
        elif self.TestData!=[]:
            return self.getTestItem(index)
    
    def __len__(self):
        if self.TrainData!=[]:
            return len(self.TrainData)
        elif self.TestData!=[]:
            return len(self.TestData)

    def get_features_class(self,label):
        return self.features[np.array(self.targets)==label]