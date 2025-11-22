import torch
import copy
import random
import numpy as np
import torch.nn as nn
from sklearn import metrics
from args.args import args_parser
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


args = args_parser()

# ---------------------Contribution: GMFAC----------------------------
def model_aggregation(weights, models):
    num_clients = len(models)
    total_weight = sum(np.abs(weights))
    normalized_weights = [(np.abs(weight) / total_weight) for weight in weights]
    print(f'Fed_Utils_Normalized_Weights:{normalized_weights}')
    
    aggregated_model = {}
    model_params = models[0].keys()
    
    flops = 0  
    for param in model_params:
            aggregated_param_value = torch.zeros_like(models[0][param], dtype=torch.float)
            for i in range(num_clients):
                flops += models[i][param].numel() 
                aggregated_param_value += normalized_weights[i] * models[i][param]
            aggregated_model[param] = aggregated_param_value
    print(f"[model_aggregation] Total FLOPs: {flops}")
    return aggregated_model

def model_global_eval(model_g, test_dataset, task_id, task_size, device):
    model_to_device(model_g, False, device)
    model_g.eval()
    test_dataset.getTestData([0, task_size * (task_id + 1)])
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=args.batch_size,drop_last=True)
    correct, total = 0, 0
    all_predicts = []
    all_labels = []
    for setp, (indexs, features, labels) in enumerate(test_loader):
        features, labels = features.cuda(device), labels.cuda(device)
        with torch.no_grad():
            outputs = model_g(features)
        outputs = torch.sigmoid(outputs)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
        all_predicts.extend(predicts)
        all_labels.extend(labels.cpu().numpy())
    all_predicts = torch.tensor(all_predicts).cpu().numpy()
    all_labels = torch.tensor(all_labels).cpu().numpy()
    accuracy = 100 * correct / total
    cm = confusion_matrix(all_labels, all_predicts)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)  
    plt.title("Confusion Matrix")
    plt.savefig(f'confusion_matrix_task_{task_id}.png')
    plt.show()
    Precision = metrics.precision_score(all_labels, all_predicts, average='weighted',zero_division=1)
    Recall = metrics.recall_score(all_labels, all_predicts, average='weighted')
    F1_Score = 2 * Precision * Recall / (Precision + Recall)
    model_g.train()

    return accuracy,Precision*100, Recall*100, F1_Score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def model_to_device(model, parallel, device):
    if parallel:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        if device == -1:
            model.to(torch.device("cpu"))
        else:
            model.to(torch.device("cuda:{}".format(device)))
    return model

def participant_buffer_storing(clients, num_clients, model_g, old_client, task_id, clients_index):
    for index in range(num_clients):
        clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0)
            else:
                clients[index].beforeTrain(task_id, 1)
            clients[index].update_exemplar_set()

def local_train(clients, index, model_g, task_id, model_old, ep_g, old_client):
    clients[index].model = copy.deepcopy(model_g)
    if index in old_client:
        clients[index].beforeTrain(task_id, 0)
    else:
        clients[index].beforeTrain(task_id, 1)
    clients[index].update_exemplar_set(ep_g,index)
    print(f'Fed_Utils Detecting New Class Signal:{clients[index].signal}')
    clients[index].train(model_old)
    local_model = clients[index].model.state_dict()
    print('*' * 60)

    return local_model, clients[index].weight 