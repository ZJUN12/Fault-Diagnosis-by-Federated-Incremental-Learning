from algorithm.FIFD import FIFD_model
from model.local_model import *
import copy
import random
import os.path as osp
import os
from utils.utils import * 
from data.data import SCADA_data
from args.args import args_parser

args = args_parser()

model_dict = {
    'Mobile': MobileNet,
}

feature_extractor = model_dict[args.local_model]()
num_clients = args.num_clients
old_client_0 = []
old_client_1 = [i for i in range(args.num_clients)]
models = []
 
setup_seed(args.seed)

model_g = network(args.numclass, feature_extractor)
model_g = model_to_device(model_g, False, args.device)
model_old = None

trainset_1 = SCADA_data('hubei')
trainset_2 = SCADA_data('xuzhou')
trainset_3 = SCADA_data('yunnan')
testset = SCADA_data('3windfarm')

if args.method == 'FIFD':
    for i in range(3):
        current_dataset = globals()[f'trainset_{i+1}']
        model_temp = FIFD_model(args.numclass, feature_extractor, args.batch_size, args.task_size, args.memory_size,
                    args.epochs_local, args.learning_rate, current_dataset, args.device)
        models.append(model_temp)
        
output_dir = osp.join(f'Diagnosis_Results')
if not osp.exists(output_dir):
    os.system('mkdir -p ' + output_dir)
if not osp.exists(output_dir):
    os.mkdir(output_dir)

out_file = open(osp.join(output_dir, 'Test' + '_' + str(args.method)+ '_' + 'Local_Model:' + str(args.local_model) + '.txt'), 'w')
log_str = 'method:{}, learning_rate:{}, exemplar_set:{}, epoch_global:{}, tasks_global:{}, balance_factor:{}, local_epochs:{}, batch_size:{}'.format(args.method, args.learning_rate, args.memory_size, args.epochs_global, args.tasks_global, args.alpha, args.epochs_local, args.batch_size)
out_file.write(log_str + '\n')
out_file.flush()

classes_learned = args.task_size
old_task_id = -1
for ep_g in range(args.epochs_global):
    model_old = copy.deepcopy(model_g)
    task_id = ep_g // args.tasks_global

    if task_id != old_task_id and old_task_id != -1:
        overall_client = len(old_client_0) + len(old_client_1)
        old_client_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.9))
        old_client_0 = [i for i in range(overall_client) if i not in old_client_1]
        num_clients = len(old_client_1) + len(old_client_0)
        print(f'old_client_0:{old_client_0}')

    if task_id != old_task_id and old_task_id != -1:
        classes_learned += args.task_size
        # update the dimension of last layer
        model_g.Incremental_learning(classes_learned)
        model_g = model_to_device(model_g, False, args.device)
    
    print('federated global round: {}, task_id: {}'.format(ep_g, task_id))
 
    w_local = []
    weights = []
    clients_index = [0,1,2]

    for c in clients_index:
        local_model, weight = local_train(models, c, model_g, task_id, model_old, ep_g, old_client_0)
        w_local.append(local_model)
        weights.append(weight)  
    print('every participant start updating their buffer set and old model...')
    participant_buffer_storing(models, num_clients, model_g, old_client_0, task_id, clients_index)
    print('updating finishes')
    print('federated aggregation...')
    if args.method == 'FIFD':
        # ----------Contribution: GMFAC--------------------
        aggregated_model = model_aggregation(weights, w_local)


    # updating global model 
    model_g.load_state_dict(aggregated_model, strict=False)
    acc_global,precision_global,recall_global,F1_score = model_global_eval(model_g, testset, task_id, args.task_size, args.device)
    log_str = 'Task: {},Round: {},Accuracy: {:.3f}%,Precision: {:.3f}%,Recall: {:.3f}%,F1_Score: {:.4f}'.format(task_id, ep_g, acc_global,precision_global,recall_global,F1_score)
    out_file.write(log_str + '\n')
    out_file.flush()
    print('classification accuracy of global model at round %d: %.3f' % (ep_g, acc_global))
    print('classification recall of global model at round %d: %.3f' % (ep_g, recall_global))
    print('classification precision of global model at round %d: %.3f' % (ep_g, precision_global))
    print('classification F1_Score of global model at round %d: %.3f' % (ep_g, F1_score)) 
    print()
    old_task_id = task_id
