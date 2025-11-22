import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='FIFD', help="name of method")
    parser.add_argument(
    '--local_model',
    type=str,
    default='Mobile',
    choices=['ResNet18', 'ResNet34', 'ResNet50', 'Mobile', 'Efficient', 'MLP'],
    help="Choose the local model: R (ResNet), MLP, DSC, Squeeze, Mobile, or Efficient"
)
    parser.add_argument('--numclass', type=int, default=1, help="number of data classes in the first task")
    parser.add_argument('--device', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--batch_size', type=int, default=128, help='size of mini-batch')
    parser.add_argument('--task_size', type=int, default=1, help='number of data classes each task')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--memory_size', type=int, default=2000, help='size of exemplar memory')
    parser.add_argument('--epochs_local', type=int, default=15, help='local epochs of each global round')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_clients', type=int, default=3, help='initial number of clients')
    parser.add_argument('--local_clients', type=int, default=3, help='number of selected clients each round')
    parser.add_argument('--epochs_global', type=int, default=6, help='total number of global rounds')
    parser.add_argument('--tasks_global', type=int, default=1, help='total number of tasks')
    parser.add_argument('--alpha',type=float,default=0.9, help='the weight of balance class loss function')
    parser.add_argument('--temp_param',type=float,default=6, help='the temperature parameter for distillation')
    args = parser.parse_args()
    return args