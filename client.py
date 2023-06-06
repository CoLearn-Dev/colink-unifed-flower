import warnings
from collections import OrderedDict
import flbenchmark.datasets
import flbenchmark.logging
import os
import sys
import json
import glob
import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import cv2
import base64
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ast
import socket

config = json.load(open(sys.argv[1], 'r'))

# load dataset
flbd = flbenchmark.datasets.FLBDatasets('~/flbenchmark.working/data')
val_dataset = None
if config['dataset'] == 'reddit':
    train_dataset, test_dataset, val_dataset = flbd.leafDatasets(config['dataset'])
elif config['dataset'] == 'femnist':
    train_dataset, test_dataset = flbd.leafDatasets(config['dataset'])
else:
    train_dataset, test_dataset = flbd.fateDatasets(config['dataset'])
train_data_base = '~/flbenchmark.working/data/'+config['dataset']+'_train'
test_data_base = '~/flbenchmark.working/data/'+config['dataset']+'_test'
val_data_base = '~/flbenchmark.working/data/'+config['dataset']+'_val'
flbenchmark.datasets.convert_to_csv(train_dataset, out_dir=train_data_base)
if test_dataset is not None:
    flbenchmark.datasets.convert_to_csv(test_dataset, out_dir=test_data_base)
if val_dataset is not None:
    flbenchmark.datasets.convert_to_csv(val_dataset, out_dir=val_data_base)

if config['dataset'] == 'reddit':
    num_class = 0
    input_len = 0
    inplanes = 0
elif config['dataset'] == 'femnist':
    num_class = 62
    input_len = (28, 28)
    inplanes = 1
elif config['dataset'] == 'celeba':
    num_class = 2
    input_len = (218, 178)
    inplanes = 3
elif config['dataset'] == 'breast_horizontal':
    num_class = 2
    input_len = 30
    inplanes = 0
elif config['dataset'] == 'default_credit_horizontal':
    num_class = 2
    input_len = 23
    inplanes = 0
elif config['dataset'] == 'give_credit_horizontal':
    num_class = 2
    input_len = 10
    inplanes = 0
elif config['dataset'] == 'student_horizontal':
    num_class = 1
    input_len = 13
    inplanes = 0
elif config['dataset'] == 'vehicle_scale_horizontal':
    num_class = 4
    input_len = 18
    inplanes = 0
else:
    raise NotImplementedError('Dataset {} is not supported.'.format(config['dataset']))

if config['model'].startswith('mlp_'):
    if config['dataset'] == 'femnist' or config['dataset'] == 'celeba':
            input_len = inplanes * input_len[0] * input_len[1]
    sp = config['model'].split('_')
    if len(sp) < 2 or len(sp) > 4:
        raise NotImplementedError('Model {} is not supported.'.format(config['model']))
elif config['model'] == 'linear_regression' or config['model'] == 'logistic_regression':
    if config['dataset'] == 'femnist' or config['dataset'] == 'celeba':
            input_len = inplanes * input_len[0] * input_len[1]
    sp = None
elif config['model'] == 'lenet':
    if config['dataset'] != 'femnist' and config['dataset'] != 'celeba':
        raise NotImplementedError('Dataset {} is not supported for {}.'.format(config['dataset'], config['model']))
    sp = None
elif config['model'] == 'lstm':
    if config['dataset'] != 'reddit':
        raise NotImplementedError('Dataset {} is not supported for {}.'.format(config['dataset'], config['model']))
    sp = None
else:
    raise NotImplementedError('Model {} is not supported.'.format(config['model']))

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Build vocab
vocab_size = 10000
embedding_dim = 160
hidden_dim = 512
oov_tok = '<OOV>'
max_length = 25
trunc_type= 'post'
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

def load_data():
    if config['dataset'] == 'reddit':
        # Dataset Pre-processing, using the same (slightly changed) code as in fedlearner
        def load_data(split, use_first_k=None):
            use_first_k = None
            with open('~/flbenchmark.working/data/reddit_%s/_main.json'%split) as inf:
                meta_info = json.load(inf)
                parties = meta_info['parties']
            if use_first_k is not None:
                parties = parties[:use_first_k]
            print ("Number of %s parties:"%split, len(parties))

            all_data = {pid:[] for pid in parties}
            for pid in parties:
                df = pd.read_csv('~/flbenchmark.working/data/reddit_%s/%s.csv'%(split, pid))
                for _, row in df.iterrows():
                    cur_frame = ast.literal_eval(row['x0'])
                    cur_x = [tok for sent in cur_frame for tok in sent if tok != '<PAD>']

                    all_data[pid].append(cur_x)
            return all_data
        
        def text_to_seq(tokenizer, data, max_length, trunc_type):
            seq_data = {}
            for pid, one_data in data.items():
                seq_data[pid] = torch.LongTensor(pad_sequences(tokenizer.texts_to_sequences(one_data), maxlen=max_length, truncating=trunc_type))
            return seq_data
        
        train_data = load_data('train')
        test_data = load_data('test')
        all_users = list(train_data.keys())
        all_train_data = [seq for one_data in train_data.values() for seq in one_data]
        tokenizer.fit_on_texts(all_train_data)
        oov_idx = tokenizer.word_index['<OOV>']
        train_data_seq = text_to_seq(tokenizer, train_data, max_length, trunc_type)
        test_data_seq = text_to_seq(tokenizer, test_data, max_length, trunc_type)
        assert set(train_data_seq.keys()) == set(test_data_seq.keys())
        all_test_seq = torch.stack([seq for one_seq in test_data_seq.values() for seq in one_seq], 0)
        return train_data_seq[all_users[int(sys.argv[2])-1]], all_test_seq, oov_idx

    if config['dataset'] == 'femnist' or config['dataset'] == 'celeba':
        x = {}
        y = {}
        x["train"] = []
        x["test"] = []
        y["train"] = []
        y["test"] = []
        for dir_path in ["train", "test"]:
            data_paths = sorted(glob.glob(os.path.join(os.path.expanduser('~'), f'flbenchmark.working/data/{config["dataset"]}_{dir_path}/*.csv')))
            if dir_path == 'train':
                data_paths = data_paths[int(sys.argv[2])-1:int(sys.argv[2])]
            for data_path in data_paths:
                data = pd.read_csv(data_path, sep=',')
                if config['dataset'] == 'celeba':
                    X = data.iloc[:, 1].to_list()
                    for i in range(len(X)):
                        img_data = base64.b64decode(X[i])
                        img_uint8 = np.asarray(bytearray(img_data), dtype='uint8')
                        img_bgr = cv2.imdecode(img_uint8, cv2.IMREAD_COLOR)
                        X[i] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).reshape(-1)
                    x[dir_path].append(np.array(X).astype(np.float32))
                else:
                    x[dir_path].append(np.array(data.iloc[:, 1:]).astype(np.float32))
                y[dir_path].append(np.array(data.iloc[:, 0]).astype(np.int32))
        train_X = x["train"][0]
        train_y = y["train"][0]
        test_X = np.concatenate(x["test"], axis=0)
        test_y = np.concatenate(y["test"], axis=0)
    else:
        client_name = 'guest' if int(sys.argv[2]) == 1 else 'host'
        if config['dataset'] == 'default_credit_horizontal' and int(sys.argv[2]) == 2:
            client_name += '_1'
        train_data = pd.read_csv(f'~/flbenchmark.working/data/{config["dataset"]}_train/{config["dataset"].replace("horizontal", "homo")}_{client_name}.csv', sep=',')
        if config["dataset"] == 'vehicle_scale_horizontal':
            train_X = np.array(train_data.iloc[:, 2:]).astype(np.float32)
            train_y = np.array(train_data.y).astype(np.int32)
            test_data = pd.read_csv(f'~/flbenchmark.working/data/{config["dataset"]}_train/{config["dataset"].replace("horizontal", "homo")}_guest.csv', sep=',')
            test_X = np.array(test_data.iloc[:, 2:]).astype(np.float32)
            test_y = np.array(test_data.y).astype(np.int32)
            test_data = pd.read_csv(f'~/flbenchmark.working/data/{config["dataset"]}_train/{config["dataset"].replace("horizontal", "homo")}_host.csv', sep=',')
            test_X = np.concatenate((test_X, np.array(test_data.iloc[:, 2:]).astype(np.float32)), axis=0)
            test_y = np.concatenate((test_y, np.array(test_data.y).astype(np.int32)), axis=0)
        else:
            test_data = pd.read_csv(f'~/flbenchmark.working/data/{config["dataset"]}_test/{config["dataset"].replace("horizontal", "homo")}_test.csv', sep=',')
            if config['dataset'] == 'student_horizontal':
                train_X = np.array(pd.concat([train_data.iloc[:, 9:], train_data.iloc[:, 1:8]], axis=1)).astype(np.float32)
                train_y = np.array(train_data.y).astype(np.float32)
                test_X = np.array(pd.concat([test_data.iloc[:, 9:], test_data.iloc[:, 1:8]], axis=1)).astype(np.float32)
                test_y = np.array(test_data.y).astype(np.float32)
            else:
                train_X = np.array(train_data.iloc[:, 2:]).astype(np.float32)
                train_y = np.array(train_data.y).astype(np.int32)
                test_X = np.array(test_data.iloc[:, 2:]).astype(np.float32)
                test_y = np.array(test_data.y).astype(np.int32)

    """Load dataset (training and test set)."""
    trainset = TensorDataset(torch.Tensor(train_X), torch.Tensor(train_y))
    testset = TensorDataset(torch.Tensor(test_X), torch.Tensor(test_y))
    return DataLoader(trainset, batch_size=config['training_param']['batch_size'], shuffle=True), DataLoader(testset, batch_size=test_y.shape[0])

if config['dataset'] == 'reddit':
    trainloader, testloader, oov_idx = load_data()
else:
    trainloader, testloader = load_data()

if config['bench_param']['mode'] == 'local' and config['dataset'] in ['celeba', 'femnist', 'reddit', 'sent140', 'shakespeare', 'synthetic']:
    torch.set_num_threads(1)

class Net(nn.Module):
    '''The standard PyTorch model we want to federate'''

    def __init__(self) -> None:
        super(Net, self).__init__()
        if config['model'] == 'linear_regression' or config['model'] == 'logistic_regression':
            self.fc = nn.Linear(input_len, num_class)
        elif sp:
            self.fc1 = nn.Linear(input_len, int(sp[1]))
            self.fc2 = nn.Linear(int(sp[1]), int(sp[2])) if len(sp) > 2 else nn.Linear(int(sp[1]), num_class)
            self.fc3 = nn.Linear(int(sp[2]), int(sp[3])) if len(sp) > 3 else (nn.Linear(int(sp[2]), num_class) if len(sp) > 2 else None)
            self.fc4 = nn.Linear(int(sp[3]), num_class) if len(sp) > 3 else None
        elif config['model'] == 'lenet':
            self.crop = transforms.CenterCrop((178, 178))
            self.resize = transforms.Resize((28, 28))
            self.conv1 = nn.Conv2d(inplanes, 6, 5, padding=2, padding_mode='reflect')
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_class)
        elif config['model'] == 'lstm':
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, 1, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if sp:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x)) if len(sp) > 2 else self.fc2(x)
            x = F.relu(self.fc3(x)) if len(sp) > 3 else (self.fc3(x) if len(sp) > 2 else x)
            x = self.fc4(x) if len(sp) > 3 else x
        elif config['model'] == 'lenet':
            x = x.reshape(-1, inplanes, input_len[0], input_len[1])
            if config['dataset'] == 'celeba':
                x = self.resize(self.crop(x))
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        elif config['model'] == 'lstm':
            x = self.embedding(x)
            x, _ = self.lstm(x)
            x = self.fc(x)
        else:
            x = F.sigmoid(self.fc(x)) if config['model'] == 'logistic_regression' else self.fc(x)

        return x.reshape(-1) if config['dataset'] == 'student_horizontal' else x


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.MSELoss() if config['dataset'] == 'student_horizontal' else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=config['training_param']['learning_rate'], momentum=config['training_param']['optimizer_param']['momentum'])
    net.train()
    logger = flbenchmark.logging.Logger(id=int(sys.argv[2]), agent_type='client')
    logger.training_round_start()
    for _ in range(epochs):
        logger.computation_start()
        if config['dataset'] == 'reddit':
            perm = np.random.permutation(len(trainloader))
            train_data_shuffle = trainloader[perm]
            st = 0
            while st < len(train_data_shuffle):
                ed = min(st+config['training_param']['batch_size'], len(train_data_shuffle))
                inp = train_data_shuffle[st:ed]
                x, y = inp[:,:-1], inp[:,1:]
                pred = net(x)

                pred_tgt = pred.view(-1, vocab_size)
                y_tgt = y.reshape(-1)
                # Remove <OOV> and <PAD>
                pred_tgt = pred_tgt[torch.logical_and(y_tgt!=oov_idx, y_tgt!=0)]
                y_tgt = y_tgt[torch.logical_and(y_tgt!=oov_idx, y_tgt!=0)]
                loss = criterion(pred_tgt, y_tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                st = ed
        else:
            for features, labels in tqdm(trainloader):
                optimizer.zero_grad()
                if config['dataset'] == 'student_horizontal':
                    criterion(net(features.to(DEVICE)), labels.float().to(DEVICE)).backward()
                else:
                    criterion(net(features.to(DEVICE)), labels.long().to(DEVICE)).backward()
                optimizer.step()
        logger.computation_end()
    logger.training_round_end()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.MSELoss() if config['dataset'] == 'student_horizontal' else torch.nn.CrossEntropyLoss()
    net.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        if config['dataset'] == 'reddit':
            st = 0
            while st < len(testloader):
                ed = min(st+config['training_param']['batch_size'], len(testloader))
                inp = testloader[st:ed]
                x, y = inp[:,:-1], inp[:,1:]
                pred = net(x)

                pred_tgt = pred.view(-1, vocab_size)
                y_tgt = y.reshape(-1)
                # Remove <OOV> and <PAD>
                pred_tgt = pred_tgt[torch.logical_and(y_tgt!=oov_idx, y_tgt!=0)]
                y_tgt = y_tgt[torch.logical_and(y_tgt!=oov_idx, y_tgt!=0)]
                loss += criterion(pred_tgt, y_tgt) * len(y_tgt)
                correct += (torch.max(pred_tgt.data, 1)[1] == y_tgt).sum().item()
                total += len(y_tgt)
                st = ed
        else:
            for features, labels in tqdm(testloader):
                outputs = net(features.to(DEVICE))
                labels = labels.to(DEVICE)
                total += labels.size(0)
                if config['dataset'] == 'student_horizontal':
                    loss += criterion(outputs, labels.float()).item()
                elif config['dataset'] == 'femnist' or config['dataset'] == 'vehicle_scale_horizontal':
                    loss += criterion(outputs, labels.long()).item()
                    correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                else:
                    loss += criterion(outputs, labels.long()).item()
    if config['dataset'] == 'student_horizontal':
        target_metric = loss / len(testloader.dataset)
    elif config['dataset'] == 'femnist' or config['dataset'] == 'celeba' or config['dataset'] == 'vehicle_scale_horizontal':
        target_metric = correct / total
    elif config['dataset'] != 'reddit':
        target_metric = json.dumps({'labels': labels.tolist(), 'logits': outputs.data[:, 1].tolist()})
    if config['dataset'] == 'reddit':
        return loss / total, correct / total
    else:
        return loss / len(testloader.dataset), target_metric


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

net = Net().to(DEVICE)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, target_metric = test(net, testloader)
        return loss, len(testloader.dataset), {"target_metric": target_metric}

while True:
    if socket.socket().connect_ex((f"{sys.argv[3]}", 8080)) == 0:
        break
    time.sleep(1)
# Start Flower client
fl.client.start_numpy_client(server_address=f"{sys.argv[3]}:8080", client=FlowerClient(), client_index=int(sys.argv[2]))
