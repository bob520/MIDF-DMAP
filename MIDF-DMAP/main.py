import logging
import os
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import paddle
from config import my_dataset, ALPHA, MY_SCHEDULER, PATIENCE, LR, EPOCH, dataset_path, D3_INFO_PATH, \
    BATCH_SIZE, OPTIM, USE_CUDA
from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config
from train_util.Focal_Loss import focal_loss
from train_util.fusion import History, model_forward
from train_util.model import theModel_final
from train_util.dataset import MYDataset
from train_util.featurizer import DownstreamCollateFn
from train_util.utils_d3 import _save_npz_data, load_bioactivity_dataset
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, sentences2vec, MolSentence
from rdkit import Chem
import pickle
from train_util.utils import get_2d
import pandas as pd
from sklearn.model_selection import train_test_split
from pahelix.datasets import InMemoryDataset
from train_util.featurizer import DownstreamTransformFn
import warnings
warnings.filterwarnings("ignore")

def create_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(
        filename=f"./{my_dataset}_Intermediate/train.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def eval(model,test_dataloader,geo_loader_test,compound_encoder,norm,loss_fn,loss_fn2,test_data_size,y_true_test,logger,test_auc,epoch_num,mode):

    test_epoch_true = 0
    epoch_test_y_predict = []
    predict_proba_test = []
    model.eval()
    with torch.no_grad():
        logger.info(f"The {mode} section begins...")
        for data, geo_data in zip(test_dataloader, geo_loader_test):
            gram, mol_vec, targets, index = data
            atom_bond_graphs, bond_angle_graphs = geo_data
            node_repr, edge_repr, graph_repr = compound_encoder(atom_bond_graphs.tensor(), bond_angle_graphs.tensor())
            graph_repr = norm(graph_repr)
            _, outputs, _ = model_forward(model, loss_fn, loss_fn2, gram, mol_vec, graph_repr, targets, index,mode="eval")
            if USE_CUDA:
                outputs=outputs.cpu().detach()
            list_auc = outputs[:, 1]
            predict_proba_test.extend(list_auc.numpy())
            y_predict = outputs.argmax(-1)
            epoch_test_y_predict.extend(y_predict)
            test_accuracy_num = (y_predict == targets).sum()
            test_epoch_true += test_accuracy_num
    test_epoch_acc = test_epoch_true / test_data_size
    p, r, f, s = precision_recall_fscore_support(y_true_test, epoch_test_y_predict, average=None)
    cm = confusion_matrix(y_true_test, epoch_test_y_predict, normalize='true')
    score = np.array(predict_proba_test)
    test_auc_epoch = roc_auc_score(y_true_test, score)
    test_auc.append(test_auc_epoch)
    logger.info(
        f"{mode} set completed，The current number of epoch is {epoch_num + 1}，The total number of hits on the {mode} set is{test_epoch_true}/{test_data_size}，acc is{test_epoch_acc}"
        f"precious is {p}，recall is {r}，f1 is {f}，AUC is {test_auc_epoch}")
    logger.info(f"The valid set confusion matrix is:\n{cm}")
    # macro
    p, r, f, s = precision_recall_fscore_support(y_true_test, epoch_test_y_predict, average="macro")
    logger.info(f"macro precious is {p}，macro recall is {r}，macro f1 is {f}")
    if mode=="valid":
        return test_auc_epoch
def train_begin():
    LOSS_FUN = "focal"
    compound_encoder_path = "./train_util/class.pdparams"
    logger = create_logger()
    logger.info("batch_size:{},lr:{},alpha:{}".format(BATCH_SIZE, LR, ALPHA))
    compound_encoder_config = load_json_config("./train_util/gnnconfig.json")
    #The GeoGNNModel model of the PaddlePaddle platform is used here to process the three-dimensional molecular information
    compound_encoder = GeoGNNModel(compound_encoder_config)
    compound_encoder.set_state_dict(paddle.load(compound_encoder_path))
    train_dataset = InMemoryDataset(
        npz_data_files=[f"./{my_dataset}_Intermediate/X_train_d3.npz"])
    valid_dataset = InMemoryDataset(
        npz_data_files=[f"./{my_dataset}_Intermediate/X_valid_d3.npz"])
    collate_fn = DownstreamCollateFn(
        atom_names=compound_encoder_config['atom_names'],
        bond_names=compound_encoder_config['bond_names'],
        bond_float_names=compound_encoder_config['bond_float_names'],
        bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
        task_type='class', is_inference=True)
    geo_loader_train = train_dataset.get_data_loader(
        batch_size=BATCH_SIZE,
        num_workers=1,
        collate_fn=collate_fn)
    geo_loader_valid = valid_dataset.get_data_loader(
        batch_size=BATCH_SIZE,
        num_workers=1,
        collate_fn=collate_fn)
    model = theModel_final()
    # if my_dataset =="BBBP":
    #     model=torch.load(f"./xxx.pth")
    #whether or not use cuda（1）
    if USE_CUDA:
        model.cuda()
        print("The CUDA device currently in use is device ", {torch.cuda.current_device()})
    train_set = MYDataset(f"./{my_dataset}_Intermediate/X_train_d1.pkl",
                          f"./{my_dataset}_Intermediate/X_train_d2.pkl",
                          f"./{my_dataset}_Intermediate/y_train.pkl")
    valid_set = MYDataset(f"./{my_dataset}_Intermediate/X_valid_d1.pkl",
                         f"./{my_dataset}_Intermediate/X_valid_d2.pkl",
                         f"./{my_dataset}_Intermediate/y_valid.pkl")
    train_data_size = len(train_set)
    valid_data_size = len(valid_set)
    train_need_batch = train_data_size / BATCH_SIZE
    valid_need_batch = valid_data_size / BATCH_SIZE
    logger.info("The trainset size is {}, and a {} batch is required to complete the training".format(train_data_size, train_need_batch))
    logger.info("The validset size is {}, and a {} batch is required to complete the validing".format(valid_data_size, valid_need_batch))

    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE)
    valid_dataloader = DataLoader(valid_set, batch_size=BATCH_SIZE)
    if LOSS_FUN == 'focal':
        loss_fn = focal_loss(alpha=ALPHA, gamma=2, num_classes=2)
        loss_fn2 = focal_loss(alpha=ALPHA, gamma=2, num_classes=2,size_average="BatchSize")
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_fn2 = torch.nn.CrossEntropyLoss(reduction=None)
    # weight_decay=0.0001
    if OPTIM == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    elif OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if MY_SCHEDULER==True:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH/3, eta_min=0.0005)
    #read label
    with open(f"./{my_dataset}_Intermediate/y_train.pkl", "rb") as f:
        y_true_train = pickle.load(f)
    with open(f"./{my_dataset}_Intermediate/y_valid.pkl", "rb") as f:
        y_true_valid = pickle.load(f)
    norm = paddle.nn.LayerNorm(compound_encoder.graph_dim)

    d1_history = History(len(train_dataloader.dataset))
    d3_history = History(len(train_dataloader.dataset))
    gram_history = History(len(train_dataloader.dataset))
    # The definitions of these variables must be placed before the start of training
    max_auc = 0
    patience = 0
    valid_auc=[]
    for i in range(EPOCH):
        train_epoch_loss = 0
        train_epoch_true = 0
        predict_proba_train = []
        epoch_train_y_predict = []
        logger.info("------- Training for round {} begins -------".format(i + 1))
        logger.info("> > > >train< < < <")
        model.train()
        for data, geo_data in zip(train_dataloader, geo_loader_train):
            atom_bond_graphs, bond_angle_graphs = geo_data
            node_repr, edge_repr, graph_repr = compound_encoder(atom_bond_graphs.tensor(), bond_angle_graphs.tensor())
            graph_repr = norm(graph_repr)
            gram,mol_vec, targets,index = data
            loss, outputs, _ = model_forward(model,loss_fn,loss_fn2,gram,mol_vec,graph_repr,targets,index,d1_history,d3_history,gram_history,mode="train")
            if USE_CUDA:
                outputs=outputs.cpu().detach()
            list_auc = outputs[:, 1]
            predict_proba_train.extend(list_auc.detach().numpy())
            train_epoch_loss = train_epoch_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if MY_SCHEDULER == True:
                scheduler.step()
            y_predict = outputs.argmax(-1)
            epoch_train_y_predict.extend(y_predict)
            accuracy_num = (y_predict == targets).sum()
            train_epoch_true += accuracy_num
        train_epoch_acc = train_epoch_true / train_data_size
        p, r, f, s = precision_recall_fscore_support(y_true_train, epoch_train_y_predict, average=None)
        score = np.array(predict_proba_train)
        train_auc_epoch = roc_auc_score(y_true_train, score)
        logger.info(f"Training set completed，The current number of epoch is {i + 1}，The total number of hits on the training set is {train_epoch_true}/{train_data_size}，acc"
                    f" is {train_epoch_acc}，precious is {p}，recall is {r}，f1 is {f},AUC is {train_auc_epoch}")
        # macro
        p, r, f, s = precision_recall_fscore_support(y_true_train, epoch_train_y_predict, average="macro")
        logger.info(f"macro precious is {p}，macro recall is {r}，macro f1 is {f}")
        cm = confusion_matrix(y_true_train, epoch_train_y_predict, normalize='true')
        logger.info(f"The train set confusion matrix is:\n{cm}")

        #valid begin
        valid_auc_epoch=eval(model,test_dataloader=valid_dataloader,geo_loader_test=geo_loader_valid,compound_encoder=compound_encoder,
             norm=norm,loss_fn=loss_fn,loss_fn2=loss_fn2,test_data_size=valid_data_size,y_true_test=y_true_valid,logger=logger,test_auc=valid_auc,epoch_num=i,mode="valid")
        if valid_auc_epoch > max_auc:
            patience  = 0
            max_auc = valid_auc_epoch
            logger.info(f"The current AUC is the best. The current number of rounds is {i+1}, store the model for this round")
            torch.save(model,f"./{my_dataset}_Intermediate/save_model/best_auc_model.pth")
        else:
            patience+=1
            logger.info(f"The current tolerance count is {patience}")
        if patience==PATIENCE:
            logger.info(f"Continuous {patience} epochs of AUC without improvement, stop training")
            break
        logger.info(f"The maximum AUC value is {max_auc}")




if __name__ == '__main__':
    if not os.path.exists(f"./{my_dataset}_Intermediate"):
        os.makedirs(f"./{my_dataset}_Intermediate/save_model")
    df = pd.read_excel(dataset_path)
    print("The total size of this dataset is ", len(df))
    # smiles trans to ecfp
    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df['ecfp'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
    # Training or using pre trained molecular word embedding models
    model = word2vec.Word2Vec.load('./Dependencies/model_300dim.pkl')
    list_mol = []
    for x in sentences2vec(df['ecfp'], model, unseen='UNK'):
        list_mol.append(x)
    label_list = df['label'].to_list()
    if my_dataset == "BACE" or my_dataset == "BBBP" or my_dataset == "BBBPuseScaffold" or my_dataset == "BACEuseScaffold":
        print("This dataset used splitter is scaffold")
        train_end = df[df['split'] == 'train'].index[-1]
        valid_end = df[df['split'] == 'valid'].index[-1]

        # +1 is because the slicing operation is left closed and right open
        X_train = list_mol[:train_end + 1]
        y_train = label_list[:train_end + 1]

        X_valid = list_mol[train_end + 1:valid_end + 1]
        y_valid = label_list[train_end + 1:valid_end + 1]

        X_test = list_mol[valid_end + 1:]
        y_test = label_list[valid_end + 1:]
    else:
        print("This dataset used splitter is random")
        X_train, X_valid_test, y_train, y_valid_test = train_test_split(list_mol, label_list, test_size=0.2,
                                                                        stratify=label_list,
                                                                        random_state=27)
        X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=0.5,
                                                            stratify=y_valid_test,
                                                            random_state=27)

    with open(f"./{my_dataset}_Intermediate/X_train_d1.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open(f"./{my_dataset}_Intermediate/X_valid_d1.pkl", "wb") as f:
        pickle.dump(X_valid, f)
    with open(f"./{my_dataset}_Intermediate/X_test_d1.pkl", "wb") as f:
        pickle.dump(X_test, f)
    with open(f"./{my_dataset}_Intermediate/y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
    with open(f"./{my_dataset}_Intermediate/y_valid.pkl", "wb") as f:
        pickle.dump(y_valid, f)
    with open(f"./{my_dataset}_Intermediate/y_test.pkl", "wb") as f:
        pickle.dump(y_test, f)
    # Obtain molecular fingerprints with a radius of 2 and further obtain more detailed information
    # Atoms exceeding the maximum range will be deleted. Prompt Outlier 119 has 122 atoms
    # It may cause errors when partitioning the dataset due to inconsistent lengths of X and y. Therefore, it is necessary to adjust the value of MAX-ATOMS based on the actual molecule with the most atoms in the molecular dataset
    get_2d(df, df['label'], my_dataset)
    if not D3_INFO_PATH:
        # These three steps will analyze the 3D features and obtain the part1.npz file, which is very slow.Please be patient and wait
        dataset = load_bioactivity_dataset(dataset_path)
        dataset.transform(DownstreamTransformFn(), num_workers=1)
        # The default storage flie is part1.npz
        dataset.save_data(f"./{my_dataset}_Intermediate/")
        data = dataset._load_npz_data_files([f"./{my_dataset}_Intermediate/part1.npz"])
    else:
        dataset = InMemoryDataset(
            npz_data_files=[D3_INFO_PATH])
        data = dataset._load_npz_data_files([D3_INFO_PATH])

    df = pd.read_excel(dataset_path)

    if my_dataset == "BACE" or my_dataset == "BBBP" or my_dataset == "BBBPuseScaffold" or my_dataset == "BACEuseScaffold":
        print("This dataset used splitter is scaffold")
        train_end = df[df['split'] == 'train'].index[-1]
        valid_end = df[df['split'] == 'valid'].index[-1]
        X_train = data[:train_end + 1]
        X_valid = data[train_end + 1:valid_end + 1]
        X_test = data[valid_end + 1:]
    else:
        print("This dataset used splitter is random")
        X_train, X_valid_test, _, y_valid_test = train_test_split(data, df['label'], test_size=0.2,
                                                                  stratify=df['label'], random_state=27)
        X_valid, X_test, _, _ = train_test_split(X_valid_test, y_valid_test, test_size=0.5, stratify=y_valid_test,
                                                 random_state=27)
    _save_npz_data(X_train, f"./{my_dataset}_Intermediate/X_train_d3.npz")
    _save_npz_data(X_valid, f"./{my_dataset}_Intermediate/X_valid_d3.npz")
    _save_npz_data(X_test, f"./{my_dataset}_Intermediate/X_test_d3.npz")
    #train
    train_begin()