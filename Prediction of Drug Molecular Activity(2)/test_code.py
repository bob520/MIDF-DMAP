import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
import paddle
from config import my_dataset, ALPHA
from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config
from train_util.Focal_Loss import focal_loss
from train_util.fusion import  model_forward
from train_util.dataset import MYDataset
from train_util.featurizer import DownstreamCollateFn
import pickle
from pahelix.datasets import InMemoryDataset
import warnings
warnings.filterwarnings("ignore")



def eval(model,test_dataloader,geo_loader_test,compound_encoder,norm,test_data_size,y_true_test,lossfn1,mode):

    test_epoch_true = 0
    epoch_test_y_predict = []
    predict_proba_test = []
    model.eval()
    with torch.no_grad():
        for data, geo_data in zip(test_dataloader, geo_loader_test):
            gram, mol_vec, targets, index = data
            atom_bond_graphs, bond_angle_graphs = geo_data
            atom_bond_graphs = atom_bond_graphs.tensor()
            bond_angle_graphs = bond_angle_graphs.tensor()
            node_repr, edge_repr, graph_repr = compound_encoder(atom_bond_graphs, bond_angle_graphs)
            graph_repr = norm(graph_repr)
            _, outputs, _ = model_forward(model, lossfn1, None, gram, mol_vec, graph_repr, targets, index,mode="eval")
            list_auc = outputs[:, 1]

            #softmax
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
    print(
        f"The total number of hits on the {mode} set is{test_epoch_true}/{test_data_size}，acc is{test_epoch_acc}"
        f"precious is {p}，recall is {r}，f1 is {f}，AUC is {test_auc_epoch}")
    print(f"The confusion matrix is:\n{cm}")
    # macro
    p, r, f, s = precision_recall_fscore_support(y_true_test, epoch_test_y_predict, average="macro")
    print(f"macro precious is {p}，macro recall is {r}，macro f1 is {f}")


if __name__ == '__main__':
    BATCH_SIZE=32
    compound_encoder_path = "./train_util/class.pdparams"
    compound_encoder_config = load_json_config("./train_util/gnnconfig.json")
    # The GeoGNNModel model of the PaddlePaddle platform is used here to process the three-dimensional molecular information
    compound_encoder = GeoGNNModel(compound_encoder_config)
    compound_encoder.set_state_dict(paddle.load(compound_encoder_path))

    test_dataset = InMemoryDataset(
        npz_data_path=f"./{my_dataset}_Intermediate/X_test_d3.npz")
    collate_fn = DownstreamCollateFn(
        atom_names=compound_encoder_config['atom_names'],
        bond_names=compound_encoder_config['bond_names'],
        bond_float_names=compound_encoder_config['bond_float_names'],
        bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
        task_type='class', is_inference=True)

    geo_loader_test = test_dataset.get_data_loader(
        batch_size=BATCH_SIZE,
        num_workers=1,
        shuffle=False,
        collate_fn=collate_fn)
    model = torch.load(f"./{my_dataset}_Intermediate/save_model/best_auc_model.pth")

    test_set = MYDataset(f"./{my_dataset}_Intermediate/X_test_d1.pkl",
                         f"./{my_dataset}_Intermediate/X_test_d2.pkl",
                         f"./{my_dataset}_Intermediate/y_test.pkl")


    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE)


    with open(f"./{my_dataset}_Intermediate/y_test.pkl", "rb") as f:
        y_true_test = pickle.load(f)
    norm = paddle.nn.LayerNorm(compound_encoder.graph_dim)

    loss_fn = focal_loss(alpha=ALPHA, gamma=2, num_classes=2)

    eval(model, test_dataloader=test_dataloader, geo_loader_test=geo_loader_test,
         compound_encoder=compound_encoder,
         norm=norm, y_true_test=y_true_test,test_data_size=len(test_set),lossfn1=loss_fn,mode="test")