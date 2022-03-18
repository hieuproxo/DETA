import argparse
import torch
import torch.utils.data as data
import torch.optim as optim

from Models.BPR import BPR
from Models.BPR_DE import BPR_DE, BPR_DETA
from Models.NeuMF import NeuMF
from Models.NeuMF_DE import NeuMF_DE, NeuMF_DETA
from Models.LightGCN import LightGCN
from Models.LightGCN_DE import LightGCN_DE, LightGCN_DETA

from Utils.dataset import implicit_CF_dataset, implicit_CF_dataset_test
from Utils.data_utils import read_LOO_settings

from run import DETA_run
import gen_graph


def run():
    # training settings
    lr, batch_size, reg, num_ns = opt.lr, opt.batch_size, opt.reg, opt.num_ns
    gpu = torch.device('cuda:' + str(opt.gpu))
    teacher_dim, student_dim = opt.teacher_dim, opt.student_dim
    num_experts = opt.num_experts

    # dataset
    data_path, dataset, seed = opt.data_path, opt.dataset, opt.seed
    user_count, item_count, train_mat, train_interactions, valid_sample, test_sample, candidates = read_LOO_settings(data_path, dataset, seed)

    train_dataset = implicit_CF_dataset(user_count, item_count, train_mat, train_interactions, num_ns)
    test_dataset = implicit_CF_dataset_test(user_count, test_sample, valid_sample, candidates)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # read teacher model
    teacher_dim = opt.teacher_dim
    if opt.model == 'BPR':
        teacher_model = BPR(user_count, item_count, teacher_dim, gpu)
    elif opt.model == 'NeuMF':
        num_layers = opt.num_layers
        teacher_model = NeuMF(user_count, item_count, teacher_dim, num_layers, gpu)
    elif opt.model == 'LightGCN':
        num_layers = opt.num_layers
        Graph = gen_graph.getSparseGraph(train_mat, user_count, item_count, gpu)
        teacher_model = LightGCN(user_count, item_count, teacher_dim, num_layers, Graph, gpu)
    else:
        assert False
        
    with torch.no_grad():
        teacher_model_path = opt.saved_models + dataset +"/" + opt.model + '_' + str(teacher_dim) + '_seed_' + str(seed)
        teacher_model = teacher_model.to(gpu)
        teacher_model.load_state_dict(torch.load(teacher_model_path, map_location='cuda:' + str(opt.gpu)))
        if opt.model == 'BPR' or opt.model == 'LightGCN':		
            teacher_user_emb, teacher_item_emb = teacher_model.get_embedding()
        elif opt.model == 'NeuMF':
            teacher_user_MF, teacher_item_MF, teacher_user_MLP, teacher_item_MLP = teacher_model.get_embedding()
        else:
            assert False
        del teacher_model

    # TA dimensions
    TA_dims = [int(student_dim + (teacher_dim - student_dim)/(opt.num_TAs+1)*(i+1)) for i in range(opt.num_TAs)]

    TAs = []
    for dim in TA_dims:
        if opt.model == 'BPR':
            TAs.append(BPR_DE(user_count, item_count, teacher_user_emb, teacher_item_emb, dim, num_experts, gpu))
        elif opt.model == 'NeuMF':
            TAs.append(NeuMF_DE(user_count, item_count, teacher_user_MF, teacher_item_MF, teacher_user_MLP, teacher_item_MLP, dim, num_layers, num_experts, gpu))
        elif opt.model == 'LightGCN':
            TAs.append(LightGCN_DE(user_count, item_count, teacher_user_emb, teacher_item_emb, dim, num_layers, Graph, num_experts, gpu))

    if opt.model == 'BPR' or opt.model == 'LightGCN':
        user_emb_TA_list = []
        item_emb_TA_list = []
    elif opt.model == 'NeuMF':
        TA_user_emb_MF_list = []
        TA_item_emb_MF_list = []
        TA_user_emb_MLP_list = []
        TA_item_emb_MLP_list = []
    else:
        assert False

    with torch.no_grad(): 
        for TA_model in TAs:
            TA_model_path = opt.saved_models + dataset + '/' + opt.model + '_DE_' + str(TA_model.student_dim) + '_seed_' + str(seed)
            TA_model = TA_model.to(gpu)
            TA_model.load_state_dict(torch.load(TA_model_path, map_location='cuda:' + str(opt.gpu)))
            if opt.model == 'BPR' or opt.model == 'LightGCN':
                user_emb_TA, item_emb_TA = TA_model.get_embedding()
                user_emb_TA_list.append(user_emb_TA)
                item_emb_TA_list.append(item_emb_TA)
            elif opt.model == 'NeuMF':
                TA_user_MF, TA_item_MF, TA_user_MLP, TA_item_MLP = TA_model.get_embedding()
                TA_user_emb_MF_list.append(TA_user_MF)
                TA_item_emb_MF_list.append(TA_item_MF)
                TA_user_emb_MLP_list.append(TA_user_MLP)
                TA_item_emb_MLP_list.append(TA_item_MLP)
            else:
                assert False

    # student model
    if opt.model == 'BPR':
        student_model = BPR_DETA(user_count, item_count, teacher_user_emb, teacher_item_emb, user_emb_TA_list, item_emb_TA_list, gpu, opt.student_dim, opt.num_experts)
    elif opt.model == 'NeuMF':
        student_model = NeuMF_DETA(user_count, item_count, teacher_user_MF, teacher_item_MF, teacher_user_MLP, teacher_item_MLP, \
                TA_user_emb_MF_list, TA_item_emb_MF_list, TA_user_emb_MLP_list, TA_item_emb_MLP_list, gpu, opt.student_dim, num_layers, opt.num_experts)
    elif opt.model == 'LightGCN':
        student_model = LightGCN_DETA(user_count, item_count, teacher_user_emb, teacher_item_emb, user_emb_TA_list, item_emb_TA_list, opt.student_dim, num_layers, Graph, opt.num_experts, gpu)
    else:
        assert False
    student_model = student_model.to(gpu)

    # training
    optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay=reg)
    DETA_run(opt, student_model, gpu, optimizer, train_loader, test_dataset, TAs, model_save_path=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--reg', type=float, default=0.001, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_ns', type=int, default=1, help='number of negative samples')
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--early_stop', type=int, default=20, help='number of epochs for early stopping')
    parser.add_argument('--es_epoch', type=int, default=0)
    parser.add_argument('--saved_models', type=str, default='Saved models/')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    
    # dataset
    parser.add_argument('--data_path', type=str, default='Data sets/')
    parser.add_argument('--dataset', type=str, default='CiteULike')
    parser.add_argument('--seed', type=int, default=0, help='dataset seed')

    # DE
    parser.add_argument('--num_experts', type=int, default=30, help='number of distillation experts')
    parser.add_argument('--lmbda_DE', type=float, default=0.01)
    parser.add_argument('--end_T', type=float, default=1e-10, help='for MTD_lmbda')
    parser.add_argument('--anneal_size', type=int, default=1e+10, help='T annealing')

    # model 
    parser.add_argument('--teacher_dim', type=int, default=200)
    parser.add_argument('--student_dim', type=int, default=20)
    parser.add_argument('--model', type=str, default='BPR')
    parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers (for NeuMF and LightGCN)')
    parser.add_argument('--num_TAs', type=int, default=8, help='number of TAs')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
    parser.add_argument('--lmbda_DE_TAS', type=float, default=0.01)

    opt = parser.parse_args()
    # print(opt)

    run()