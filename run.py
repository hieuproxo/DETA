import time
from copy import deepcopy
import torch

from Utils.evaluation import evaluation, LOO_print_result, print_final_result
from Utils.data_utils import T_annealing


def no_KD_run(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path=None):
    """
    Training without KD
    ----------

    Parameters
    ----------
    opt: parse arguments
    model: model
    gpu: gpu device
    optimizer: optimizer
    train_loader: training dataset
    test_dataset: test dataset
    model_save_path (str): path for saving models
    """

    max_epoch, early_stop, es_epoch = opt.max_epoch, opt.early_stop, opt.es_epoch
    template = {'best_score': -999, 'best_result': -1, 'final_result': -1}
    eval_dict = {'05': deepcopy(template), '10': deepcopy(template), '15': deepcopy(template), '20': deepcopy(template), 'early_stop': 0, 'early_stop_max': early_stop, 'final_epoch': 0}
    
    print('\nTraining model with dim =', model.dim, '...\n')

    # begin training
    for epoch in range(max_epoch):
        tic1 = time.time()
        train_loader.dataset.negative_sampling()
        epoch_loss = []

        # mini-batch training
        for batch_user, batch_pos_item, batch_neg_item in train_loader:
            batch_user = batch_user.to(gpu)
            batch_pos_item = batch_pos_item.to(gpu)
            batch_neg_item = batch_neg_item.to(gpu)

            # forward propagation
            model.train()
            output = model(batch_user, batch_pos_item, batch_neg_item)

            # batch loss
            batch_loss = model.get_loss(output)
            epoch_loss.append(batch_loss)

            # update model parameters
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        # total loss in an epoch 
        epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
        toc1 = time.time()

        # evaluation
        is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
        LOO_print_result(epoch, max_epoch, epoch_loss, eval_results, is_improved=is_improved, train_time = toc1-tic1, test_time = elapsed)

        # save model
        if model_save_path != None and is_improved:
            torch.save(model.state_dict(), model_save_path)
        
        # early stopping
        if eval_dict['early_stop'] >= eval_dict['early_stop_max']:
            break

    print("BEST EPOCH::", eval_dict['final_epoch'])
    print_final_result(eval_dict)


def DE_run(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path=None):
    """
    Training using DE for KD
    ----------

    Parameters
    ----------
    opt: parse arguments
    model: model
    gpu: gpu device
    optimizer: optimizer
    train_loader: training dataset
    test_dataset: test dataset
    model_save_path (str): path for saving models
    """

    max_epoch, early_stop, es_epoch = opt.max_epoch, opt.early_stop, opt.es_epoch
    template = {'best_score':-999, 'best_result':-1, 'final_result':-1}
    eval_dict = {'05': deepcopy(template), '10':deepcopy(template), '15':deepcopy(template), '20':deepcopy(template), 'early_stop':0,  'early_stop_max':early_stop, 'final_epoch':0}

    print('\nTraining model with dim =', model.dim, 'using DE ...\n')

    current_T = opt.end_T * opt.anneal_size

    # begin training
    for epoch in range(max_epoch):
        tic1 = time.time()
        train_loader.dataset.negative_sampling()
        epoch_loss = []

        model.T = current_T
        
        # mini-batch training
        for batch_user, batch_pos_item, batch_neg_item in train_loader:			
            batch_user = batch_user.to(gpu)
            batch_pos_item = batch_pos_item.to(gpu)
            batch_neg_item = batch_neg_item.to(gpu)
            
            # forward propagation
            model.train()
            output = model(batch_user, batch_pos_item, batch_neg_item)
            base_loss = model.get_loss(output)

            # DE loss
            if opt.model == 'BPR' or opt.model == 'LightGCN':
                DE_loss_user = model.get_DE_loss(batch_user.unique(), is_user=True)
                DE_loss_pos = model.get_DE_loss(batch_pos_item.unique(), is_user=False)
                DE_loss_neg = model.get_DE_loss(batch_neg_item.unique(), is_user=False)
                DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg)*0.5
            elif opt.model == 'NeuMF':
                DE_loss_user_MF = model.get_DE_loss(batch_user.unique(), is_MF=True, is_user=True)
                DE_loss_pos_MF = model.get_DE_loss(batch_pos_item.unique(), is_MF=True, is_user=False)
                DE_loss_neg_MF = model.get_DE_loss(batch_neg_item.unique(), is_MF=True, is_user=False)

                # DE_loss_user_MLP = model.get_DE_loss(batch_user.unique(), is_MF=False, is_user=True)
                # DE_loss_pos_MLP = model.get_DE_loss(batch_pos_item.unique(), is_MF=False, is_user=False)
                # DE_loss_neg_MLP = model.get_DE_loss(batch_neg_item.unique(), is_MF=False, is_user=False)

                # DE_loss = DE_loss_user_MF + DE_loss_user_MLP + (DE_loss_pos_MF + DE_loss_neg_MF + DE_loss_pos_MLP + DE_loss_neg_MLP) * 0.5
                DE_loss = DE_loss_user_MF + (DE_loss_pos_MF + DE_loss_neg_MF)*0.5
                # DE_loss = DE_loss_user_MLP + (DE_loss_pos_MLP + DE_loss_neg_MLP) * 0.5
            
            # batch loss
            batch_loss = base_loss + DE_loss*opt.lmbda_DE

            epoch_loss.append(batch_loss)
            
            # update model parameters
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # compute total loss in an epoch 
        epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
        toc1 = time.time()
        
        # evaluation
        is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
        LOO_print_result(epoch, max_epoch, epoch_loss, eval_results, is_improved=is_improved, train_time = toc1-tic1, test_time = elapsed)
        
        # save model
        if model_save_path != None and is_improved:
            torch.save(model.state_dict(), model_save_path)

        # early stopping
        if (eval_dict['early_stop'] >= eval_dict['early_stop_max']):
            break

        # annealing
        current_T = T_annealing(epoch, max_epoch, opt.end_T * opt.anneal_size, opt.end_T)
        if current_T < opt.end_T:
            current_T = opt.end_T

    print("BEST EPOCH::", eval_dict['final_epoch'])
    print_final_result(eval_dict)


# training using multiple TAs
def DETA_run(opt, model, gpu, optimizer, train_loader, test_dataset, TAs, model_save_path=None):
    """
    Training using DE for KD
    ----------

    Parameters
    ----------
    opt: parse arguments
    model: model
    gpu: gpu device
    optimizer: optimizer
    train_loader: training dataset
    test_dataset: test dataset
    TAs: list of TA models
    model_save_path (str): path for saving models
    """

    max_epoch, early_stop, es_epoch = opt.max_epoch, opt.early_stop, opt.es_epoch
    template = {'best_score':-999, 'best_result':-1, 'final_result':-1}
    eval_dict = {'05': deepcopy(template), '10':deepcopy(template), '15':deepcopy(template), '20':deepcopy(template), 'early_stop':0,  'early_stop_max':early_stop, 'final_epoch':0}

    print('\nTraining model using DETAwith dim =', model.dim, '...\n')

    current_T = opt.end_T * opt.anneal_size

    # begin training
    for epoch in range(max_epoch):
        
        tic1 = time.time()
        train_loader.dataset.negative_sampling()
        epoch_loss = []

        model.T = current_T
        
        # mini-batch training
        for batch_user, batch_pos_item, batch_neg_item in train_loader:			
            batch_user = batch_user.to(gpu)
            batch_pos_item = batch_pos_item.to(gpu)
            batch_neg_item = batch_neg_item.to(gpu)
            
            # forward propagation
            model.train()
            output = model(batch_user, batch_pos_item, batch_neg_item)
            base_loss = model.get_loss(output)

            # DE loss from teacher
            if opt.model == 'BPR' or opt.model == 'LightGCN':
                DE_loss_user = model.get_DE_loss(batch_user.unique(), is_user=True)
                DE_loss_pos = model.get_DE_loss(batch_pos_item.unique(), is_user=False)
                DE_loss_neg = model.get_DE_loss(batch_neg_item.unique(), is_user=False)
                DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg)*0.5
            elif opt.model == 'NeuMF':
                DE_loss_user_MF = model.get_DE_loss(batch_user.unique(), is_MF=True, is_user=True)
                DE_loss_pos_MF = model.get_DE_loss(batch_pos_item.unique(), is_MF=True, is_user=False)
                DE_loss_neg_MF = model.get_DE_loss(batch_neg_item.unique(), is_MF=True, is_user=False)
                DE_loss = DE_loss_user_MF + (DE_loss_pos_MF + DE_loss_neg_MF)*0.5

            # dropout
            dropout = torch.bernoulli(torch.Tensor([opt.dropout]*(len(TAs)+1)))        
            # add DE loss from teacher to batch loss
            batch_loss = base_loss + dropout[0].item()*DE_loss*opt.lmbda_DE

            # DE loss from TAs
            for i in range(len(TAs)):   
                if opt.model == 'BPR' or opt.model == 'LightGCN':
                    DE_loss_user_TAS = model.get_DE_TA_loss(batch_user.unique(), TA_id=i, is_user=True)
                    DE_loss_pos_TAS = model.get_DE_TA_loss(batch_pos_item.unique(), TA_id=i, is_user=False)
                    DE_loss_neg_TAS = model.get_DE_TA_loss(batch_neg_item.unique(), TA_id=i, is_user=False)
                    DE_loss_TAS = DE_loss_user_TAS + 0.5*(DE_loss_pos_TAS + DE_loss_neg_TAS)
                elif opt.model == 'NeuMF':
                    DE_loss_user_MF_TAS = model.get_DE_TA_loss(batch_user.unique(), TA_id=i, is_MF=True, is_user=True)
                    DE_loss_pos_MF_TAS = model.get_DE_TA_loss(batch_pos_item.unique(), TA_id=i, is_MF=True, is_user=False)
                    DE_loss_neg_MF_TAS = model.get_DE_TA_loss(batch_neg_item.unique(), TA_id=i, is_MF=True, is_user=False)
                    DE_loss_TAS = DE_loss_user_MF_TAS + 0.5*(DE_loss_pos_MF_TAS + DE_loss_neg_MF_TAS)

                # accumulate batch loss
                batch_loss += dropout[i+1].item()*DE_loss_TAS*opt.lmbda_DE_TAS

            epoch_loss.append(batch_loss)
            
            # update model parameters
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # total loss in an epoch 
        epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
        toc1 = time.time()
        
        # evaluation
        is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
        LOO_print_result(epoch, max_epoch, epoch_loss, eval_results, is_improved=is_improved, train_time = toc1-tic1, test_time = elapsed)
        
        # save model
        if model_save_path != None and is_improved:
            torch.save(model.state_dict(), model_save_path)

        # early stopping
        if (eval_dict['early_stop'] >= eval_dict['early_stop_max']):
            break

        # annealing
        current_T = T_annealing(epoch, max_epoch, opt.end_T * opt.anneal_size, opt.end_T)
        if current_T < opt.end_T:
            current_T = opt.end_T


    print("BEST EPOCH::", eval_dict['final_epoch'])
    print_final_result(eval_dict)

