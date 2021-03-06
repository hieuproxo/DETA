import torch
import torch.utils.data as data
import numpy as np

from Utils.data_utils import *

class implicit_CF_dataset(data.Dataset):
    def __init__(self, user_count, item_count, rating_mat, interactions, num_ns):
        """
        Parameters
        ----------
        user_count (int): number of users
        item_count (int): number of items
        rating_mat (dict): user-item rating matrix
        interactions (list): total train interactions, each instance has a form of (user, item, 1)
        num_ns (int): number of negative samples
        """

        super(implicit_CF_dataset, self).__init__()

        self.user_count = user_count
        self.item_count = item_count
        self.rating_mat = rating_mat
        self.num_ns = num_ns
        self.interactions = interactions

    
    def negative_sampling(self):
        self.train_arr = []
        sample_list = np.random.choice(list(range(self.item_count)), size = 10*len(self.interactions)*self.num_ns)

        sample_idx = 0
        for user, pos_item, _ in self.interactions: 
            ns_count = 0

            while True:
                neg_item = sample_list[sample_idx]
                if not is_visited(self.rating_mat, user, neg_item):  
                    self.train_arr.append((user, pos_item, neg_item))
                    sample_idx += 1
                    ns_count += 1
                    if ns_count == self.num_ns:
                        break
                        
                sample_idx += 1


    def __len__(self):
        return len(self.interactions) * self.num_ns
    

    def __getitem__(self, idx):
        return self.train_arr[idx][0], self.train_arr[idx][1], self.train_arr[idx][2]
            
             
class implicit_CF_dataset_test(data.Dataset):
    def __init__(self, user_count, test_sample, valid_sample, candidates, batch_size=1024):
        """
        Parameters
        ----------
        user_count (int): number of users
        test_sample (dict): sampled test item for each user
        valid_sample (dict): sampled valid item for each user
        candidates (dict): sampled candidate items for each user
        batch_size (int): batch size for mini-batch training
        """

        super(implicit_CF_dataset_test, self).__init__()

        self.user_count = user_count
        self.test_item = []
        self.valid_item = []
        self.candidates = []

        num_candidates = len(candidates[0])
        
        for user in range(user_count):
            if user not in test_sample:
                self.test_item.append([0])
                self.valid_item.append([0])
                self.candidates.append([0]*num_candidates)
            else:
                self.test_item.append([int(test_sample[user])])
                self.valid_item.append([int(valid_sample[user])])
                self.candidates.append(list(candidates[user].keys()))

        self.test_item = torch.LongTensor(self.test_item) 
        self.valid_item = torch.LongTensor(self.valid_item) 
        self.candidates = torch.LongTensor(self.candidates) 

        self.user_list = torch.LongTensor(list(test_sample.keys()))

        self.batch_start = 0
        self.batch_size = batch_size
        

    def get_next_batch_users(self):
        """
        Get the next batch of test users
        ----------

        Returns
        ----------
        self.user_list[batch_start: batch_start + batch_size] (1-D LongTensor): next batch of users
        is_last_batch (boolean): is current batch last batch
        """

        batch_start = self.batch_start
        batch_end = self.batch_start + self.batch_size

        # if it is the last batch
        if batch_end >= len(self.user_list):
            batch_end = len(self.user_list)
            self.batch_start = 0
            is_last_batch = True
        else:
            self.batch_start += self.batch_size
            is_last_batch = False

        return self.user_list[batch_start: batch_end], is_last_batch


    def get_next_batch(self, batch_user):
        """
        Get next test batch (i.e., test samples, valid samples, candidates)
        ----------

        Parameters
        ----------
        batch_user (1D LongTensor): current batch of test users
        
        Returns
        -------
        batch_test_items (2D LongTensor(): batch of test items
        batch_valid_items (2D LongTensor): batch of valid items
        batch_candidates (2D LongTensor): batch of candidates
        """

        batch_test_items = torch.index_select(self.test_item, 0, batch_user)
        batch_valid_items = torch.index_select(self.valid_item, 0, batch_user)
        batch_candidates = torch.index_select(self.candidates, 0, batch_user)

        return batch_test_items, batch_valid_items, batch_candidates