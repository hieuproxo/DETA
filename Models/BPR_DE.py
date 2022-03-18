import torch
import torch.nn as nn

from Models.BPR import BPR


class Expert(nn.Module):
    def __init__(self, dims):
        super(Expert, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )
    
    def forward(self, x):
        return self.mlp(x)


class BPR_DE(BPR):
    def __init__(self, user_count, item_count, teacher_user_emb, teacher_item_emb, student_dim, num_experts, gpu):
        """
        Parameters
        ----------
        user_count (int): number of users
        item_count (int): number of items
        teacher_user_emb (2D FloatTensor): teacher user embeddings
        teacher_item_emb (2D FloatTensor): teacher item embeddings
        student_dim (int): dimension of embedding vectors of student model
        num_experts (int): number of DEs
        gpu: gpu device
        """

        BPR.__init__(self, user_count, item_count, student_dim, gpu)

        self.student_dim = student_dim
        self.gpu = gpu

        # teacher embeddings
        self.teacher_user_emb = nn.Embedding.from_pretrained(teacher_user_emb)
        self.teacher_item_emb = nn.Embedding.from_pretrained(teacher_item_emb)

        # fix the teacher embeddings
        self.teacher_user_emb.weight.requires_grad = False
        self.teacher_item_emb.weight.requires_grad = False

        # get the teacher dimension
        self.teacher_dim = self.teacher_user_emb.weight.size(1)

        # expert configuration
        self.num_experts = num_experts
        # define dimensions of the expert network
        expert_dims = [self.student_dim, (self.teacher_dim + self.student_dim)//2, self.teacher_dim]
        # for self-distillation
        if self.teacher_dim == self.student_dim:
            expert_dims = [self.student_dim, self.student_dim // 2, self.teacher_dim]
        
        # user/item experts
        self.user_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
        self.item_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])

        # user/item selection networks
        self.user_selection_net = nn.Sequential(
            nn.Linear(self.teacher_dim, num_experts),
            nn.Softmax(dim=1)
        )
        self.item_selection_net = nn.Sequential(
            nn.Linear(self.teacher_dim, num_experts),
            nn.Softmax(dim=1)
        )

        print('Teacher dim:', self.teacher_dim)
        print('Student dim:', self.student_dim)
        print('Expert dims:', expert_dims)
        
        # Gumbel-Softmax temperature
        self.T = 0.
        self.sm = nn.Softmax(dim = 1)


    def get_DE_loss(self, batch_entity, is_user=True):
        """
        Compute DE loss
        ----------

        Parameters
        ----------
        batch_entity (1-D Long Tensor): batch of users/items
        is_user (Bolean): distilling from user or item embeddings

        Returns
        ----------
        DE_loss (float): DE loss
        """

        if is_user:
            s = self.user_emb(batch_entity)
            t = self.teacher_user_emb(batch_entity) 

            experts = self.user_experts
            selection_net = self.user_selection_net
        else:
            s = self.item_emb(batch_entity)
            t = self.teacher_item_emb(batch_entity)

            experts = self.item_experts
            selection_net = self.item_selection_net
        
        selection_dist = selection_net(t) 

        if self.num_experts == 1:
            selection_result = 1
        else:
            # expert selection
            g = torch.distributions.Gumbel(0, 1).sample(selection_dist.size()).to(self.gpu) 
            eps = 1e-10
            selection_dist = selection_dist + eps
            selection_dist = self.sm((selection_dist.log() + g) / self.T) 

            selection_dist = torch.unsqueeze(selection_dist, 1) 
            selection_result = selection_dist.repeat(1, self.teacher_dim, 1) 

        expert_outputs = [experts[i](s).unsqueeze(-1) for i in range(self.num_experts)]
        expert_outputs = torch.cat(expert_outputs, -1) 

        expert_outputs = expert_outputs*selection_result 

        expert_outputs = expert_outputs.sum(2) 

        DE_loss = ((t - expert_outputs)**2).sum(-1).sum() 

        return DE_loss
        

class BPR_DETA(BPR):
    def __init__(self, user_count, item_count, teacher_user_emb, teacher_item_emb, TA_user_emb_list, TA_item_emb_list, student_dim, num_experts, gpu):
        """
        Parameters
        ----------
        user_count (int): number of users
        item_count (int): number of items
        teacher_user_emb (2D FloatTensor): teacher user embeddings
        teacher_item_emb (2D FloatTensor): teacher item embeddings
        TA_user_emb_list (list of 2D FloatTensors): list of TA user embeddings
        TA_item_emb_list (list of 2D FloatTensors): list of TA item embeddings
        student_dim (int): dimension of embedding vectors of student model
        num_experts (int): number of DEs
        gpu: gpu device
        """
        
        BPR.__init__(self, user_count, item_count, student_dim, gpu)

        self.student_dim = student_dim
        self.gpu = gpu

        # teacher embedding
        self.user_emb_teacher = nn.Embedding.from_pretrained(teacher_user_emb) 
        self.item_emb_teacher = nn.Embedding.from_pretrained(teacher_item_emb)

        # lists of TA embeddings
        self.user_emb_TAs = nn.ModuleList([nn.Embedding.from_pretrained(TA_user_emb) for TA_user_emb in TA_user_emb_list]) 
        self.item_emb_TAs = nn.ModuleList([nn.Embedding.from_pretrained(TA_item_emb) for TA_item_emb in TA_item_emb_list]) 

        # fix the teacher embeddings
        self.user_emb_teacher.weight.requires_grad = False
        self.item_emb_teacher.weight.requires_grad = False

        # fix the TA embeddings
        for user_emb_TA, item_emb_TA in zip(self.user_emb_TAs, self.item_emb_TAs): 
            user_emb_TA.weight.requires_grad = False
            item_emb_TA.weight.requires_grad = False

        # get the teacher dimension
        self.teacher_dim = self.user_emb_teacher.weight.size(1)
        # get the TA dimensions
        self.TAs_dim = [TA_user_emb.size(1) for TA_user_emb in TA_user_emb_list]

        # expert configuration
        self.num_experts = num_experts
        expert_dims = [self.student_dim, (self.teacher_dim + self.student_dim)//2, self.teacher_dim]
        expert_dims_TAs = [[self.student_dim, (TA_dim + self.student_dim)//2, TA_dim] for TA_dim in self.TAs_dim]
        
        # experts for KD from teacher
        self.user_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
        self.item_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])

        # experts for KD from TAs
        self.user_experts_TAs = nn.ModuleList([nn.ModuleList([Expert(expert_dims_TA).to(self.gpu) for i in range(self.num_experts)]) for expert_dims_TA in expert_dims_TAs])
        self.item_experts_TAs = nn.ModuleList([nn.ModuleList([Expert(expert_dims_TA).to(self.gpu) for i in range(self.num_experts)]) for expert_dims_TA in expert_dims_TAs])

        # selection networks for KD from teacher
        self.user_selection_net = nn.Sequential(
            nn.Linear(self.teacher_dim, num_experts),
            nn.Softmax(dim=1)
        )
        self.item_selection_net = nn.Sequential(
            nn.Linear(self.teacher_dim, num_experts),
            nn.Softmax(dim=1)
        )

        # selection networks for KD from TAs
        self.user_selection_net_TAs = nn.ModuleList([nn.Sequential(
            nn.Linear(self.TAs_dim[i], num_experts),
            nn.Softmax(dim=1)
        ) for i in range(len(TA_user_emb_list))])

        self.item_selection_net_TAs = nn.ModuleList([nn.Sequential(
            nn.Linear(self.TAs_dim[i], num_experts),
            nn.Softmax(dim=1)
        ) for i in range(len(TA_user_emb_list))])

        print('Teacher dim:', self.teacher_dim)
        print('Student dim:', self.student_dim)
        print('TA dims: ', end='')
        for i in range(len(TA_user_emb_list)):
            print(self.TAs_dim[i], '', end='')
        print()

        # Gumbel-Softmax temperature
        self.T = 0.
        self.sm = nn.Softmax(dim = 1)


    def get_DE_loss(self, batch_entity, is_user=True):
        """
        Compute DE loss
        ----------

        Parameters
        ----------
        batch_entity (1-D Long Tensor): batch of users/items
        is_user (Bolean): distilling from user or item embeddings

        Returns
        ----------
        DE_loss (float): DE loss
        """

        if is_user:
            s = self.user_emb(batch_entity) 
            t = self.teacher_user_emb(batch_entity) 

            experts = self.user_experts
            selection_net = self.user_selection_net
        else:
            s = self.item_emb(batch_entity)
            t = self.teacher_item_emb(batch_entity)

            experts = self.item_experts
            selection_net = self.item_selection_net
        
        selection_dist = selection_net(t) 

        if self.num_experts == 1:
            selection_result = 1
        else:
            # expert selection
            g = torch.distributions.Gumbel(0, 1).sample(selection_dist.size()).to(self.gpu) 
            eps = 1e-10
            selection_dist = selection_dist + eps
            selection_dist = self.sm((selection_dist.log() + g) / self.T) 

            selection_dist = torch.unsqueeze(selection_dist, 1) 
            selection_result = selection_dist.repeat(1, self.teacher_dim, 1)

        expert_outputs = [experts[i](s).unsqueeze(-1) for i in range(self.num_experts)] 
        expert_outputs = torch.cat(expert_outputs, -1)

        expert_outputs = expert_outputs*selection_result 
        expert_outputs = expert_outputs.sum(2) 

        DE_loss = ((t - expert_outputs)**2).sum(-1).sum() 

        return DE_loss
    

    def get_DETA_loss(self, batch_entity, TA_id, is_user=True):
        """
        Compute DETA loss
        ----------

        Parameters
        ----------
        batch_entity (1-D Long Tensor): batch of users/items
        TA_id (int): index of TA in the TA list
        is_user (Bolean): distilling from user or item embeddings

        Returns
        ----------
        DETA_loss (float): DETA loss
        """

        if is_user:
            s = self.user_emb(batch_entity)
            t = self.user_emb_TAs[TA_id](batch_entity) 

            experts = self.user_experts_TAs[TA_id]
            selection_net = self.user_selection_net_TAs[TA_id]
            
        else:
            s = self.item_emb(batch_entity)
            t = self.item_emb_TAs[TA_id](batch_entity)

            experts = self.item_experts_TAs[TA_id]
            selection_net = self.item_selection_net_TAs[TA_id]
        
        selection_dist = selection_net(t) 

        if self.num_experts == 1:
            selection_result = 1
        else:
            # expert selection
            g = torch.distributions.Gumbel(0, 1).sample(selection_dist.size()).to(self.gpu) 
            eps = 1e-10
            selection_dist = selection_dist + eps
            selection_dist = self.sm((selection_dist.log() + g) / self.T) 

            selection_dist = torch.unsqueeze(selection_dist, 1) 
            selection_result = selection_dist.repeat(1, self.TAs_dim[TA_id], 1) 
            

        expert_outputs = [experts[i](s).unsqueeze(-1) for i in range(self.num_experts)] 
        expert_outputs = torch.cat(expert_outputs, -1) 

        expert_outputs = expert_outputs*selection_result
        expert_outputs = expert_outputs.sum(2) 

        DETA_loss = ((t - expert_outputs)**2).sum(-1).sum()

        return DETA_loss
