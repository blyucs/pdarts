import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES_NORMAL,PRIMITIVES_REDUCE
from genotypes import Genotype


class MixedOp(nn.Module):

    def __init__(self, C, stride, reduction, switch, p):
        super(MixedOp, self).__init__()
        self.m_ops = nn.ModuleList()
        self.p = p
        if reduction:
            PRIMITIVES = PRIMITIVES_REDUCE
        else:
            PRIMITIVES = PRIMITIVES_NORMAL
        for i in range(len(switch)):
            if switch[i]:
                primitive = PRIMITIVES[i]
                op = OPS[primitive](C, stride, False)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                if isinstance(op, Identity) and p > 0:
                    op = nn.Sequential(op, nn.Dropout(self.p))
                self.m_ops.append(op)
                
    def update_p(self):
        for op in self.m_ops:
            if isinstance(op, nn.Sequential):
                if isinstance(op[0], Identity):
                    op[1].p = self.p
                    
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.m_ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, switches, p):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.p = p
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        switch_count = 0
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, reduction, switch=switches[switch_count], p=self.p)
                self.cell_ops.append(op)
                switch_count = switch_count + 1
    
    def update_p(self):
        for op in self.cell_ops:
            op.p = self.p
            op.update_p()

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self.cell_ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, switches_normal=[], switches_reduce=[], p=0.0):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.p = p
        self.switches_normal = switches_normal
        self.switches_reduce = switches_reduce
        switch_normal_ons = []
        switch_reduce_ons = []
        for i in range(len(switches_normal)):
            ons = 0
            for j in range(len(switches_normal[i])):
                if switches_normal[i][j]:
                    ons = ons + 1
            switch_normal_ons.append(ons)
            ons = 0
        self.switch_normal_on = switch_normal_ons[0]

        for i in range(len(switches_reduce)):
            ons = 0
            for j in range(len(switches_reduce[i])):
                if switches_reduce[i][j]:
                    ons = ons + 1
            switch_reduce_ons.append(ons)
            ons = 0
        self.switch_reduce_on = switch_reduce_ons[0]

        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
    
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_reduce, self.p)
            else:
                reduction = False
                cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches_normal, self.p)
#            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.forward_type = True
        self._initialize_alphas()
        self.baseline = 0
        # self.baseline_decay_weight = 0.99
        self.baseline_decay_weight = 0.95
        self.rl_batch_size = 10
        self.rl_interval_steps = 1
    def forward(self, input):
        # print("fuck %d" % self.forward_type)
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if self.alphas_reduce.size(1) == 1:
                    if self.forward_type == True:
                        weights = F.softmax(self.alphas_reduce, dim=0)
                    else:
                        weights = self.sub_alphas_reduce
                else:
                    if self.forward_type == True:  #  this
                        weights = F.softmax(self.alphas_reduce, dim=-1)
                    else:
                        weights = self.sub_alphas_reduce
            else:
                if self.alphas_normal.size(1) == 1:
                    if self.forward_type == True:
                        weights = F.softmax(self.alphas_normal, dim=0)
                    else:
                        weights = self.sub_alphas_normal
                else:
                    if self.forward_type == True: # this
                        weights = F.softmax(self.alphas_normal, dim=-1)
                    else:
                        weights = self.sub_alphas_normal
            s0, s1 = s1, cell(s0, s1, weights)
            # print(weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

    def update_p(self):
        for cell in self.cells:
            cell.p = self.p
            cell.update_p()
    
    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target) 

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        # num_ops = self.switch_on
        self.alphas_normal = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k, self.switch_normal_on)))
        self.alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3*np.random.randn(k, self.switch_reduce_on)))
        # self.alphas_normal = nn.Parameter(torch.Tensor(k, num_ops))
        # self.alphas_reduce = nn.Parameter(torch.Tensor(k, num_ops))
        # for i in range(k):
        #     self.alphas_normal.data[i].normal_(0, 1e-3)
        #     self.alphas_reduce.data[i].normal_(0, 1e-3)

        self.sub_alphas_normal =torch.FloatTensor(1e-3*np.random.randn(k, self.switch_normal_on))
        self.sub_alphas_reduce =torch.FloatTensor(1e-3*np.random.randn(k, self.switch_reduce_on))
        # self.sub_alphas_normal =torch.FloatTensor(k, num_ops)
        # self.sub_alphas_reduce =torch.FloatTensor(k, num_ops)

        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
        self.normal_log_prob = Variable(torch.zeros(k, dtype = torch.float), requires_grad = True)
        self.reduce_log_prob = Variable(torch.zeros(k, dtype = torch.float), requires_grad = True)

    def arch_parameters(self):
        return self._arch_parameters

    @property
    def probs_over_ops(self):
        self.normal_probs = F.softmax(self.alphas_normal, dim=1)  # softmax to probability
        self.reduce_probs = F.softmax(self.alphas_reduce, dim=1)
        return self.normal_probs, self.reduce_probs

    def set_log_prob(self):
        normal_probs, reduce_probs = self.probs_over_ops


        # normal_sample = torch.utils.data.sampler.SubsetRandomSampler(normal_probs)[0]
        # reduce_sample = torch.utils.data.sampler.SubsetRandomSampler(reduce_probs)[0]


        # self.log_prob = torch.log(probs[sample])
        # self.current_prob_over_ops = probs
        if 0:  #
            normal_sample = torch.multinomial(normal_probs, 1)
            reduce_sample = torch.multinomial(reduce_probs, 1)
        else: # random sample
            normal_sample = torch.LongTensor(len(normal_probs)).cuda()
            reduce_sample = torch.LongTensor(len(normal_probs)).cuda()
            for i in range(len(normal_probs)):
                normal_sample[i] = torch.from_numpy(np.random.choice([_i for _i in range(self.switch_normal_on)], 1))[0]
                reduce_sample[i] = torch.from_numpy(np.random.choice([_i for _i in range(self.switch_reduce_on)], 1))[0]
            normal_sample = normal_sample.unsqueeze(1)
            reduce_sample = reduce_sample.unsqueeze(1)

        self.normal_log_prob = torch.log(torch.gather(normal_probs,1,normal_sample))
        self.reduce_log_prob = torch.log(torch.gather(reduce_probs,1,reduce_sample))
        return normal_sample, reduce_sample

    def set_sub_net(self, switch_normal, switch_reduce):
        for i in range(len(switch_normal)):
            for j in range(self.switch_normal_on):
                if switch_normal[i][j]:
                    self.sub_alphas_normal[i][j] = 1.0
                else:
                    self.sub_alphas_normal[i][j] = 0

        for i in range(len(switch_reduce)):
            for j in range(self.switch_reduce_on):
                if switch_reduce[i][j]:
                    self.sub_alphas_reduce[i][j] = 1.0
                else:
                    self.sub_alphas_reduce[i][j] = 0
        self.forward_type = False

    def restore_super_net(self):
        self.forward_type = True
