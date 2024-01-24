import torch
import torch.nn as nn
import torch.nn.functional as F
from baseline_models import Medical_2DNet,Medical_3DNet,Medical_base_2DNet,Medical_base_3DNet
import numpy as np
from Models.fundus_swin_network import build_model as fundus_build_model
from Models.unetr import UNETR_base_3DNet

# TMC loss function
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)

def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C
# TMC
class TMC(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(TMC, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        classifier_Fundus_dims = classifiers_dims[1]
        self.res2net_2DNet = Medical_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        self.resnet_3DNet = Medical_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.Classifiers= nn.ModuleList([self.res2net_2DNet, self.resnet_3DNet])
        # self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])
        self.sp = nn.Softplus()

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a,u_a,b_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a,u_a,b_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a,u_a,b_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a,u_a,b_a

    def forward(self, X, y, global_step):
        evidence = self.infer(X)
        loss = 0
        alpha = dict()
        for v_num in range(len(X)):
            alpha[v_num] = evidence[v_num] + 1
            loss += ce_loss(y, alpha[v_num], self.classes, global_step, self.lambda_epochs)
        alpha_a,u_a,b_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs)
        loss = torch.mean(loss)
        # return evidence, evidence_a, loss, u_a, b_a
        return evidence, evidence_a, loss, u_a

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for m_num in range(self.modalties):
            backbone_output = self.Classifiers[m_num](input[m_num])
            evidence[m_num] = self.sp(backbone_output)
        return evidence

class TMC_WAO(nn.Module):

    def __init__(self, classes, modalties, classifiers_dims, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(TMC_WAO, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        classifier_Fundus_dims = classifiers_dims[1]
        self.res2net_2DNet = Medical_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        self.resnet_3DNet = Medical_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.Classifiers= nn.ModuleList([self.res2net_2DNet, self.resnet_3DNet])
        # self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])
        self.sp = nn.Softplus()

    def WAO_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def WAO_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.classes/S[v]

            # WAO
            # b^0 @ b^(0+1)
            weight_b = 1 / 2 * (b[0] + b[1])
            weight_u = 1 / 2 * (u[0] + u[1])

            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub) + C.view(-1, 1).expand(b[0].shape) * weight_b
            # calculate u^a
            u_a = torch.mul(u[0], u[1]) + C.view(-1, 1).expand(u[0].shape) * weight_u

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a, u_a, b_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a, u_a, b_a = WAO_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a, u_a, b_a = WAO_Combin_two(alpha_a, alpha[v+1])
        return alpha_a,u_a,b_a

    def forward(self, X, y, global_step):
        evidence = self.infer(X)
        loss = 0
        alpha = dict()
        for v_num in range(len(X)):
            alpha[v_num] = evidence[v_num] + 1
            loss += ce_loss(y, alpha[v_num], self.classes, global_step, self.lambda_epochs)
        alpha_a, u_a, b_a = self.WAO_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs)
        loss = torch.mean(loss)
        return evidence, evidence_a, loss,u_a,b_a

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for m_num in range(self.modalties):
            backbone_output = self.Classifiers[m_num](input[m_num])
            evidence[m_num] = self.sp(backbone_output)
        return evidence

# Evidential T-loss
def add_to_logging_dict(logging_dict, header, values):
    for index, a in enumerate(values):
        a_mean = torch.mean(a).detach().cpu().numpy()
        a_variance = torch.var(a).detach().cpu().numpy()
        # print(names[index], " " , a_mean, a_variance)
        logging_dict[header[index] + "_mean"] = a_mean
        # logging_dict[header[index]+"_variance"] = a_variance
    return logging_dict

def NIG_NLL(it, y, mu, v, alpha, beta):
    epsilon = 1e-16
    twoBlambda = 2 * beta * (1 + v)

    a1 = 0.5723649429247001 - 0.5 * torch.log(v + epsilon)
    # a1 = 0.5 * torch.log(np.pi / torch.max(v,epsilon))
    a2a = - alpha * torch.log(2 * beta + epsilon)
    a2b = - alpha * torch.log(1 + v)
    a3 = (alpha + 0.5) * torch.log(v * (y - mu) ** 2 + twoBlambda + epsilon)
    a4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

    a2 = a2a + a2b

    nll = a1 + a2 + a3 + a4

    # nll = torch.exp(nll)
    likelihood = (np.pi / v) ** (0.5) / (twoBlambda ** alpha) * ((v * (y - mu) ** 2 + twoBlambda) ** (alpha + 0.5))
    # nll = 1 * (y - mu)**2
    likelihood *= torch.exp(a4)
    # nll = likelihood

    mse = (mu - y) ** 2
    mse += 1e-15
    mse = torch.log(mse)

    # header = ['y', 'mu', 'v', 'alpha', 'beta', 'nll', 'mse', 'a1', 'a2a', 'a2b', 'a2', 'a3', 'a4', 'likelihood',
    #           'twoblambda']
    # values = [y, mu, v, alpha, beta, nll, mse, a1, a2a, a2b, a2, a3, a4, likelihood, twoBlambda]
    #
    # logging_dict = {}
    # logging_dict['Iteration'] = it
    # logging_dict = add_to_logging_dict(logging_dict, header, values)
    #
    # is_nan = torch.stack([torch.isnan(x) * 1 for x in values])

    # return nll, logging_dict
    return nll

def ST_NLL(it, y, mu, sigma, v):
    epsilon = 1e-16

    a1 = torch.log(sigma + epsilon)
    a2 = 0.5723649429247001 + 0.5 * torch.log(v + epsilon)
    a3a =  0.5 * (v+1)
    a3 = a3a * torch.log(1+((y - mu) ** 2/(v*sigma*sigma)) )
    a4 = torch.lgamma(v/2) - torch.lgamma(v/2 + 0.5)

    nll = a1 + a2 + a3 + a4


    # header = ['y', 'mu', 'sigma', 'v', 'a1', 'a2', 'a3a', 'a3', 'a4']
    # values = [y, mu, sigma, v, a1, a2, a3a, a3, a4]

    # logging_dict = {}
    # logging_dict['Iteration'] = it
    # logging_dict = add_to_logging_dict(logging_dict, header, values)
    #
    # is_nan = torch.stack([torch.isnan(x) * 1 for x in values])

    # return nll, logging_dict
    return nll

def NIG_Reg(y, mu, v, alpha,beta):
    error = torch.abs(y - mu)

    # alternatively can do
    # error = (y-gamma)**2

    evi = v + alpha + 1 / (beta + 1e-15)
    reg = error * evi.detach()
    # reg = error * evi
    return reg,evi

def ST_Reg(y, mu, sigma, v):
    error = torch.abs(y - mu)
    epsilon = 1e-16

    # alternatively can do
    # error = (y-gamma)**2
    # mv = v.clone()
    # mv = mv.view(mv.size(0)*mv.size(1), -1)
    evi = 1 / ((sigma*sigma) + epsilon) + v
    # evi = v

    # reg = error * evi.detach()

    reg = error * evi

    return reg,evi

def ST_ICE(y, mu, sigma, v):
    # if y is not one_hot
    # c = len(torch.unique(y))
    # label = F.one_hot(y, num_classes=c)
    # lossCE = torch.sum(label * (torch.log(mu)), dim=1, keepdim=True)

    # if y is the one_hot label
    mu_softplus = F.softplus(mu)
    lossCE = torch.sum(y * (torch.log(mu_softplus)), dim=1, keepdim=True)

    # CE_loss = nn.CrossEntropyLoss()
    # lossCE = CE_loss(mu, y)
    return lossCE

def ST_CE(y, mu, sigma, v):
    CE_loss = nn.CrossEntropyLoss()
    lossCE = CE_loss(mu, y)
    return lossCE

def NIG_ICE(y, mu, v, alpha,beta):
    # if y is not one_hot
    # c = len(torch.unique(y))
    # label = F.one_hot(y, num_classes=c)
    # lossCE = torch.sum(label * (torch.log(mu)), dim=1, keepdim=True)

    # if y is the one_hot label
    mu_softplus = F.softplus(mu)
    lossCE = torch.sum(y * (torch.log(mu_softplus)), dim=1, keepdim=True)

    # CE_loss = nn.CrossEntropyLoss()
    # lossCE = CE_loss(mu, y)
    return lossCE

def NIG_CE(y, mu, v, alpha,beta):
    CE_loss = nn.CrossEntropyLoss()
    lossCE = CE_loss(mu, y)
    return lossCE

def ST_Entropy(v):
    S = 0.5*(v + 1)*(torch.digamma(0.5*(v+1)) - torch.digamma(0.5*v))
    S += torch.log(torch.sqrt(v)*torch.distributions.beta.Beta(0.5*v, 0.5))
    return S

def calculate_evidential_U_loss_constraints(it, y, mu, v, alpha, beta, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    nig_nll = NIG_NLL(it, y, mu, v, alpha,beta)

    nig_reg,nig_conf = NIG_Reg(y, mu, v, alpha,beta)

    nig_ce = NIG_CE(y, mu, v, alpha,beta)
    # nig_ce = NIG_ICE(y, mu, v, alpha,beta)

    nig_confa = 0.5*(beta / ((alpha - 1) + 1e-15))
    nig_confb = 0.5*torch.log((beta / ((alpha - 1) + 1e-15)))
    # ev_sum = nig_nll  + lambda_coef*nig_reg
    # ev_sum = nig_nll  + nig_ce
    # ev_sum = nig_nll  + 0.5*nig_ce
    # ev_sum = nig_nll  + 0.7*nig_conf
    ev_sum = nig_nll  + 0.01*nig_ce*nig_conf
    # ev_sum = nig_nll  + lambda_coef*nig_ce
    # ev_sum = nig_nll  + 0.5*(nig_ce*nig_confa+nig_confb)
    # ev_sum = nig_nll  + 0.5*nig_ce + lambda_coef*nig_reg

    evidential_loss = torch.mean(ev_sum)

    header = ['nig_nll', 'nig_reg', 'nig_loss', 'mse']
    values = [nig_nll, nig_reg, ev_sum, (y-mu)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_MMNIG_loss_constraints(it, y, mu, v, alpha, beta, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    nig_nll = NIG_NLL(it, y, mu, v, alpha,beta)

    nig_reg,nig_conf = NIG_Reg(y, mu, v, alpha,beta)

    nig_ce = NIG_CE(y, mu, v, alpha,beta)
    # nig_ce = NIG_ICE(y, mu, v, alpha,beta)

    nig_confa = 0.5*(beta / ((alpha - 1) + 1e-15))
    nig_confb = 0.5*torch.log((beta / ((alpha - 1) + 1e-15)))
    # ev_sum = nig_nll  + lambda_coef*nig_reg
    # ev_sum = nig_nll  + nig_ce
    ev_sum = nig_nll  + 0.5*nig_ce
    # ev_sum = nig_nll  + 0.7*nig_conf
    # ev_sum = nig_nll  + 0.5*nig_ce*nig_conf
    # ev_sum = nig_nll  + lambda_coef*nig_ce
    # ev_sum = nig_nll  + 0.5*(nig_ce*nig_confa+nig_confb)
    # ev_sum = nig_nll  + 0.5*nig_ce + lambda_coef*nig_reg

    evidential_loss = torch.mean(ev_sum)

    header = ['nig_nll', 'nig_reg', 'nig_loss', 'mse']
    values = [nig_nll, nig_reg, ev_sum, (y-mu)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_lamda_loss_constraints(it, y, mu, v, alpha, beta, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    nig_nll = NIG_NLL(it, y, mu, v, alpha,beta)

    nig_reg,nig_conf = NIG_Reg(y, mu, v, alpha,beta)

    nig_ce = NIG_CE(y, mu, v, alpha,beta)
    # nig_ce = NIG_ICE(y, mu, v, alpha,beta)

    nig_confa = 0.5*(beta / ((alpha - 1) + 1e-15))
    nig_confb = 0.5*torch.log((beta / ((alpha - 1) + 1e-15)))
    # ev_sum = nig_nll  + lambda_coef*nig_reg
    # ev_sum = nig_nll  + nig_ce
    ev_sum = nig_nll  + 0.01*nig_ce
    # ev_sum = nig_nll  + 0.7*nig_conf
    # ev_sum = nig_nll  + 0.5*nig_ce*nig_conf
    # ev_sum = nig_nll  + lambda_coef*nig_ce
    # ev_sum = nig_nll  + 0.5*(nig_ce*nig_confa+nig_confb)
    # ev_sum = nig_nll  + 0.5*nig_ce + lambda_coef*nig_reg

    evidential_loss = torch.mean(ev_sum)

    header = ['nig_nll', 'nig_reg', 'nig_loss', 'mse']
    values = [nig_nll, nig_reg, ev_sum, (y-mu)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_lamda_NLL_loss_constraints(it, y, mu, v, alpha, beta, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    nig_nll = NIG_NLL(it, y, mu, v, alpha,beta)

    nig_reg,nig_conf = NIG_Reg(y, mu, v, alpha,beta)

    nig_ce = NIG_CE(y, mu, v, alpha,beta)
    # nig_ce = NIG_ICE(y, mu, v, alpha,beta)

    nig_confa = 0.5*(beta / ((alpha - 1) + 1e-15))
    nig_confb = 0.5*torch.log((beta / ((alpha - 1) + 1e-15)))
    # ev_sum = nig_nll  + lambda_coef*nig_reg
    # ev_sum = nig_nll  + nig_ce
    ev_sum = nig_nll
    # ev_sum = nig_nll  + 0.7*nig_conf
    # ev_sum = nig_nll  + 0.5*nig_ce*nig_conf
    # ev_sum = nig_nll  + lambda_coef*nig_ce
    # ev_sum = nig_nll  + 0.5*(nig_ce*nig_confa+nig_confb)
    # ev_sum = nig_nll  + 0.5*nig_ce + lambda_coef*nig_reg

    evidential_loss = torch.mean(ev_sum)

    header = ['nig_nll', 'nig_reg', 'nig_loss', 'mse']
    values = [nig_nll, nig_reg, ev_sum, (y-mu)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_loss_constraints(it, y, mu, v, alpha, beta, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    nig_nll = NIG_NLL(it, y, mu, v, alpha,beta)

    nig_reg,nig_conf = NIG_Reg(y, mu, v, alpha,beta)

    nig_ce = NIG_CE(y, mu, v, alpha,beta)
    # nig_ce = NIG_ICE(y, mu, v, alpha,beta)

    nig_confa = 0.5*(beta / ((alpha - 1) + 1e-15))
    nig_confb = 0.5*torch.log((beta / ((alpha - 1) + 1e-15)))
    # ev_sum = nig_nll  + lambda_coef*nig_reg
    # ev_sum = nig_nll  + nig_ce
    # ev_sum = nig_nll  + 0.5*nig_ce
    ev_sum = nig_nll  + 0.01*nig_ce*nig_conf
    # ev_sum = nig_nll  + 0.5*nig_ce*nig_conf

    # ev_sum = nig_nll  + 0.7*nig_conf
    # ev_sum = nig_nll  + 0.5*nig_ce*nig_conf
    # ev_sum = nig_nll  + lambda_coef*nig_ce
    # ev_sum = nig_nll  + 0.5*(nig_ce*nig_confa+nig_confb)
    # ev_sum = nig_nll  + 0.5*nig_ce + lambda_coef*nig_reg

    evidential_loss = torch.mean(ev_sum)

    header = ['nig_nll', 'nig_reg', 'nig_loss', 'mse']
    values = [nig_nll, nig_reg, ev_sum, (y-mu)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_loss_constraints2(it, y, mu, v, alpha, beta, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    nig_nll = NIG_NLL(it, y, mu, v, alpha,beta)

    nig_reg,nig_conf = NIG_Reg(y, mu, v, alpha,beta)

    nig_ce = NIG_CE(y, mu, v, alpha,beta)
    # nig_ce = NIG_ICE(y, mu, v, alpha,beta)

    nig_confa = 0.5*(beta / ((alpha - 1) + 1e-15))
    nig_confb = 0.5*torch.log((beta / ((alpha - 1) + 1e-15)))
    # ev_sum = nig_nll  + lambda_coef*nig_reg
    # ev_sum = nig_nll  + nig_ce
    # ev_sum = nig_nll  + 0.5*nig_ce
    # ev_sum = nig_nll  + 0.01*nig_ce*nig_conf
    ev_sum = nig_nll  + 0.5*nig_ce*nig_conf

    # ev_sum = nig_nll  + 0.7*nig_conf
    # ev_sum = nig_nll  + 0.5*nig_ce*nig_conf
    # ev_sum = nig_nll  + lambda_coef*nig_ce
    # ev_sum = nig_nll  + 0.5*(nig_ce*nig_confa+nig_confb)
    # ev_sum = nig_nll  + 0.5*nig_ce + lambda_coef*nig_reg

    evidential_loss = torch.mean(ev_sum)

    header = ['nig_nll', 'nig_reg', 'nig_loss', 'mse']
    values = [nig_nll, nig_reg, ev_sum, (y-mu)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_Reg_loss_constraints(it, y, mu, v, alpha, beta, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    nig_nll = NIG_NLL(it, y, mu, v, alpha,beta)

    nig_reg,nig_conf = NIG_Reg(y, mu, v, alpha,beta)

    nig_ce = NIG_CE(y, mu, v, alpha,beta)
    # nig_ce = NIG_ICE(y, mu, v, alpha,beta)

    nig_confa = 0.5*(beta / ((alpha - 1) + 1e-15))
    nig_confb = 0.5*torch.log((beta / ((alpha - 1) + 1e-15)))
    # ev_sum = nig_nll  + lambda_coef*nig_reg
    # ev_sum = nig_nll  + nig_ce
    ev_sum = nig_nll  + 0.01*nig_reg*nig_conf
    # ev_sum = nig_nll  + 0.01*nig_ce*nig_conf

    # ev_sum = nig_nll  + 0.7*nig_conf
    # ev_sum = nig_nll  + 0.5*nig_ce*nig_conf
    # ev_sum = nig_nll  + lambda_coef*nig_ce
    # ev_sum = nig_nll  + 0.5*(nig_ce*nig_confa+nig_confb)
    # ev_sum = nig_nll  + 0.5*nig_ce + lambda_coef*nig_reg

    evidential_loss = torch.mean(ev_sum)

    header = ['nig_nll', 'nig_reg', 'nig_loss', 'mse']
    values = [nig_nll, nig_reg, ev_sum, (y-mu)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_Reg_Ce_loss_constraints(it, y, mu, v, alpha, beta, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    nig_nll = NIG_NLL(it, y, mu, v, alpha,beta)

    nig_reg,nig_conf = NIG_Reg(y, mu, v, alpha,beta)

    nig_ce = NIG_CE(y, mu, v, alpha,beta)
    # nig_ce = NIG_ICE(y, mu, v, alpha,beta)

    nig_confa = 0.5*(beta / ((alpha - 1) + 1e-15))
    nig_confb = 0.5*torch.log((beta / ((alpha - 1) + 1e-15)))
    # ev_sum = nig_nll  + lambda_coef*nig_reg
    # ev_sum = nig_nll  + nig_ce
    # ev_sum = nig_nll  + 0.01*nig_reg*nig_conf
    ev_sum = nig_nll  + 0.5*nig_ce + 0.01*nig_reg*nig_conf

    # ev_sum = nig_nll  + 0.7*nig_conf
    # ev_sum = nig_nll  + 0.5*nig_ce*nig_conf
    # ev_sum = nig_nll  + lambda_coef*nig_ce
    # ev_sum = nig_nll  + 0.5*(nig_ce*nig_confa+nig_confb)
    # ev_sum = nig_nll  + 0.5*nig_ce + lambda_coef*nig_reg

    evidential_loss = torch.mean(ev_sum)

    header = ['nig_nll', 'nig_reg', 'nig_loss', 'mse']
    values = [nig_nll, nig_reg, ev_sum, (y-mu)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_mmst_loss_constraints(it, y, mu, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, mu, sigma, v)

    st_ce = ST_CE(y, mu, sigma, v)

    # nig_confa = 0.5*(beta / ((alpha - 1) + 1e-15))
    # nig_confb = 0.5*torch.log((beta / ((alpha - 1) + 1e-15)))
    # ev_sum = nig_nll  + lambda_coef*nig_reg
    # ev_sum = nig_nll  + nig_ce
    ev_sum = st_nll  + 0.5*st_ce
    # ev_sum = nig_nll  + 0.7*nig_conf
    # ev_sum = nig_nll  + 0.5*nig_ce*nig_conf
    # ev_sum = nig_nll  + lambda_coef*nig_ce
    # ev_sum = nig_nll  + 0.5*(nig_ce*nig_confa+nig_confb)
    # ev_sum = nig_nll  + 0.5*nig_ce + lambda_coef*nig_reg

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_loss', 'st_ce']
    values = [st_nll,  ev_sum, st_ce]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_mmst_Reg_loss_constraints(it, y, mu, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, mu, sigma, v)

    st_ce = ST_CE(y, mu, sigma, v)
    
    st_Reg,st_conf = ST_Reg(y, mu, sigma, v)

    
    # nig_confa = 0.5*(beta / ((alpha - 1) + 1e-15))
    # nig_confb = 0.5*torch.log((beta / ((alpha - 1) + 1e-15)))
    # ev_sum = nig_nll  + lambda_coef*nig_reg
    # ev_sum = nig_nll  + nig_ce
    # ev_sum = st_nll  + 0.5*st_ce
    
    ev_sum = st_nll  + 0.01*st_Reg*st_conf

    
    # ev_sum = nig_nll  + 0.7*nig_conf
    # ev_sum = nig_nll  + 0.5*nig_ce*nig_conf
    # ev_sum = nig_nll  + lambda_coef*nig_ce
    # ev_sum = nig_nll  + 0.5*(nig_ce*nig_confa+nig_confb)
    # ev_sum = nig_nll  + 0.5*nig_ce + lambda_coef*nig_reg

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_loss', 'st_ce']
    values = [st_nll,  ev_sum, st_ce]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss


def calculate_evidential_mmst_Reg_Ce_loss_constraints(it, y, mu, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, mu, sigma, v)

    st_ce = ST_CE(y, mu, sigma, v)

    st_Reg, st_conf = ST_Reg(y, mu, sigma, v)

    # nig_confa = 0.5*(beta / ((alpha - 1) + 1e-15))
    # nig_confb = 0.5*torch.log((beta / ((alpha - 1) + 1e-15)))
    # ev_sum = nig_nll  + lambda_coef*nig_reg
    # ev_sum = nig_nll  + nig_ce
    # ev_sum = st_nll  + 0.5*st_ce

    ev_sum = st_nll + 0.5*st_ce + 0.01 * st_Reg * st_conf

    # ev_sum = nig_nll  + 0.7*nig_conf
    # ev_sum = nig_nll  + 0.5*nig_ce*nig_conf
    # ev_sum = nig_nll  + lambda_coef*nig_ce
    # ev_sum = nig_nll  + 0.5*(nig_ce*nig_confa+nig_confb)
    # ev_sum = nig_nll  + 0.5*nig_ce + lambda_coef*nig_reg

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_loss', 'st_ce']
    values = [st_nll, ev_sum, st_ce]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss
def calculate_evidential_stU_loss_constraints(it, y, u, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, u, sigma, v)

    st_reg,ST_conf = ST_Reg(y, u, sigma, v)

    ST_ce = ST_CE(y, u, sigma, v)
    # ST_ce = ST_ICE(y, u, sigma, v)

    ST_confa = 0.5/(sigma*sigma)
    ST_confb = 0.5*torch.log(sigma*sigma)
    # ev_sum = st_nll  + lambda_coef*st_reg
    # ev_sum = st_nll  + ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce
    ev_sum = st_nll  + 0.01*ST_ce*ST_conf

    # ev_sum = st_nll  + 0.7*ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*ST_conf
    # ev_sum = st_nll  + lambda_coef*ST_ce

    # ev_sum = st_nll  + 0.5*(ST_ce*ST_confa+ST_confb)
    # ev_sum = st_nll  + 0.5*ST_ce + lambda_coef*st_reg

    # ev_sum = st_nll  + lambda_coef*st_reg + ST_ce

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_reg', 'st_loss', 'mse']
    values = [st_nll, st_reg, ev_sum, (y-u)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_st_loss_constraints(it, y, u, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, u, sigma, v)

    st_reg,ST_conf = ST_Reg(y, u, sigma, v)

    ST_ce = ST_CE(y, u, sigma, v)
    # ST_ce = ST_ICE(y, u, sigma, v)

    ST_confa = 0.5/(sigma*sigma)
    ST_confb = 0.5*torch.log(sigma*sigma)
    # ev_sum = st_nll  + lambda_coef*st_reg
    # ev_sum = st_nll  + ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*st_reg
    ev_sum = st_nll  + 0.5*ST_ce

    # ev_sum = st_nll  + 0.7*ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*ST_conf
    # ev_sum = st_nll  + lambda_coef*ST_ce

    # ev_sum = st_nll  + 0.5*(ST_ce*ST_confa+ST_confb)
    # ev_sum = st_nll  + 0.5*ST_ce + lambda_coef*st_reg

    # ev_sum = st_nll  + lambda_coef*st_reg + ST_ce

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_reg', 'st_loss', 'mse']
    values = [st_nll, st_reg, ev_sum, (y-u)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_st_loss_constraints1(it, y, u, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, u, sigma, v)

    st_reg,ST_conf = ST_Reg(y, u, sigma, v)

    ST_ce = ST_CE(y, u, sigma, v)
    # ST_ce = ST_ICE(y, u, sigma, v)

    ST_confa = 0.5/(sigma*sigma)
    ST_confb = 0.5*torch.log(sigma*sigma)
    # ev_sum = st_nll  + lambda_coef*st_reg
    # ev_sum = st_nll  + ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*st_reg
    # ev_sum = st_nll  + 0.5*ST_ce
    ev_sum = st_nll  + 0.1*ST_ce

    # ev_sum = st_nll  + 0.7*ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*ST_conf
    # ev_sum = st_nll  + lambda_coef*ST_ce

    # ev_sum = st_nll  + 0.5*(ST_ce*ST_confa+ST_confb)
    # ev_sum = st_nll  + 0.5*ST_ce + lambda_coef*st_reg

    # ev_sum = st_nll  + lambda_coef*st_reg + ST_ce

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_reg', 'st_loss', 'mse']
    values = [st_nll, st_reg, ev_sum, (y-u)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_st_loss_constraints2(it, y, u, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, u, sigma, v)

    st_reg,ST_conf = ST_Reg(y, u, sigma, v)

    ST_ce = ST_CE(y, u, sigma, v)
    # ST_ce = ST_ICE(y, u, sigma, v)

    ST_confa = 0.5/(sigma*sigma)
    ST_confb = 0.5*torch.log(sigma*sigma)
    # ev_sum = st_nll  + lambda_coef*st_reg
    # ev_sum = st_nll  + ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*st_reg
    # ev_sum = st_nll  + 0.5*ST_ce
    ev_sum = st_nll

    # ev_sum = st_nll  + 0.7*ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*ST_conf
    # ev_sum = st_nll  + lambda_coef*ST_ce

    # ev_sum = st_nll  + 0.5*(ST_ce*ST_confa+ST_confb)
    # ev_sum = st_nll  + 0.5*ST_ce + lambda_coef*st_reg

    # ev_sum = st_nll  + lambda_coef*st_reg + ST_ce

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_reg', 'st_loss', 'mse']
    values = [st_nll, st_reg, ev_sum, (y-u)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_st_loss_constraints3(it, y, u, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, u, sigma, v)

    st_reg,ST_conf = ST_Reg(y, u, sigma, v)

    ST_ce = ST_CE(y, u, sigma, v)
    # ST_ce = ST_ICE(y, u, sigma, v)

    ST_confa = 0.5/(sigma*sigma)
    ST_confb = 0.5*torch.log(sigma*sigma)
    # ev_sum = st_nll  + lambda_coef*st_reg
    # ev_sum = st_nll  + ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*st_reg
    # ev_sum = st_nll  + 0.5*ST_ce
    ev_sum = st_nll  + 0.3*ST_ce

    # ev_sum = st_nll  + 0.7*ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*ST_conf
    # ev_sum = st_nll  + lambda_coef*ST_ce

    # ev_sum = st_nll  + 0.5*(ST_ce*ST_confa+ST_confb)
    # ev_sum = st_nll  + 0.5*ST_ce + lambda_coef*st_reg

    # ev_sum = st_nll  + lambda_coef*st_reg + ST_ce

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_reg', 'st_loss', 'mse']
    values = [st_nll, st_reg, ev_sum, (y-u)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_st_loss_constraints7(it, y, u, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, u, sigma, v)

    st_reg,ST_conf = ST_Reg(y, u, sigma, v)

    ST_ce = ST_CE(y, u, sigma, v)
    # ST_ce = ST_ICE(y, u, sigma, v)

    ST_confa = 0.5/(sigma*sigma)
    ST_confb = 0.5*torch.log(sigma*sigma)
    # ev_sum = st_nll  + lambda_coef*st_reg
    # ev_sum = st_nll  + ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*st_reg
    # ev_sum = st_nll  + 0.5*ST_ce
    ev_sum = st_nll  + 0.7*ST_ce

    # ev_sum = st_nll  + 0.7*ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*ST_conf
    # ev_sum = st_nll  + lambda_coef*ST_ce

    # ev_sum = st_nll  + 0.5*(ST_ce*ST_confa+ST_confb)
    # ev_sum = st_nll  + 0.5*ST_ce + lambda_coef*st_reg

    # ev_sum = st_nll  + lambda_coef*st_reg + ST_ce

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_reg', 'st_loss', 'mse']
    values = [st_nll, st_reg, ev_sum, (y-u)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_st_loss_constraints10(it, y, u, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, u, sigma, v)

    st_reg,ST_conf = ST_Reg(y, u, sigma, v)

    ST_ce = ST_CE(y, u, sigma, v)
    # ST_ce = ST_ICE(y, u, sigma, v)

    ST_confa = 0.5/(sigma*sigma)
    ST_confb = 0.5*torch.log(sigma*sigma)
    # ev_sum = st_nll  + lambda_coef*st_reg
    # ev_sum = st_nll  + ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*st_reg
    # ev_sum = st_nll  + 0.5*ST_ce
    ev_sum = st_nll  + ST_ce

    # ev_sum = st_nll  + 0.7*ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*ST_conf
    # ev_sum = st_nll  + lambda_coef*ST_ce

    # ev_sum = st_nll  + 0.5*(ST_ce*ST_confa+ST_confb)
    # ev_sum = st_nll  + 0.5*ST_ce + lambda_coef*st_reg

    # ev_sum = st_nll  + lambda_coef*st_reg + ST_ce

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_reg', 'st_loss', 'mse']
    values = [st_nll, st_reg, ev_sum, (y-u)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_st_loss_constraints_evi(it, y, u, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, u, sigma, v)

    st_reg,ST_conf = ST_Reg(y, u, sigma, v)

    ST_ce = ST_CE(y, u, sigma, v)
    # ST_ce = ST_ICE(y, u, sigma, v)

    ST_confa = 0.5/(sigma*sigma)
    ST_confb = 0.5*torch.log(sigma*sigma)
    # ev_sum = st_nll  + lambda_coef*st_reg
    # ev_sum = st_nll  + ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*st_reg
    # ev_sum = st_nll  + 0.5*ST_ce
    ev_sum = st_nll  + 0.01*ST_ce*ST_conf

    # ev_sum = st_nll  + 0.7*ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*ST_conf
    # ev_sum = st_nll  + lambda_coef*ST_ce

    # ev_sum = st_nll  + 0.5*(ST_ce*ST_confa+ST_confb)
    # ev_sum = st_nll  + 0.5*ST_ce + lambda_coef*st_reg

    # ev_sum = st_nll  + lambda_coef*st_reg + ST_ce

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_reg', 'st_loss', 'mse']
    values = [st_nll, st_reg, ev_sum, (y-u)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_st_loss_constraints_evi2(it, y, u, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, u, sigma, v)

    st_reg,ST_conf = ST_Reg(y, u, sigma, v)

    ST_ce = ST_CE(y, u, sigma, v)
    # ST_ce = ST_ICE(y, u, sigma, v)

    ST_confa = 0.5/(sigma*sigma)
    ST_confb = 0.5*torch.log(sigma*sigma)
    # ev_sum = st_nll  + lambda_coef*st_reg
    # ev_sum = st_nll  + ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*st_reg
    # ev_sum = st_nll  + 0.5*ST_ce
    ev_sum = st_nll  + 0.5*ST_ce*ST_conf

    # ev_sum = st_nll  + 0.7*ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*ST_conf
    # ev_sum = st_nll  + lambda_coef*ST_ce

    # ev_sum = st_nll  + 0.5*(ST_ce*ST_confa+ST_confb)
    # ev_sum = st_nll  + 0.5*ST_ce + lambda_coef*st_reg

    # ev_sum = st_nll  + lambda_coef*st_reg + ST_ce

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_reg', 'st_loss', 'mse']
    values = [st_nll, st_reg, ev_sum, (y-u)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_st_Reg_loss_constraints(it, y, u, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, u, sigma, v)

    st_reg,ST_conf = ST_Reg(y, u, sigma, v)

    ST_ce = ST_CE(y, u, sigma, v)
    # ST_ce = ST_ICE(y, u, sigma, v)

    ST_confa = 0.5/(sigma*sigma)
    ST_confb = 0.5*torch.log(sigma*sigma)
    # ev_sum = st_nll  + lambda_coef*st_reg
    # ev_sum = st_nll  + ST_ce
    ev_sum = st_nll  + 0.01*st_reg*ST_conf

    # ev_sum = st_nll  + 0.7*ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*ST_conf
    # ev_sum = st_nll  + lambda_coef*ST_ce

    # ev_sum = st_nll  + 0.5*(ST_ce*ST_confa+ST_confb)
    # ev_sum = st_nll  + 0.5*ST_ce + lambda_coef*st_reg

    # ev_sum = st_nll  + lambda_coef*st_reg + ST_ce

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_reg', 'st_loss', 'mse']
    values = [st_nll, st_reg, ev_sum, (y-u)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_st_Reg_Ce_loss_constraints(it, y, u, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, u, sigma, v)

    st_reg,ST_conf = ST_Reg(y, u, sigma, v)

    ST_ce = ST_CE(y, u, sigma, v)
    # ST_ce = ST_ICE(y, u, sigma, v)

    ST_confa = 0.5/(sigma*sigma)
    ST_confb = 0.5*torch.log(sigma*sigma)
    # ev_sum = st_nll  + lambda_coef*st_reg
    # ev_sum = st_nll  + ST_ce
    ev_sum = st_nll  + 0.5*ST_ce  + 0.01*st_reg*ST_conf

    # ev_sum = st_nll  + 0.7*ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*ST_conf
    # ev_sum = st_nll  + lambda_coef*ST_ce

    # ev_sum = st_nll  + 0.5*(ST_ce*ST_confa+ST_confb)
    # ev_sum = st_nll  + 0.5*ST_ce + lambda_coef*st_reg

    # ev_sum = st_nll  + lambda_coef*st_reg + ST_ce

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_reg', 'st_loss', 'mse']
    values = [st_nll, st_reg, ev_sum, (y-u)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_st_lamda_loss_constraints(it, y, u, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, u, sigma, v)

    st_reg,ST_conf = ST_Reg(y, u, sigma, v)

    ST_ce = ST_CE(y, u, sigma, v)
    # ST_ce = ST_ICE(y, u, sigma, v)

    ST_confa = 0.5/(sigma*sigma)
    ST_confb = 0.5*torch.log(sigma*sigma)
    # ev_sum = st_nll  + lambda_coef*st_reg
    # ev_sum = st_nll  + ST_ce
    ev_sum = st_nll  + 0.01*ST_ce

    # ev_sum = st_nll  + 0.7*ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*ST_conf
    # ev_sum = st_nll  + lambda_coef*ST_ce

    # ev_sum = st_nll  + 0.5*(ST_ce*ST_confa+ST_confb)
    # ev_sum = st_nll  + 0.5*ST_ce + lambda_coef*st_reg

    # ev_sum = st_nll  + lambda_coef*st_reg + ST_ce

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_reg', 'st_loss', 'mse']
    values = [st_nll, st_reg, ev_sum, (y-u)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_evidential_st_lamda_NLL_loss_constraints(it, y, u, sigma, v, lambda_coef=1.0):
    logging_dict = {}
    # nig_nll, logging_dict = NIG_NLL(it, y, mu, v, alpha,beta)
    st_nll = ST_NLL(it, y, u, sigma, v)

    st_reg,ST_conf = ST_Reg(y, u, sigma, v)

    ST_ce = ST_CE(y, u, sigma, v)
    # ST_ce = ST_ICE(y, u, sigma, v)

    ST_confa = 0.5/(sigma*sigma)
    ST_confb = 0.5*torch.log(sigma*sigma)
    # ev_sum = st_nll  + lambda_coef*st_reg
    # ev_sum = st_nll  + ST_ce
    ev_sum = st_nll

    # ev_sum = st_nll  + 0.7*ST_ce
    # ev_sum = st_nll  + 0.5*ST_ce*ST_conf
    # ev_sum = st_nll  + lambda_coef*ST_ce

    # ev_sum = st_nll  + 0.5*(ST_ce*ST_confa+ST_confb)
    # ev_sum = st_nll  + 0.5*ST_ce + lambda_coef*st_reg

    # ev_sum = st_nll  + lambda_coef*st_reg + ST_ce

    evidential_loss = torch.mean(ev_sum)

    header = ['st_nll', 'st_reg', 'st_loss', 'mse']
    values = [st_nll, st_reg, ev_sum, (y-u)**2]
    logging_dict = add_to_logging_dict(logging_dict, header, values)

    # return evidential_loss, logging_dict
    return evidential_loss

def calculate_cml_st_loss(u,sigma, v, y, loc, scale_2, df,input_num):
    confidence_list = []
    U_confidence_list = []
    predict_list = []
    epsilon = 1e-16

    # Modality confidence
    for num in range(input_num):
        modality_mean = u[num]
        U_confidence = 1 / ((sigma[num]*sigma[num]) + epsilon) + v[num]
        probability = F.softmax(modality_mean)
        confidence, predict = torch.max(probability, axis=1)
        # print(predict.shape)
        predict_list.append(predict)
        confidence_list.append(confidence)
        U_confidence_list.append(U_confidence)

    # Combine modality confidence
    combine_modality_mean = loc
    combine_U_confidence = 1 / ((scale_2*scale_2) + epsilon) + df
    combine_probability = F.softmax(combine_modality_mean)
    combine_confidence, combine_predict = torch.max(combine_probability, axis=1)
    predict_list.append(combine_predict)
    confidence_list.append(combine_confidence)
    U_confidence_list.append(combine_U_confidence)

    # cal uncertainty_loss
    cml_loss = 0
    for num in range(input_num):
        sign = (~(predict_list[num]!=y)).long()  # trick 1
        cml_loss += torch.nn.ReLU()(torch.sub(confidence_list[num], confidence_list[input_num]) * sign - epsilon).sum()

    # return evidential_loss, logging_dict
    return cml_loss

def calculate_cml_u_st_loss(u,sigma, v, y, loc, scale_2, df,input_num):
    confidence_list = []
    U_confidence_list = []
    predict_list = []
    epsilon = 1e-16

    # Modality confidence
    for num in range(input_num):
        modality_mean = u[num]
        U = 1 / ((sigma[num]*sigma[num]) + epsilon) + v[num]
        U_confidence = torch.exp(-U)
        probability = F.softmax(modality_mean)
        confidence, predict = torch.max(probability, axis=1)
        idex = predict.cpu().detach().int().numpy()
        # print(predict.shape)
        predict_list.append(predict)
        confidence_list.append(confidence)
        U_confidence_list.append(U_confidence[0,idex])

    # Combine modality confidence
    combine_modality_mean = loc
    combine_U = 1 / ((scale_2*scale_2) + epsilon) + df
    combine_U_confidence = torch.exp(-combine_U)

    combine_probability = F.softmax(combine_modality_mean)
    combine_confidence, combine_predict = torch.max(combine_probability, axis=1)
    combine_idex = combine_predict.cpu().detach().int().numpy()
    predict_list.append(combine_predict)
    confidence_list.append(combine_confidence)
    U_confidence_list.append(combine_U_confidence[0,combine_idex])

    # cal uncertainty_loss
    cml_loss = 0
    for num in range(input_num):
        sign = (~(predict_list[num]!=y)).long()  # trick 1
        cml_loss += torch.nn.ReLU()(torch.sub(U_confidence_list[num], U_confidence_list[input_num]) * sign - epsilon).sum()

    # return evidential_loss, logging_dict
    return cml_loss


def calculate_cml_loss(gamma, v, alpha, beta, y, loc, scale_2, df,input_num):
    confidence_list = []
    U_confidence_list = []
    predict_list = []
    epsilon = 1e-16

    # Modality confidence
    for num in range(input_num):
        modality_mean = gamma[num]
        U_confidence = v[num] + alpha[num] + 1 / (beta[num] + 1e-15)
        probability = F.softmax(modality_mean)
        confidence, predict = torch.max(probability, axis=1)
        # print(predict.shape)
        predict_list.append(predict)
        confidence_list.append(confidence)
        U_confidence_list.append(U_confidence)

    # Combine modality confidence
    combine_modality_mean = loc
    combine_U_confidence = 1 / ((scale_2*scale_2) + epsilon) + df
    combine_probability = F.softmax(combine_modality_mean)
    combine_confidence, combine_predict = torch.max(combine_probability, axis=1)
    predict_list.append(combine_predict)
    confidence_list.append(combine_confidence)
    U_confidence_list.append(combine_U_confidence)

    # cal uncertainty_loss
    cml_loss = 0
    for num in range(input_num):
        sign = (~(predict_list[num]!=y)).long()  # trick 1
        cml_loss += torch.nn.ReLU()(torch.sub(confidence_list[num], confidence_list[input_num]) * sign - epsilon).sum()

    # return evidential_loss, logging_dict
    return cml_loss

def calculate_cml_u_loss(gamma, v, alpha, beta, y, loc, scale_2, df,input_num):
    confidence_list = []
    U_confidence_list = []
    predict_list = []
    epsilon = 1e-16

    # Modality confidence
    for num in range(input_num):
        modality_mean = gamma[num]
        U = v[num] + alpha[num] + 1 / (beta[num] + 1e-15)
        U_confidence = U
        probability = F.softmax(modality_mean)
        confidence, predict = torch.max(probability, axis=1)
        idex = predict.cpu().detach().int().numpy()
        # print(predict.shape)
        predict_list.append(predict)
        confidence_list.append(confidence)
        U_confidence_list.append(U_confidence[0,idex])

    # Combine modality confidence
    combine_modality_mean = loc
    combine_U = 1 / ((scale_2*scale_2) + epsilon) + df
    # combine_U = df
    combine_U_confidence = combine_U

    combine_probability = F.softmax(combine_modality_mean)
    combine_confidence, combine_predict = torch.max(combine_probability, axis=1)
    combine_idex = combine_predict.cpu().detach().int().numpy()
    predict_list.append(combine_predict)
    confidence_list.append(combine_confidence)
    U_confidence_list.append(combine_U_confidence[0,combine_idex])

    # cal uncertainty_loss
    cml_loss = 0
    for num in range(input_num):
        sign = (~(predict_list[num]!=y)).long()  # trick 1
        cml_loss += torch.nn.ReLU()(torch.sub(U_confidence_list[num], U_confidence_list[input_num]) * sign - epsilon).sum()

    # return evidential_loss, logging_dict
    return cml_loss

def normal_para(x):
    k = dict()
    for m in range(len(x)):
        k[m] = x[m]/torch.max(x[m])
    return k

class EyeMost_prior(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims,args, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(ECNP_ST_Beta_fusion_C, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_base_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        self.resnet_3DNet = Medical_base_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.Classifiers= nn.ModuleList([self.res2net_2DNet, self.resnet_3DNet])
        # self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])

        # ---Evidential
        self.transform_gamma = nn.Sequential(nn.ReLU(), nn.Linear(2048, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_v = nn.Sequential(nn.ReLU(), nn.Linear(2048, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_alpha = nn.Sequential(nn.ReLU(), nn.Linear(2048, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_beta = nn.Sequential(nn.ReLU(), nn.Linear(2048, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_gamma_3D = nn.Sequential(nn.ReLU(), nn.Linear(8192, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_v_3D = nn.Sequential(nn.ReLU(), nn.Linear(8192, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_alpha_3D = nn.Sequential(nn.ReLU(), nn.Linear(8192, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_beta_3D = nn.Sequential(nn.ReLU(), nn.Linear(8192, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))

        # self.transform_gamma_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
        #                                      nn.Linear(64, self.classes))
        # self.transform_v_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
        #                                      nn.Linear(64, self.classes))
        # self.transform_alpha_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
        #                                      nn.Linear(64, self.classes))
        # self.transform_beta_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
        #                                      nn.Linear(64, self.classes))
        self._ev_dec_beta_min = args.ev_dec_beta_min
        self._ev_dec_alpha_max = args.ev_dec_alpha_max
        self._ev_dec_v_max = args.ev_dec_v_max
        self.ev_st_u_min = args.ev_st_u_min
        self.args = args

    def evidence(self, x):

        return F.softplus(x)

    def ST_Combin(self, gamma,v,alpha,beta):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        df, loc, scale_2 = dict(), dict(), dict()
        min_e = 1e-8
        # df_a ,loc_a, scale_2_a = dict(), dict(), dict()
        for m in range(2):
            df[m] = 2 * alpha[m] # v
            loc[m] = gamma[m] # u
            scale_2[m] = (beta[m] * (1 + v[m])) / v[m] / alpha[m] # sigma
        b, d = df[0].shape
        # normalization parameter
        df_normal = normal_para(df)
        scale_2_normal = normal_para(scale_2)
        # initialize

        # 1. matching student-t
        df_a = torch.zeros(df[0].shape).cuda()
        loc_a = torch.zeros(loc[0].shape).cuda()
        scale_2_a = torch.zeros(scale_2[0].shape).cuda()
        for i_b in range(b):
            m_0 = 0 # sign for minnum_v number in modality 1
            m_1 = 0 # sign for minnum_v number in modality 2
            for i_d in range(d):
                if df[0][i_b][i_d] < df[1][i_b][i_d]:
                # if df_normal[0][i_b][i_d] < df_normal[1][i_b][i_d]:
                    m_0 = m_0+1
                else:
                    m_1 = m_1+1

            if m_0 > m_1:
                # if scale_2_normal[0][i_b][i_d]/df_normal[0][i_b][i_d] > scale_2_normal[1][i_b][i_d]/df_normal[1][i_b][i_d]:
                # if scale_2[0][i_b][i_d] / df[0][i_b][i_d] > scale_2[1][i_b][i_d] / df[1][i_b][i_d]:
                df_combine = df[0][i_b]
                evi_0 = 1 / ((scale_2[0][i_b] * scale_2[0][i_b]) + min_e) + df[0][i_b]
                evi_1 = 1 / ((scale_2[1][i_b] * scale_2[1][i_b]) + min_e) + df[1][i_b]
                total_evi = evi_0 + evi_1
                loc_combine = torch.div(evi_0*loc[0][i_b],total_evi)+torch.div(evi_1*loc[1][i_b],total_evi)
                # loc_combine = loc[0][i_b]

                # loc_combine = (loc[0][i_b][i_d]+loc[1][i_b][i_d])/2
                # loc_combine = loc[0][i_b][i_d]*torch.div(df[1][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])+loc[1][i_b][i_d]*torch.div(df[0][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])
                # loc_combine = (loc[0][i_b][i_d]*torch.div(df_normal[1][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])+loc[1][i_b][i_d])*torch.div(df_normal[0][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])

                scale_M_2 = torch.div(scale_2[1][i_b] * df[1][i_b] * (df[0][i_b] - 2),
                                                ((df[1][i_b] - 2) * (df[0][i_b])))
                scale_2_combine = (scale_2[0][i_b] + scale_M_2)/2
                # scale_2_combine = scale_2[1][i_b][i_d] * df[1][i_b][i_d] * (df[0][i_b][i_d] -2)
                # scale_2_combine = torch.div(scale_2[1][i_b][i_d] * df[1][i_b][i_d] * (df[0][i_b][i_d] -2) , df[1][i_b][i_d]-2)

            elif m_0 == m_1:
                if min(df[0][i_b]) < min(df[1][i_b]):
                # if sum(df[0][i_b]) < sum(df[1][i_b]):

                    df_combine = df[0][i_b]
                    # loc_combine = loc[0][i_b]
                    evi_0 = 1 / ((scale_2[0][i_b] * scale_2[0][i_b]) + min_e) + df[0][i_b]
                    evi_1 = 1 / ((scale_2[1][i_b] * scale_2[1][i_b]) + min_e) + df[1][i_b]
                    total_evi = evi_0 + evi_1
                    loc_combine = torch.div(evi_0*loc[0][i_b],total_evi)+torch.div(evi_1*loc[1][i_b],total_evi)
                    scale_M_2 = torch.div(scale_2[1][i_b] * df[1][i_b] * (df[0][i_b] - 2),
                                                    ((df[1][i_b] - 2) * (df[0][i_b])))
                    scale_2_combine = (scale_2[0][i_b] + scale_M_2)/2

                else:
                    df_combine = df[1][i_b]
                    # loc_combine = loc[1][i_b]
                    evi_0 = 1 / ((scale_2[0][i_b] * scale_2[0][i_b]) + min_e) + df[0][i_b]
                    evi_1 = 1 / ((scale_2[1][i_b] * scale_2[1][i_b]) + min_e) + df[1][i_b]
                    total_evi = evi_0 + evi_1
                    loc_combine = torch.div(evi_0 * loc[0][i_b], total_evi) + torch.div(evi_1 * loc[1][i_b], total_evi)

                    scale_M_2 = torch.div(scale_2[0][i_b] * df[0][i_b] * (df[1][i_b] - 2),
                                                    ((df[0][i_b] - 2) * (df[1][i_b])))
                    scale_2_combine = (scale_2[1][i_b] + scale_M_2)/2

            else:
                df_combine = df[1][i_b]
                # loc_combine = loc[1][i_b]
                evi_0 = 1 / ((scale_2[0][i_b] * scale_2[0][i_b]) + min_e) + df[0][i_b]
                evi_1 = 1 / ((scale_2[1][i_b] * scale_2[1][i_b]) + min_e) + df[1][i_b]
                total_evi = evi_0 + evi_1
                loc_combine = torch.div(evi_0 * loc[0][i_b], total_evi) + torch.div(evi_1 * loc[1][i_b], total_evi)
                # loc_combine = (loc[0][i_b][i_d]+loc[1][i_b][i_d])/2
                # loc_combine = loc[1][i_b][i_d]*torch.div(df[0][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])+loc[0][i_b][i_d]*torch.div(df[1][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])
                #  loc_combine = (loc[1][i_b][i_d]*torch.div(df_normal[0][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])+loc[0][i_b][i_d])*torch.div(df_normal[1][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])

                scale_M_2 = torch.div(scale_2[0][i_b] * df[0][i_b] * (df[1][i_b] - 2) , ((df[0][i_b] - 2) * (df[1][i_b])))
                # scale_2_combine = torch.div(1 , ((df[0][i_b][i_d] - 2) * (df[1][i_b][i_d]-0)))
                # scale_2_combine = (df[0][i_b][i_d] - 2) * (df[1][i_b][i_d]-0)
                # scale_2_combine = torch.div(scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2) , df[0][i_b][i_d]-2)
                # scale_2_combine = scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2)
                # scale_2_combine = torch.div(scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2) , df[1][i_b][i_d]-0)
                # scale_2_combine = scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2)
                scale_2_combine = (scale_2[1][i_b] + scale_M_2) / 2

            df_a[i_b] = df_combine
            loc_a[i_b] = loc_combine
            scale_2_a[i_b] = scale_2_combine

            # df_a = df_combine
            # loc_a = loc_combine
            # scale_2_a = scale_2_combine
        # loss = nn.functional.mse_loss(scale_2_a, scale_2_a)

        # 2. combine two matching st:
        # alpha_a = df_a /2
        # beta_a = alpha_a
        # gamma_a = loc_a
        # v_a = 1/(scale_2_a-1)

        # return gamma_a,u_a,alpha_a
        # loss_1 = nn.functional.mse_loss(v_a, v_a)
        return loc_a, scale_2_a, df_a

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        gamma = dict()
        v = dict()
        alpha = dict()
        beta = dict()

        for m_num in range(self.modalties):
            backbone_output = self.Classifiers[m_num](input[m_num])
            batch_size, d = backbone_output.shape
            if d<=2048:
                inc_gamma = self.transform_gamma(backbone_output)
                logv = self.transform_v(backbone_output)
                logalpha = self.transform_alpha(backbone_output)
                logbeta = self.transform_beta(backbone_output)
            else:
                inc_gamma = self.transform_gamma_3D(backbone_output)
                logv = self.transform_v_3D(backbone_output)
                logalpha = self.transform_alpha_3D(backbone_output)
                logbeta = self.transform_beta_3D(backbone_output)
            gamma[m_num] = inc_gamma
            v[m_num] = self.evidence(logv)  # + 1.0
            alpha[m_num] = self.evidence(logalpha)
            alpha[m_num] = alpha[m_num] + 1
            beta[m_num] = self.evidence(logbeta)

            # The constraints
            alpha_thr = self._ev_dec_alpha_max * torch.ones(alpha[m_num] .shape).to(alpha[m_num] .device)
            alpha[m_num] = torch.min(alpha[m_num], alpha_thr)
            v_thr = self._ev_dec_v_max * torch.ones(v[m_num].shape).to(v[m_num].device)
            v[m_num] = torch.min(v[m_num], v_thr)
            beta_min = self._ev_dec_beta_min * torch.ones(beta[m_num].shape).to(beta[m_num].device)
            beta[m_num] = beta[m_num] + beta_min

        return gamma, v, alpha, beta

    def forward(self, X, y, global_step):
        # loss = torch.zeros(0, device=y.device)
        loss = 0

        # gamma = dict()
        # v = dict()
        # alpha = dict()
        # beta = dict()
        # df = dict()
        # loc = dict()
        # scale = dict()
        debug_save_logging_dict = dict()
        gamma, v, alpha, beta = self.infer(X)
        one_hot_y = torch.zeros(y.size(0), self.classes).cuda().scatter_(1, y.unsqueeze(1), 1)
        annealing_coef = min(0.5, global_step / self.lambda_epochs)

        for m_num in range(len(X)):
            # m_loss, debug_save_logging_dict[m_num] = calculate_evidential_loss_constraints(global_step, y[m_num], gamma[m_num], v[m_num], alpha[m_num],
            #                                                                           beta[m_num],
            #                                                                           lambda_coef=self.args.nig_nll_reg_coef)
            loss += calculate_evidential_loss_constraints(global_step, one_hot_y, gamma[m_num], v[m_num], alpha[m_num],
                                                                                      beta[m_num],
                                                                                      lambda_coef=annealing_coef)
            # m_loss, debug_save_logging_dict[m_num] = calculate_evidential_loss_constraints(global_step, one_hot_y, gamma[m_num], v[m_num], alpha[m_num],
            #                                                                           beta[m_num],
            #                                                                           lambda_coef=self.args.nig_nll_reg_coef)
            # loss += m_loss
        # loss_T =  torch.mean(loss_T)

        loc_a, scale_2_a, df_a = self.ST_Combin(gamma,v,alpha,beta)
        # df (float or Tensor) – degrees of freedom v
        # loc (float or Tensor) – mean of the distribution u
        # scale (float or Tensor) – scale of the distribution sigma

        # evidence_a = alpha_a - 1
        # loss_a = nn.functional.mse_loss(v_a, v_a)
        loss += calculate_evidential_st_loss_constraints(global_step, one_hot_y, loc_a, scale_2_a, df_a,
                                                                                      annealing_coef)
        # loss += m_a_loss
        # loss = torch.mean(loss)

        df_v = df_a
        loc_u = loc_a
        loc_u_min = self.ev_st_u_min * torch.ones(loc_u.shape).to(loc_u.device)
        loc_u = loc_u + loc_u_min
        scale_sigma = scale_2_a
        dist = torch.distributions.studentT.StudentT(df=df_v, loc=loc_u, scale=scale_sigma)
        # av_epis = scale_sigma * scale_sigma * (1+2/(df_v-2))
        av_epis = scale_sigma * (1+2/(df_v-2))
        if self.mode == "test":
            return dist, loc_u, loss, av_epis, gamma, v, alpha, beta
        else:
            return dist, loc_u, loss, av_epis

class EyeMost(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims,args, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(EyeMost, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.mode = args.mode
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_base_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        self.resnet_3DNet = Medical_base_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.Classifiers= nn.ModuleList([self.res2net_2DNet, self.resnet_3DNet])
        # self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])

        # ---Evidential
        self.transform_gamma = nn.Sequential(nn.ReLU(), nn.Linear(2048, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_v = nn.Sequential(nn.ReLU(), nn.Linear(2048, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_alpha = nn.Sequential(nn.ReLU(), nn.Linear(2048, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_beta = nn.Sequential(nn.ReLU(), nn.Linear(2048, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        if args.dataset == 'OLIVES':
            self.transform_gamma_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
                                                    nn.Linear(64, self.classes))
            self.transform_v_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
                                                nn.Linear(64, self.classes))
            self.transform_alpha_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
                                                    nn.Linear(64, self.classes))
            self.transform_beta_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
                                                   nn.Linear(64, self.classes))
        else:
            self.transform_gamma_3D = nn.Sequential(nn.ReLU(), nn.Linear(8192, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
            self.transform_v_3D = nn.Sequential(nn.ReLU(), nn.Linear(8192, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
            self.transform_alpha_3D = nn.Sequential(nn.ReLU(), nn.Linear(8192, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
            self.transform_beta_3D = nn.Sequential(nn.ReLU(), nn.Linear(8192, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))

        # self.transform_gamma_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
        #                                      nn.Linear(64, self.classes))
        # self.transform_v_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
        #                                      nn.Linear(64, self.classes))
        # self.transform_alpha_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
        #                                      nn.Linear(64, self.classes))
        # self.transform_beta_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
        #                                      nn.Linear(64, self.classes))
        self._ev_dec_beta_min = args.ev_dec_beta_min
        self._ev_dec_alpha_max = args.ev_dec_alpha_max
        self._ev_dec_v_max = args.ev_dec_v_max
        self.ev_st_u_min = args.ev_st_u_min
        self.args = args

    def evidence(self, x):

        return F.softplus(x)

    def ST_Combin(self, gamma,v,alpha,beta):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        df, loc, scale_2 = dict(), dict(), dict()
        min_e = 1e-8
        # df_a ,loc_a, scale_2_a = dict(), dict(), dict()
        for m in range(2):
            df[m] = 2 * alpha[m] # v
            loc[m] = gamma[m] # u
            scale_2[m] = (beta[m] * (1 + v[m])) / v[m] / alpha[m] # sigma
        b, d = df[0].shape
        # normalization parameter
        df_normal = normal_para(df)
        scale_2_normal = normal_para(scale_2)
        # initialize

        # 1. matching student-t
        df_a = torch.zeros(df[0].shape).cuda()
        loc_a = torch.zeros(loc[0].shape).cuda()
        scale_2_a = torch.zeros(scale_2[0].shape).cuda()
        for i_b in range(b):
            m_0 = 0 # sign for minnum_v number in modality 1
            m_1 = 0 # sign for minnum_v number in modality 2
            for i_d in range(d):
                if df[0][i_b][i_d] < df[1][i_b][i_d]:
                # if df_normal[0][i_b][i_d] < df_normal[1][i_b][i_d]:
                    m_0 = m_0+1
                else:
                    m_1 = m_1+1

            if m_0 > m_1:
                # if scale_2_normal[0][i_b][i_d]/df_normal[0][i_b][i_d] > scale_2_normal[1][i_b][i_d]/df_normal[1][i_b][i_d]:
                # if scale_2[0][i_b][i_d] / df[0][i_b][i_d] > scale_2[1][i_b][i_d] / df[1][i_b][i_d]:
                df_combine = df[0][i_b]
                evi_0 = df[0][i_b] + min_e
                evi_1 = df[1][i_b] + min_e
                total_evi = evi_0 + evi_1
                loc_combine = torch.div(evi_0*loc[0][i_b],total_evi)+torch.div(evi_1*loc[1][i_b],total_evi)
                # loc_combine = loc[0][i_b]

                # loc_combine = (loc[0][i_b][i_d]+loc[1][i_b][i_d])/2
                # loc_combine = loc[0][i_b][i_d]*torch.div(df[1][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])+loc[1][i_b][i_d]*torch.div(df[0][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])
                # loc_combine = (loc[0][i_b][i_d]*torch.div(df_normal[1][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])+loc[1][i_b][i_d])*torch.div(df_normal[0][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])

                scale_M_2 = torch.div(scale_2[1][i_b] * df[1][i_b] * (df[0][i_b] - 2),
                                                ((df[1][i_b] - 2) * (df[0][i_b])))
                scale_2_combine = (scale_2[0][i_b] + scale_M_2)/2
                # scale_2_combine = scale_2[1][i_b][i_d] * df[1][i_b][i_d] * (df[0][i_b][i_d] -2)
                # scale_2_combine = torch.div(scale_2[1][i_b][i_d] * df[1][i_b][i_d] * (df[0][i_b][i_d] -2) , df[1][i_b][i_d]-2)

            elif m_0 == m_1:
                if min(df[0][i_b]) < min(df[1][i_b]):
                # if sum(df[0][i_b]) < sum(df[1][i_b]):

                    df_combine = df[0][i_b]
                    # loc_combine = loc[0][i_b]
                    evi_0 = df[0][i_b] + min_e
                    evi_1 = df[1][i_b] + min_e
                    total_evi = evi_0 + evi_1
                    loc_combine = torch.div(evi_0 * loc[0][i_b], total_evi) + torch.div(evi_1 * loc[1][i_b], total_evi)
                    scale_M_2 = torch.div(scale_2[1][i_b] * df[1][i_b] * (df[0][i_b] - 2),
                                                    ((df[1][i_b] - 2) * (df[0][i_b])))
                    scale_2_combine = (scale_2[0][i_b] + scale_M_2)/2

                else:
                    df_combine = df[1][i_b]
                    # loc_combine = loc[1][i_b]
                    evi_0 = df[0][i_b] + min_e
                    evi_1 = df[1][i_b] + min_e
                    total_evi = evi_0 + evi_1
                    loc_combine = torch.div(evi_0 * loc[0][i_b], total_evi) + torch.div(evi_1 * loc[1][i_b], total_evi)

                    scale_M_2 = torch.div(scale_2[0][i_b] * df[0][i_b] * (df[1][i_b] - 2),
                                                    ((df[0][i_b] - 2) * (df[1][i_b])))
                    scale_2_combine = (scale_2[1][i_b] + scale_M_2)/2

            else:
                df_combine = df[1][i_b]
                # loc_combine = loc[1][i_b]
                evi_0 = df[0][i_b] + min_e
                evi_1 = df[1][i_b] + min_e
                total_evi = evi_0 + evi_1
                loc_combine = torch.div(evi_0*loc[0][i_b],total_evi)+torch.div(evi_1*loc[1][i_b],total_evi)
                # loc_combine = (loc[0][i_b][i_d]+loc[1][i_b][i_d])/2
                # loc_combine = loc[1][i_b][i_d]*torch.div(df[0][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])+loc[0][i_b][i_d]*torch.div(df[1][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])
                #  loc_combine = (loc[1][i_b][i_d]*torch.div(df_normal[0][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])+loc[0][i_b][i_d])*torch.div(df_normal[1][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])

                scale_M_2 = torch.div(scale_2[0][i_b] * df[0][i_b] * (df[1][i_b] - 2) , ((df[0][i_b] - 2) * (df[1][i_b])))
                # scale_2_combine = torch.div(1 , ((df[0][i_b][i_d] - 2) * (df[1][i_b][i_d]-0)))
                # scale_2_combine = (df[0][i_b][i_d] - 2) * (df[1][i_b][i_d]-0)
                # scale_2_combine = torch.div(scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2) , df[0][i_b][i_d]-2)
                # scale_2_combine = scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2)
                # scale_2_combine = torch.div(scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2) , df[1][i_b][i_d]-0)
                # scale_2_combine = scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2)
                scale_2_combine = (scale_2[1][i_b] + scale_M_2) / 2

            df_a[i_b] = df_combine
            loc_a[i_b] = loc_combine
            scale_2_a[i_b] = scale_2_combine

            # df_a = df_combine
            # loc_a = loc_combine
            # scale_2_a = scale_2_combine
        # loss = nn.functional.mse_loss(scale_2_a, scale_2_a)

        # 2. combine two matching st:
        # alpha_a = df_a /2
        # beta_a = alpha_a
        # gamma_a = loc_a
        # v_a = 1/(scale_2_a-1)

        # return gamma_a,u_a,alpha_a
        # loss_1 = nn.functional.mse_loss(v_a, v_a)
        return loc_a, scale_2_a, df_a

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        gamma = dict()
        v = dict()
        alpha = dict()
        beta = dict()

        for m_num in range(self.modalties):
            backbone_output = self.Classifiers[m_num](input[m_num])
            batch_size, d = backbone_output.shape
            if d<=2048:
                inc_gamma = self.transform_gamma(backbone_output)
                logv = self.transform_v(backbone_output)
                logalpha = self.transform_alpha(backbone_output)
                logbeta = self.transform_beta(backbone_output)
            else:
                inc_gamma = self.transform_gamma_3D(backbone_output)
                logv = self.transform_v_3D(backbone_output)
                logalpha = self.transform_alpha_3D(backbone_output)
                logbeta = self.transform_beta_3D(backbone_output)
            gamma[m_num] = inc_gamma
            v[m_num] = self.evidence(logv)  # + 1.0
            alpha[m_num] = self.evidence(logalpha)
            alpha[m_num] = alpha[m_num] + 1
            beta[m_num] = self.evidence(logbeta)

            # The constraints
            alpha_thr = self._ev_dec_alpha_max * torch.ones(alpha[m_num] .shape).to(alpha[m_num] .device)
            alpha[m_num] = torch.min(alpha[m_num], alpha_thr)
            v_thr = self._ev_dec_v_max * torch.ones(v[m_num].shape).to(v[m_num].device)
            v[m_num] = torch.min(v[m_num], v_thr)
            beta_min = self._ev_dec_beta_min * torch.ones(beta[m_num].shape).to(beta[m_num].device)
            beta[m_num] = beta[m_num] + beta_min

        return gamma, v, alpha, beta

    def forward(self, X, y, global_step):
        # loss = torch.zeros(0, device=y.device)
        loss = 0

        # gamma = dict()
        # v = dict()
        # alpha = dict()
        # beta = dict()
        # df = dict()
        # loc = dict()
        # scale = dict()
        debug_save_logging_dict = dict()
        gamma, v, alpha, beta = self.infer(X)
        one_hot_y = torch.zeros(y.size(0), self.classes).cuda().scatter_(1, y.unsqueeze(1), 1)
        annealing_coef = min(0.5, global_step / self.lambda_epochs)

        for m_num in range(len(X)):
            # m_loss, debug_save_logging_dict[m_num] = calculate_evidential_loss_constraints(global_step, y[m_num], gamma[m_num], v[m_num], alpha[m_num],
            #                                                                           beta[m_num],
            #                                                                           lambda_coef=self.args.nig_nll_reg_coef)
            loss += calculate_evidential_loss_constraints(global_step, one_hot_y, gamma[m_num], v[m_num], alpha[m_num],
                                                                                      beta[m_num],
                                                                                      lambda_coef=annealing_coef)
            # m_loss, debug_save_logging_dict[m_num] = calculate_evidential_loss_constraints(global_step, one_hot_y, gamma[m_num], v[m_num], alpha[m_num],
            #                                                                           beta[m_num],
            #                                                                           lambda_coef=self.args.nig_nll_reg_coef)
            # loss += m_loss
        # loss_T =  torch.mean(loss_T)

        loc_a, scale_2_a, df_a = self.ST_Combin(gamma,v,alpha,beta)
        # df (float or Tensor) – degrees of freedom v
        # loc (float or Tensor) – mean of the distribution u
        # scale (float or Tensor) – scale of the distribution sigma

        # evidence_a = alpha_a - 1
        # loss_a = nn.functional.mse_loss(v_a, v_a)
        loss += calculate_evidential_st_loss_constraints(global_step, one_hot_y, loc_a, scale_2_a, df_a,
                                                                                      annealing_coef)
        # loss += m_a_loss
        # loss = torch.mean(loss)

        df_v = df_a
        loc_u = loc_a
        loc_u_min = self.ev_st_u_min * torch.ones(loc_u.shape).to(loc_u.device)
        loc_u = loc_u + loc_u_min
        scale_sigma = scale_2_a
        dist = torch.distributions.studentT.StudentT(df=df_v, loc=loc_u, scale=scale_sigma)
        # av_epis = scale_sigma * scale_sigma * (1+2/(df_v-2))
        av_epis = scale_sigma * (1+2/(df_v-2))
        if self.mode == "test":
            return dist, loc_u, loss, av_epis, gamma, v, alpha, beta
        else:
            return dist, loc_u, loss, av_epis

class Base_transformer(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims, args):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(Base_transformer, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.mode = args.mode
        # ---- 2D Transformer Backbone ----
        self.transformer_2DNet = fundus_build_model() # SWIN-Transformer

        # ---- 3D Transformer Backbone ----
        # UNETR from MONAI
        self.transformer_3DNet = UNETR_base_3DNet(num_classes=self.classes)

        # ---Evidential
        self.fc = nn.Sequential(nn.ReLU(), nn.Linear(1024+768, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))

        self.ce_loss = nn.CrossEntropyLoss()

        self.args = args

    def forward(self, X, y):
        # loss = torch.zeros(0, device=y.device)
        backboneout_1 = self.transformer_2DNet(X[0])
        backboneout_2 = self.transformer_3DNet(X[1])
        combine_features = torch.cat([backboneout_1,backboneout_2],1)
        pred = self.fc(combine_features)
        loss = self.ce_loss(pred, y)

        loss = torch.mean(loss)
        return pred, loss


class EyeMost_Plus_transformer(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims, args, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(EyeMost_Plus_transformer, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.mode = args.mode
        self.lambda_epochs = lambda_epochs
        # ---- 2D Transformer Backbone ----
        self.transformer_2DNet = fundus_build_model() # SWIN-Transformer

        # ---- 3D Transformer Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        # UNETR from MONAI
        self.transformer_3DNet = UNETR_base_3DNet(num_classes=self.classes)
        self.Classifiers= nn.ModuleList([self.transformer_2DNet, self.transformer_3DNet])
        # self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])

        # ---Evidential
        self.transform_gamma = nn.Sequential(nn.ReLU(), nn.Linear(1024, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_v = nn.Sequential(nn.ReLU(), nn.Linear(1024, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_alpha = nn.Sequential(nn.ReLU(), nn.Linear(1024, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_beta = nn.Sequential(nn.ReLU(), nn.Linear(1024, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))

        self.transform_gamma_3D = nn.Sequential(nn.ReLU(), nn.Linear(768, 64), nn.ReLU(),
                                            nn.Linear(64, self.classes))
        self.transform_v_3D = nn.Sequential(nn.ReLU(), nn.Linear(768, 64), nn.ReLU(),
                                            nn.Linear(64, self.classes))
        self.transform_alpha_3D = nn.Sequential(nn.ReLU(), nn.Linear(768, 64), nn.ReLU(),
                                            nn.Linear(64, self.classes))
        self.transform_beta_3D = nn.Sequential(nn.ReLU(), nn.Linear(768, 64), nn.ReLU(),
                                            nn.Linear(64, self.classes))

        self._ev_dec_beta_min = args.ev_dec_beta_min
        self._ev_dec_alpha_max = args.ev_dec_alpha_max
        self._ev_dec_v_max = args.ev_dec_v_max
        self.ev_st_u_min = args.ev_st_u_min
        self.args = args

    def evidence(self, x):

        return F.softplus(x)

    def ST_Combin(self, gamma,v,alpha,beta):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        df, loc, scale_2 = dict(), dict(), dict()
        min_e = 1e-8
        # df_a ,loc_a, scale_2_a = dict(), dict(), dict()
        for m in range(2):
            df[m] = 2 * alpha[m] # v
            loc[m] = gamma[m] # u
            scale_2[m] = (beta[m] * (1 + v[m])) / v[m] / alpha[m] # sigma
        b, d = df[0].shape
        # normalization parameter
        df_normal = normal_para(df)
        scale_2_normal = normal_para(scale_2)
        # initialize

        # 1. matching student-t
        df_a = torch.zeros(df[0].shape).cuda()
        loc_a = torch.zeros(loc[0].shape).cuda()
        scale_2_a = torch.zeros(scale_2[0].shape).cuda()
        for i_b in range(b):
            m_0 = 0 # sign for minnum_v number in modality 1
            m_1 = 0 # sign for minnum_v number in modality 2
            for i_d in range(d):
                if df[0][i_b][i_d] < df[1][i_b][i_d]:
                # if df_normal[0][i_b][i_d] < df_normal[1][i_b][i_d]:
                    m_0 = m_0+1
                else:
                    m_1 = m_1+1

            if m_0 > m_1:
                # if scale_2_normal[0][i_b][i_d]/df_normal[0][i_b][i_d] > scale_2_normal[1][i_b][i_d]/df_normal[1][i_b][i_d]:
                # if scale_2[0][i_b][i_d] / df[0][i_b][i_d] > scale_2[1][i_b][i_d] / df[1][i_b][i_d]:
                df_combine = df[0][i_b]
                evi_0 = df[0][i_b] + min_e
                evi_1 = df[1][i_b] + min_e
                total_evi = evi_0 + evi_1
                loc_combine = torch.div(evi_0*loc[0][i_b],total_evi)+torch.div(evi_1*loc[1][i_b],total_evi)
                # loc_combine = loc[0][i_b]

                # loc_combine = (loc[0][i_b][i_d]+loc[1][i_b][i_d])/2
                # loc_combine = loc[0][i_b][i_d]*torch.div(df[1][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])+loc[1][i_b][i_d]*torch.div(df[0][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])
                # loc_combine = (loc[0][i_b][i_d]*torch.div(df_normal[1][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])+loc[1][i_b][i_d])*torch.div(df_normal[0][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])

                scale_M_2 = torch.div(scale_2[1][i_b] * df[1][i_b] * (df[0][i_b] - 2),
                                                ((df[1][i_b] - 2) * (df[0][i_b])))
                scale_2_combine = (scale_2[0][i_b] + scale_M_2)/2
                # scale_2_combine = scale_2[1][i_b][i_d] * df[1][i_b][i_d] * (df[0][i_b][i_d] -2)
                # scale_2_combine = torch.div(scale_2[1][i_b][i_d] * df[1][i_b][i_d] * (df[0][i_b][i_d] -2) , df[1][i_b][i_d]-2)

            elif m_0 == m_1:
                if min(df[0][i_b]) < min(df[1][i_b]):
                # if sum(df[0][i_b]) < sum(df[1][i_b]):

                    df_combine = df[0][i_b]
                    # loc_combine = loc[0][i_b]
                    evi_0 = df[0][i_b] + min_e
                    evi_1 = df[1][i_b] + min_e
                    total_evi = evi_0 + evi_1
                    loc_combine = torch.div(evi_0 * loc[0][i_b], total_evi) + torch.div(evi_1 * loc[1][i_b], total_evi)
                    scale_M_2 = torch.div(scale_2[1][i_b] * df[1][i_b] * (df[0][i_b] - 2),
                                                    ((df[1][i_b] - 2) * (df[0][i_b])))
                    scale_2_combine = (scale_2[0][i_b] + scale_M_2)/2

                else:
                    df_combine = df[1][i_b]
                    # loc_combine = loc[1][i_b]
                    evi_0 = df[0][i_b] + min_e
                    evi_1 = df[1][i_b] + min_e
                    total_evi = evi_0 + evi_1
                    loc_combine = torch.div(evi_0 * loc[0][i_b], total_evi) + torch.div(evi_1 * loc[1][i_b], total_evi)

                    scale_M_2 = torch.div(scale_2[0][i_b] * df[0][i_b] * (df[1][i_b] - 2),
                                                    ((df[0][i_b] - 2) * (df[1][i_b])))
                    scale_2_combine = (scale_2[1][i_b] + scale_M_2)/2

            else:
                df_combine = df[1][i_b]
                # loc_combine = loc[1][i_b]
                evi_0 = df[0][i_b] + min_e
                evi_1 = df[1][i_b] + min_e
                total_evi = evi_0 + evi_1
                loc_combine = torch.div(evi_0 * loc[0][i_b], total_evi) + torch.div(evi_1 * loc[1][i_b], total_evi)
                # loc_combine = (loc[0][i_b][i_d]+loc[1][i_b][i_d])/2
                # loc_combine = loc[1][i_b][i_d]*torch.div(df[0][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])+loc[0][i_b][i_d]*torch.div(df[1][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])
                #  loc_combine = (loc[1][i_b][i_d]*torch.div(df_normal[0][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])+loc[0][i_b][i_d])*torch.div(df_normal[1][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])

                scale_M_2 = torch.div(scale_2[0][i_b] * df[0][i_b] * (df[1][i_b] - 2) , ((df[0][i_b] - 2) * (df[1][i_b])))
                # scale_2_combine = torch.div(1 , ((df[0][i_b][i_d] - 2) * (df[1][i_b][i_d]-0)))
                # scale_2_combine = (df[0][i_b][i_d] - 2) * (df[1][i_b][i_d]-0)
                # scale_2_combine = torch.div(scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2) , df[0][i_b][i_d]-2)
                # scale_2_combine = scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2)
                # scale_2_combine = torch.div(scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2) , df[1][i_b][i_d]-0)
                # scale_2_combine = scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2)
                scale_2_combine = (scale_2[1][i_b] + scale_M_2) / 2

            df_a[i_b] = df_combine
            loc_a[i_b] = loc_combine
            scale_2_a[i_b] = scale_2_combine

            # df_a = df_combine
            # loc_a = loc_combine
            # scale_2_a = scale_2_combine
        # loss = nn.functional.mse_loss(scale_2_a, scale_2_a)

        # 2. combine two matching st:
        # alpha_a = df_a /2
        # beta_a = alpha_a
        # gamma_a = loc_a
        # v_a = 1/(scale_2_a-1)

        # return gamma_a,u_a,alpha_a
        # loss_1 = nn.functional.mse_loss(v_a, v_a)
        return loc_a, scale_2_a, df_a

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        gamma = dict()
        v = dict()
        alpha = dict()
        beta = dict()

        for m_num in range(self.modalties):
            backbone_output = self.Classifiers[m_num](input[m_num])
            batch_size, d = backbone_output.shape
            if d>768:
                inc_gamma = self.transform_gamma(backbone_output)
                logv = self.transform_v(backbone_output)
                logalpha = self.transform_alpha(backbone_output)
                logbeta = self.transform_beta(backbone_output)
            else:
                inc_gamma = self.transform_gamma_3D(backbone_output)
                logv = self.transform_v_3D(backbone_output)
                logalpha = self.transform_alpha_3D(backbone_output)
                logbeta = self.transform_beta_3D(backbone_output)
            gamma[m_num] = inc_gamma
            v[m_num] = self.evidence(logv)  # + 1.0
            alpha[m_num] = self.evidence(logalpha)
            alpha[m_num] = alpha[m_num] + 1
            beta[m_num] = self.evidence(logbeta)

            # The constraints
            alpha_thr = self._ev_dec_alpha_max * torch.ones(alpha[m_num] .shape).to(alpha[m_num] .device)
            alpha[m_num] = torch.min(alpha[m_num], alpha_thr)
            v_thr = self._ev_dec_v_max * torch.ones(v[m_num].shape).to(v[m_num].device)
            v[m_num] = torch.min(v[m_num], v_thr)
            beta_min = self._ev_dec_beta_min * torch.ones(beta[m_num].shape).to(beta[m_num].device)
            beta[m_num] = beta[m_num] + beta_min

        return gamma, v, alpha, beta

    def forward(self, X, y, global_step):
        # loss = torch.zeros(0, device=y.device)
        loss = 0

        # gamma = dict()
        # v = dict()
        # alpha = dict()
        # beta = dict()
        # df = dict()
        # loc = dict()
        # scale = dict()
        debug_save_logging_dict = dict()
        gamma, v, alpha, beta = self.infer(X)
        one_hot_y = torch.zeros(y.size(0), self.classes).cuda().scatter_(1, y.unsqueeze(1), 1)
        annealing_coef = min(0.5, global_step / self.lambda_epochs)
        modality_num = len(X)
        for m_num in range(len(X)):
            # m_loss, debug_save_logging_dict[m_num] = calculate_evidential_loss_constraints(global_step, y[m_num], gamma[m_num], v[m_num], alpha[m_num],
            #                                                                           beta[m_num],
            #                                                                           lambda_coef=self.args.nig_nll_reg_coef)
            loss += calculate_evidential_loss_constraints(global_step, one_hot_y, gamma[m_num], v[m_num], alpha[m_num],
                                                                                      beta[m_num],
                                                                                      lambda_coef=annealing_coef)
            # m_loss, debug_save_logging_dict[m_num] = calculate_evidential_loss_constraints(global_step, one_hot_y, gamma[m_num], v[m_num], alpha[m_num],
            #                                                                           beta[m_num],
            #                                                                           lambda_coef=self.args.nig_nll_reg_coef)
            # loss += m_loss
        # loss_T =  torch.mean(loss_T)

        loc_a, scale_2_a, df_a = self.ST_Combin(gamma,v,alpha,beta)
        # df (float or Tensor) – degrees of freedom v
        # loc (float or Tensor) – mean of the distribution u
        # scale (float or Tensor) – scale of the distribution sigma

        # evidence_a = alpha_a - 1
        # loss_a = nn.functional.mse_loss(v_a, v_a)
        loss += calculate_evidential_st_loss_constraints(global_step, one_hot_y, loc_a, scale_2_a, df_a,
                                                                                      annealing_coef)
        # loss += m_a_loss
        # loss = torch.mean(loss)

        # CML loss
        cml_loss = calculate_cml_loss(gamma, v, alpha, beta, y, loc_a, scale_2_a, df_a,modality_num)
        weight_beta  = 10

        loss += weight_beta * cml_loss

        df_v = df_a
        loc_u = loc_a
        loc_u_min = self.ev_st_u_min * torch.ones(loc_u.shape).to(loc_u.device)
        loc_u = loc_u + loc_u_min
        scale_sigma = scale_2_a
        dist = torch.distributions.studentT.StudentT(df=df_v, loc=loc_u, scale=scale_sigma)
        # av_epis = scale_sigma * scale_sigma * (1+2/(df_v-2))
        av_epis = scale_sigma * (1+2/(df_v-2))
        if self.mode == "test":
            return dist, loc_u, loss, av_epis, gamma, v, alpha, beta
        else:
            return dist, loc_u, loss, av_epis

class EyeMost_Plus(nn.Module):
    def __init__(self, classes, modalties, classifiers_dims, args, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of modalties
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(EyeMost_Plus, self).__init__()
        self.modalties = modalties
        self.classes = classes
        self.mode = args.mode
        self.lambda_epochs = lambda_epochs
        # ---- 2D Res2Net Backbone ----
        self.res2net_2DNet = Medical_base_2DNet(num_classes=self.classes)

        # ---- 3D ResNet Backbone ----
        classifier_OCT_dims = classifiers_dims[0]
        self.resnet_3DNet = Medical_base_3DNet(classifier_OCT_dims,num_classes=self.classes)
        self.Classifiers= nn.ModuleList([self.res2net_2DNet, self.resnet_3DNet])
        # self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])

        # ---Evidential
        self.transform_gamma = nn.Sequential(nn.ReLU(), nn.Linear(2048, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_v = nn.Sequential(nn.ReLU(), nn.Linear(2048, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_alpha = nn.Sequential(nn.ReLU(), nn.Linear(2048, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        self.transform_beta = nn.Sequential(nn.ReLU(), nn.Linear(2048, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
        if args.dataset == 'OLIVES':
            self.transform_gamma_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
                                                    nn.Linear(64, self.classes))
            self.transform_v_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
                                                nn.Linear(64, self.classes))
            self.transform_alpha_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
                                                    nn.Linear(64, self.classes))
            self.transform_beta_3D = nn.Sequential(nn.ReLU(), nn.Linear(3072, 64), nn.ReLU(),
                                                   nn.Linear(64, self.classes))
        else:
            self.transform_gamma_3D = nn.Sequential(nn.ReLU(), nn.Linear(8192, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
            self.transform_v_3D = nn.Sequential(nn.ReLU(), nn.Linear(8192, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
            self.transform_alpha_3D = nn.Sequential(nn.ReLU(), nn.Linear(8192, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))
            self.transform_beta_3D = nn.Sequential(nn.ReLU(), nn.Linear(8192, 64), nn.ReLU(),
                                             nn.Linear(64, self.classes))

        self._ev_dec_beta_min = args.ev_dec_beta_min
        self._ev_dec_alpha_max = args.ev_dec_alpha_max
        self._ev_dec_v_max = args.ev_dec_v_max
        self.ev_st_u_min = args.ev_st_u_min
        self.args = args

    def evidence(self, x):

        return F.softplus(x)

    def ST_Combin(self, gamma,v,alpha,beta):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        df, loc, scale_2 = dict(), dict(), dict()
        min_e = 1e-8
        # df_a ,loc_a, scale_2_a = dict(), dict(), dict()
        for m in range(2):
            df[m] = 2 * alpha[m] # v
            loc[m] = gamma[m] # u
            scale_2[m] = (beta[m] * (1 + v[m])) / v[m] / alpha[m] # sigma
        b, d = df[0].shape
        # normalization parameter
        df_normal = normal_para(df)
        scale_2_normal = normal_para(scale_2)
        # initialize

        # 1. matching student-t
        df_a = torch.zeros(df[0].shape).cuda()
        loc_a = torch.zeros(loc[0].shape).cuda()
        scale_2_a = torch.zeros(scale_2[0].shape).cuda()
        for i_b in range(b):
            m_0 = 0 # sign for minnum_v number in modality 1
            m_1 = 0 # sign for minnum_v number in modality 2
            for i_d in range(d):
                if df[0][i_b][i_d] < df[1][i_b][i_d]:
                # if df_normal[0][i_b][i_d] < df_normal[1][i_b][i_d]:
                    m_0 = m_0+1
                else:
                    m_1 = m_1+1

            if m_0 > m_1:
                # if scale_2_normal[0][i_b][i_d]/df_normal[0][i_b][i_d] > scale_2_normal[1][i_b][i_d]/df_normal[1][i_b][i_d]:
                # if scale_2[0][i_b][i_d] / df[0][i_b][i_d] > scale_2[1][i_b][i_d] / df[1][i_b][i_d]:
                df_combine = df[0][i_b]
                evi_0 = df[0][i_b] + min_e
                evi_1 = df[1][i_b] + min_e
                total_evi = evi_0 + evi_1
                loc_combine = torch.div(evi_0*loc[0][i_b],total_evi)+torch.div(evi_1*loc[1][i_b],total_evi)
                # loc_combine = loc[0][i_b]

                # loc_combine = (loc[0][i_b][i_d]+loc[1][i_b][i_d])/2
                # loc_combine = loc[0][i_b][i_d]*torch.div(df[1][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])+loc[1][i_b][i_d]*torch.div(df[0][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])
                # loc_combine = (loc[0][i_b][i_d]*torch.div(df_normal[1][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])+loc[1][i_b][i_d])*torch.div(df_normal[0][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])

                scale_M_2 = torch.div(scale_2[1][i_b] * df[1][i_b] * (df[0][i_b] - 2),
                                                ((df[1][i_b] - 2) * (df[0][i_b])))
                scale_2_combine = (scale_2[0][i_b] + scale_M_2)/2
                # scale_2_combine = scale_2[1][i_b][i_d] * df[1][i_b][i_d] * (df[0][i_b][i_d] -2)
                # scale_2_combine = torch.div(scale_2[1][i_b][i_d] * df[1][i_b][i_d] * (df[0][i_b][i_d] -2) , df[1][i_b][i_d]-2)

            elif m_0 == m_1:
                if min(df[0][i_b]) < min(df[1][i_b]):
                # if sum(df[0][i_b]) < sum(df[1][i_b]):

                    df_combine = df[0][i_b]
                    # loc_combine = loc[0][i_b]
                    evi_0 = df[0][i_b] + min_e
                    evi_1 = df[1][i_b] + min_e
                    total_evi = evi_0 + evi_1
                    loc_combine = torch.div(evi_0 * loc[0][i_b], total_evi) + torch.div(evi_1 * loc[1][i_b], total_evi)
                    scale_M_2 = torch.div(scale_2[1][i_b] * df[1][i_b] * (df[0][i_b] - 2),
                                                    ((df[1][i_b] - 2) * (df[0][i_b])))
                    scale_2_combine = (scale_2[0][i_b] + scale_M_2)/2

                else:
                    df_combine = df[1][i_b]
                    # loc_combine = loc[1][i_b]
                    evi_0 = df[0][i_b] + min_e
                    evi_1 = df[1][i_b] + min_e
                    total_evi = evi_0 + evi_1
                    loc_combine = torch.div(evi_0 * loc[0][i_b], total_evi) + torch.div(evi_1 * loc[1][i_b], total_evi)

                    scale_M_2 = torch.div(scale_2[0][i_b] * df[0][i_b] * (df[1][i_b] - 2),
                                                    ((df[0][i_b] - 2) * (df[1][i_b])))
                    scale_2_combine = (scale_2[1][i_b] + scale_M_2)/2

            else:
                df_combine = df[1][i_b]
                # loc_combine = loc[1][i_b]
                evi_0 = df[0][i_b] + min_e
                evi_1 = df[1][i_b] + min_e
                total_evi = evi_0 + evi_1
                loc_combine = torch.div(evi_0 * loc[0][i_b], total_evi) + torch.div(evi_1 * loc[1][i_b], total_evi)
                # loc_combine = (loc[0][i_b][i_d]+loc[1][i_b][i_d])/2
                # loc_combine = loc[1][i_b][i_d]*torch.div(df[0][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])+loc[0][i_b][i_d]*torch.div(df[1][i_b][i_d],df[0][i_b][i_d]+df[1][i_b][i_d])
                #  loc_combine = (loc[1][i_b][i_d]*torch.div(df_normal[0][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])+loc[0][i_b][i_d])*torch.div(df_normal[1][i_b][i_d],df_normal[0][i_b][i_d]+df_normal[1][i_b][i_d])

                scale_M_2 = torch.div(scale_2[0][i_b] * df[0][i_b] * (df[1][i_b] - 2) , ((df[0][i_b] - 2) * (df[1][i_b])))
                # scale_2_combine = torch.div(1 , ((df[0][i_b][i_d] - 2) * (df[1][i_b][i_d]-0)))
                # scale_2_combine = (df[0][i_b][i_d] - 2) * (df[1][i_b][i_d]-0)
                # scale_2_combine = torch.div(scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2) , df[0][i_b][i_d]-2)
                # scale_2_combine = scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2)
                # scale_2_combine = torch.div(scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2) , df[1][i_b][i_d]-0)
                # scale_2_combine = scale_2[0][i_b][i_d] * df[0][i_b][i_d] * (df[1][i_b][i_d] - 2)
                scale_2_combine = (scale_2[1][i_b] + scale_M_2) / 2

            df_a[i_b] = df_combine
            loc_a[i_b] = loc_combine
            scale_2_a[i_b] = scale_2_combine

            # df_a = df_combine
            # loc_a = loc_combine
            # scale_2_a = scale_2_combine
        # loss = nn.functional.mse_loss(scale_2_a, scale_2_a)

        # 2. combine two matching st:
        # alpha_a = df_a /2
        # beta_a = alpha_a
        # gamma_a = loc_a
        # v_a = 1/(scale_2_a-1)

        # return gamma_a,u_a,alpha_a
        # loss_1 = nn.functional.mse_loss(v_a, v_a)
        return loc_a, scale_2_a, df_a

    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        gamma = dict()
        v = dict()
        alpha = dict()
        beta = dict()

        for m_num in range(self.modalties):
            backbone_output = self.Classifiers[m_num](input[m_num])
            batch_size, d = backbone_output.shape
            if d<=2048:
                inc_gamma = self.transform_gamma(backbone_output)
                logv = self.transform_v(backbone_output)
                logalpha = self.transform_alpha(backbone_output)
                logbeta = self.transform_beta(backbone_output)
            else:
                inc_gamma = self.transform_gamma_3D(backbone_output)
                logv = self.transform_v_3D(backbone_output)
                logalpha = self.transform_alpha_3D(backbone_output)
                logbeta = self.transform_beta_3D(backbone_output)
            gamma[m_num] = inc_gamma
            v[m_num] = self.evidence(logv)  # + 1.0
            alpha[m_num] = self.evidence(logalpha)
            alpha[m_num] = alpha[m_num] + 1
            beta[m_num] = self.evidence(logbeta)

            # The constraints
            alpha_thr = self._ev_dec_alpha_max * torch.ones(alpha[m_num] .shape).to(alpha[m_num] .device)
            alpha[m_num] = torch.min(alpha[m_num], alpha_thr)
            v_thr = self._ev_dec_v_max * torch.ones(v[m_num].shape).to(v[m_num].device)
            v[m_num] = torch.min(v[m_num], v_thr)
            beta_min = self._ev_dec_beta_min * torch.ones(beta[m_num].shape).to(beta[m_num].device)
            beta[m_num] = beta[m_num] + beta_min

        return gamma, v, alpha, beta

    def forward(self, X, y, global_step):
        # loss = torch.zeros(0, device=y.device)
        loss = 0

        # gamma = dict()
        # v = dict()
        # alpha = dict()
        # beta = dict()
        # df = dict()
        # loc = dict()
        # scale = dict()
        debug_save_logging_dict = dict()
        gamma, v, alpha, beta = self.infer(X)
        one_hot_y = torch.zeros(y.size(0), self.classes).cuda().scatter_(1, y.unsqueeze(1), 1)
        annealing_coef = min(0.5, global_step / self.lambda_epochs)
        modality_num = len(X)
        for m_num in range(len(X)):
            # m_loss, debug_save_logging_dict[m_num] = calculate_evidential_loss_constraints(global_step, y[m_num], gamma[m_num], v[m_num], alpha[m_num],
            #                                                                           beta[m_num],
            #                                                                           lambda_coef=self.args.nig_nll_reg_coef)
            loss += calculate_evidential_loss_constraints(global_step, one_hot_y, gamma[m_num], v[m_num], alpha[m_num],
                                                                                      beta[m_num],
                                                                                      lambda_coef=annealing_coef)
            # m_loss, debug_save_logging_dict[m_num] = calculate_evidential_loss_constraints(global_step, one_hot_y, gamma[m_num], v[m_num], alpha[m_num],
            #                                                                           beta[m_num],
            #                                                                           lambda_coef=self.args.nig_nll_reg_coef)
            # loss += m_loss
        # loss_T =  torch.mean(loss_T)

        loc_a, scale_2_a, df_a = self.ST_Combin(gamma,v,alpha,beta)
        # df (float or Tensor) – degrees of freedom v
        # loc (float or Tensor) – mean of the distribution u
        # scale (float or Tensor) – scale of the distribution sigma

        # evidence_a = alpha_a - 1
        # loss_a = nn.functional.mse_loss(v_a, v_a)
        loss += calculate_evidential_st_loss_constraints(global_step, one_hot_y, loc_a, scale_2_a, df_a,
                                                                                      annealing_coef)
        # loss += m_a_loss
        # loss = torch.mean(loss)

        # CML loss
        cml_loss = calculate_cml_loss(gamma, v, alpha, beta, y, loc_a, scale_2_a, df_a,modality_num)
        weight_beta  = 10

        loss += weight_beta * cml_loss

        df_v = df_a
        loc_u = loc_a
        loc_u_min = self.ev_st_u_min * torch.ones(loc_u.shape).to(loc_u.device)
        loc_u = loc_u + loc_u_min
        scale_sigma = scale_2_a
        dist = torch.distributions.studentT.StudentT(df=df_v, loc=loc_u, scale=scale_sigma)
        # av_epis = scale_sigma * scale_sigma * (1+2/(df_v-2))
        av_epis = scale_sigma * (1+2/(df_v-2))
        if self.mode == "test":
            return dist, loc_u, loss, av_epis, gamma, v, alpha, beta
        else:
            return dist, loc_u, loss, av_epis
