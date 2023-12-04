import os
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import TMC,TMC_WAO,ECNP_ST,MM_ST,MM_NIG_ST,MM_ST_NIG,MM_NIG,MM_UNIG,ECNP_ST_Beta_CMLU2,ECNP_ST_Beta_CMLU3,ECNP_IST,ECNP_UST,MM_UST,ECNP_ST_Beta_fusion_C_DF_CML_lamda0,ECNP_ST_Beta_fusion_C_DF_CML_2D,ECNP_ST_Beta_fusion_C_DF_CML_3D,ECNP_ST_Beta_fusion_C_DF_CML_lamda1,ECNP_ST_Beta_fusion_C_DF_CML_lamda3,ECNP_ST_Beta_fusion_C_DF_CML_lamda7,ECNP_ST_Beta_fusion_C_DF_CML_lamda10,MM_ST_fusion_C_DF,ECNP_UVST,ECNP_ST_Beta_fusion1,ECNP_ST_Beta_fusion4,ECNP_ST_Beta_fusion_C_DF_CMLU_DF,ECNP_ST_Beta_fusion_C_DF_CML_variance,ECNP_ST_Beta_fusion_C_DF_CML_loss_evi3,ECNP_ST_Beta_fusion_C_DF_CML_loss_evi2, ECNP_ST_Beta_fusion_C_DF_CML_loss,ECNP_ST_Beta_fusion_C_DF_CML_loss_evi,ECNP_ST_lamda,ECNP_ST_Beta_fusion,ECNP_ST_Beta_fusion2,ECNP_ST_Beta_fusion3,ECNP_ST_Beta_fusion_C_DF_CML,ECNP_ST_Beta_fusion_C_DF_CMLU,ECNP_ST_Beta_fusion_C,ECNP_ST_Beta_fusion_C_DF,ECNP_ST_Beta_fusion_C_CMLU,ECNP_ST_lamda_NLL,ECNP_ST_Beta_CMLU4,ECNP_ST_Beta_CMLU5,MM_ST_Reg_Ce_CML,MM_ST_Reg_Ce_CMLU,ECNP_ST_Beta,MM_ST_Reg,MM_ST_Reg_CML,MM_ST_Reg_CMLU,MM_ST_Reg_Ce,ECNP_ST_Beta_loss,ECNP_ST_Beta_Reg_CMLU,MM_NIG_ablation,ECNP_ST_Beta_CML,ECNP_ST_Beta_Reg_CML,ECNP_ST_Beta_CMLU,ECNP_ST_Beta_Reg,ECNP_ST_Beta_Reg_Ce
from sklearn.model_selection import KFold
from data import Multi_modal_data,GAMMA_sub1_dataset,GAMMA_dataset,OLIVES_dataset
import warnings
import matplotlib.pyplot as plt
import numpy as np
from metrics import cal_ece,cal_ece_our
from metrics2 import calc_aurc_eaurc,calc_nll_brier
from scipy.io import loadmat
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
import torch.nn as nn
import seaborn as sns
import torch.nn.functional as F
import math
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from numpy import log, sqrt
from scipy.special import psi, beta
import logging
def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

def st_entropy(nu):
    S = 0.5 * (nu + 1) * (psi(0.5 * (nu + 1)) - psi(0.5 * nu))
    S += log(sqrt(nu) * beta(0.5 * nu, 0.5))
    log_S = log(S)
    return log_S

def Uentropy(logits,c):

    pc = F.softmax(logits, dim=1)
    logits = F.log_softmax(logits, dim=1)
    u_all = -pc * logits / math.log(c)
    NU = torch.sum(u_all[:,1:u_all.shape[1]], dim=1)
    return NU

def entropy(sigma_2,predict_id):
    id = int(predict_id)
    entropy_list = 0.5 * np.log(2 * np.pi * np.exp(1.) * (sigma_2.cpu().detach().float().numpy()))
    # entropy_list = np.log(sigma_2.cpu().detach().float().numpy())

    NU = entropy_list[0,id]
    return NU

def loss_plot(args,loss):
    num = args.end_epochs
    x = [i for i in range(num)]
    plot_save_path = r'results/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = plot_save_path+str(args.model_name)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.end_epochs)+'_loss.jpg'
    list_loss = list(loss)
    plt.figure()
    plt.plot(x,loss,label='loss')
    plt.legend()
    plt.savefig(save_loss)

def metrics_plot(arg,name,*args):
    num = arg.end_epochs
    names = name.split('&')
    metrics_value = args
    i=0
    x = [i for i in range(num)]
    plot_save_path = r'results/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = plot_save_path + str(arg.model_name) + '_' + str(arg.batch_size) + '_' + str(arg.dataset) + '_' + str(arg.end_epochs) + '_'+name+'.jpg'
    plt.figure()
    for l in metrics_value:
        plt.plot(x,l,label=str(names[i]))
        #plt.scatter(x,l,label=str(l))
        i+=1
    plt.legend()
    plt.savefig(save_metrics)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def convert_u_list(list):

    npu_list = np.zeros(len(list))
    for j in range(len(list)):
        npu_list[j] = (list[j])

    return npu_list

def find_in_u(list_acc,u_list):
    c_list = []
    inc_list = []

    for i in range(len(list_acc)):
        if list_acc[i] == 1:
            c_list.append(i)
        else:
            inc_list.append(i)

    c_u_list = np.zeros(len(c_list))
    for j in range(len(c_list)):
        c_u_list[j] = (u_list[c_list[j]])

    inc_u_list = np.zeros(len(inc_list))
    for j in range(len(inc_list)):
        inc_u_list[j] = (u_list[inc_list[j]])
    return c_u_list,inc_u_list

def find_in_entropy(list_acc,entropy_list):
    c_list = []
    inc_list = []

    for i in range(len(list_acc)):
        if list_acc[i] == 1:
            c_list.append(i)
        else:
            inc_list.append(i)

    c_entropy_list = np.zeros(len(c_list))
    for j in range(len(c_list)):
        c_entropy_list[j] = (entropy_list[c_list[j]])

    inc_entropy_list = np.zeros(len(inc_list))
    for j in range(len(inc_list)):
        inc_entropy_list[j] = (entropy_list[inc_list[j]])
    return c_entropy_list,inc_entropy_list


def train(epoch,train_loader,model):
    model.train()
    loss_meter = AverageMeter()
    # loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].cuda())
        target = Variable(target.long().cuda())
        # target = Variable(np.array(target)).cuda())

        # refresh the optimizer
        optimizer.zero_grad()
        evidences, evidence_a, loss, _ = model(data, target, epoch)
        print("total loss %f"%loss)
        # compute gradients and take step
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        # for i in range(0,len(loss_meter)):
        #     loss_list = loss_list.append(loss_meter[i].avg)
    return loss_meter

def val(current_epoch,val_loader,model,best_acc):
    model.eval()
    loss_meter = AverageMeter()
    correct_num, data_num = 0, 0
    for batch_idx, (data, target) in enumerate(val_loader):
        for m_num in range(len(data)):
            data[m_num] = Variable(data[m_num].float().cuda())
        data_num += target.size(0)
        with torch.no_grad():
            target = Variable(target.long().cuda())
            evidences, evidence_a, loss,_ = model(data, target, epoch)
            _, predicted = torch.max(evidence_a.data, 1)
            correct_num += (predicted == target).sum().item()
            loss_meter.update(loss.item())
    aver_acc = correct_num / data_num
    print('====> acc: {:.4f}'.format(aver_acc))
    if evidence_a.shape[1] >2:
        if aver_acc > best_acc:
            print('aver_acc:{} > best_acc:{}'.format(aver_acc, best_acc))
            best_acc = aver_acc
            print('===========>save best model!')
            file_name = os.path.join(args.save_dir,
                                 args.model_name + '_' + args.dataset + '_' + args.folder + '_best_epoch.pth')
            torch.save({
            'epoch': current_epoch,
            'state_dict': model.state_dict(),
            },
            file_name)
        return loss_meter.avg, best_acc

    else:
        if (current_epoch + 1) % int(args.end_epochs - 1) == 0 \
                or (current_epoch + 1) % int(args.end_epochs - 2) == 0 \
                or (current_epoch + 1) % int(args.end_epochs - 3) == 0:
            file_name = os.path.join(args.save_dir,
                                 args.model_name + '_' + args.dataset + '_' + args.folder + '_epoch_{}.pth'.format(
                                     current_epoch))
            torch.save({
            'epoch': current_epoch,
            'state_dict': model.state_dict(),
            },
            file_name)
        if aver_acc > best_acc:
            print('aver_acc:{} > best_acc:{}'.format(aver_acc, best_acc))
            best_acc = aver_acc
            print('===========>save best model!')
            file_name = os.path.join(args.save_dir,
                                 args.model_name + '_' + args.dataset + '_' + args.folder + '_best_epoch.pth')
            torch.save({ 'epoch': current_epoch,'state_dict': model.state_dict()},file_name)
        # if (current_epoch + 1) % int(args.end_epochs - 1) == 0 \
        #         or (current_epoch + 1) % int(args.end_epochs - 2) == 0 \
        #         or (current_epoch + 1) % int(args.end_epochs - 3) == 0:
        #     file_name = os.path.join(args.save_dir,
        #                          args.model_name + '_' + args.dataset + '_' + args.folder + '_epoch_{}.pth'.format(
        #                              current_epoch))
        #     torch.save({
        #     'epoch': current_epoch,
        #     'state_dict': model.state_dict(),
        #     },
        #     file_name)
        #     if aver_acc > best_acc:
        #         print('aver_acc:{} > best_acc:{}'.format(aver_acc, best_acc))
        #         best_acc = aver_acc
        #         print('===========>save best model!')
        #         file_name = os.path.join(args.save_dir,
        #                          args.model_name + '_' + args.dataset + '_' + args.folder + '_best_epoch.pth')
        #         torch.save({ 'epoch': current_epoch,'state_dict': model.state_dict()},file_name)
        return loss_meter.avg, best_acc

def test(args, test_loader,model,epoch):
    if args.num_classes == 2:
        if args.test_epoch > 100:
            load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 args.model_name + '_' + args.dataset +'_'+ args.folder + '_best_epoch.pth')
        else:
            load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 args.model_name + '_' + args.dataset +'_'+ args.folder + '_epoch_{}.pth'.format(args.test_epoch))
    else:
        if args.test_epoch > 100:
            load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 args.model_name + '_' + args.dataset +'_'+ args.folder + '_best_epoch.pth')
        else:
            load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 args.model_name + '_' + args.dataset +'_'+ args.folder + '_epoch_{}.pth'.format(args.test_epoch))

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(
            os.path.join(args.save_dir + '/' + args.model_name +'_'+args.dataset+ '_epoch_' + str(args.test_epoch))))
    else:
        print('There is no resume file to load!')
    model.eval()
    list_acc = []
    u_list =[]
    Fundus_au_list= []
    OCT_au_list= []
    OCT_eu_list= []
    Fundus_eu_list= []
    AU_two = dict()
    EU_two = dict()
    u_entropy_list = []
    correct_list = []
    entropy_list =[]
    ece_list = []
    nll_list = []
    brier_list = []
    label_list = []
    prediction_list = []
    probability_list = []
    evidence_list = []
    one_hot_label_list = []
    one_hot_probability_list = []
    # sp = nn.Softplus()
    correct_num, data_num = 0, 0
    start_time = time.time()
    time_list=[]
    for batch_idx, (data, target) in enumerate(test_loader):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].float().cuda())
        data_num += target.size(0)
        with torch.no_grad():
            target = Variable(target.long().cuda())
            evidences, evidence_a, _, u_a= model(data, target, epoch)
            # evidences, evidence_a, _, u_a, u, sigma, v = model(data, target, epoch)

            # ## NIG
            # AU_two[0] = beta[0]/(alpha[0]-1)
            # AU_two[1] = beta[1]/(alpha[1]-1)
            #
            # EU_two[0] = beta[0]/(alpha[0]-1)/v[0]
            # EU_two[1] = beta[1]/(alpha[1]-1)/v[1]
            #
            # ## ST
            # AU_two[0] = v[0]/(v[0]-2)
            # AU_two[1] = v[1]/(v[1]-2)
            #
            # EU_two[0] = sigma[0] * sigma[0] * (1+2/(v[0]-2))
            # EU_two[1] = sigma[1] * sigma[1] * (1+2/(v[1]-2))

            elapsed_time = time.time() - start_time
            time_list.append(elapsed_time)
            # probability_list.append(b_a.cpu().detach().float().numpy())
            correct_pred, predicted = torch.max(evidence_a.data, 1)
            correct_num += (predicted == target).sum().item()
            correct = (predicted == target)

            list_acc.append((predicted == target).sum().item())
            if args.model_name =="ResNet_ECNP"or args.model_name =="ResNet_ECNP_IST" or args.model_name =="ResNet_ECNP_Beta_Reg_Ce" or args.model_name =="ResNet_MMST_Reg_CML" or args.model_name =="ResNet_MMST_Reg_CMLU" \
                    or args.model_name =="ResNet_ECNP_Beta"or args.model_name =="ResNet_MMST_Reg" or args.model_name =="ResNet_MMST_Reg_Ce"  or args.model_name =="ECNP_ST_Beta_Reg_CMLU" or args.model_name == "ResNet_ECNP_Beta_Reg"  \
                    or args.model_name == "ResNet_ECNP_Beta_Reg_CML" or args.model_name =="ResNet_ECNP_Beta_loss"or args.model_name =="ResNet_ECNP_Beta_CML" or args.model_name =="ResNet_ECNP_Beta_CMLU"or args.model_name =="ResNet_ECNP_Beta_CML_50"\
                    or args.model_name =="ResNet_ECNP_lamda_NLL" or args.model_name =="ResNet_ECNPU"or args.model_name =="ResNet_ECNP_UST"or args.model_name =="ResNet_ECNP_UVST" or args.model_name =="ResNet_MMST"or args.model_name =="ResNet_MMUST"\
                    or args.model_name =="ResNet_MM_NIGST"or args.model_name =="ResNet_MM_STNIG"or args.model_name =="ResNet_MM_NIG"or args.model_name =="ResNet_MM_UNIG"or args.model_name =="ResNet_MM_NIG_ablation"or args.model_name =="ResNet_MMST_Reg_CML"\
                    or args.model_name =="ResNet_MMST_Reg_CMLU"or args.model_name =="ResNet_MMST_Reg_Ce_CML" or args.model_name =="ResNet_MMST_Reg_Ce_CMLU"or args.model_name =="ResNet_ECNP_Beta_Reg_CMLU"or args.model_name =="ResNet_ECNP_Beta_CMLU"\
                    or args.model_name =="ResNet_ECNP_Beta_CMLU3"or args.model_name =="ResNet_ECNP_Beta_CMLU2"or args.model_name =="ResNet_ECNP_Beta_CMLU4"or args.model_name =="ResNet_ECNP_Beta_CMLU5"\
                    or args.model_name =="ResNet_ECNP_ST_Beta_fusion"or args.model_name =="ResNet_ECNP_ST_Beta_fusion1"or args.model_name =="ResNet_ECNP_ST_Beta_fusion4"or args.model_name =="ResNet_ECNP_ST_Beta_fusion2"or args.model_name =="ResNet_ECNP_ST_Beta_fusion3"or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C"\
                    or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF"or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF_CMLU"or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF_CMLU_DF"\
                    or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_CMLU" or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF_CML_lamda0"or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF_CML_lamda1"or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF_CML_lamda3"\
                    or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF_CML_lamda7"or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF_CML_lamda10"\
                    or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_CML" or args.model_name =="ResNet_MM_ST_fusion_C_DF"or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF_CML"\
                    or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF_CML_loss"or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF_CML_variance"\
                    or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF_CML_loss_evi"or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF_CML_loss_evi2"\
                    or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF_CML_loss_evi3"or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF_CML_2D"or args.model_name =="ResNet_ECNP_ST_Beta_fusion_C_DF_CML_3D":

                u_list.append(u_a[0][predicted.cpu().detach().float().numpy()]) #
                # Fundus_au_list.append(AU_two[0][0][predicted.cpu().detach().float().numpy()])
                # OCT_au_list.append(AU_two[1][0][predicted.cpu().detach().float().numpy()])
                # Fundus_eu_list.append(EU_two[0][0][predicted.cpu().detach().float().numpy()])
                # OCT_eu_list.append(EU_two[1][0][predicted.cpu().detach().float().numpy()])

                if args.model_name =="ResNet_MMST":
                    u_entropy_list.append(st_entropy(evidences.df.cpu().detach().float().numpy()))
            elif args.model_name =="ResNet_TMC":
                u_list.append(u_a[0])
            else:
                print('There is no this model name')
                raise NameError
            probability = torch.softmax(evidence_a, dim=1).cpu().detach().float().numpy()
            one_hot_label = F.one_hot(target, num_classes=args.num_classes).squeeze(dim=0).cpu().detach().float().numpy()
            # NLL brier
            nll, brier = calc_nll_brier(probability, evidence_a, target, one_hot_label)
            nll_list.append(nll)
            brier_list.append(brier)
            prediction_list.append(predicted.cpu().detach().float().numpy())
            label_list.append(target.cpu().detach().float().numpy())
            correct_list.append(correct.cpu().detach().float().numpy())
            one_hot_label_list.append(F.one_hot(target, num_classes=args.num_classes).squeeze(dim=0).cpu().detach().float().numpy())
            probability_list.append(torch.softmax(evidence_a, dim=1).cpu().detach().float().numpy()[:,1])
            evidence_list.append(evidence_a.cpu().detach().float().numpy())
            # probability_list.append(sp(evidence_a).cpu().detach().float().numpy()[:,1])
            one_hot_probability_list.append(torch.softmax(evidence_a, dim=1).data.squeeze(dim=0).cpu().detach().float().numpy())
            # one_hot_probability_list.append(sp(evidence_a).data.squeeze(dim=0).cpu().detach().float().numpy())
            # one_hot_probability_list.append(evidence_a.data.squeeze(dim=0).cpu().detach().float().numpy())
            # one_hot_probability_list.append(F.softplus(evidence_a).data.squeeze(dim=0).cpu().detach().float().numpy())
            # epoch_auc_list.append(epoch_auc)
            # ece
            ece_list.append(cal_ece_our(torch.squeeze(evidence_a), target))
            # entropy
            # entropy_list.append(Uentropy(evidence_a,args.num_classes))
            entropy_list.append(entropy(evidence_a,predicted.cpu().detach().float().numpy()))

    # one_hot_probability_list = one_hot_probability_list/max(one_hot_probability_list)
    # probability_list = probability_list/max(probability_list)

    np_u_list = convert_u_list(u_list)
    # np_OCT_au_list = convert_u_list(OCT_au_list)
    # np_Fundus_au_list = convert_u_list(Fundus_au_list)
    # np_OCT_eu_list = convert_u_list(OCT_eu_list)
    # np_Fundus_eu_list = convert_u_list(Fundus_eu_list)
    logging.info('Single sample test time consumption {:.2f} seconds!'.format(sum(time_list)/len(time_list)))
    print('Single sample test time consumption {:.2f} seconds!'.format(sum(time_list)/len(time_list)))
    if args.num_classes > 2:
        epoch_auc = metrics.roc_auc_score(one_hot_label_list,one_hot_probability_list, multi_class='ovo')
    else:
        epoch_auc = metrics.roc_auc_score(label_list,probability_list)

    # correct_list = (prediction_list==label_list)
    aurc, eaurc = calc_aurc_eaurc(probability_list, correct_list)

    normalized_u_list=[]
    normalized_au_list = []
    xmin = min(u_list)
    xmax = max(u_list)

    for i in range(len(u_list)):
        normalized_u_list.append((u_list[i] - xmin) / (xmax-xmin))

    c_u, inc_u = find_in_u(list_acc,u_list)

    # Fundus
    # for i in range(len(np_Fundus_au_list)):
    #     normalized_au_list.append((np_Fundus_au_list[i] - xmin) / (xmax-xmin))
    # c_u, inc_u = find_in_u(list_acc,np_Fundus_au_list)
    # normalized_c_u, normalized_inc_u = find_in_u(list_acc,normalized_au_list)

    mean_inc_u = inc_u.mean()
    mean_c_u = c_u.mean()
    # np.append(inc_u, values=max(c_u+0.00002))
    # c_u[c_u>=0.0005] = mean_inc_u-0.00002

    normalized_c_u, normalized_inc_u = find_in_u(list_acc,normalized_u_list)


    c_entropy, inc_entropy = find_in_entropy(list_acc,entropy_list)

    mean_inc_u = normalized_inc_u.mean()
    mean_c_u = normalized_c_u.mean()
    mean_inc_entropy = inc_entropy.mean()
    mean_c_entropy = c_entropy.mean()
    # np.append(normalized_inc_u, values=max(normalized_c_u)+0.02)
    # normalized_c_u[normalized_c_u>=0.6] = mean_c_u
    # null_c_u =  np.delete(c_u,np.where(c_u<0.00025))
    # null_inc_u = np.delete(inc_u,np.where(inc_u<0.00025))
    # np.savez(r'./results/Our_OCT_normal.npz', c_u=c_u, inc_u=inc_u,normalized_c_u=normalized_c_u,normalized_inc_u=normalized_inc_u)

    # Uni- and Fusion
    # plt.figure(0)
    # sns.kdeplot(np_u_list,label="Fusion Uncertianty", color='red', shade=True)
    # sns.kdeplot(np_OCT_au_list,label="OCT Uncertianty", color='blue', shade=True)
    # sns.kdeplot(np_Fundus_au_list,label="Fundus Uncertianty", color='orange', shade=True)
    # # plt.xlabel(x_level,fontsize=15)
    # # plt.ylabel(y_level,fontsize=15)
    # # plt.title('Uncertainty of Predictions',fontsize=15)
    # # plt.xticks(x,x_names,fontsize=10)
    # # plt.xlim(x1, x2)
    # # plt.ylim(y1,y2)
    # plt.xlabel('Uncertainty', fontsize=15)
    # # plt.xlabel('Aleatoric Uncertainty', fontsize=15)
    # plt.ylabel('Density', fontsize=15)
    # plt.legend(fontsize=15)   #显示标签
    # plt.show()
    # #
    # # plt.figure(1)
    # sns.kdeplot(np_u_list,label="Fusion Uncertianty", color='red', shade=True)
    # sns.kdeplot(np_OCT_eu_list,label="OCT Uncertianty", color='blue', shade=True)
    # sns.kdeplot(np_Fundus_eu_list,label="Fundus Uncertianty", color='orange', shade=True)
    # # plt.xlabel(x_level,fontsize=15)
    # # plt.ylabel(y_level,fontsize=15)
    # # plt.title('Uncertainty of Predictions',fontsize=15)
    # # plt.xticks(x,x_names,fontsize=10)
    # # plt.xlim(x1, x2)
    # # plt.ylim(y1,y2)
    # plt.xlabel('Epistemic Uncertainty', fontsize=15)
    # plt.ylabel('Density', fontsize=15)
    # plt.legend(fontsize=15)   #显示标签
    # plt.show()
    # sns.kdeplot(np_u_list,label="Fusion Uncertianty", color='red', shade=True)
    # sns.kdeplot(np_OCT_au_list+np_OCT_eu_list,label="OCT Uncertianty", color='blue', shade=True)
    # sns.kdeplot(np_Fundus_au_list+np_Fundus_eu_list,label="Fundus Uncertianty", color='orange', shade=True)
    # # plt.xlabel(x_level,fontsize=15)
    # # plt.ylabel(y_level,fontsize=15)
    # # plt.title('Uncertainty of Predictions',fontsize=15)
    # # plt.xticks(x,x_names,fontsize=10)
    # # plt.xlim(x1, x2)
    # # plt.ylim(y1,y2)
    # plt.xlabel('All Uncertainty', fontsize=15)
    # plt.ylabel('Density', fontsize=15)
    # plt.legend(fontsize=15)   #显示标签
    # plt.show()

    # sns.kdeplot(c_u,label="Correct classification samples", color='blue', shade=True)
    # sns.kdeplot(inc_u,label="Mis-classification samples", color='red', shade=True)
    # # plt.xlabel(x_level,fontsize=15)
    # # plt.ylabel(y_level,fontsize=15)
    # # plt.title('Uncertainty of Predictions',fontsize=15)
    # # plt.xticks(x,x_names,fontsize=10)
    # # plt.xlim(x1, x2)
    # # plt.ylim(y1,y2)
    # plt.xlabel('Uncertainty', fontsize=15)
    # plt.ylabel('Density', fontsize=15)
    # plt.legend(fontsize=15)   #显示标签
    # plt.show()
    #
    # sns.kdeplot(normalized_c_u,label="Correct classification samples", color='blue', shade=True)
    # sns.kdeplot(normalized_inc_u,label="Mis-classification samples", color='red', shade=True)
    # # plt.xlabel(x_level,fontsize=15)
    # # plt.ylabel(y_level,fontsize=15)
    # # plt.title('Uncertainty of Predictions',fontsize=15)
    # # plt.xticks(x,x_names,fontsize=10)
    # # plt.xlim(x1, x2)
    # # plt.ylim(y1,y2)
    # plt.xlabel('Normalized Uncertainty', fontsize=15)
    # plt.ylabel('Density', fontsize=15)
    # plt.legend(fontsize=15)   #显示标签
    # plt.show()
    #
    # sns.kdeplot(c_entropy,label="Correct classification samples", color='blue', shade=True)
    # sns.kdeplot(inc_entropy,label="Mis-classification samples", color='red', shade=True)
    # # plt.xlabel(x_level,fontsize=15)
    # # plt.ylabel(y_level,fontsize=15)
    # # plt.title('Uncertainty of Predictions',fontsize=15)
    # # plt.xticks(x,x_names,fontsize=10)
    # # plt.xlim(x1, x2)
    # # plt.ylim(y1,y2)
    # plt.xlabel('Entropy', fontsize=15)
    # plt.ylabel('Density', fontsize=15)
    # plt.legend(fontsize=15)   #显示标签
    # plt.show()

    avg_acc = correct_num/data_num
    avg_ece = sum(ece_list)/len(ece_list)
    avg_nll = sum(nll_list)/len(nll_list)
    avg_brier = sum(brier_list)/len(brier_list)

    avg_kappa = cohen_kappa_score(prediction_list, label_list)
    F1_Score = f1_score(y_true=label_list, y_pred=prediction_list, average='weighted')
    Recall_Score = recall_score(y_true=label_list, y_pred=prediction_list, average='weighted')

    if not os.path.exists(os.path.join(args.save_dir, "{}_{}_{}".format(args.model_name,args.dataset,args.folder))):
        os.makedirs(os.path.join(args.save_dir, "{}_{}_{}".format(args.model_name,args.dataset,args.folder)))

    with open(os.path.join(args.save_dir,"{}_{}_{}_Metric.txt".format(args.model_name,args.dataset,args.folder)),'w') as Txt:
        Txt.write("Acc: {}, AUC: {}, AURC: {}, EAURC: {},  NLL: {}, BRIER: {}, F1_Score: {}, Recall_Score: {}, Kappa_Score: {}, ECE: {}\n".format(
            round(avg_acc,6),round(epoch_auc,6),round(aurc,6),round(eaurc,6),round(avg_nll,6),round(avg_brier,6),round(F1_Score,6),round(Recall_Score,6),round(avg_kappa,6),round(avg_ece,6)
        ))
    # print(
    #     "Acc: {:.4f}, AUC: {:.4f}, AURC: {:.4f}, EAURC: {:.4f}, NLL: {:.4f}, BRIER: {:.4f},  F1_Score: {:.4f}, Recall_Score: {:.4f}, kappa: {:.4f}, ECE: {:.4f}".format(
    #         avg_acc, epoch_auc, aurc, eaurc, avg_nll, avg_brier, F1_Score, Recall_Score, avg_kappa, avg_ece))
    # print('====> mean_inc_u: {:.4f},mean_c_u: {:.4f}\n'.format(mean_inc_u, mean_c_u))
    return avg_acc,epoch_auc,aurc, eaurc, avg_nll,avg_brier,F1_Score,Recall_Score,avg_kappa,avg_ece

if __name__ == "__main__":

    # kkk= math.log(math.pi)
    # filename = './datasets/handwritten_6views.mat'
    # image = loadmat(filename)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--end_epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--test_epoch', type=int, default=101, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lambda_epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    # parser.add_argument('--end_epochs', type=int, default=200, metavar='N',
    #                     help='number of epochs to train [default: 500]')
    # parser.add_argument('--test_epoch', type=int, default=198, metavar='N',
    #                     help='number of epochs to train [default: 500]')
    # parser.add_argument('--lambda_epochs', type=int, default=100, metavar='N',dir
    #                     help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--modal_number', type=int, default=2, metavar='N',
                        help='modalties number')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate') # ResNet_ECNP_ST_Beta_fusion_C_DF_CML_loss ResNet_ECNP_ST_Beta_fusion_C_DF_CML_loss_evi ResNet_ECNP_ST_Beta_fusion_C_DF_CML ResNet_ECNP_ST_Beta_fusion_C_DF_CML_2D ResNet_ECNP_ST_Beta_fusion_C_DF_CML_3D
    parser.add_argument('--save_dir', default='./results', type=str) # ResNet_MMST /ResNet_MMST_Reg /ResNet_MMST_Reg_Ce/ResNet_MMST_Reg/ResNet_MMST_Reg_CML//ResNet_MMST_Reg_CMLU /ResNet_MMST_Reg_Ce/ ResNet_TMC  / ResNet_ECNP / ResNet_ECNP_Beta_CML / ResNet_ECNP_Beta_Reg /ResNet_ECNP_Beta_Reg_Ce / ResNet_ECNP_Beta_Reg_CML / ResNet_ECNP_Beta_Reg_CMLU/ ResNet_MMST_Reg_Ce_CMLU/ ResNet_MMST_Reg_Ce_CML
    parser.add_argument("--model_name", default="ResNet_ECNP_ST_Beta_fusion_C_DF_CML", type=str, help="ResNet_TMC/ResNet_TMC_WAO/ResNet_ECNP/ResNet_ECNP_Beta/ResNet_ECNP_Beta_Reg_CML/ResNet_ECNP_Beta_loss/ResNet_ECNP_lamda/ResNet_ECNP_lamda_NLL/ResNet_ECNP_UVST/ResNet_MMST/ResNet_MMUST/ResNet_MM_NIGST/ResNet_MM_STNIG/ResNet_MM_NIG/ResNet_MM_UNIG/ResNet_ECNP_IST/ResNet_ECNP_UST")
    parser.add_argument("--dataset", default="OLIVES", type=str, help="MMOCTF/Gamma/MGamma/OLIVES")
    # parser.add_argument('--num_classes', type=int, default=2, metavar='N',
    #                     help='class number: MMOCTF: 2 /Gamma: 3')
    parser.add_argument("--condition", default="noise", type=str, help="noise/normal")
    parser.add_argument("--condition_name", default="Gaussian", type=str, help="Gaussian/SaltPepper/All")
    parser.add_argument("--Condition_SP_Variance", default=0.005, type=int, help="Variance: 0.01/0.1")
    parser.add_argument("--Condition_G_Variance", default=0.05, type=int, help="Variance: 15/1/0.1")
    parser.add_argument("--folder", default="folder0", type=str, help="folder0/folder1/folder2/folder3/folder4")
    parser.add_argument("--mode", default="test", type=str, help="train/test/train&test")
    # -- for ECNP parameters
    parser.add_argument('-rps', '--representation_size', type=int, default=128, help='Representation size for context')
    parser.add_argument('-hs', '--hidden_size', type=int, default=128, help='Model hidden size')
    parser.add_argument('-ev_dec_beta_min', '--ev_dec_beta_min', type=float, default=0.2,
                        help="EDL Decoder beta minimum value")
    parser.add_argument('-ev_dec_alpha_max', '--ev_dec_alpha_max', type=float, default=20.0,
                        help="EDL output alpha maximum value")
    parser.add_argument('-ev_dec_v_max', '--ev_dec_v_max', type=float, default=20.0, help="EDL output v maximum value")
    parser.add_argument('-nig_nll_reg_coef', '--nig_nll_reg_coef', type=float, default=0.1,
                        help="EDL nll reg balancing factor")
    parser.add_argument('-nig_nll_ker_reg_coef', '--nig_nll_ker_reg_coef', type=float, default=1.0,
                        help='EDL kernel reg balancing factor')
    parser.add_argument('-ev_st_u_min', '--ev_st_u_min', type=float, default=0.0001,
                        help="EDL st output sigma minnum value")
    parser.add_argument('-ev_st_sigma_min', '--ev_st_sigma_min', type=float, default=0.2,
                        help="EDL st output sigma minnum value")
    parser.add_argument('-ev_st_v_max', '--ev_st_v_max', type=float, default=30.0, help="EDL output v maximum value")

    # Condition_G_Variance =[0] # Fundus  & all`
    Condition_G_Variance = [0,0.1,0.2,0.3,0.4,0.5] # Fundus OCT & OLIVES` Our
    # Condition_G_Variance = [0.1,0.2,0.3,0.4,0.5] # Fundus OCT & OLIVES` Our
    # Condition_G_Variance = [0,0.01, 0.03, 0.05, 0.07, 0.1,0.3,0.5] # OCT  & GAMMA
    # Condition_G_Variance =[0,0.1,0.2,0.3,0.4,0.5] # Fundus  Fundus & GAMMA`

    # Condition_G_Variance =[0, 0.01, 0.03, 0.05, 0.07, 0.1] # Our Fundus  & all
    # Condition_G_Variance =[0.1,0.2,0.3,0.4,0.5] # Our Fundus  & all

    # Condition_G_Variance =[0.01, 0.03, 0.05, 0.07, 0.1] # Our Fundus  & all
    # Condition_G_Variance =[0.2, 0.3,0.4, 0.5] # Our Fundus  & all

    # Condition_G_Variance =[0, 0.01, 0.03, 0.05, 0.07, 0.1] # Fundus  & all
    # Condition_G_Variance =[0, 0.05, 0.1, 0.3, 0.5] # OCT
    # Condition_G_Variance =[0.05, 0.1, 0.3, 0.5] # OLIVES

    # Condition_G_Variance =[0,0.1,0.2,0.3,0.4,0.5] # Fundus  & all`
    # Condition_G_Variance = [0, 0.05, 0.1]
    # Condition_G_Variance =[0.03, 0.07] # Fundus  & all

    # Condition_G_Variance =[0.1]
    seed_num = list(range(1,11))
    condition_level = ['normal','noise']

    args = parser.parse_args()
    args.seed_idx = 11

    if args.dataset =="MMOCTF":
        args.data_path = '/home/zou_ke/projects_data/Multi-OF/2000/'
        args.modalties_name = ["FUN", "OCT"]
        args.num_classes = 2
        args.dims = [[(128, 256, 128)], [(512, 512)]]
        args.modalties = len(args.dims)
        train_loader = torch.utils.data.DataLoader(
        Multi_modal_data(args.data_path, args.modal_number,args.modalties_name, 'train',args.condition,args, folder=args.folder), batch_size=args.batch_size)
        val_loader = torch.utils.data.DataLoader(
        Multi_modal_data(args.data_path,  args.modal_number, args.modalties_name, 'val',args.condition,args, folder=args.folder), batch_size=1)
        test_loader = torch.utils.data.DataLoader(
        Multi_modal_data(args.data_path, args.modal_number, args.modalties_name, 'test',args.condition,args, folder=args.folder), batch_size=1)
        N_mini_batches = len(train_loader)
        print('The number of training images = %d' % N_mini_batches)
    elif args.dataset =="OLIVES":
        args.data_path = '/home/zou_ke/projects_data/OLIVES/OLIVES/'
        # args.data_path = '/home/zou_ke/projects_data/OLIVES2/OLIVES/'
        # args.data_path = '/home/zou_ke/projects_data/OLIVES3/OLIVES/'

        args.modalties_name = ["FUN", "OCT"]
        args.num_classes = 2
        args.dims = [[(48, 248, 248)], [(512, 512)]]
        args.modalties = len(args.dims)
        train_loader = torch.utils.data.DataLoader(
        OLIVES_dataset(args.data_path, args.modal_number,args.modalties_name, 'train',args.condition,args, folder=args.folder), batch_size=args.batch_size)
        val_loader = torch.utils.data.DataLoader(
        OLIVES_dataset(args.data_path,  args.modal_number, args.modalties_name, 'val',args.condition,args, folder=args.folder), batch_size=1)
        test_loader = torch.utils.data.DataLoader(
        OLIVES_dataset(args.data_path, args.modal_number, args.modalties_name, 'test',args.condition,args, folder=args.folder), batch_size=1)
        N_mini_batches = len(train_loader)
        print('The number of training images = %d' % N_mini_batches)
    elif args.dataset =="Gamma":
        args.modalties_name = ["FUN", "OCT"]
        args.dims = [[(128, 256, 128)], [(512, 512)]]
        args.num_classes = 3
        args.modalties = len(args.dims)
        args.base_path = '/home/zou_ke/projects_data/Multi-OF/Gamma/'
        args.data_path = args.base_path + 'multi-modality_images/'
        filelists = os.listdir(args.data_path)
        kf = KFold(n_splits=5, shuffle=True, random_state=10)
        y = kf.split(filelists)
        count = 0
        train_filelists = [[], [], [], [], []]
        val_filelists = [[], [], [], [], []]
        for tidx, vidx in y:
            train_filelists[count], val_filelists[count] = np.array(filelists)[tidx], np.array(filelists)[vidx]
            count = count + 1
        f_folder =  args.folder[-1]

        train_dataset = GAMMA_sub1_dataset(dataset_root = args.data_path,
                                           oct_img_size = args.dims[0],
                                           fundus_img_size =  args.dims[1],
                                           mode = 'train',
                                           label_file = args.base_path + 'glaucoma_grading_training_GT.xlsx',
                                           filelists = np.array(train_filelists[f_folder]))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = GAMMA_sub1_dataset(dataset_root=args.data_path,
                                           oct_img_size = args.dims[0],
                                           fundus_img_size =  args.dims[1],
                                           mode = 'val',
                                           label_file = args.base_path + 'glaucoma_grading_training_GT.xlsx',
                                           filelists = np.array(val_filelists[f_folder]),)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
        test_dataset = val_dataset
        test_loader = val_loader
    elif args.dataset =="MGamma":
        args.modalties_name = ["FUN", "OCT"]
        args.dims = [[(128, 256, 128)], [(512, 512)]]
        args.num_classes = 3
        args.modalties = len(args.dims)
        args.base_path = '/home/zou_ke/projects_data/Multi-OF/Gamma/'
        args.data_path = '/home/zou_ke/projects_data/Multi-OF/MGamma/'
        filelists = os.listdir(args.data_path)
        # kf = KFold(n_splits=5, shuffle=True, random_state=10)
        kf = KFold(n_splits=5, shuffle=True, random_state=10)

        y = kf.split(filelists)
        count = 0
        train_filelists = [[], [], [], [], []]
        val_filelists = [[], [], [], [], []]
        for tidx, vidx in y:
            train_filelists[count], val_filelists[count] = np.array(filelists)[tidx], np.array(filelists)[vidx]
            count = count + 1
        f_folder =  int(args.folder[-1])
        train_dataset = GAMMA_dataset(args,dataset_root = args.data_path,
                                           oct_img_size = args.dims[0],
                                           fundus_img_size =  args.dims[1],
                                           mode = 'train',
                                           label_file = args.base_path + 'glaucoma_grading_training_GT.xlsx',
                                           filelists = np.array(train_filelists[f_folder]))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = GAMMA_dataset(args,dataset_root=args.data_path,
                                           oct_img_size = args.dims[0],
                                           fundus_img_size =  args.dims[1],
                                           mode = 'val',
                                           label_file = args.base_path + 'glaucoma_grading_training_GT.xlsx',
                                           filelists = np.array(val_filelists[f_folder]),)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
        test_dataset = val_dataset
        test_loader = val_loader
    else:
        print('There is no this dataset name')
        raise NameError

    if args.model_name =="ResNet_TMC":
        model = TMC(args.num_classes, args.modalties, args.dims, args.lambda_epochs)
        # model = ECNP_WAO(args.num_classes, args.modalties, args.dims, args.lambda_epochs)
    elif args.model_name =="ResNet_ECNP_Beta":
        model = ECNP_ST_Beta(args.num_classes, args.modalties, args.dims, args, args.lambda_epochs)
    elif args.model_name == "ResNet_ECNP_ST_Beta_fusion_C_DF":  # fusion rule for mean Our
        model = ECNP_ST_Beta_fusion_C_DF(args.num_classes, args.modalties, args.dims, args, args.lambda_epochs)
    elif args.model_name == "ResNet_ECNP_ST_Beta_fusion_C_CMLU":  # fusion rule for mean Our
        model = ECNP_ST_Beta_fusion_C_CMLU(args.num_classes, args.modalties, args.dims, args, args.lambda_epochs)
    elif args.model_name == "ResNet_ECNP_ST_Beta_fusion_C_DF_CML":  # fusion rule for mean Our
        model = ECNP_ST_Beta_fusion_C_DF_CML(args.num_classes, args.modalties, args.dims, args, args.lambda_epochs)
    elif args.model_name =="ResNet_ECNP_Beta_CML":
        model = ECNP_ST_Beta_CML(args.num_classes, args.modalties, args.dims, args, args.lambda_epochs)
    else:
        print('There is no this model name')
        raise NameError
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    model.cuda()
    best_acc = 0
    loss_list = []
    acc_list = []

    if args.mode =='train&test':
        epoch = 0
        for epoch in range(args.start_epoch, args.end_epochs + 1):
            print('===========Train begining!===========')
            print('Epoch {}/{}'.format(epoch, args.end_epochs - 1))
            epoch_loss = train(epoch,train_loader,model)
            print("epoch %d avg_loss:%0.3f" % (epoch, epoch_loss.avg))
            val_loss, best_acc = val(epoch,val_loader,model,best_acc)
            loss_list.append(epoch_loss.avg)
            acc_list.append(best_acc)
        loss_plot(args, loss_list)
        metrics_plot(args, 'acc', acc_list)
        test_acc,test_acclist = test(args,test_loader,model,epoch)
        # args.model_name = "ResNet_ECNP_ST_Beta_fusion_C_DF_CMLU"
        # epoch = 0
        # for epoch in range(args.start_epoch, args.end_epochs + 1):
        #     print('===========Train begining!===========')
        #     print('Epoch {}/{}'.format(epoch, args.end_epochs - 1))
        #     epoch_loss = train(epoch,train_loader,model)
        #     print("epoch %d avg_loss:%0.3f" % (epoch, epoch_loss.avg))
        #     val_loss, best_acc = val(epoch,val_loader,model,best_acc)
        #     loss_list.append(epoch_loss.avg)
        #     acc_list.append(best_acc)
        # loss_plot(args, loss_list)
        # metrics_plot(args, 'acc', acc_list)
        # test_acc,test_acclist = test(args,test_loader,model,epoch)

    elif args.mode == 'test':
        epoch = args.test_epoch
        for i in range(len(Condition_G_Variance)):
            args.Condition_G_Variance = Condition_G_Variance[i]
            print("Gaussian noise: %f" % args.Condition_G_Variance)
            acc_list,auc_list,aurc_list,eaurc_list,nll_list, brier_list,\
            F1_list,Rec_list,kap_list,ECE_list = [],[],[],[],[],[],[],[],[],[]

            for j in range(len(seed_num)):
            # for j in range(1):

                args.seed_idx = seed_num[j]
                # print("seed_idx: %d" % args.seed_idx)

                if args.dataset == "MMOCTF":
                    args.data_path = '/home/zou_ke/projects_data/Multi-OF/2000/'
                    args.modalties_name = ["FUN", "OCT"]
                    args.num_classes = 2
                    args.dims = [[(128, 256, 128)], [(512, 512)]]
                    args.modalties = len(args.dims)

                    test_loader = torch.utils.data.DataLoader(
                        Multi_modal_data(args.data_path, args.modal_number, args.modalties_name, 'test', args.condition,
                                         args, folder=args.folder), batch_size=1)
                    N_mini_batches = len(test_loader)
                    # print('The number of testing images = %d' % N_mini_batches)
                elif args.dataset == "OLIVES":
                    args.data_path = '/home/zou_ke/projects_data/OLIVES/OLIVES/'
                    # args.data_path = '/home/zou_ke/projects_data/OLIVES2/OLIVES/'
                    # args.data_path = '/home/zou_ke/projects_data/OLIVES3/OLIVES/'

                    args.modalties_name = ["FUN", "OCT"]
                    args.num_classes = 2
                    args.dims = [[(48, 248, 248)], [(512, 512)]]
                    args.modalties = len(args.dims)

                    test_loader = torch.utils.data.DataLoader(
                        OLIVES_dataset(args.data_path, args.modal_number, args.modalties_name, 'test', args.condition,
                                       args, folder=args.folder), batch_size=1)
                    N_mini_batches = len(test_loader)
                    # print('The number of testing images = %d' % N_mini_batches)
                elif args.dataset == "MGamma":
                    args.modalties_name = ["FUN", "OCT"]
                    args.dims = [[(128, 256, 128)], [(512, 512)]]
                    args.num_classes = 3
                    args.modalties = len(args.dims)
                    args.base_path = '/home/zou_ke/projects_data/Multi-OF/Gamma/'
                    args.data_path = '/home/zou_ke/projects_data/Multi-OF/MGamma/'
                    filelists = os.listdir(args.data_path)
                    kf = KFold(n_splits=5, shuffle=True, random_state=10)
                    y = kf.split(filelists)
                    count = 0
                    train_filelists = [[], [], [], [], []]
                    val_filelists = [[], [], [], [], []]
                    for tidx, vidx in y:
                        train_filelists[count], val_filelists[count] = np.array(filelists)[tidx], np.array(filelists)[
                            vidx]
                        count = count + 1
                    f_folder = int(args.folder[-1])
                    val_dataset = GAMMA_dataset(args, dataset_root=args.data_path,
                                                oct_img_size=args.dims[0],
                                                fundus_img_size=args.dims[1],
                                                mode='val',
                                                label_file=args.base_path + 'glaucoma_grading_training_GT.xlsx',
                                                filelists=np.array(val_filelists[f_folder]), )

                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
                    test_dataset = val_dataset
                    test_loader = val_loader
                    N_mini_batches = len(test_loader)
                    # print('The number of testing images = %d' % N_mini_batches)
                else:
                    print('There is no this dataset name')
                    raise NameError
                test_acc, test_auc, test_aurc, test_eaurc, test_nll, test_brier,test_F1, test_Rec, test_kappa, test_ece\
                    = test(args, test_loader, model, epoch)

                acc_list.append(test_acc)
                auc_list.append(test_auc)
                aurc_list.append(test_aurc)
                eaurc_list.append(test_eaurc)
                nll_list.append(test_nll)
                brier_list.append(test_brier)
                F1_list.append(test_F1)
                Rec_list.append(test_Rec)
                kap_list.append(test_kappa)
                ECE_list.append(test_ece)

            acc_list_mean,acc_list_std = np.mean(acc_list),np.std(acc_list)
            auc_list_mean,auc_list_std = np.mean(auc_list),np.std(auc_list)
            aurc_list_mean,aurc_list_std = np.mean(aurc_list),np.std(aurc_list)
            eaurc_list_mean,eaurc_list_std = np.mean(eaurc_list),np.std(eaurc_list)
            nll_list_mean,nll_list_std = np.mean(nll_list),np.std(nll_list)
            brier_list_mean,brier_list_std = np.mean(brier_list),np.std(brier_list)
            F1_list_mean,F1_list_std = np.mean(F1_list),np.std(F1_list)
            Rec_list_mean,Rec_list_std = np.mean(Rec_list),np.std(Rec_list)
            kap_list_mean,kap_list_std = np.mean(kap_list),np.std(kap_list)
            ECE_list_mean,ECE_list_std = np.mean(ECE_list),np.std(ECE_list)
            print(
                "Mean_Std_Acc: {:.4f} +- {:.4f}, Mean_Std_AUC: {:.4f} +- {:.4f},Mean_Std_AURC: {:.4f} +- {:.4f}, "
                "Mean_Std_EAURC: {:.4f} +- {:.4f}, Mean_Std_nll: {:.4f} +- {:.4f}, Mean_Std_brier: {:.4f} +- {:.4f}, Mean_Std_F1_Score: {:.4f} +- "
                "{:.4f}, Mean_Std_Recall_Score: {:.4f} +- {:.6f}, Mean_Std_kappa: {:.4f} +- {:.4f}, Mean_Std_ECE: {:.4f} +- {:.4f}".format(
                    acc_list_mean, acc_list_std, auc_list_mean,auc_list_std, aurc_list_mean, aurc_list_std, eaurc_list_mean, eaurc_list_std,
                    nll_list_mean, nll_list_std,brier_list_mean,brier_list_std,F1_list_mean, F1_list_std,Rec_list_mean,Rec_list_std,kap_list_mean,kap_list_std,ECE_list_mean,ECE_list_std))
            logging.info(
                "Mean_Std_Acc: {:.4f} +- {:.4f}, Mean_Std_AUC: {:.4f} +- {:.4f},Mean_Std_AURC: {:.4f} +- {:.4f}, "
                "Mean_Std_EAURC: {:.4f} +- {:.4f}, Mean_Std_nll: {:.4f} +- {:.4f}, Mean_Std_brier: {:.4f} +- {:.4f}, Mean_Std_F1_Score: {:.4f} +- "
                "{:.4f}, Mean_Std_Recall_Score: {:.4f} +- {:.4f}, Mean_Std_kappa: {:.4f} +- {:.4f}, Mean_Std_ECE: {:.4f} +- {:.4f}".format(
                    acc_list_mean, acc_list_std, auc_list_mean,auc_list_std, aurc_list_mean, aurc_list_std, eaurc_list_mean, eaurc_list_std,
                    nll_list_mean, nll_list_std,brier_list_mean,brier_list_std,F1_list_mean, F1_list_std,Rec_list_mean,Rec_list_std,kap_list_mean,kap_list_std,ECE_list_mean,ECE_list_std))
