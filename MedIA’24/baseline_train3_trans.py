import os
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from baseline_models import Res2Net2D,ResNet3D,Multi_ResNet,Multi_EF_ResNet,Multi_CBAM_ResNet,Multi_dropout_ResNet
from model import Base_transformer
from sklearn.model_selection import KFold
from data import Multi_modal_data,GAMMA_dataset,OLIVES_dataset
from metrics import cal_ece
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from metrics2 import calc_aurc_eaurc,calc_nll_brier
import torch.nn.functional as F
import logging
import time

warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def find_in_u(list_acc,in_list,u_list,class_num=0):
    for i in range(len(list_acc)):
        if list_acc[i] == class_num:
            in_list.append(i)
    in_u_list = np.zeros(len(in_list))
    for j in range(len(in_list)):
        in_u_list[j] = (u_list[in_list[j]])
    return in_u_list

def train(epoch,train_loader,model):
    model.train()
    loss_meter = AverageMeter()
    # loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].float().cuda())
        target = Variable(target.long().cuda())
        # target = Variable(np.array(target)).cuda())

        # refresh the optimizer
        optimizer.zero_grad()
        pred, loss = model(data, target)
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
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].float().cuda())
        data_num += target.size(0)
        with torch.no_grad():
            target = Variable(target.long().cuda())
            pred, loss = model(data, target)
            predicted = pred.argmax(dim=-1)
            # _, predicted = torch.max(pred.data, 1)
            correct_num += (predicted == target).sum().item()
            loss_meter.update(loss.item())
    aver_acc = correct_num / data_num
    print('====> acc: {:.4f}'.format(aver_acc))
    if pred.shape[1] >2:
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
        return loss_meter.avg, best_acc

    else:
        if aver_acc > best_acc \
                or (current_epoch + 1) % int(args.end_epochs - 1) == 0 \
                or (current_epoch + 1) % int(args.end_epochs - 2) == 0 \
                or (current_epoch + 1) % int(args.end_epochs - 3) == 0:
            print('aver_acc:{} > best_acc:{}'.format(aver_acc, best_acc))
            best_acc = aver_acc
            print('===========>save best model!')
            file_name = os.path.join(args.save_dir,
                                 args.model_name + '_' + args.dataset + '_' + args.folder + '_epoch_{}.pth'.format(
                                     current_epoch))
            torch.save({
            'epoch': current_epoch,
            'state_dict': model.state_dict(),
            },
            file_name)
        return loss_meter.avg, best_acc


def test(args, test_loader,model,epoch):
    if args.num_classes == 2:
        load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 args.model_name + '_' + args.dataset +'_'+ args.folder + '_epoch_{}.pth'.format(args.test_epoch))
    else:
        load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 args.model_name + '_' + args.dataset +'_'+ args.folder + '_best_epoch.pth')

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
    in_list = []
    label_list = []
    ece_list = []
    prediction_list = []
    probability_list = []
    one_hot_label_list = []
    one_hot_probability_list = []
    correct_num, data_num = 0, 0
    epoch_auc = 0
    start_time = time.time()
    time_list = []
    correct_list=[]
    nll_list= []
    brier_list = []
    for batch_idx, (data, target) in enumerate(test_loader):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].float().cuda())
        data_num += target.size(0)
        pred = torch.zeros(1,args.num_classes).cuda()
        with torch.no_grad():
            target = Variable(target.long().cuda())
            if args.model_name =='Multi_dropout_ResNet':
                dropout_times = 10
                for i in range(dropout_times):
                    pred_i, _ = model(data, target)
                    pred += pred_i
                pred = pred/dropout_times
            else:
                pred, _ = model(data, target)
            elapsed_time = time.time() - start_time
            time_list.append(elapsed_time)
            predicted = pred.argmax(dim=-1)
            correct_num += (predicted == target).sum().item()
            correct = (predicted == target)

            list_acc.append((predicted == target).sum().item())
            prediction_list.append(predicted.cpu().detach().float().numpy())
            label_list.append(target.cpu().detach().float().numpy())
            # label_list = F.one_hot(target, num_classes=args.num_classes).cpu().detach().float().numpy()
            probability = torch.softmax(pred, dim=1).cpu().detach().float().numpy()
            probability_list.append(torch.softmax(pred, dim=1).cpu().detach().float().numpy()[:,1])
            correct_list.append(correct.cpu().detach().float().numpy())

            one_hot_probability_list.append(torch.softmax(pred, dim=1).squeeze(dim=0).cpu().detach().float().numpy())
            # one_hot_probability_list.append(pred.data.squeeze(dim=0).cpu().detach().float().numpy())
            one_hot_label = F.one_hot(target, num_classes=args.num_classes).squeeze(dim=0).cpu().detach().float().numpy()
            one_hot_label_list.append(one_hot_label)
            ece_list.append(cal_ece(torch.squeeze(pred), target))

            if args.model_name == 'Multi_dropout_ResNet':
                # NLL brier
                nll, brier = calc_nll_brier(probability, pred, target, one_hot_label)
                nll_list.append(nll)
                brier_list.append(brier)


    logging.info('Single sample test time consumption {:.2f} seconds!'.format(sum(time_list)/len(time_list)))
    print('Single sample test time consumption {:.2f} seconds!'.format(sum(time_list)/len(time_list)))
    if args.num_classes > 2:
        epoch_auc = metrics.roc_auc_score(one_hot_label_list, one_hot_probability_list, multi_class='ovo')
    else:
        epoch_auc = metrics.roc_auc_score(label_list, probability_list)
        # epoch_auc = metrics.roc_auc_score(label_list, probability_list)
    # fpr, tpr, thresholds = roc_curve(label_list, probability_list, pos_label=1)

    # roc_auc = auc(fpr, tpr)
    avg_acc = correct_num/data_num
    avg_ece = sum(ece_list)/len(ece_list)
    aurc, eaurc = calc_aurc_eaurc(probability_list, correct_list)

    # epoch_auc = metrics.roc_auc_score(label_list, prediction_list)
    avg_kappa = cohen_kappa_score(prediction_list, label_list)
    F1_Score = f1_score(y_true=label_list, y_pred=prediction_list, average='weighted')
    Recall_Score = recall_score(y_true=label_list, y_pred=prediction_list, average='weighted')



    if not os.path.exists(os.path.join(args.save_dir, "{}_{}_{}".format(args.model_name,args.dataset,args.folder))):
        os.makedirs(os.path.join(args.save_dir, "{}_{}_{}".format(args.model_name,args.dataset,args.folder)))

    if args.model_name == 'Multi_dropout_ResNet':
        avg_nll = sum(nll_list) / len(nll_list)
        avg_brier = sum(brier_list) / len(brier_list)
        with open(os.path.join(args.save_dir, "{}_{}_{}_Metric.txt".format(args.model_name, args.dataset, args.folder)),
                  'w') as Txt:
            Txt.write(
                "Acc: {}, AUC: {}, AURC: {}, EAURC: {},  NLL: {}, BRIER: {}, F1_Score: {}, Recall_Score: {}, Kappa_Score: {}, ECE: {}\n".format(
                    round(avg_acc, 6), round(epoch_auc, 6), round(aurc, 6), round(eaurc, 6), round(avg_nll, 6),
                    round(avg_brier, 6), round(F1_Score, 6), round(Recall_Score, 6), round(avg_kappa, 6),
                    round(avg_ece, 6)
                ))
        # print(
        #     "Acc: {:.4f}, AUC: {:.4f}, AURC: {:.4f}, EAURC: {:.4f}, NLL: {:.4f}, BRIER: {:.4f},  F1_Score: {:.4f}, Recall_Score: {:.4f}, kappa: {:.4f}, ECE: {:.4f}".format(
        #         avg_acc, epoch_auc, aurc, eaurc, avg_nll, avg_brier, F1_Score, Recall_Score, avg_kappa, avg_ece))
        # print('====> mean_inc_u: {:.4f},mean_c_u: {:.4f}\n'.format(mean_inc_u, mean_c_u))
        return avg_acc, epoch_auc, aurc, eaurc, avg_nll, avg_brier, F1_Score, Recall_Score, avg_kappa, avg_ece
    else:
        with open(os.path.join(args.save_dir,"{}_{}_{}_Metric.txt".format(args.model_name,args.dataset,args.folder)),'w') as Txt:
            Txt.write("Acc: {}, AUC: {},  AURC: {}, EAURC: {}, F1_Score: {}, Recall_Score: {}, Kappa_Score: {}, ECE: {}\n".format(
            round(avg_acc,6),round(epoch_auc,6),round(aurc,6),round(eaurc,6),round(F1_Score,6),round(Recall_Score,6),round(avg_kappa,6),round(avg_ece,6)
        ))
        # print("Acc: {:.6f}, AUC: {:.6f},AURC: {}, EAURC: {}, F1_Score: {:.6f}, Recall_Score: {:.6f}, kappa: {:.6f}, ECE: {:.6f}".format(
        #         avg_acc,epoch_auc,aurc, eaurc, F1_Score,Recall_Score,avg_kappa,avg_ece))
        # logging.info("Acc: {:.6f}, AUC: {:.6f},AURC: {:.6f}, EAURC: {:.6f}, F1_Score: {:.6f}, Recall_Score: {:.6f}, kappa: {:.6f}, ECE: {:.6f}\n".format(
        #         avg_acc,epoch_auc,aurc, eaurc,F1_Score,Recall_Score,avg_kappa,avg_ece))
        # return list_acc, avg_acc
        return avg_acc, epoch_auc,aurc, eaurc,F1_Score,Recall_Score,avg_kappa,avg_ece


def test_ensemble(args, test_loader,models,epoch):
    if args.dataset == 'MGamma':
        deepen_times = 4
    else:
        deepen_times = 5

    # load ensemble models
    load_model=[]
    # load_model[0]=.23
    for i in range(deepen_times):
        print(i+1)
        if args.num_classes == 2:
            load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 'Multi_DE' +str(i+1) + '_ResNet_' + args.dataset +'_'+ args.folder + '_epoch_{}.pth'.format(args.test_epoch))
        else:
            load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 'Multi_DE' +str(i+1) + '_ResNet_' + args.dataset +'_'+ args.folder +  '_best_epoch.pth')

        load_model.append(torch.load(load_file))
        # KK =model[i]
        models[i].load_state_dict(load_model[i]['state_dict'])
    print('Successfully load all ensemble models')
    for model in models:
        model.eval()
    list_acc = []
    u_list =[]
    in_list = []
    label_list = []
    ece_list=[]
    prediction_list = []
    probability_list = []
    one_hot_label_list = []
    one_hot_probability_list = []
    correct_list=[]
    correct_num, data_num = 0, 0
    epoch_auc = 0
    start_time = time.time()
    time_list= []
    nll_list= []
    brier_list = []
    for batch_idx, (data, target) in enumerate(test_loader):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].float().cuda())
        data_num += target.size(0)
        pred = torch.zeros(1,args.num_classes).cuda()
        with torch.no_grad():
            target = Variable(target.long().cuda())
            for i in range(deepen_times):
                # print('ensemble model:{}'.format(i))
                pred_i, _ = models[i](data, target)
                pred += pred_i
            pred = pred/deepen_times
            elapsed_time = time.time() - start_time
            time_list.append(elapsed_time)
            predicted = pred.argmax(dim=-1)
            correct_num += (predicted == target).sum().item()
            correct = (predicted == target)

            list_acc.append((predicted == target).sum().item())
            prediction_list.append(predicted.cpu().detach().float().numpy())
            label_list.append(target.cpu().detach().float().numpy())
            correct_list.append(correct.cpu().detach().float().numpy())

            # label_list = F.one_hot(target, num_classes=args.num_classes).cpu().detach().float().numpy()
            probability = torch.softmax(pred, dim=1).cpu().detach().float().numpy()
            probability_list.append(torch.softmax(pred, dim=1).cpu().detach().float().numpy()[:,1])
            one_hot_probability_list.append(torch.softmax(pred, dim=1).squeeze(dim=0).cpu().detach().float().numpy())
            # one_hot_probability_list.append(pred.data.squeeze(dim=0).cpu().detach().float().numpy())
            one_hot_label = F.one_hot(target, num_classes=args.num_classes).squeeze(dim=0).cpu().detach().float().numpy()
            one_hot_label_list.append(one_hot_label)
            ece_list.append(cal_ece(torch.squeeze(pred), target))
            # NLL brier
            nll, brier = calc_nll_brier(probability, pred, target, one_hot_label)
            nll_list.append(nll)
            brier_list.append(brier)
    logging.info('Single sample test time consumption {:.2f} seconds!'.format(sum(time_list)/len(time_list)))
    print('Single sample test time consumption {:.2f} seconds!'.format(sum(time_list)/len(time_list)))
    if args.num_classes > 2:
        epoch_auc = metrics.roc_auc_score(one_hot_label_list, one_hot_probability_list, multi_class='ovo')
    else:
        epoch_auc = metrics.roc_auc_score(label_list, probability_list)
        # epoch_auc = metrics.roc_auc_score(label_list, probability_list)
    # fpr, tpr, thresholds = roc_curve(label_list, probability_list, pos_label=1)
    # roc_auc = auc(fpr, tpr)
    avg_acc = correct_num/data_num
    avg_ece = sum(ece_list)/len(ece_list)
    # epoch_auc = metrics.roc_auc_score(label_list, prediction_list)
    avg_kappa = cohen_kappa_score(prediction_list, label_list)
    F1_Score = f1_score(y_true=label_list, y_pred=prediction_list, average='weighted')
    Recall_Score = recall_score(y_true=label_list, y_pred=prediction_list, average='weighted')
    aurc, eaurc = calc_aurc_eaurc(probability_list, correct_list)



    if not os.path.exists(os.path.join(args.save_dir, "{}_{}_{}".format(args.model_name,args.dataset,args.folder))):
        os.makedirs(os.path.join(args.save_dir, "{}_{}_{}".format(args.model_name,args.dataset,args.folder)))

    avg_nll = sum(nll_list) / len(nll_list)
    avg_brier = sum(brier_list) / len(brier_list)
    with open(os.path.join(args.save_dir, "{}_{}_{}_Metric.txt".format(args.model_name, args.dataset, args.folder)),
              'w') as Txt:
        Txt.write(
            "Acc: {}, AUC: {}, AURC: {}, EAURC: {},  NLL: {}, BRIER: {}, F1_Score: {}, Recall_Score: {}, Kappa_Score: {}, ECE: {}\n".format(
                round(avg_acc, 6), round(epoch_auc, 6), round(aurc, 6), round(eaurc, 6), round(avg_nll, 6),
                round(avg_brier, 6), round(F1_Score, 6), round(Recall_Score, 6), round(avg_kappa, 6),
                round(avg_ece, 6)
            ))
    # print(
    #     "Acc: {:.4f}, AUC: {:.4f}, AURC: {:.4f}, EAURC: {:.4f}, NLL: {:.4f}, BRIER: {:.4f},  F1_Score: {:.4f}, Recall_Score: {:.4f}, kappa: {:.4f}, ECE: {:.4f}".format(
    #         avg_acc, epoch_auc, aurc, eaurc, avg_nll, avg_brier, F1_Score, Recall_Score, avg_kappa, avg_ece))
    # print('====> mean_inc_u: {:.4f},mean_c_u: {:.4f}\n'.format(mean_inc_u, mean_c_u))
    return avg_acc, epoch_auc, aurc, eaurc, avg_nll, avg_brier, F1_Score, Recall_Score, avg_kappa, avg_ece

if __name__ == "__main__":

    # filename = './datasets/handwritten_6views.mat'
    # image = loadmat(filename)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--end_epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--test_epoch', type=int, default=198, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lambda_epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--modal_number', type=int, default=2, metavar='N',
                        help='modalties number')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--save_dir', default='./results', type=str)
    parser.add_argument("--model_name", default="Base_transformer", type=str, help="Base_transformer/ResNet3D/Res2Net2D/Multi_ResNet/Multi_dropout_ResNet/Multi_DE_ResNet/Multi_CBAM_ResNet/Multi_EF_ResNet")
    parser.add_argument("--dataset", default="MMOCTF", type=str, help="MMOCTF/MGamma/Gamma/OLIVES")
    parser.add_argument("--folder", default="folder0", type=str, help="folder0/folder1/folder2/folder3/folder4")
    parser.add_argument("--mode", default="test", type=str, help="train/test/train&test")
    parser.add_argument("--model_base", default="transformer", type=str, help="transformer/cnn")
    parser.add_argument("--condition", default="noise", type=str, help="noise/normal")
    parser.add_argument("--condition_name", default="Gaussian", type=str, help="Gaussian/SaltPepper/All")
    parser.add_argument("--Condition_SP_Variance", default=0.005, type=int, help="Variance: 0.01/0.1")
    parser.add_argument("--Condition_G_Variance", default=0.05, type=int, help="Variance: 15/1/0.1")

    args = parser.parse_args()
    args.seed_idx = 11

    if args.model_name == "Multi_DE6_ResNet" or args.model_name == "Multi_DE5_ResNet":
        args.batch_size = 8

    # Condition_G_Variance = [0,0.01, 0.03, 0.05, 0.07,0.1]
    # Condition_G_Variance = [0,0.2,0.3,0.4,0.5]
    # Condition_G_Variance = [0,0.1,0.2,0.3,0.4,0.5] # Fundus OCT & OLIVES` Our
    # Condition_G_Variance = [0.1,0.2,0.3,0.4,0.5] # Fundus OCT & OLIVES` Our
    # Condition_G_Variance = [0, 0.05, 0.1, 0.3, 0.5] # OCT
    Condition_G_Variance = [0.1,0.2,0.3,0.4,0.5] # Fundus OCT & OLIVES`
    # Condition_G_Variance = [0,0.01, 0.03, 0.05, 0.07, 0.1,0.3,0.5] # OCT  & GAMMA
    # Condition_G_Variance = [0,0.01, 0.03, 0.05, 0.07] # OCT  & GAMMA
    # Condition_G_Variance = [0,0.01, 0.03, 0.05, 0.07, 0.1,0.3,0.5] # OCT  & GAMMA
    # Condition_G_Variance =[0,0.1,0.2,0.3,0.4,0.5] # Fundus  Fundus & GAMMA`

    if args.dataset =="MMOCTF":
        args.data_path = '/data/zou_ke/projects_data/Multi-OF/2000/'
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
        args.data_path = '/data/zou_ke/projects_data/OLIVES/OLIVES/'
        # args.data_path = '/data/zou_ke/projects_data/OLIVES2/OLIVES/'
        # args.data_path = '/data/zou_ke/projects_data/OLIVES3/OLIVES/'

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
    elif args.dataset =="MGamma":
        args.modalties_name = ["FUN", "OCT"]
        args.dims = [[(128, 256, 128)], [(512, 512)]]
        args.num_classes = 3
        args.modalties = len(args.dims)
        args.base_path = '/data/zou_ke/projects_data/Multi-OF/Gamma/'
        args.data_path = '/data/zou_ke/projects_data/Multi-OF/MGamma/'
        filelists = os.listdir(args.data_path)
        # kf = KFold(n_splits=4, shuffle=True, random_state=10)
        kf = KFold(n_splits=5, shuffle=True, random_state=10)

        y = kf.split(filelists)
        count = 0
        train_filelists = [[], [], [], [], []]
        val_filelists = [[], [], [], [], []]
        for tidx, vidx in y:
            train_filelists[count], val_filelists[count] = np.array(filelists)[tidx], np.array(filelists)[vidx]
            count = count + 1
        f_folder = int(args.folder[-1])
        train_dataset = GAMMA_dataset(args, dataset_root=args.data_path,
                                      oct_img_size=args.dims[0],
                                      fundus_img_size=args.dims[1],
                                      mode='train',
                                      label_file=args.base_path + 'glaucoma_grading_training_GT.xlsx',
                                      filelists=np.array(train_filelists[f_folder]))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = GAMMA_dataset(args, dataset_root=args.data_path,
                                    oct_img_size=args.dims[0],
                                    fundus_img_size=args.dims[1],
                                    mode='val',
                                    label_file=args.base_path + 'glaucoma_grading_training_GT.xlsx',
                                    filelists=np.array(val_filelists[f_folder]), )

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
        test_dataset = val_dataset
        test_loader = val_loader
    else:
        print('There is no this dataset name')
        raise NameError

    if args.model_name =="ResNet3D":
        args.modalties_name = ["OCT"]
        args.modal_number = 1
        args.dims = [(128, 256,128)]
        model = ResNet3D(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name =="Res2Net2D":
        args.modalties_name = ["FUN"]
        args.modal_number = 1
        args.dims = [(512, 512)]
        args.modalties = len(args.dims)
        model = Res2Net2D(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name =="Multi_ResNet":
        args.modalties_name = ["FUN", "OCT"]
        args.modalties = len(args.dims)
        model = Multi_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name =="Base_transformer":
        args.modalties_name = ["FUN", "OCT"]
        args.modalties = len(args.dims)
        model = Base_transformer(args.num_classes, args.modal_number, args.dims, args)
    elif args.model_name =="Multi_EF_ResNet":
        args.modalties_name = ["FUN", "OCT"]
        if args.dataset == 'OLIVES':
            args.dims = [[(48+3, 248, 248)], [(512, 512)]] # OLIVES
        else:
            args.dims = [[(128 + 3, 256, 128)], [(512, 512)]]  # Our
        args.modalties = len(args.dims)
        model = Multi_EF_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name =="Multi_CBAM_ResNet":
        args.modalties_name = ["FUN", "OCT"]
        args.modalties = len(args.dims)
        model = Multi_CBAM_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name =="Multi_dropout_ResNet":
        args.modalties_name = ["FUN", "OCT"]
        args.modalties = len(args.dims)
        model = Multi_dropout_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name == "Multi_DE1_ResNet":
        args.lr = 0.0001
        args.modalties_name = ["FUN", "OCT"]
        model = Multi_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name == "Multi_DE2_ResNet":
        args.lr = 0.0003
        args.modalties_name = ["FUN", "OCT"]
        model = Multi_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name == "Multi_DE3_ResNet":
        args.lr = 0.001
        args.modalties_name = ["FUN", "OCT"]
        model = Multi_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name == "Multi_DE4_ResNet":
        args.lr = 0.0002
        args.modalties_name = ["FUN", "OCT"]
        model = Multi_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name == "Multi_DE5_ResNet":
        args.lr = 0.00001
        args.modalties_name = ["FUN", "OCT"]
        # args.modalties = len(args.dims)
        model = Multi_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
    elif args.model_name == "Multi_DE_ResNet":
        args.modalties_name = ["FUN", "OCT"]
        # args.modalties = len(args.dims)
        models = []
        model = Multi_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
        # model2 = Multi_ensemble_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
        # model3 = Multi_ensemble_3D_ResNet(args.num_classes, args.modal_number, args.dims, args.lambda_epochs)
        models.append(model)
        models.append(model)
        models.append(model)
        models.append(model)
        models.append(model)
    else:
        print('There is no this model name')
        raise NameError

    N_mini_batches = len(train_loader)
    print('The number of training images = %d' % N_mini_batches)

    seed_num = list(range(1,11))

    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'results', args.dataset +'_'+ args.model_name)
    log_file = log_dir + '.txt'
    log_args(log_file)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    model.cuda()
    best_acc = 0
    loss_list = []
    acc_list = []
    if args.mode =='train&test':
        epoch = 0
        print('===========Train begining!===========')
        for epoch in range(args.start_epoch, args.end_epochs + 1):
            print('Epoch {}/{}'.format(epoch, args.end_epochs))
            epoch_loss = train(epoch,train_loader,model)
            print("epoch %d avg_loss:%0.3f" % (epoch, epoch_loss.avg))
            val_loss, best_acc = val(epoch,val_loader,model,best_acc)
            loss_list.append(epoch_loss.avg)
            acc_list.append(best_acc)
        loss_plot(args, loss_list)
        metrics_plot(args, 'acc', acc_list)
        test_acc, test_acclist  = test(args,test_loader,model,epoch)
    elif args.mode == 'test':
        epoch = args.test_epoch
        for i in range(len(Condition_G_Variance)):
            args.Condition_G_Variance = Condition_G_Variance[i]
            print("Gaussian noise: %f"%args.Condition_G_Variance)
            logging.info("Gaussian noise: %f" % args.Condition_G_Variance)
            acc_list, auc_list, aurc_list, eaurc_list, nll_list, brier_list, \
            F1_list, Rec_list, kap_list, ECE_list = [], [], [], [], [], [], [], [], [], []
            for j in range(len(seed_num)):
                args.seed_idx = seed_num[j]
                # print("seed_idx: %d" % args.seed_idx)
                # logging.info("seed_idx: %d" % args.seed_idx)

                if args.dataset == "MMOCTF":
                    args.data_path = '/data/zou_ke/projects_data/Multi-OF/2000/'
                    args.modalties_name = ["FUN", "OCT"]
                    args.modal_number = len(args.modalties_name)
                    args.num_classes = 2
                    args.dims = [[(128, 256, 128)], [(512, 512)]]
                    args.modalties = len(args.dims)
                    train_loader = torch.utils.data.DataLoader(
                        Multi_modal_data(args.data_path, args.modal_number, args.modalties_name, 'train',
                                         args.condition, args, folder=args.folder), batch_size=args.batch_size)
                    val_loader = torch.utils.data.DataLoader(
                        Multi_modal_data(args.data_path, args.modal_number, args.modalties_name, 'val', args.condition,
                                         args, folder=args.folder), batch_size=1)
                    test_loader = torch.utils.data.DataLoader(
                        Multi_modal_data(args.data_path, args.modal_number, args.modalties_name, 'test', args.condition,
                                         args, folder=args.folder), batch_size=1)
                    N_mini_batches = len(train_loader)
                    print('The number of training images = %d' % N_mini_batches)
                elif args.dataset == "Gamma":
                    args.modalties_name = ["FUN", "OCT"]
                    args.dims = [[(128, 256, 128)], [(512, 512)]]

                    args.num_classes = 3
                    args.modalties = len(args.dims)
                    args.base_path = '/data/zou_ke/projects_data/Multi-OF/Gamma/'
                    args.data_path = args.base_path + 'multi-modality_images/'
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
                    train_dataset = GAMMA_sub1_dataset(dataset_root=args.data_path,
                                                       oct_img_size=args.dims[0],
                                                       fundus_img_size=args.dims[1],
                                                       mode='train',
                                                       label_file=args.base_path + 'glaucoma_grading_training_GT.xlsx',
                                                       filelists=np.array(train_filelists[0]))

                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
                    val_dataset = GAMMA_sub1_dataset(dataset_root=args.data_path,
                                                     oct_img_size=args.dims[0],
                                                     fundus_img_size=args.dims[1],
                                                     mode='val',
                                                     label_file=args.base_path + 'glaucoma_grading_training_GT.xlsx',
                                                     filelists=np.array(val_filelists[0]), )

                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
                    test_dataset = val_dataset
                    test_loader = val_loader
                elif args.dataset == "MGamma":
                    args.modalties_name = ["FUN", "OCT"]
                    args.dims = [[(128, 256, 128)], [(512, 512)]]
                    args.num_classes = 3
                    args.modalties = len(args.dims)
                    args.base_path = '/data/zou_ke/projects_data/Multi-OF/Gamma/'
                    args.data_path = '/data/zou_ke/projects_data/Multi-OF/MGamma/'
                    filelists = os.listdir(args.data_path)
                    # kf = KFold(n_splits=5, shuffle=True, random_state=10)
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
                    train_dataset = GAMMA_dataset(args, dataset_root=args.data_path,
                                                  oct_img_size=args.dims[0],
                                                  fundus_img_size=args.dims[1],
                                                  mode='train',
                                                  label_file=args.base_path + 'glaucoma_grading_training_GT.xlsx',
                                                  filelists=np.array(train_filelists[f_folder]))

                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
                    val_dataset = GAMMA_dataset(args, dataset_root=args.data_path,
                                                oct_img_size=args.dims[0],
                                                fundus_img_size=args.dims[1],
                                                mode='val',
                                                label_file=args.base_path + 'glaucoma_grading_training_GT.xlsx',
                                                filelists=np.array(val_filelists[f_folder]), )

                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
                    test_dataset = val_dataset
                    test_loader = val_loader
                elif args.dataset == "OLIVES":
                    args.data_path = '/data/zou_ke/projects_data/OLIVES/OLIVES/'
                    # args.data_path = '/data/zou_ke/projects_data/OLIVES2/OLIVES/'
                    # args.data_path = '/data/zou_ke/projects_data/OLIVES3/OLIVES/'

                    args.modalties_name = ["FUN", "OCT"]
                    args.num_classes = 2
                    args.dims = [[(48, 248, 248)], [(512, 512)]]
                    args.modalties = len(args.dims)

                    test_loader = torch.utils.data.DataLoader(
                        OLIVES_dataset(args.data_path, args.modal_number, args.modalties_name, 'test', args.condition,
                                       args, folder=args.folder), batch_size=1)
                    N_mini_batches = len(test_loader)
                    print('The number of testing images = %d' % N_mini_batches)
                else:
                    print('There is no this dataset name')
                    raise NameError
                if args.model_name == 'Multi_DE_ResNet' or args.model_name == 'Multi_dropout_ResNet':
                    if args.model_name == 'Multi_DE_ResNet':
                        test_acc, test_auc, test_aurc, test_eaurc, test_nll, test_brier,test_F1, test_Rec, test_kappa, test_ece\
                            = test_ensemble(args,test_loader,models,epoch)
                    else:
                        test_acc, test_auc, test_aurc, test_eaurc, test_nll, test_brier,test_F1, test_Rec, test_kappa, test_ece \
                            = test(args,test_loader,model,epoch)
                    nll_list.append(test_nll)
                    brier_list.append(test_brier)
                else:
                    test_acc, test_auc, test_aurc, test_eaurc, test_F1, test_Rec, test_kappa, test_ece \
                        = test(args, test_loader, model, epoch)
                acc_list.append(test_acc)
                auc_list.append(test_auc)
                aurc_list.append(test_aurc)
                eaurc_list.append(test_eaurc)
                F1_list.append(test_F1)
                Rec_list.append(test_Rec)
                kap_list.append(test_kappa)
                ECE_list.append(test_ece)
            if args.model_name == 'Multi_DE_ResNet' or args.model_name == 'Multi_dropout_ResNet':
                acc_list_mean, acc_list_std = np.mean(acc_list), np.std(acc_list)
                auc_list_mean, auc_list_std = np.mean(auc_list), np.std(auc_list)
                aurc_list_mean, aurc_list_std = np.mean(aurc_list), np.std(aurc_list)
                eaurc_list_mean, eaurc_list_std = np.mean(eaurc_list), np.std(eaurc_list)
                nll_list_mean, nll_list_std = np.mean(nll_list), np.std(nll_list)
                brier_list_mean, brier_list_std = np.mean(brier_list), np.std(brier_list)
                F1_list_mean, F1_list_std = np.mean(F1_list), np.std(F1_list)
                Rec_list_mean, Rec_list_std = np.mean(Rec_list), np.std(Rec_list)
                kap_list_mean, kap_list_std = np.mean(kap_list), np.std(kap_list)
                ECE_list_mean, ECE_list_std = np.mean(ECE_list), np.std(ECE_list)
                print(
                    "Mean_Std_Acc: {:.4f} +- {:.4f}, Mean_Std_AUC: {:.4f} +- {:.4f},Mean_Std_AURC: {:.4f} +- {:.4f}, "
                    "Mean_Std_EAURC: {:.4f} +- {:.4f}, Mean_Std_nll: {:.4f} +- {:.4f}, Mean_Std_brier: {:.4f} +- {:.4f}, Mean_Std_F1_Score: {:.4f} +- "
                    "{:.4f}, Mean_Std_Recall_Score: {:.4f} +- {:.6f}, Mean_Std_kappa: {:.4f} +- {:.4f}, Mean_Std_ECE: {:.4f} +- {:.4f}".format(
                        acc_list_mean, acc_list_std, auc_list_mean, auc_list_std, aurc_list_mean, aurc_list_std,
                        eaurc_list_mean, eaurc_list_std,
                        nll_list_mean, nll_list_std, brier_list_mean, brier_list_std, F1_list_mean, F1_list_std,
                        Rec_list_mean, Rec_list_std, kap_list_mean, kap_list_std, ECE_list_mean, ECE_list_std))
                logging.info(
                    "Mean_Std_Acc: {:.4f} +- {:.4f}, Mean_Std_AUC: {:.4f} +- {:.4f},Mean_Std_AURC: {:.4f} +- {:.4f}, "
                    "Mean_Std_EAURC: {:.4f} +- {:.4f}, Mean_Std_nll: {:.4f} +- {:.4f}, Mean_Std_brier: {:.4f} +- {:.4f}, Mean_Std_F1_Score: {:.4f} +- "
                    "{:.4f}, Mean_Std_Recall_Score: {:.4f} +- {:.4f}, Mean_Std_kappa: {:.4f} +- {:.4f}, Mean_Std_ECE: {:.4f} +- {:.4f}".format(
                        acc_list_mean, acc_list_std, auc_list_mean, auc_list_std, aurc_list_mean, aurc_list_std,
                        eaurc_list_mean, eaurc_list_std,
                        nll_list_mean, nll_list_std, brier_list_mean, brier_list_std, F1_list_mean, F1_list_std,
                        Rec_list_mean, Rec_list_std, kap_list_mean, kap_list_std, ECE_list_mean, ECE_list_std))
            else:
                acc_list_mean,acc_list_std = np.mean(acc_list),np.std(acc_list)
                auc_list_mean,auc_list_std = np.mean(auc_list),np.std(auc_list)
                aurc_list_mean,aurc_list_std = np.mean(aurc_list),np.std(aurc_list)
                eaurc_list_mean,eaurc_list_std = np.mean(eaurc_list),np.std(eaurc_list)
                F1_list_mean,F1_list_std = np.mean(F1_list),np.std(F1_list)
                Rec_list_mean,Rec_list_std = np.mean(Rec_list),np.std(Rec_list)
                kap_list_mean,kap_list_std = np.mean(kap_list),np.std(kap_list)
                ECE_list_mean,ECE_list_std = np.mean(ECE_list),np.std(ECE_list)
                print(
                    "Mean_Std_Acc: {:.4f} +- {:.6f}, Mean_Std_AUC: {:.4f} +- {:.4f},Mean_Std_AURC: {:.4f} +- {:.4f}, "
                    "Mean_Std_EAURC: {:.4f} +- {:.4f}, Mean_Std_F1_Score: {:.4f} +- "
                    "{:.4f}, Mean_Std_Recall_Score: {:.4f} +- {:.4f}, Mean_Std_kappa: {:.4f} +- {:.4f}, Mean_Std_ECE: {:.4f} +- {:.4f}".format(
                        acc_list_mean, acc_list_std, auc_list_mean,auc_list_std, aurc_list_mean, aurc_list_std, eaurc_list_mean, eaurc_list_std,
                        F1_list_mean, F1_list_std,Rec_list_mean,Rec_list_std,kap_list_mean,kap_list_std,ECE_list_mean,ECE_list_std))
                logging.info(
                    "Mean_Std_Acc: {:.4f} +- {:.4f}, Mean_Std_AUC: {:.4f} +- {:.4f},Mean_Std_AURC: {:.4f} +- {:.4f}, "
                    "Mean_Std_EAURC: {:.4f} +- {:.4f}, Mean_Std_F1_Score: {:.4f} +- {:.4f}, Mean_Std_Recall_Score: {:.4f} +- {:.4f}, "
                    "Mean_Std_kappa: {:.4f} +- {:.4f}, Mean_Std_ECE: {:.4f} +- {:.4f}".format(
                        acc_list_mean, acc_list_std, auc_list_mean,auc_list_std, aurc_list_mean, aurc_list_std, eaurc_list_mean, eaurc_list_std,
                        F1_list_mean, F1_list_std,Rec_list_mean,Rec_list_std,kap_list_mean,kap_list_std,ECE_list_mean,ECE_list_std))




