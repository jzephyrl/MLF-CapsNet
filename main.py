"""
Keras implementation of Multi-level Features Guided Capsule Network (MLF-CapsNet).
This file trains a MLF-CapsNet on DEAP/DREAMER dataset with the parameters as mentioned in paper.
We have developed this code using the following GitHub repositories:
- Xifeng Guo's CapsNet code (https://github.com/XifengGuo/CapsNet-Keras)

Usage:
       python capsulenet-multi-gpu.py --gpus 2

"""
#每个被试者训练一个模型，使用胶囊网络，原始参数和较少参数

# from capsulelayers import CapsuleLayer, PrimaryCap, Length

import pandas as pd
import time
import pickle
import numpy as np
import numpy as np
# import tensorflow as tf
import os
# from keras import callbacks
# from keras.utils.vis_utils import plot_model
# from keras.utils import multi_gpu_mod
import torch
from torchvision.transforms import Compose, ToTensor
import torch_geometric.transforms as T
# setting the hyper parameters
import argparse
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from CapsuleNet import CapsuleNetwork
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
def deap_load(data_file,dimention,debaseline):
    rnn_suffix = ".mat_win_128_rnn_dataset.pkl"
    label_suffix = ".mat_win_128_labels.pkl"
    arousal_or_valence = dimention
    with_or_without = debaseline # 'yes','not'
    dataset_dir = "../deap_shuffled_data/" + with_or_without + "_" + arousal_or_valence + "/"

    ###load training set
    with open(dataset_dir + data_file + rnn_suffix, "rb") as fp:
        rnn_datasets = pickle.load(fp)
    with open(dataset_dir + data_file + label_suffix, "rb") as fp:
        labels = pickle.load(fp)
        labels = np.transpose(labels)

    # print("type(labels):",type(labels))
    # print("label:",labels)
    # labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)


    # shuffle data
    index = np.array(range(0, len(labels)))
    np.random.shuffle(index)
    rnn_datasets = rnn_datasets[index]  # .transpose(0,2,1)
    labels = labels[index]
    # print("rnn_datasets:",rnn_datasets.shape)
    datasets = rnn_datasets.reshape(-1, 1,128, 32).astype('float32')
    labels = labels.astype('float32')
    # print("labels:",labels)
    return datasets , labels

def dreamer_load(sub,dimention,debaseline):
    if debaseline == 'yes':
        dataset_suffix = "f_dataset.pkl"
        label_suffix = "_labels.pkl"
        dataset_dir = "/home/bsipl_5/experiment/Data/data_pre(-base)/" + dimention + "/"
    else:
        dataset_suffix = "_rnn_dataset.pkl"
        label_suffix = "_labels.pkl"
        dataset_dir = '/home/bsipl_5/experiment/ijcnn-master/dreamer_shuffled_data/' + 'no_' + dimention + '/'

    ###load training set
    with open(dataset_dir + sub + dataset_suffix, "rb") as fp:
        datasets = pickle.load(fp)
    with open(dataset_dir + sub + '_' + dimention + label_suffix, "rb") as fp:
        labels = pickle.load(fp)
        labels = np.transpose(labels)

    labels = labels > 3
    labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)


    # shuffle data
    index = np.array(range(0, len(labels)))
    np.random.shuffle(index)
    datasets = datasets[index]  # .transpose(0,2,1)
    labels = labels[index]

    datasets = datasets.reshape(-1, 128, 14, 1).astype('float32')
    labels = labels.astype('float32')

    return datasets , labels

#create dataset
def EEG_Dataset(feat, label):
    dataset = []
    for i in range(feat.shape[0]):
        dataset.append(Data(x=torch.tensor(feat[i], dtype=torch.float), y=label[i]))
    return dataset



time_start_whole = time.time()

dataset_name = 'deap' #'deap' # dreamer
# subjects =['s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13','s14','s15','s16']#,'s05']#,'s06','s07','s08']#,'s09','s10','s11','s12','s13','s14','s15','s16'，'s17','s18','s19','s20','s21','s22','s23','s24','s25','s26','s27','s28',]
subjects = ['s02']#,'s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23','s24','s25','s26','s28','s29','s30','s31','s32'] 
dimentions = ['dominance']#,'arousal','dominance']
debaseline = 'yes' # yes or not
tune_overfit = 'tune_overfit'
model_version = 'v2' # v0:'CapsNet', v1:'MLF-CapsNet(w/o)', v2:'MLF-CapsNet'


#save model
# def get_assigned_file(checkpoint_dir,num):
#     assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
#     return assign_file

# def get_resume_file(checkpoint_dir):
#     filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
#     if len(filelist) == 0:
#         return None

#     filelist =  [ x  for x in filelist if os.path.basename(x) != 'best.tar' ]
#     epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
#     max_epoch = np.max(epochs)
#     resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
#     return resume_file

# def get_best_file(checkpoint_dir):    
#     best_file = os.path.join(checkpoint_dir, 'best.tar')
#     if os.path.isfile(best_file):
#         return best_file
#     else:
#         return get_resume_file(checkpoint_dir)



def to_one_hot(x, length):
    batch_size = len(x)
    x_one_hot = torch.zeros(batch_size, length)
    for i in range(batch_size):
        x_one_hot[i, int(x[i])] = 1.0
    return x_one_hot

if __name__ == "__main__":
    for dimention in dimentions:
        Test_Acc=[]
        for subject in subjects:
            import numpy as np
            # import tensorflow as tf
            import os
            # from keras import callbacks
            # from keras.utils.vis_utils import plot_model

            # setting the hyper parameters
            import argparse
            parser = argparse.ArgumentParser(description="Capsule Network on " + dataset_name)
            parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
            parser.add_argument('--epochs', default=40, type=int)  # v0:20, v2:40
            parser.add_argument('--batch_size', default=35, type=int)
            parser.add_argument('--lam_regularize', default=0.0, type=float,
                                help="The coefficient for the regularizers")
            parser.add_argument('-r', '--routings', default=3, type=int,
                                help="Number of iterations used in routing algorithm. should > 0")
            parser.add_argument('--debug', default=0, type=int,
                                help="Save weights by TensorBoard")
            parser.add_argument('--save_dir', default='./result_'+ dataset_name + '/sub_dependent_'+ model_version +'/') # other
            parser.add_argument('-t', '--testing', action='store_true',
                                help="Test the trained model on testing dataset")
            parser.add_argument('-w', '--weights', default=None,
                                help="The path of the saved weights. Should be specified when testing")
            parser.add_argument('--lr', default=0.00001, type=float,
                                help="Initial learning rate")  # v0:0.0001, v2:0.00001
            # parser.add_argument('--lam_regularize', default=0.0, type=float,
                                # help="The coefficient for the regularizers")
            # parser.add_argument('--gpus', default=2, type=int)
            parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
            parser.add_argument('--stop_epoch'  , default=400, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
            parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
            parser.add_argument('--save_freq'   , default=10, type=int, help='Save frequency')
            parser.add_argument('--model'       , default='Capsule',      help='model') 
            args = parser.parse_args()
            GPU_IDX=0            
            DEVICE = torch.device('cuda:{}'.format(GPU_IDX) if torch.cuda.is_available() else 'cpu')
            print(time.asctime(time.localtime(time.time())))
            print(args)
            start_epoch = args.start_epoch
            stop_epoch = args.stop_epoch

            # print("sub:",subject)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            checkpoint_dir = '%s%s/checkpoints'%(args.save_dir,args.model)
            #load dataset
            if dataset_name == 'dreamer':         
                # load dreamer data
                datasets,labels = dreamer_load(subject,dimention,debaseline)
            else:  # load deap data
                datasets,labels = deap_load(subject,dimention,debaseline)
            # print("datasets:",datasets.shape)
            # print("labels:",labels.shape)
            args.save_dir = args.save_dir + '/' + debaseline + '/' + subject + '_' + dimention + str(args.epochs)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            #十折交叉验证
            fold = 10
            test_accuracy_allfold =[]
            # train_used_time_allfold = np.zeros(shape=[0], dtype=float)
            # test_used_time_allfold = np.zeros(shape=[0], dtype=float)
            for curr_fold in range(fold):
                fold_size = datasets.shape[0] // fold
                indexes_list = [i for i in range(len(datasets))]
                #indexes = np.array(indexes_list)
                split_list = [i for i in range(curr_fold * fold_size, (curr_fold + 1) * fold_size)]
                # print("split_list:",split_list)
                split = np.array(split_list)
                x_test = datasets[split]
                y_test = labels[split]

                split = np.array(list(set(indexes_list) ^ set(split_list)))
                # print("split:",split)
                x_train = datasets[split]
                y_train = labels[split]
                # print("x_train:",x_train.shape) [2160,128,32,1]
                # print("x_train.shape[1:]:",x_train.shape[1:]) (128,32,1)
                
                #create dataloader
                Train=EEG_Dataset(x_train,y_train)
                Trainloader=DataLoader(Train,batch_size=35,shuffle=False)
                print("Trainloader:",len(Trainloader.dataset))
                Test=EEG_Dataset(x_test,y_test)
                Testloader=DataLoader(Test,batch_size=35,shuffle=False)

       
                # define model
                model=CapsuleNetwork(eeg_width=128,
                         eeg_height=32,
                         eeg_channels=1,
                         conv_inputs=1,
                         conv_outputs=256,
                         num_primary_units=8,
                         primary_unit_size=32*60*12,
                         num_output_units=2, # one for each digit
                         output_unit_size=16).to(DEVICE)
                
                # plot_model(model, to_file=args.save_dir+'/model_fold'+str(curr_fold)+'.png', show_shapes=True)

                # define muti-gpu model
                # train
                train_start_time = time.time()
                for epoch in range(1,args.epochs+1):
                    correct=0
                    optimizer = optim.Adam(model.parameters(), lr=args.lr)
                    for i,data in enumerate(tqdm(Trainloader)):
                        # label_indeices=np.array(data.y)
                        model.train()
                        data=data.to(DEVICE,non_blocking=True)
                        label_indeices=torch.as_tensor(data.y,dtype=torch.long).to(DEVICE)
                        label=torch.as_tensor(to_one_hot(data.y,2),dtype=torch.long).to(DEVICE)
                        # print("label:",label)
                        optimizer.zero_grad()
                        output=model(data)
                        # print("v_eeg:",v_eeg)
                        loss=model.loss(output,label)
                        loss.backward()
                        # last_loss=loss.data[0]
                        optimizer.step()
                        v_eeg = torch.sqrt((output**2).sum(dim=2, keepdim=True))
                        pred = v_eeg.data.max(1, keepdim=True)[1]
                        # print("pred:",pred)
                        # correct+=(pred==label_indeices).sum().item()
                        correct += pred.eq(label_indeices.view_as(pred)).sum()
                        # print("loss:",loss)
                        # print("correct:",correct)
                    train_acc=100.*correct/len(Trainloader.dataset)
                    print('subject:{} | curr_fold:{} | Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} train_acc: {}'.format(subject,curr_fold,epoch,i * len(data), len(Trainloader.dataset),100. * i / len(Trainloader),loss, train_acc))
                # if not os.path.isdir(checkpoint_dir):
                #     os.makedirs(checkpoint_dir)

                # if (epoch % args.save_freq==0) or (epoch==stop_epoch-1):
                #     outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
                #     torch.save({'epoch':epoch, 'state':model.state_dict() }, outfile)
                train_used_time_fold = time.time() - train_start_time
                # model.save_weights(args.save_sdir + '/trained_model_fold'+str(curr_fold)+'.h5')
                # print('Trained model saved to \'%s/trained_model_fold%s.h5\'' % (args.save_dir,curr_fold))
                print('Train time: ', train_used_time_fold)

                #no-valid
                #test
                print('-' * 30 + 'fold  ' + str(curr_fold) + '  Begin: test' + '-' * 30)
                test_start_time = time.time()
                model.eval()
                with torch.no_grad():
                    test_loss=0
                    correct=0
                    for i,data in enumerate(Testloader,0):
                        data=data.to(DEVICE,non_blocking=True)
                        label_indeices=torch.as_tensor(data.y,dtype=torch.long).to(DEVICE)
                        label=torch.as_tensor(to_one_hot(data.y,2),dtype=torch.long).to(DEVICE)
                        output=model(data)
                        test_loss+=model.loss(output,label)
                        v_eeg = torch.sqrt((output**2).sum(dim=2, keepdim=True))
                        pred = v_eeg.data.max(1, keepdim=True)[1]
                        correct += pred.eq(label_indeices.view_as(pred)).sum()
                    test_loss /= len(Testloader.dataset)
                    test_acc_fold=(100.*correct/len(Testloader.dataset)).item()
                    # y_pred = eval_model.predict(x_test, batch_size=100)  # batch_size = 100
                    test_used_time_fold = time.time() - test_start_time
                    # test_acc_fold = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
                    #print('shape of y_pred: ',y_pred.shape[0])
                    print('(' + time.asctime(time.localtime(time.time())) + ') Test acc:', test_acc_fold,'Test loss:',test_loss.item(), 'Test time: ',test_used_time_fold )
                    print('-' * 30 + 'fold  ' + str(curr_fold) + '  End: test' + '-' * 30)
                    test_accuracy_allfold.append(test_acc_fold)
                    # train_used_time_allfold = np.append(train_used_time_allfold, train_used_time_fold)
                    # test_used_time_allfold = np.append(test_used_time_allfold, test_used_time_fold)
    
                # K.clear_session()

            # summary = pd.DataFrame({'fold': range(1,fold+1), 'Test accuracy': test_accuracy_allfold, 'train time': train_used_time_allfold, 'test time': test_used_time_allfold})
            # hyperparam = pd.DataFrame({'average acc of 10 folds': np.mean(test_accuracy_allfold), 'average train time of 10 folds': np.mean(train_used_time_allfold), 'average test time of 10 folds': np.mean(test_used_time_allfold),'epochs': args.epochs, 'lr':args.lr, 'batch size': args.batch_size},index=['dimention/sub'])
            # writer = pd.ExcelWriter(args.save_dir + '/'+'summary'+ '_'+subject+'.xlsx')
            # summary.to_excel(writer, 'Result', index=False)
            # hyperparam.to_excel(writer, 'HyperParam', index=False)
            # writer.save()
            Test_Acc.append(np.mean(test_accuracy_allfold))
            print('suject {} 10 fold average accuracy: {} '.format(subject,np.mean(test_accuracy_allfold)))
        print('Test_Acc:',np.mean(Test_Acc))
            # print('suject {} 10 fold average accuracy: {} '.format(subject, torch.mean(torch.stack(test_accuracy_allfold))))
        
            # print('suject {} 10 fold average train time:{} '.format(subject,np.mean(train_used_time_allfold)) )
            # print('suject {} 10 fold average test time:{} '.format(subject,np.mean(test_used_time_allfold)))