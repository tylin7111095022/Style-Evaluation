# 當創建新環境時，需要將ultralytics/nn/modules內的Classify class做更動
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision.datasets import ImageFolder
#custom
from utils import split_dataset
from models import initialize_model, get_optim
from models.loss import get_loss_fn

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, dest='mname', default='resnet', help='deep learning model will be used')
    parser.add_argument("--loss", choices=["asl", "bce"],default="bce", help='loss function when training')
    parser.add_argument("--optim", type=str, default='Adam', help='optimizer')
    parser.add_argument("--in_channel", type=int, default=3, dest="inch",help="the number of input channel of model")
    parser.add_argument("--img_size", type=int, default=128,help="image size")
    parser.add_argument("--nclass", type=int, default=2,help="the number of class for classification task")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--epoches", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size during training")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate for optimizer")
    parser.add_argument("--weight_path", type=str,dest='wpath', default='./best.pth', help="path of model we trained best")
    parser.add_argument("--is_parallel", type=bool, default=False, dest="paral",help="parallel calculation at multiple gpus")
    parser.add_argument("--device", type=str, default='cuda:0', help='device trainging deep learning')
    return parser.parse_args()

def main():
    #設置Logging架構
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    log_filename = 'log.txt'
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    
    hparam = get_args()
    logging.info(hparam)
    # TRANSPIPE = transforms.Compose([transforms.Resize((hparam.img_size,hparam.img_size))])

    #資料隨機分選訓練、測試集
    
    train_pipe = transforms.Compose([transforms.Resize((hparam.img_size,hparam.img_size)),
                                          transforms.ToTensor()])
    
    fundus_dataset = ImageFolder(root=r"D:\tsungyu\chromosome_data\cyclegan_data\real_img",transform=train_pipe)
    logging.info(fundus_dataset.class_to_idx)
    trainset, testset = split_dataset(fundus_dataset,test_ratio=0.2,seed=20230823)

    model = initialize_model(hparam.mname,num_classes=hparam.nclass,use_pretrained=True)
    if torch.cuda.device_count() > 1 and hparam.paral:
          logging.info(f"use {torch.cuda.device_count()} GPUs!")
          model = torch.nn.DataParallel(model)
    
    # logging.info(model)
    optimizer = get_optim(optim_name=hparam.optim,model=model,lr=hparam.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                              mode='min',
                                                              factor=0.5,
                                                              patience=3,
                                                              min_lr=1e-6,
                                                              verbose =True)
    device = torch.device( hparam.device if torch.cuda.is_available() else 'cpu')
    criteria = get_loss_fn(hparam.loss)

    #training
    history = training(model,trainset, testset, criteria, optimizer,lr_scheduler, device, hparam)
    #plotting
    plot_history(history['train_history'],history['validationn_history'],saved=True)

    fig, ax = plt.subplots(1,1, figsize = (15, 8))
    ax[0].plot(torch.arange(1,hparam.epoches+1,dtype=torch.int64),history['acc_history'])
    ax[0].grid(visible=True)
    ax[0].set_ylim(0.0, 1.0)
    ax[0].set_title(f"accuracy History")

    plt.tight_layout()
    fig.savefig("accuracy")

    return

def evaluate(model,dataset, loss_fn,device,hparam):
    model.eval()
    model = model.to(device)
    total_loss = 0
    ys_true = ys_pred = []
    dataloader = DataLoader(dataset=dataset,batch_size=hparam.batch_size,shuffle=False)
    for img_data,labels in dataloader:
        img_data = img_data.to(device)
        labels = labels.to(device).squeeze() #變為一軸
        logits = model(img_data).squeeze()
        loss = loss_fn(logits,labels)
        total_loss += loss.item()
        labels_arr = labels.cpu().numpy()
        y_pred = torch.softmax(logits.detach(),dim=1).squeeze().argmax(1)
        y_pred = y_pred.cpu().numpy()
        ys_true.append(labels_arr.astype(np.int64))
        ys_pred.append(y_pred)

    acc = accuracy_score(ys_true, ys_pred)

    mean_loss = total_loss/ ((len(dataset)//hparam.batch_size)+1)

    return mean_loss, acc

def training(model, trainset, testset, loss_fn, optimizer,lr_scheduler, device, hparam):
    model = model.to(device)
    dataloader = DataLoader(trainset,batch_size=hparam.batch_size,shuffle=True,)

    logging.info(f'''Starting training:
        Model:          {hparam.mname}
        Optimizer:      {hparam.optim}
        Epochs:         {hparam.epoches}
        Batch size:     {hparam.batch_size}
        Training size:  {len(trainset)}
        Testing size:   {len(testset)}
        Image size:     {hparam.img_size}
        Device:         {device.type}
        Initial learning rate:  {hparam.lr}
    ''')

    train_history = []
    validationn_history = []
    acc_history = []
    for epoch in range(1,hparam.epoches+1):
        model.train()
        epoch_loss = 0
        for img_data, labels in tqdm(dataloader):
            img_data = img_data.to(device)
            labels = labels.to(device)
            logits = model(img_data).squeeze()
            # print(f"label shape: {labels.shape}, logits shape: {logits.shape}")
            loss = loss_fn(logits,labels)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info(f'train Loss for epoch {epoch}: {epoch_loss/((len(trainset)//hparam.batch_size)+1):.4f}')
        train_history.append(epoch_loss/((len(trainset)//hparam.batch_size)+1))

        test_mean_loss, acc = evaluate(model=model,dataset=testset,loss_fn=loss_fn,device=device,hparam=hparam)
        logging.info(f"test acc is {acc*100}%.")
    
        lr_scheduler.step(test_mean_loss) # lr_scheduler 參照 test_mean_loss

        validationn_history.append(test_mean_loss)
        acc_history.append(acc)
        
        #儲存最佳的模型
        if epoch == 1:
            criterion = acc
            torch.save(model.state_dict(), hparam.wpath)
            logging.info(f'at epoch {epoch}, BESTMODEL.pth saved!')
        elif(acc >= criterion):
            criterion = acc
            torch.save(model.state_dict(),hparam.wpath)
            logging.info(f'at epoch {epoch}, BESTMODEL.pth saved!')
            
        torch.save(model.state_dict(),"./last.pth")

    return dict(train_history=train_history,
                validationn_history=validationn_history,
                acc_history=acc_history,
                )

def plot_history(trainingloss:list,validationloss:list, saved:bool=False,figname:str='history'):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (12, 6))
    ax.plot(torch.arange(1,len(trainingloss)+1,dtype=torch.int64), trainingloss, marker=".")
    ax.plot(torch.arange(1,len(validationloss)+1,dtype=torch.int64), validationloss, marker=".")
    ax.grid(visible=True)
    ax.legend(['TrainingLoss', 'ValidationLoss'])
    ax.set_title(f"Train History")

    fig.tight_layout()
    fig.show()
    if saved:
        fig.savefig(figname)

    return

if __name__ == '__main__':
    main()
