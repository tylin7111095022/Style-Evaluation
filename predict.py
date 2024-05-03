from models import initialize_model
import torch
from torchvision import transforms
import argparse
import os
import warnings
from tqdm import tqdm
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ChromosomeDataset, PairDataset

warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, dest='mname', default='resnet152', help='deep learning model will be used')
    parser.add_argument("--in_channel", type=int, default=3, dest="inch",help="the number of input channel of model")
    parser.add_argument("--image_dirroot", type=str, dest="imgroot", default=r'D:\tsungyu\chromosome_data\cyclegan_data\fake_zong', )
    parser.add_argument("--img_size", type=int, default=128,help="image size")
    parser.add_argument("--nclass", type=int, default=2,help="the number of class for classification task")
    parser.add_argument("--weight_path", type=str,dest='wpath', default=r'log\resnet152\best.pth', help="path of model we want to load")
    parser.add_argument("--class_ndx", type=int, default=1,help="label index (chang_label = 0 zong_label = 1)")

    return parser.parse_args()

def main():
    # hparam = get_args()
    # model = initialize_model(hparam.mname,num_classes=hparam.nclass,use_pretrained=True)
    # model.load_state_dict(torch.load(hparam.wpath))
    # correct_portion = evaluate_imgs(model,hparam)
    # print(f"correct_portion is {correct_portion}")
    eval_style(k=128,imgsize=128,weight=r"log\resnet152\best.pth")
    pass
    
def eval_style(k:int,imgsize:int, weight):
    encoder = initialize_model("encoder")
    encoder = load_weight(encoder,weight)
    dataset = ChromosomeDataset(dataset_root=r"D:\\tsungyu\\chromosome_data\\cyclegan_data\\", imgsize=imgsize)
    pairset = PairDataset(domain1_dir=r"D:\\tsungyu\\chromosome_data\\cyclegan_data\\real_chang\\",
                          domain2_dir=r"D:\\tsungyu\\chromosome_data\\cyclegan_data\\fake_zong\\",
                          imgsize=imgsize)

    diff_f_index = get_max_diff_f(encoder,pairset,k=k)
    print(len(dataset))
    print(dataset.class2ndx)
    tsne(encoder=encoder,dataset=dataset,f_index=diff_f_index)
    
    
def evaluate_imgs(net, hparam):
    """chang_label = 0 zong_label = 1"""
    net.eval() #miou 計算不用 eval mode 因為 running mean and running std 誤差可能在訓練過程紀錄的時候過大
    test_imgs = os.listdir(hparam.imgroot)
    total_count = len(test_imgs)
    correct_c = 0
    for p in tqdm(test_imgs):
        img = load_img(os.path.join(hparam.imgroot,p),hparam)#加入批次軸
        img = img.to(dtype=torch.float32)
        truth = hparam.class_ndx
        #print('shape of truth: ',truth.shape)
        with torch.no_grad():
            logit = net(img)
            # print(logit.shape)
            print(torch.softmax(logit.detach(),dim=1))
            y_pred = torch.softmax(logit.detach(),dim=1).argmax(1).item()
            # print(f"y_pred {y_pred} label {truth}")
        if y_pred == truth:
            correct_c += 1
    return correct_c / total_count

def load_img(imgpath,args):
    trans_pipe = transforms.Compose([transforms.Resize((args.img_size,args.img_size)),transforms.ToTensor()])
    img = Image.open(imgpath)
    img_t = trans_pipe(img)
    img_t = img_t[None] # 加入batch軸
    
    return img_t


def _encode_embedding(net, dataset,f_index=None, device="cpu"):
    dev = torch.device(device)
    net.to(dev)
    loader = DataLoader(dataset,batch_size=10,num_workers=4)
    embeddings = []
    labels = []
    with torch.no_grad():
        for data, label in tqdm(loader):
            data = data.to(dev)
            embedding = net(data).detach().cpu()
            G = gram_matrix(embedding)
            embedding = torch.stack([torch.diag(G[i]) for i in range(G.size(0))],dim=0)
            if f_index is not None:
                embedding = torch.gather(embedding, dim=1, index=f_index.repeat(embedding.size(0), 1))
            embeddings.append(embedding)
            labels.append(label)
    embeddings = torch.concat(embeddings, dim=0)
    labels = torch.concat(labels, dim=0)
    embeddings = embeddings.numpy()
    labels = labels.numpy()

    return embeddings, labels
        

def tsne(encoder,dataset,f_index=None, device="cuda:0"):
    embedded, labels = _encode_embedding(encoder,dataset,f_index, device)
    print(embedded.shape, labels.shape)
    embedded = TSNE(n_components=2,perplexity=30).fit_transform(embedded)
    print(embedded.shape, labels.shape)
    df = pd.DataFrame(embedded, columns=['dimension1', 'dimension2'])
    df["y"] = labels
    fig, ax = plt.subplots(figsize=(18, 18))
    sns.scatterplot(data=df, x='dimension1', y='dimension2', hue=df['y'], palette='deep', ax=ax)
    ax.set_title('t-SNE')
    fig.savefig("tsne.png")

    return

def get_max_diff_f(net, pairset, k:int, device="cuda:0" ):
    dev = torch.device(device)
    net.to(dev)
    loader = DataLoader(pairset,batch_size=10,num_workers=4)
    total_diff = torch.zeros(256)
    with torch.no_grad():
        for domain1, domain2 in tqdm(loader):
            domain1 = domain1.to(dev)
            domain2 = domain2.to(dev)
            embe1 = net(domain1).detach().cpu()
            embe2 = net(domain2).detach().cpu()
            G1 = gram_matrix(embe1)
            G2 = gram_matrix(embe2)
            embe1 = torch.stack([torch.diag(G1[i]) for i in range(G1.size(0))],dim=0)
            embe2 = torch.stack([torch.diag(G2[i]) for i in range(G2.size(0))],dim=0)
            diff = torch.sum(embe1 - embe2,dim=0)
            total_diff += diff

    _, indice = torch.topk(total_diff, k)     
    return indice


def load_weight(net, path):
    model_param_dict = net.state_dict()
    weight = torch.load(path)
    pretrain_weight = {k:v for k, v in weight.items() if k in model_param_dict.keys()}
    model_param_dict.update(pretrain_weight)
    net.load_state_dict(model_param_dict)
    return net

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.reshape(a ,b, c * d)  # resize F_XL into \hat F_XL

    G = torch.matmul(features , features.permute(0,2,1))  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G




if __name__ == "__main__":
    main()
    
