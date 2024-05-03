from torch.utils.data import Dataset
import torch
import os
from PIL import Image 
from torchvision import transforms

def get_image_label(root):
    class_name = os.listdir(root)
    class2ndx = {name:idx for idx,name in enumerate(class_name)}
    num_class = len(class_name)
    json_file = {}
    json_file["img_label"] = {}
    json_file["img_path"] = {}
    for dirname, dirs, files in os.walk(root):
        for file in files:
            if not file in json_file["img_label"].keys():
                path = os.path.join(dirname, file)
                ndx = class2ndx[os.path.basename(dirname)]
                json_file["img_label"][path] = ndx
                json_file["img_path"][path] = path #當img_name第一次被讀取時建立該image的路徑，避免圖片被重複讀取
            
    return json_file, class2ndx

class ChromosomeDataset(Dataset):
    def __init__(self, dataset_root, imgsize:int,transforms=None):
        super(ChromosomeDataset,self).__init__()
        self.json_file, self.class2ndx = get_image_label(dataset_root)
        self.ndx2img = {ndx:key for ndx,key in enumerate(self.json_file["img_label"].keys())}
        self.transforms = transforms
        self.imgsize = imgsize
    def __getitem__(self, ndx):
        key = self.ndx2img[ndx]
        imgpath = self.json_file["img_path"][key]
        img = Image.open(imgpath) # convert("L") 轉換為灰階
        if self.transforms is None:
            self.transforms = transforms.Compose([transforms.Resize((self.imgsize,self.imgsize)),
                                          transforms.ToTensor()])
            img_t = self.transforms(img)
        else:
            img_t = self.transforms(img)

        imglabel = torch.tensor(self.json_file["img_label"][key],dtype=torch.float32)

        return img_t, imglabel
    
    def __len__(self):
        return len(self.json_file["img_label"])
    
class PairDataset(Dataset):
    def __init__(self, domain1_dir,domain2_dir, imgsize:int,transforms=None):
        super(PairDataset,self).__init__()
        self.domain1_dir = domain1_dir
        self.domain2_dir = domain2_dir
        self.transforms = transforms
        self.imgsize = imgsize
        self.imgs= os.listdir(domain1_dir)
        
    def __getitem__(self, ndx):
        imgpath1 = os.path.join(self.domain1_dir, self.imgs[ndx])
        imgpath2 = os.path.join(self.domain2_dir, self.imgs[ndx])

        img1 = Image.open(imgpath1) # convert("L") 轉換為灰階
        img2 = Image.open(imgpath2) # convert("L") 轉換為灰階

        if self.transforms is None:
            self.transforms = transforms.Compose([transforms.Resize((self.imgsize,self.imgsize)),
                                          transforms.ToTensor()])
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
        else:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return img1, img2
    
    def __len__(self):
        return len(self.imgs)