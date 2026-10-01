from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class imagefolderdataset(Dataset):
    def __init__(self,root, transform=None):
        self.root=root
        self.transform=transform
        self.files= list(os.listdir(root))
        self.files=[p for p in self.files if p.endswith('.jpg') or p.endswith('.png') or p.endswith('.jpeg')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path=os.path.join(self.root,self.files[idx])
        img=Image.open(img_path).convert('RGB')
        if self.transform:
            img=self.transform(img)
        return img

def get_transformer(final_size,size, crop):
    transform_list=[]
    if size>0:
        transform_list.append(transforms.Resize((size,size)))

    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize((final_size,final_size)))


    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def adaptive_instent_normalization(c_feats,s_feats):
    size=c_feats.size()
    style_mean,style_std=calc_mean_std(s_feats)
    content_mean,content_std=calc_mean_std(c_feats)
    normalize_content_feat=(c_feats-content_mean.expand(size))/content_std.expand(size)
    return normalize_content_feat*style_std.expand(size)+style_mean.expand(size)



def calc_mean_std(feat,eps=1e-5):
    size=feat.size()
    assert(len(size)==4)
    batch_size,channel= size[:2]
    feat_mean=feat.view(batch_size,channel,-1).mean(dim=2).view(batch_size,channel,1,1)
    feat_var=feat.view(batch_size,channel,-1).var(dim=2, unbiased=False)+eps
    feat_std=feat_var.sqrt().view(batch_size,channel,1,1)
    return feat_mean,feat_std

