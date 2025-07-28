import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

# without the use of "convert("RGB") the image was being converted to a tensor with 4 channels 'RGBA' a for alpha layer"
imsize = 512 if torch.cuda.is_available() else 128
loader = transforms.Compose([transforms.Lambda(lambda img : img.convert("RGB")),
                             transforms.Resize(imsize),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
unloader = transforms.ToPILImage()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.select_features = ['0','5','10','19','28'] # selecting the conv layers
        self.vgg = models.vgg19(pretrained=True).features #loading a pretrained vgg19 and accessing it's features layer i.e conv(feature extraction) layer excluding classifer(fully connected layer)
    
    def forward(self,output):
        features = []
        for name , layer in self.vgg._modules.items():
            output = layer(output)
            if name in self.select_features:
                features.append(output)
        return features


def image_loader(image_path):
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device,torch.float)

def img_show(tensor,title=None):
    image = tensor.cpu().clone()
    denormalization = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    image = image.squeeze(0)
    image = denormalization(image).clamp(0, 1)
    image = unloader(image)
    plt.imshow(image)
    if title is not None :
        plt.title(title)

def get_content_loss(content,target):
    return torch.mean((content-target)**2)/2

def gram_matrix(input):
    _,c,h,w = input.size()
    input = input.view(c,h*w)
    G = torch.mm(input,input.t())
    return G

def get_style_loss(style,target):
    _,c,h,w = target.size()
    Gt = gram_matrix(target)
    Gs = gram_matrix(style)
    return torch.mean((Gt-Gs)**2)/((c*h*w))

def Adam_optimizer(x,y):
    return optim.Adam([x],lr=y)

def lbfgs_optimizer(x):
   return optim.LBFGS([x])

def save(target, i):
    denormalization = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    img = target.clone().squeeze()
    img = denormalization(img).clamp(0, 1)
    save_image(img, f'result_{i}.png')

def train(style_img,content_img):
    vgg = VGG().to(device).eval()

    style_image = image_loader(style_img)
    content_image = image_loader(content_img)
    #assert style_image.size() == content_image.size(),"content and style img need to of the the same size"

    #model = models.vgg19(pretrained = True).features
    target_image = content_image.clone().requires_grad_(True)
    #optimizer_1 = Adam_optimizer([target_image],0.001)

    optimizer = lbfgs_optimizer(target_image)
    t_loss, s_loss, c_loss = [], [], []
    steps = 300
    beta = 1000000 
    alpha = 1

    for step in range(steps):
        def closure():

            # below this till backward() will work for adam outside of the function no need of closure in adam
            #setting parameters to zero
            optimizer.zero_grad()

            #obtaining the feature vector representation for every image
            target_feature = vgg(target_image)
            style_feature = vgg(style_image)
            content_feature = vgg(content_image)
            
            style_loss = 0
            content_loss = 0

            for target,content,style in zip(target_feature,content_feature,style_feature):
                content_loss += get_content_loss(content,target)
                style_loss += get_style_loss(style,target)

            total_loss = alpha*content_loss + beta*style_loss
            #compute the gradient
            total_loss.backward()

            return total_loss

        #updating the parameters
        optimizer.step(closure)

        if step % 15 == 0:
            with torch.no_grad():
                target_feature = vgg(target_image)
                style_feature = vgg(style_image)
                content_feature = vgg(content_image)

            style_loss = 0
            content_loss = 0

            for target, content, style in zip(target_feature, content_feature, style_feature):
                content_loss += get_content_loss(content, target)
                style_loss += get_style_loss(style, target)

            total_loss = alpha * content_loss + beta * style_loss
            print(f'step: {step}, content loss: {content_loss.item():.4f}, style loss: {style_loss.item():.4f}')
            c_loss.append(content_loss.item())
            s_loss.append(style_loss.item())
            t_loss.append(total_loss.item())
        if step%30 == 0:
            save(target_image, step)
        
    return target_image

#img_show(target_image,title="output image")

def normalize(l):
    return [(x - min(l)) / (max(l) - min(l)) for x in l]

def plot_loss_graphs(s_loss,c_loss,t_loss):
    s = [i for i in range(0,300,15)]

    fig , axs = plt.subplots(2,2,layout='constrained',figsize=(10,8))
    cbl = axs[0][0]
    cbl.plot(s, normalize(s_loss), label='Style Loss (Normalized)')
    cbl.plot(s, normalize(c_loss), label='Content Loss (Normalized)')
    cbl.plot(s, normalize(t_loss), label='Total Loss (Normalized)')
    cbl.grid(True)
    cbl.legend()

    sl = axs[0][1]
    sl.plot(s,s_loss,label='style_loss')
    sl.grid(True)
    sl.legend()

    cl = axs[1][0]
    cl.plot(s,c_loss,label='content loss')
    cl.grid(True)
    cl.legend()

    tl = axs[1][1]
    tl.plot(s,t_loss,label='total loss')
    tl.grid(True)
    tl.legend()

    tl.set_xlabel("Iterations")
    cbl.set_ylabel("Loss")
    cl.set_xlabel("Iterations")
    cl.set_ylabel("Loss")

    plt.show()
