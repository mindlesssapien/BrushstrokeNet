import torch 

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