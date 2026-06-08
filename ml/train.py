from ml.models.vgg import VGG
from ml.utils.image_utils import image_loader, img_show
from ml.utils.loss_utils import get_content_loss, get_style_loss
from ml.utils.optimizer import adam_optimizer, lbfgs_optimizer, save
from ml.utils.plot_utils import plot_losses
import ml.config as config
import torch
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True # said that it makes traingin on gpu faster ?

def train_nst(content_path, style_path, output_path):
    content = image_loader(content_path)
    style = image_loader(style_path)

    target = content.clone().requires_grad_(True).to(device)

    model = VGG().to(device).eval()

    content_features = model(content)
    style_features = model(style)

    optimizer = lbfgs_optimizer(target)
    # optimizer = adam_optimizer(target, lr=0.003)

    style_loss_list = []
    content_loss_list = []
    total_loss_list = []
    step = [0]
    num_steps = config.STEPS

    print("starting neural style transfer")
    while step[0] < num_steps:

        def closure():
            optimizer.zero_grad()

            target_features = model(target)

            style_loss = 0
            content_loss = 0

            for t, c, s in zip(target_features, content_features, style_features):
                content_loss += get_content_loss(t, c)
                style_loss += get_style_loss(t, s)

            total_loss = config.CONTENT_WEIGHT * content_loss + config.STYLE_WEIGHT * style_loss
            total_loss.backward()

            step[0] += 1

            style_loss_list.append(style_loss.item())
            content_loss_list.append(content_loss.item())
            total_loss_list.append(total_loss.item())

            if step[0] % 50 == 0:
                print(f"Step {step[0]}:")
                print(f"Style Loss : {style_loss.item():.4f}")
                print(f"Content Loss: {content_loss.item():.4f}")
                print(f"Total Loss  : {total_loss.item():.4f}")
                save(target, step[0])

            return total_loss

        optimizer.step(closure)

    print("optimization finished!")

    img_show(target, title="Final Output")

    return output_path

#plot_losses(style_loss_list, content_loss_list, total_loss_list)
