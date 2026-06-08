import matplotlib.pyplot as plt

def normalize(l):
    return [(x - min(l)) / (max(l) - min(l)) for x in l]

def plot_losses(style_loss, content_loss, total_loss):
    steps = [i for i in range(0, len(style_loss)*15, 15)]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, normalize(style_loss), label='Style Loss')
    plt.plot(steps, normalize(content_loss), label='Content Loss')
    plt.plot(steps, normalize(total_loss), label='Total Loss')
    plt.legend()
    plt.grid(True)
    plt.show()