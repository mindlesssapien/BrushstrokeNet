import matplotlib.pyplot as plt


def plot_history(history, path=None, title=None):
    """history: list of dicts with keys step, content, style, total (as returned by run_nst)."""
    steps = [h["step"] for h in history]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, key in zip(axs, ["content", "style", "total"]):
        ax.plot(steps, [h[key] for h in history])
        ax.set_yscale("log")
        ax.set_title(f"{key} loss")
        ax.set_xlabel("loss evaluations")
        ax.grid(True)
    if title:
        fig.suptitle(title)
    if path:
        fig.savefig(path, dpi=120)
        plt.close(fig)
    else:
        plt.show()
