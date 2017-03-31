import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

targrt_dict = {1: 'b', 2:'r', 3: 'g', 4:'c', 5:'m', 6:'y', 7:'k', 8:'w'}


def get_zs_for_target(target, current_latents):
    return [z for z, t in current_latents if t == target]


def get_zs_for_all_targets(current_latents):
    return {t: get_zs_for_target(t, current_latents) for t in targrt_dict}


def plot_all_scats(zs_by_target):
    scat_dict = {}
    for t in targrt_dict:
        zs = zs_by_target[t]
        xs = [z[0, 0] for z in zs]
        ys = [z[1, 0] for z in zs]
        scat = plt.scatter(xs, ys, color=targrt_dict[t])
        scat_dict[t] = scat
    return scat_dict

def animate_2D(all_latents):
    """Animate the path of the latent variables as they move
    through a 2D z-space

    :param all_latents - A list, of lists of latent variables
    The outer index corresponds to the number of epochs, while
    the inner index corresponds to n, the number of data points.
    This function assumes that the latent space is only 2D."""


    fig, ax = plt.subplots()

    epochs = len(all_latents)
    # ax.set_xlim(-3.0, 3.0)
    # ax.set_ylim(0.0, 0.5)
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    # ax.set_xlim(-20.0, 20.0)
    # ax.set_ylim(-20.0, 20.0)
    current_latents = all_latents[0]
    # ones = [z for z, t in current_latents if t == 5]
    # threes = [z for z, t in current_latents if t == 9]
    # one0s = [z[0, 0] for z in ones]
    # one1s = [z[1, 0] for z in ones]
    # three0s = [z[0, 0] for z in threes]
    # three1s = [z[1, 0] for z in threes]
    # scat1 = plt.scatter(one0s, one1s, color='r')
    # scat3 = plt.scatter(three0s, three1s, color='b')
    zs_by_target = get_zs_for_all_targets(current_latents)
    scat_dict = plot_all_scats(zs_by_target)
    scats = scat_dict.values()

    def init():
        return scats

    def update(frame):
        """Call back used by the animation to acess the
        appropriate latent at the appropriate time. Each frame
        corresponds to an epoch of training."""
        current_latents = all_latents[frame+1]
        # ones = [z for z, t in current_latents if t == 5]
        # threes = [z for z, t in current_latents if t == 9]
        # scat1.set_offsets(ones)
        # scat3.set_offsets(threes)
        zs_by_target = get_zs_for_all_targets(current_latents)
        for t in targrt_dict:
            scat_dict[t].set_offsets(zs_by_target[t])
        return scats

    ani = FuncAnimation(fig, update, frames=epochs-1, init_func=init, interval=500, repeat=False)

    plt.show()

# def gen_latents():
#     return [np.random.randn(2, 1) for i in xrange(20)]
#
# all_latents = [gen_latents() for i in xrange(30)]
# animate_2D(all_latents)


