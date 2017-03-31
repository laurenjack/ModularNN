import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.cm as cm

colors = cm.rainbow(np.linspace(0, 1, 10))
target_dict = {i: colors[i] for i in xrange(10)}
print "hello"

def generate__m_samples_in_line(m, n, gg, final_latents):
    """Pick a random direction in the latent space to generate m samples from,
    do this n times"""
    for i in xrange(n):
        dir = np.random.randn(gg.d, 1)
        dir = dir / np.linalg.norm(dir)
        step = 4.0/float(m-1)
        c = -2.0
        for j in xrange(m):
            show_sample(c * dir, gg, final_latents)
            c += step

def generate_20_samples_in_line_n_times(gg, final_latents, n):
    for i in xrange(n):
        generate_20_samples_in_line(gg, final_latents)

def generate_20_samples_in_line(gg, final_latents):
    #Create the random direction which samples will be drawn from
    dir = np.random.randn(gg.d, 1)
    dir = dir / np.linalg.norm(dir)
    #Prepare to draw samples evenly along this line
    cs = np.arange(-2.0, 2.0, 4.0/float(30))
    #Show distribution of Z's with the random direction shown as line
    plt.figure(1)
    ax = plt.gca()
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    plot_latent_dist(plt, final_latents)
    xs = [dir[0,0]*c for c in cs]
    ys = [dir[1,0]*c for c in cs]
    plt.plot(xs, ys, color='r')

    #Next figure, show 20 observations along that line
    plt.figure(2)
    sub_num = 1
    for c in cs:
        image = gg.generate_as_image(c * dir)
        sub_plot = plt.subplot(5, 6, sub_num)
        sub_plot.yaxis.set_visible(False)
        sub_plot.xaxis.set_visible(False)
        sub_plot.imshow(image, interpolation='nearest', cmap='Greys')
        sub_num += 1

    plt.show()


def generate_n_samples(n, gg, final_latents):
    for i in xrange(n):
        generate_sample(gg, final_latents)

def generate_sample(gg, final_latents):
    """Generate a sample from a Gaussian Generator, illustrating where that
    sample was drawn from in the image, and the generated image"""
    #Sample a random vector from a d-dimensional isotropic Gaussian
    z = np.random.randn(gg.d, 1)
    show_sample(z, gg, final_latents)

def show_sample(z, gg, final_latents):
    # Use the generator to make generate an X
    X = gg.network.feedforward(z)
    # Put the X in the form of a 2D image
    width = int(float(gg.d_out) ** 0.5 + 0.5)
    image = X.reshape(width, width)

    # Graph the latent variables on a scatter plot
    latent_scatter = plt.subplot(121)
    ax = plt.gca()
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    zs_by_target = get_zs_for_all_targets(final_latents)
    plot_all_scats(zs_by_target, latent_scatter)
    latent_scatter.scatter([z[0, 0]], [z[1, 0]], color='r', marker='x')

    # Next door to the scatter plot, draw the image
    image_sub = plt.subplot(122)
    image_sub.imshow(image, interpolation='nearest', cmap='Greys')

    # Show the graph
    plt.show()

def plot_latent_dist(plot, final_latents):
    zs_by_target = get_zs_for_all_targets(final_latents)
    plot_all_scats(zs_by_target, plot)

def get_zs_for_target(target, current_latents):
    return [z for z, t in current_latents if t == target]


def get_zs_for_all_targets(current_latents):
    return {t: get_zs_for_target(t, current_latents) for t in target_dict}


def plot_all_scats(zs_by_target, sub_plot):
    scat_dict = {}
    for t in target_dict:
        zs = zs_by_target[t]
        xs = [z[0, 0] for z in zs]
        ys = [z[1, 0] for z in zs]
        scat = sub_plot.scatter(xs, ys, color=target_dict[t])
        scat_dict[t] = scat
    return scat_dict