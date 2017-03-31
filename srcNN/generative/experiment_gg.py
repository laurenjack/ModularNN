import data_gg as data_loader
import factory_gg as factory
import train_gg as train
import latent_animator as la
import highlight_sampler as hs

"""Responsible for running experiments that test the capability of
Gaussian Generators"""

#Load the data
xs, targets = data_loader.load_mnist([3,6])
#xs = [0.25 + x/2.0 for x in xs]
# for i in xrange(5):
#     X = random.choice(xs)
#     width = int(math.sqrt(X.shape[0]) + 0.5)
#     image = X.reshape(width, width)
#     plt.imshow(image, interpolation='nearest', cmap='Greys')
#     plt.show()
#xs, targets = data_loader.load_mnist([3, 6])
#Create the Gaussian generator
gg = factory.simple_gg([2, 784, 784, 784], ['relu', 'relu', 'sig'], [0.01, 0.01, 0.1])

#Train the Gaussian Generator, and animate the training
all_latents = train.train(gg, xs, targets, 20, 150, 0.1, 1.0, is_mean=False)
la.animate_2D(all_latents)

#hs.generate_n_samples(3, gg, all_latents[-1])
hs.generate_20_samples_in_line_n_times(gg, all_latents[-1], 10)

#Generate some images!
# for i in xrange(10):
#     gg.generate_image()








