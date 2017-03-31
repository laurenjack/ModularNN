import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

d = 2
n = 4
directions = 30
super_n = 100000

#Sample a bunch of n Gaussians
#order_dict = {}
magnitudes = np.zeros(super_n)
avg_fst = np.zeros(n)
avg_snd = np.zeros(n)
current = np.zeros(n)
in_rand_directions = [np.zeros(n) for i in xrange(directions)]
for i in xrange(super_n):
    sample = np.random.randn(n, 1)
    sample_mat = np.concatenate([sample] * n, axis=1)
    diffs = (sample_mat - sample_mat.transpose()) ** 2
    magnitudes[i] = np.sum(diffs)
    #magnitudes[i] = np.sum(sample ** 2)# np.linalg.norm(sample)

plt.hist(magnitudes, 200, normed=0, facecolor='green', alpha=0.75)
plt.show()
    #sort the samples by their first and second
#     fst = sorted(samples, key = lambda s: s[0])
#     for j in xrange(n):
#         current[j] = fst[j][0]
#     avg_fst += current
#     snd = sorted(samples, key = lambda s: s[1])
#     for j in xrange(n):
#         current[j] = snd[j][1]
#     avg_snd += current
#     #sort according to random direction
#     for j in xrange(directions):
#         dir = np.random.uniform(-1, 1, 2)
#         dir = dir/np.linalg.norm(dir)
#         for k in xrange(n):
#             current[k] = dir.dot(samples[k])
#         current = sorted(current)
#         in_rand_directions[j] += current
#
# print avg_fst/super_n
# print avg_snd/super_n
# for sum in in_rand_directions:
#     print sum/super_n


# for key in sorted(order_dict.iterkeys()):
#     print "%s: %s" % (key, order_dict[key])
#
# #Find standard dev of counts
# sse = 0.0
# for count in order_dict.itervalues():
#     sse += math.pow(count - 833.33, 2)
# variance = sse/120.0
# print math.pow(variance, 0.5)