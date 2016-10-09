import random

from srcthesis.visual import visual_domain as vd

#Third party libraries
import numpy as np

class NetworkRunner:

    def sgd_experiment(self, network, hyperParams, experiment):
        train = experiment.train
        batchSize = hyperParams.batchSize
        epochs = hyperParams.epochs
        eta = hyperParams.eta
        test = experiment.test
        node_stats = vd.create_stats(network, experiment)
        return self.__sgd(network, train, batchSize, epochs, eta, node_stats, test)

    def sgd(self, network, training_data, mini_batch_size, epochs, eta, test_data=None):
        self.__sgd(network, training_data, mini_batch_size, epochs, eta, vd.NullStats(), test_data)

    def __sgd(self, network, training_data, mini_batch_size, epochs, eta, node_stats, test_data=None):
        """Train the neural network using mini-batch stochastic
            gradient descent.  The ``training_data`` is a list of tuples
            ``(x, y)`` representing the training inputs and the desired
            outputs.  The other non-optional parameters are
            self-explanatory.  If ``test_data`` is provided then the
            network will be evaluated against the test data after each
            epoch, and partial progress printed out.  This is useful for
            tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        node_stats.update()
        for j in xrange(epochs):
            random.shuffle(training_data)
            # Make a list of lists, i.e. a lsit of distinct training data subsets
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                # run backpropagation on the neural net using the current batch
                # of training data and the learning rate eta
                network.update_mini_batch(mini_batch, eta)
                node_stats.update()
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(network, test_data), n_test)
            else:
                pass
                #print "Epoch {0} complete.".format(j)
        return node_stats

    def sgd_two_eta(self, network, training_data, mini_batch_size, epochs, fst, scnd, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            # Make a list of lists, i.e. a lsit of distinct training data subsets
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                # run backpropagation on the neural net using the current batch
                # of training data and the learning rate eta
                network.update_mini_batch(mini_batch, fst, scnd)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(network, test_data), n_test)
            else:
                print "Epoch {0} complete.".format(j)

    def sgd_tracking_error(self, network, training_data, mini_batch_size, epochs, test_data=None, report_grad_epochs=[]):
        #train_errors = []
        test_errors = []
        if test_data: n_test = len(test_data)
        n = len(training_data)
        #test, no_correct = self.__class_error_test(test_data, n_test, network)
        #test_errors.append(no_correct)
        reported_dws = []
        reported_dbs = []
        for j in xrange(epochs):

            random.shuffle(training_data)
            # Report the gradients for a randomly selected batch
            if j in report_grad_epochs:
                mini = training_data[0:100]
                dw, db = network.get_grads_for_mini_batch(mini)
                reported_dws.append(dw)
                reported_dbs.append(db)

            # if (j+1)%30 == 0:
            #     for act in network.activations:
            #         act.opt().decay_learning_rate(1.0/3.0)
            # Make a list of lists, i.e. a lsit of distinct training data subsets
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                # run backpropagation on the neural net using the current batch
                # of training data and the learning rate eta
                network.update_mini_batch(mini_batch)

            if test_data:
                #tr = self.__class_error_train(training_data, n, network)
                test, no_correct = self.__class_error_test(test_data, n_test, network)
                #train_errors.append(tr)
                test_errors.append(test);
                print "Epoch {0}: {1} / {2}".format(
                    j, no_correct, n_test)
            else:
                print "Epoch {0} complete.".format(j)

        if not report_grad_epochs:
            return test_errors
        else:
            return test_errors, reported_dws, reported_dbs

    def evaluate(self, network, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(network.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def evaluate_train(self, network, test_data):
        test_results = [(np.argmax(network.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    def __random_sub_set(self, train, n):
        random.sample(train, n/5)

    def __class_error_train(self, training_set, n, network):
        fifth_train = random.sample(training_set, n / 5)
        no_correct = self.evaluate_train(network, fifth_train)
        return 1 - no_correct/float(len(fifth_train))

    def __class_error_test(self, test_set, n_test, network):
        no_correct = self.evaluate(network, test_set)
        return 1 - no_correct/float(n_test), no_correct

    # total_cost = 0
    # for x, y in data_set:
    #     a = network.feedforward(x)
    #     cost = network.cost(a, y)
    #     total_cost += cost
    # return total_cost / len(data_set)


