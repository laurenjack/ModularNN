import srcthesis.network.network_factory as nf
import srcthesis.network.network_runner as nr
import mnist_loader

def grid_search_3_1(sizes, act_strings, eta1_grid_1, eta2_grid_2, ws_grid):
    # Load the data
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    runner = nr.NetworkRunner()

    #File to store results
    f = load_file(act_strings, sizes)

    for eta1 in eta1_grid_1:
        for eta2 in eta2_grid_2:
            for ws in ws_grid:
                hypers = [eta1, (eta2, ws), eta1]
                network = nf.mix_network(sizes, act_strings, hypers)
                val_er = runner.sgd_tracking_error(network, training_data, 10, 20, validation_data)
                to_file(f, hypers, val_er)

def load_file(act_strings, sizes):
    #Prep the name
    name = ''
    for a in act_strings:
        name += a+'-'
    for s in sizes[:-1]:
        name += str(s)+'-'
    name+= str(sizes[-1])

    #Create the file
    return open(name, 'w')

def to_file(f, hypers, val_error):
    f.write(str(hypers)+'\n')
    for err in val_error:
        f.write(str(err)+'\n')
    f.write('\n')
    return f

grid_search_3_1([784, 30, 30, 10], ['sig', 'or', 'sm'], [3.0], [0.1], [1.0])