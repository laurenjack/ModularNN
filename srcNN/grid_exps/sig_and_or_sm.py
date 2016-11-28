import grid_search as gd

gd.grid_search_2_logics([784,30,30,30,10], ['sig','and', 'or', 'sm'], [1.0, 0.3, 0.1, 0.03], [0.01, 0.001, 0.0001], [0.01, 0.001, 0.0001])