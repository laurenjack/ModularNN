def get_best(name):
    f = open(name, 'r')
    min_err = 1
    min_name = None
    count = 0
    for line in f:
        count += 1
        if line.startswith('['):
           current_name = line
           continue
        if (count+1) % 52 == 0:
            err = float(line.rstrip())
            if err < min_err:
                min_err = err
                min_name = current_name.rstrip()
    print min_name+": "+str(min_err)

get_best('../grid_exps/sig-sig-sm')
