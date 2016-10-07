def get_best(name):
    f = open(name, 'r')
    min_err = 1
    min_name = None
    count = 1
    for line in f:
        if line.startswith('['):
           current_name = line
           continue
        if count % 52 == 0:
            err = float(line.rstrip())
            if err < min_err:
                min_err = err
                min_name = current_name.rstrip()
        count+=1
    print min_name+": "+str(min_err)
