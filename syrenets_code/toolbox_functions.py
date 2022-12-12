

def sum_triangular(val):
    if val == 1:
        return 1
    else:
        return sum_triangular(val-1) + val

def get_size( n_inp):
    inp_sum = sum_triangular(n_inp)
    inp_prod = sum_triangular(n_inp)
    inp_sin = n_inp
    inp_cos = n_inp
    # return inp_sum + inp_prod + inp_sin + inp_cos + n_inp + 1
    # return inp_sum + inp_prod + inp_sin + 1
    return inp_sum + inp_prod + inp_sin + inp_cos #+ 1