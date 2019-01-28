import numpy as np

def calc_F1(traj_act, traj_rec, noloop=False):
    '''Compute recall, precision and F1 for recommended trajectories'''
    assert (isinstance(noloop, bool))
    assert (len(traj_act) > 0)
    assert (len(traj_rec) > 0)

    if noloop == True:
        intersize = len(set(traj_act) & set(traj_rec))
    else:
        match_tags = np.zeros(len(traj_act), dtype=np.bool)
        for poi in traj_rec:
            for j in range(len(traj_act)):
                if match_tags[j] == False and poi == traj_act[j]:
                    match_tags[j] = True
                    break
        intersize = np.nonzero(match_tags)[0].shape[0]

    recall = intersize * 1.0 / len(traj_act)
    precision = intersize * 1.0 / len(traj_rec)
    Denominator=recall+precision
    if Denominator==0:
        Denominator=1
    F1 = 2 * precision * recall * 1.0 / Denominator
    return F1


# cpdef float calc_pairsF1(y, y_hat):
def  calc_pairsF1(y, y_hat):
    assert (len(y) > 0)
    # assert (len(y) == len(set(y)))  # no loops in y
    # cdef int n, nr, nc, poi1, poi2, i, j
    # cdef double n0, n0r
    n = len(y)
    nr = len(y_hat)
    n0 = n * (n - 1) / 2
    n0r = nr * (nr - 1) / 2

    # y determines the correct visiting order
    order_dict = dict()
    for i in range(n):
        order_dict[y[i]] = i

    nc = 0
    for i in range(nr):
        poi1 = y_hat[i]
        for j in range(i + 1, nr):
            poi2 = y_hat[j]
            if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                if order_dict[poi1] < order_dict[poi2]: nc += 1


    precision = (1.0 * nc) / (1.0 * n0r)
    recall = (1.0 * nc) / (1.0 * n0)
    if nc == 0:
        F1 = 0
    else:
        F1 = 2. * precision * recall / (precision + recall)
    return float(F1)

def  calc_pairsF12(y, y_hat):
    f1=0
    for i in range(len(y)):
         if (y[i]==y_hat[i]):
             f1+=1
    return float(float(f1)/float(len(y))) #float type
