import numpy as np

def comp_next(n, k, a, more, h, t ):
    if not more:
        t = n
        h = 0
        if len(a) == 0:
            a = [n]
        else:
            a[0] = n
            a[1:k+1] = 0
    else:
        if 1 < t:
            h = 0
        
        h = h + 1
        t = a[h]
        a[0] = t - 1
        a[h+1] = a[h+1] + 1

    if a[k-1] == n:
        more = False
    else:
        more = True

    return a, more, h, t

def level_to_order_open(dim_num, level):
    order = np.zeros([dim_num, 1])
    for dim in range(0,dim_num):
        if level[dim] < 0:
            order[dim,0] = -1
        elif level[dim] == 0:
            order[dim,0] = 1
        else:
            order[dim,0] = 2**(level[dim]+1) - 1

    return order[0]

def sparse_grid_herm_size(dim_num, level_max):
    if level_max == 0:
        return 1

    point_num = 0
    level_min = max(0, level_max + 1 - dim_num)

    for level in range(level_min, level_max+1):
        level_1d = []
        more = False
        h = 0
        t = 0

        while True:
            level_1d, more, h, t = comp_next(level, dim_num, level_1d, more, h, t)
            order_1d = level_to_order_open(dim_num, level_1d)

            for dim in range(0,dim_num):
                if level_min < level and 1 < order_1d[dim]:
                    order_1d[dim] = order_1d[dim] - 1

            point_num = point_num + np.prod(order_1d[0:dim_num])

            if not more:
                break

    return point_num

if __name__ == '__main__':
    level_1d, more, h, t = comp_next(1, 2, [], False, 0, 0)
  #  order_1d = level_to_order_open(1,[1])
    #print(order_1d)