import numpy as np

def comp_next(n, k, a, more, h, t ):
    if not more:
        t = n
        h = 0
        if len(a) == 0:
            a = [n]
        else:
            a[0] = n
        a.extend([0]*(k-1))
    else:
        if 1 < t:
            h = 0
        
        h = h + 1
        t = a[h-1]
        a[h-1] = 0
        a[0] = t - 1
        a[h] = a[h] + 1

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

    return order

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

    return int(point_num)

def sparse_grid_herm(dim_num, level_max, point_num):
    grid_point = np.zeros([dim_num, point_num])
    grid_weight = np.zeros([1, point_num])

    point_num2 = 0
    level_min = max(0, level_max + 1 - dim_num)

    for level in range(level_min, level_max+1):
        pass

def dice(u, ground_truth):
    """
    Computes the Sorensen-Dice coefficient of a segmentation relative to a known ground
    truth.
    """ 
    if type(ground_truth) != np.ndarray:
        return 0

    tp = np.sum(u * ground_truth)       # Number of true positives
    fp = np.sum(u * (1-ground_truth))   # Number of false positives
    fn = np.sum((1-u) * ground_truth)   # Number of false negatives

    dice = 2*tp / (2*tp + fp + fn)

    return dice

def PSNR(im, image):
    """
    Computes the Peak Signal-to-Noise Ratio of "im" relative to "image".
    """
    MSE = 1/np.size(im) * np.sum((im - image)**2)
    PSNR = 20*np.log10(np.max(image) / np.sqrt(MSE))

    return PSNR

if __name__ == '__main__':
    a = np.array([[[1,2],[3,4]], [[1,2],[3,4]], [[1,2],[3,4]]])
    b = np.array([[[1,2],[3,4]], [[1,2],[3,4.1]], [[1,2],[3,4]]])

    print(PSNR(a,b))



