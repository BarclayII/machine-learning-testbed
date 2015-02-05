import numpy as NP

def fetch(m, x1, y1, x2, y2):
    """
    Fetch a matrix from (x1, y1) (inclusive) to (x2, y2) (exclusive),
    filling absent elements with zero.

    Both original matrix and new matrix should be square matrices.
    """
    if m.shape[0] != m.shape[1]:
        raise ValueError("Original matrix should be a square matrix")
    size_m = m.shape[0]

    width = x2 - x1
    height = y2 - y1
    if width != height:
        raise ValueError("Result matrix should be a square matrix")

    result = NP.zeros((width, height))
    
    left_m = max(x1, 0)
    top_m = max(y1, 0)
    right_m = min(x2, size_m)
    bottom_m = min(y2, size_m)

    left_r = left_m - x1
    top_r = top_m - y1
    right_r = right_m - x1
    bottom_r = bottom_m - y1

    result[left_r:right_r, top_r:bottom_r] = m[left_m:right_m, top_m:bottom_m]
    return result


def glimpse(env, gw, loc):
    """
    Returns some glimpses centered at loc=(x,y) of a certain environment.

    The glimpses are taken as described in the DeepMind paper.

    Parameters:
        @env:   Environment matrix
        @gw:    Width of the inner-most glimpse.
                The number of glimpses is floor(log2(env.width / gw))
                if @gw is even.  Currently this implementation doesn't
                accept odd @gw.
        @loc:   Center position, which would be automatically rounded to
                something like (6.5, 7.5)

    Returns:
        A list of gw*gw matrices, each represents a glimpse, from
        inner to outer.
    """
    # Check validity of x and y
    if gw % 2 == 1:
        raise ValueError("Odd @gw not supported")

    x, y = loc[0], loc[1]

    x = round(x - 0.5) + 0.5
    y = round(y - 0.5) + 0.5

    size = env.shape[0]
    stride = 1
    res = []
    k = gw

    while k <= size:
        left = (int(round(2 * x)) - k + 1) / 2
        right = (int(round(2 * x)) + k - 1) / 2
        top = (int(round(2 * y)) - k + 1) / 2
        bottom = (int(round(2 * y)) + k - 1) / 2
        try:
            M = fetch(env, left, top, right + 1, bottom + 1)
            R = NP.zeros((gw, gw))
            for i in range(0, gw):
                for j in range(0, gw):
                    R[i, j] = M[i*stride:(i+1)*stride, j*stride:(j+1)*stride].sum() / (stride ** 2)
        except:
            #print 'Fetching %d,%d-%d,%d' % (left, top, right, bottom)
            #print 'center=', (x, y), ', k=%d, stride=%d, gw=%d' % (k, stride, gw)
            R = NP.zeros((gw, gw))
        res.append(R)
        k = k * 2
        stride = stride * 2

    return res
