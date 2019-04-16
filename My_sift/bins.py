import cv2
import numpy as np
import matplotlib.pyplot as plt
from conf import *
from scipy.interpolate import interp1d
from scipy.optimize import minimize,minimize_scalar,basinhopping


def Gauss_size(sigma):
    size = int(6*sigma + 1)
    size = size if size%2==1 else size+1
    return size


def Gauss_blur_(img, sigma, size):
    #P90 关于返回图像深度
    res = cv2.GaussianBlur(img, (size, size), sigma)
    return res


def fun_1(a):
    b = 1 if a>int(a) else 0
    return int(a) + b


def get_sigma(s, octave, sigma0=SIGMA):
    return sigma0*2**(octave + s/S)


def resize(img):
    w, l = img.shape[:2]
    l2 = fun_1(np.log2(l))
    l2 = 2**l2
    w2 = fun_1(np.log2(w))
    w2 = 2**w2
    res = cv2.copyMakeBorder(img, 0, w2-w, 0, l2-l, cv2.BORDER_REPLICATE)
    return res


def resize2(img):
    res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return res


def resize3(img, n):
    res = cv2.resize(img, None, fx=n, fy=n, interpolation=cv2.INTER_LINEAR)
    return res


def to_gray(img):
    res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return res


def get_Gauss_pyramid(img, octave):
    Gauss_pyramid = {}
    n = 0.5
    for o in range(1, octave+1):
        G = img.copy()
        octave_now = [G]
        for s in range(1, S+4):
            sigma = get_sigma(s, o)
            size = Gauss_size(sigma)
            #print('Size:', size)
            G = Gauss_blur_(G, sigma, size)
            octave_now.append(G)
        Gauss_pyramid[str(o)] = [resize3(G, n) for G in octave_now]
        n *= 0.5
    return Gauss_pyramid


def get_DoG_pyramid(Gauss_pyramid, octave):
    DoG_pyramid = {}
    for o in range(1, octave+1):
        imgs = Gauss_pyramid[str(o)]
        octave_now = []
        for i in range(len(imgs) - 1):
            DoG_img = cv2.subtract(imgs[i+1], imgs[i])
            octave_now.append(DoG_img)
        DoG_pyramid[str(o)] = octave_now
    return DoG_pyramid


def Get_octave(img_size):

    res = np.log2(min(img_size)) - 3
    return int(res)


def point_compara(imgs, x, y, value, sym = 'inner', max=True, min=True):
    (r, l) = imgs[1].shape
    for a in [1, 0, -1]:
        for b in [1, 0, -1]:
            if sym == 'inner' and a == b == 0:
                continue
            if x+a < 0 or x+a >= r-1 or y+b < 0 or y+b >= l-1:
                continue
            if imgs[1][x+a, y+b] < value:
                min = False
            if imgs[1][x+a, y+b] > value:
                max = False
    return (max, min)


def find_points_from_DoG(DoGpyramid, octave):
    '''
    :param DoGpyramid:
    :param octave:
    :return:point_set:点集合，字典结构，key为每个DoG的层，里面为列表，每个元素为(r,l,i),第三个为octave内索引
    '''
    point_set = {}
    for o in range(1, octave+1):
        imgs = DoGpyramid[str(o)]
        points_now = []
        for i in range(1, len(imgs[1:-1]) + 1):
            sub_imgs = imgs[i-1:i+1]
            (r, l) = sub_imgs[1].shape
            for r_ in range(1, r):
                for l_ in range(1, l):
                    value = sub_imgs[1][r_, l_]
                    if value < PRE_COLOR_THRES:
                        continue
                    (max, min) = point_compara(sub_imgs, r_, l_, value)
                    if any((min, max)):
                        (max1, min1) = point_compara(sub_imgs, r_, l_, value,'', max, min)
                        (max2, min2) = point_compara(sub_imgs, r_, l_, value, '', max1, min1)
                        if any((max2, min2))and any((max1, min1)): #
                            points_now.append((r_, l_, i))
        point_set[str(o)] = points_now
    return point_set


def cacl_D_x_hat(D, x_hat, T):
    res = D + 0.5*np.dot(T.T, x_hat)
    return res


#Hessian矩阵需要检查
def get_x_hat(imgs, point, iter=1):
    '''
    :param imgs:
    :param point:
    :param iter:
    :return:指定的一个点的插值估计值
    '''
    if iter > ITER:
        return None
    try:
        (r, c, s) = point
        f0 = float(imgs[s][r, c])
        f1 = float(imgs[s][r, c+1])
        f3 = float(imgs[s][r, c+1])
        f2 = float(imgs[s][r+1, c])
        f4 = float(imgs[s][r-1, c])
        f5 = float(imgs[s][r-1, c+1])
        f6 = float(imgs[s][r+1, c+1])
        f7 = float(imgs[s][r+1, c-1])
        f8 = float(imgs[s][r-1, c-1])

        f21 = float(imgs[s+1][r, c])
        f23 = float(imgs[s-1][r, c])
        f25 = float(imgs[s+1][r-1, c])
        f28 = float(imgs[s-1][r-1, c])
        f26 = float(imgs[s+1][r+1, c])
        f27 = float(imgs[s-1][r+1, c])

        f32 = imgs[s-1][r, c]
        f34 = imgs[s+1][r-1, c]
        f35 = float(imgs[s+1][r, c+1])
        f36 = float(imgs[s-1][r, c+1])
        f37 = float(imgs[s-1][r, c-1])
        f38 = float(imgs[s+1][r, c-1])
    except IndexError:
        return None
    H2 = np.mat([
        [f2 + f4 - 2 * f0, (f8 + f6 - f5 - f7) / 4],
        [(f8 + f6 - f5 - f7) / 4, f1 + f3 - 2 * f0]
    ])
    H2_det = np.linalg.det(H2)
    H2_tr = np.trace(H2)
    compare_1 = H2_tr**2/H2_det
    if compare_1 > (R + 1)**2/R:
        return None
    H = np.mat([
        [f2+f4-2*f0, (f8+f6-f5-f7)/4, (f28+f26-f25-f27)/4],
        [(f8+f6-f5-f7)/4, f1+f3-2*f0, (f8+f6-f5-f7)/4],
        [(f28+f26-f25-f27)/4, (f38+f36-f35-f37)/4, f21+f23-2*f0]
    ])
    T = 0.5*np.mat([
        [f2-f4],
        [f1-f3],
        [f21-f23]
    ])
    x_hat = -np.dot(np.linalg.pinv(H), T)
    if max(np.abs(x_hat))<OFFSET_THRES:
        D_hat = cacl_D_x_hat(f0, x_hat, T)
        if D_hat < CONTRAST_THRES:
            return None
        (x, y) = (r + T[0,0], c + T[1,0])
        print('x_hat:', (x, y))
        return (int(x), int(y), s)
    else:
        print('Refitting point, iter:', iter)
        (r_, c_, s_) = (x_hat[0,0], x_hat[1, 0], x_hat[2, 0])
        (x, y, z) = (int(r+round(r_)), int(c+round(c_)), int(s+round(s_)))
        return get_x_hat(imgs, (x, y, z), iter+1)


def get_real_points(DoG_pyramid, points_dir, octave):
    real_points = {}
    for o in range(1, 1+octave):
        points = points_dir[str(o)]
        imgs = DoG_pyramid[str(o)]
        points_now = []
        for point in points:
            x_hat = get_x_hat(imgs, point)
            if x_hat is not None:
                points_now.append(x_hat)
            else:
                print('A point deleted')
        real_points[str(o)] = points_now
    return real_points


def grad_L_xy(imgs, point):
    (r, l, s) = point
    f1 = float(imgs[s][r, l+1])
    f2 = float(imgs[s][r+1, l])
    f3 = float(imgs[s][r, l-1])
    f4 = float(imgs[s][r-1, l])
    grad_L = ((f1 - f3)/2, (f2 - f4)/2)
    return grad_L


def amplitude_theta_xy(imgs, point):
    '''
    用于获取梯度赋值，角度
    :param imgs:
    :param point:
    :return:
    '''
    (r, l, s) = point
    # s += 1#注意这里imd是Gauss_pyramid，不是DoG
    L_1 = float(imgs[s][r, l+1])
    L_2 = float(imgs[s][r+1, l])
    L_3 = float(imgs[s][r, l-1])
    L_4 = float(imgs[s][r-1, l])
    amplitude = np.square((L_1 - L_3)**2 + (L_2 - L_4)**2)
    el = 1e-9
    theta = np.arctan((L_2 - L_4)/(L_1 - L_3 + el))
    if L_1 - L_3 < 0 and theta < 0:
        theta += np.pi
    elif L_1 - L_3 < 0 and theta > 0:
        theta -= np.pi
    return amplitude, theta


def max_x(func, board, accuracy=ACCURACY):
    xs, xe = board
    max_x_ = -19
    max_y_ = -1
    length = (xe - xs)/accuracy
    x = xs
    for _ in range(int(length)):
        x += accuracy
        y = func(x)
        if y > max_y_:
            max_x_ = x
            max_y_ = y
    return max_x_


def get_main_direct(Gauss_pyramid, DoG_pyramid, point_dir, octave):
    '''
    :param Gauss_pyramid:
    :param DoG_pyramid:
    :param point_dir:
    :param octave:
    :return:point_dir: point的结构添加了梯度的赋值，方向描述
    '''
    point_dir_new  = {}
    for o in range(1, 1+octave):
        points = point_dir[str(o)]
        # imgs_DoG = DoG_pyramid[str(o)]
        imgs_Gauss = Gauss_pyramid[str(o)]
        points_new = []
        for point in points:
            #每个兴趣点添加一个主方向描述
            (r, l, s) = point
            sigma = get_sigma(s, o)
            radius = 3*1.5*sigma
            # print('rad:', radius)
            points_direc = {}
            for r_ in range(int(-radius-1), int(+radius+1)):
                for l_ in range(int(-radius-1), int(+radius+1)):
                    if r_**2 + l_**2 > radius**2:continue
                    try:
                        # print('point:',(r+r_, l+l_, s))
                        neibor_point = (r+r_, l+l_, s)
                        #梯度
                        # grad_L = grad_L_xy(imgs_DoG, neibor_point)
                        m, theta = amplitude_theta_xy(imgs_Gauss, neibor_point)
                        #10度一个方向分割
                        index = int(theta*18/np.pi + 0.01)
                        #可以使用collection的有初值的字典类型
                        if index in points_direc.keys():
                            points_direc[index] += m
                        else:
                            points_direc[index] = m
                    except IndexError:
                        continue
            #获取方向直方图，插值寻找极值
            points_grads = np.array(list(points_direc.items()))
            x = points_grads[:, 0].reshape(-1)
            y = points_grads[:, 1].reshape(-1)
            # print('角度：', points_direc.keys())
            # plt.scatter(x, y)
            # plt.show()
            #二次插值
            histor = interp1d(x, y, kind='quadratic', bounds_error=False, fill_value=0)
            # t = np.arange(-37, 37, 0.01)
            # plt.plot(t, histor(t))
            # plt.show()
            #寻找拟合后的函数的最大值，为主方向
            theta = max_x(histor, (-19, 19))
            # reus = basinhopping(lambda x:-1*histor(x), x0=np.array([-18,]))
            # print(reus)
            # print(reus.x)
            # (theta, m) = reus.x, -reus.fun
            #point的结构添加了梯度的方向描述
            point = (r, l, s, theta)
            points_new.append(point)
        point_dir_new[o] = points_new
    return point_dir_new

if __name__ == '__main__':

    cv2.waitKey()