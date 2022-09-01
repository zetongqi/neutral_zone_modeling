import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from sklearn.datasets import make_blobs
from numpy import linalg as LA
import collections
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm
import plotly.graph_objects as go


def gaussian_kernel(x1, x2, sigma=1):
    """! Gaussian kernel function
    @param x1
    @param x2
    @return val
    """
    return np.exp(-LA.norm(x1-x2, axis=-1)**2 / (2*sigma**2))


def get_gaussian_kernel_matrix(X):
    """! return Gaussian kernel function gram matrix
    @param X
    @return K
    """
    m,n = X.shape
    avg_mean_l2_norm = np.mean(LA.norm(X, axis=1))
    K = np.zeros((m, m))
    for i, x_i in enumerate(X):
        # use mean l2 norm of the input data as scaling factor sigma
        K[i, :] = gaussian_kernel(x_i, X, avg_mean_l2_norm)
    return K


def linear_kernel_func(x1, x2):
    """! linear kernel function
    @param x1
    @param x2
    @return val
    """
    return np.matmul(x1, x2.T)


def get_linear_kernel_mat(X):
    """! return Gaussian kernel function gram matrix
    @param X
    @return K
    """
    m,n = X.shape
    K = np.zeros((m, m))
    for i, x_i in enumerate(X):
        K[i, :] = linear_kernel_func(x_i, X)
    return K


"""! class for fitting a hypersphere of high-dimensional data points.
"""
class hypersphere(object):
    def __init__(self):
        return
    
    def fit(self, X):
        """! fit a hypersphere on high-dimensional data points
        @param X:            [n X d] high-dimensional data points where n is the number of samples, d
                                is the dimension of the data points
        @return X_surface:       points on the surface of the hypersphere
        @return alpha_surface:   weights of the convex combination of the surface points
        @return idx:             indices of the input X that are on the sphere surface
        """
        n, d = X.shape
        PMat = get_gaussian_kernel_matrix(X)
        P = matrix(PMat)
        qMat = -gaussian_kernel(X, X)
        q = matrix(qMat)
        G = matrix(-np.eye(n))
        h = matrix(np.zeros(n))
        A = matrix(np.ones(n), (1, n))
        b = matrix(1.)
        sol = solvers.qp(P,q,G,h,A,b)
        
        self.alpha = np.array(sol['x'])
        self.loss = np.array(sol['primal objective'])
        self.idx = np.where(self.alpha > 1e-4)[0]
        
        X_surface = X[self.idx]
        alpha_surface = self.alpha[self.idx]
        center = np.sum(X_surface * alpha_surface, axis=0)
        dists = []
        for p in X_surface:
            dist = LA.norm(center - p, axis=-1)
            dists.append(dist)
        r = max(dists) 
        self.center = center
        self.r = r
        return X_surface, alpha_surface, self.idx
    
    def get_solution(self):
        """! return the solution
        @return alpha:       weights of the convex combination of all input points
        @return loss:        optimization loss
        """
        return self.alpha, self.loss
    
    def get_center(self):
        """! return the center of the hypersphere
        @return center
        """
        return self.center
    
    def get_radius(self):
        """! return the radius of the hypersphere
        @return radius
        """
        return self.r


def contain(t, A, USE_GUROBI=False):
    """! check if a test point t is contrained by convex hull defined by extreme points A
    @param t:            test point t
    @param A:            data points A defining a convex hull
    @param USE_GUROBI:   if using Gurobi to solve(turn on when a Gurobi license is available)
    @return:             bool indicating if a test point t is in A
    """
    if USE_GUROBI:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        K = get_linear_kernel_mat(A)
        m = gp.Model(env=env)
        m.setParam('NonConvex', 2)
        alpha = m.addMVar(K.shape[0])
        c0 = m.addConstrs((alpha[i] >= 0 for i in range(alpha.shape[0])))
        c1 = m.addConstr(alpha.sum() == 1)
        q = linear_kernel_func(t, A)
        m.setObjective(alpha @ K @ alpha - 2*(alpha @ q) + linear_kernel_func(t, t), GRB.MINIMIZE)
        m.optimize()
        return np.abs(np.array(m.getObjective().getValue())) < 1e-4
    else:
        m, n = A.shape
        K = get_linear_kernel_mat(A)
        q = linear_kernel_func(t, A)
        
        P_mat = matrix(K)
        q_mat = matrix(-q)
        G_mat = matrix(-np.eye(m))
        h_mat = matrix(np.zeros(m))
        A_mat = matrix(np.ones(m), (1, m))
        b_mat = matrix(1.0)
        
        # supress solver log
        solvers.options['show_progress'] = False
        sol = solvers.qp(P_mat, q_mat, G_mat, h_mat, A_mat, b_mat)
        alphas = np.array(sol['x'])
        diff = np.matmul(A.T, alphas).reshape(n) - t
        diff_2norm = np.linalg.norm(diff, 2)
        return diff_2norm < 0.01


def convex_combination(t, A, USE_GUROBI=False):
    """! check if a test point t is contrained by convex hull defined by extreme points A, and
         return it's convex combination
    @param t:            test point t
    @param A:            data points A defining a convex hull
    @param USE_GUROBI:   if using Gurobi to solve(turn on when a Gurobi license is available)
    @return:             bool indicating if a test point t is in A
    @return:             convex combination weights
    """
    if USE_GUROBI:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        K = get_linear_kernel_mat(A)
        m = gp.Model(env=env)
        m.setParam('NonConvex', 2)
        alpha = m.addMVar(K.shape[0])
        c0 = m.addConstrs((alpha[i] >= 0 for i in range(alpha.shape[0])))
        c1 = m.addConstr(alpha.sum() == 1)
        q = linear_kernel_func(t, A)
        m.setObjective(alpha @ K @ alpha - 2*(alpha @ q) + linear_kernel_func(t, t), GRB.MINIMIZE)
        m.optimize()
        return np.abs(np.array(m.getObjective().getValue())) < 0.01, alpha.X
    else:
        m, n = A.shape
        K = get_linear_kernel_mat(A)
        q = linear_kernel_func(t, A)
        
        P_mat = matrix(K)
        q_mat = matrix(-q)
        G_mat = matrix(-np.eye(m))
        h_mat = matrix(np.zeros(m))
        A_mat = matrix(np.ones(m), (1, m))
        b_mat = matrix(1.0)
        
        # supress solver log
        solvers.options['show_progress'] = False
        sol = solvers.qp(P_mat, q_mat, G_mat, h_mat, A_mat, b_mat)
        alphas = np.array(sol['x'])
        diff = np.matmul(A.T, alphas).reshape(n) - t
        diff_2norm = np.linalg.norm(diff, 2)
        return diff_2norm < 0.01, alphas


def entropy(x):
    """! calculating the entropy of a convex combination weight vector
    @param x:            weights that sum up to one and x[i] >= 0
    @return:             entropy
    """
    return np.sum(-x*np.log(x))


def max_entropy(d):
    """! calculating the max entropy an d-dimensional convex combination vector
    @param d:            dimensions
    @return:             max entropy
    """
    return np.log(d)


def entropy_ratio(x):
    """! calculating the entropy ratio(entropy / max entropy)
    @param x:            convex combination weight vector
    @return:             entropy ratio
    """
    return entropy(x) / max_entropy(x.shape[0])


def distance_to_center(y, X_surf, alpha_surf):
    """! calculating the distance between a point and the center of the convex hull
         defined by the extreme points
    @param y:            test point
    @param X_surf:       convex hull surface points
    @param alpha_surf:   convex hull points linear combination weight vector
    @return:             distance
    """
    dist = np.sqrt(gaussian_kernel(y, y) - 2 * np.matmul(alpha_surf, gaussian_kernel(y, X_surf)))
    return dist


def hypersphere_sort(X, x_surf, a_surf):
    """! sort surface points on a hypersphere based on distance to the center of the hypersphere
    @param X:            points to be sorted
    @param X_surf:       convex hull surface points
    @param alpha_surf:   convex hull points linear combination weight vector
    @return:             sorted X
    """
    return sorted(X, key=lambda x: distance_to_center(x, x_surf, a_surf))


def reverse_idx(num_idx, idx):
    """! return the inverse oridinal set
    @param num_idx:      maximum value in the ordinal set
    @param idx:          set to be inversed
    @return:             inverse set
    """
    # num_idx: index in [0, num_idx-1], e.g. [0, 9]
    # idx = [1, 2, 4]
    # retutn: [0, 3, 5, 6, 7, 8, 9]
    return [i for i in range(num_idx) if i not in idx]


def get_extreme_points(X):
    """! finding the extreme points of the convex hull specified by X
    @param X:            convex hull
    @return:             extreme points of convex hull X
    """
    hs = hypersphere()
    x_surf, a_surf, idx = hs.fit(X)
    print('sphere points number', x_surf.shape[0])
    a_surf = a_surf.reshape(a_surf.shape[0])
    v0 = x_surf
    L_star = hypersphere_sort(X[reverse_idx(X.shape[0], idx)], x_surf, a_surf)
    for x in tqdm(np.array(L_star), total=len(L_star)):
        if not contain(x, v0):
            v0 = np.append(v0, x.reshape((1, x.shape[0])), axis=0)
    print('number after expansion', v0.shape[0])
    v = []
    for idx, x in tqdm(enumerate(v0), total=v0.shape[0]):
        v0_x = np.delete(v0, idx, 0)
        if not contain(x, v0_x):
            v.append(x)
    v = np.array(v)
    print('final number', v.shape[0])
    return v


def find_neutral_zone(v0, v1):
    # finding the neutral zone extreme points
    pts = []
    for x in v0:
        if contain(x, v1):
            pts.append(x)
    for x in v1:
        if contain(x, v0):
            pts.append(x)
    pts = np.array(pts)
    print('neutral zone points num', pts.shape[0])
    return pts


def sample_from_cube(v, sample_num=2000):
    # finding the contactness of the data convex hull and the neutral zone
    min0 = list(np.min(v, axis=0))
    max0 = list(np.max(v, axis=0))
    x_uniform = np.random.uniform(min0, max0, size=(sample_num, v.shape[1]))
    return x_uniform

def comvex_combinations_test(v, X):
    entropy_ratios = []
    for x in tqdm(X, total=X.shape[0]):
        contained, alpha = convex_combination(x, v)
        if contained:
            entropy_ratios.append(entropy_ratio(alpha))
    return np.array(entropy_ratios)


def evaluate_neutral_zone(pts, X0, X1, v0, v1, sample_num = 2000):
    # set random seed
    np.random.seed(0)

    # calculating the percentages data in each class taht falls in the neutral zone
    neutral_cnt0 = 0
    cnt0 = 0
    for x in X0:
        if contain(x, pts):
            neutral_cnt0 += 1
        if contain(x, v0):
            cnt0 += 1

    neutral_cnt1 = 0
    cnt1 = 0
    for x in X1:
        if contain(x, pts):
            neutral_cnt1 += 1
        if contain(x, v1):
            cnt1 += 1
    print(((cnt0 + cnt1) / (X0.shape[0] + X1.shape[0]))*100, 'percent of data contained by convex hull')
    print((neutral_cnt0 / X0.shape[0])*100, 'percent of data in class 0 fall in neutral zone')
    print((neutral_cnt1 / X1.shape[0])*100, 'percent of data in class 1 fall in neutral zone')

    # finding the contactness of the data convex hull and the neutral zone
    min0 = list(np.min(pts, axis=0))
    max0 = list(np.max(pts, axis=0))
    x_uniform = np.random.uniform(min0, max0, size=(sample_num, pts.shape[1]))

    cnt_in_hull = 0
    neutral_zone_cnt = 0
    for x in x_uniform:
        test0 = contain(x, v0)
        test1 = contain(x, v1)
        if test0 or test1:
            cnt_in_hull += 1
        if test0 and test1:
            neutral_zone_cnt += 1
    print('convex hull takes up', (cnt_in_hull / x_uniform.shape[0])*100, 'percent of high-dimensional cube')
    print('neutral zone takes up', (neutral_zone_cnt / cnt_in_hull)*100, 'percent of the convex hull')


def sort_data_to_dict(X, Y):
    dic = {}
    for x, y in zip(X, Y):
        if y not in dic:
            dic[y] = [x]
        else:
            dic[y].append(x)

    for key in dic:
        dic[key] = np.array(dic[key])
    dic = collections.OrderedDict(sorted(dic.items()))

    return dic


def get_outpost_info(pts, i, X0, X1):
    pts_hat = np.delete(pts, i, 0)
    idx0 = []
    for idx, x in enumerate(X0):
        if contain(x, pts):
            idx0.append(idx)
    idx1 = []
    for idx, x in enumerate(X1):
        if contain(x, pts):
            idx1.append(idx)

    idx0_hat = []
    for idx, x in enumerate(X0):
        if contain(x, pts_hat):
            idx0_hat.append(idx)
    idx1_hat = []
    for idx, x in enumerate(X1):
        if contain(x, pts_hat):
            idx1_hat.append(idx)

    lost_class0_num = np.sum([1 for idx in idx0 if idx not in idx0_hat])
    lost_class1_num = np.sum([1 for idx in idx1 if idx not in idx1_hat])

    ratio = max(lost_class0_num, lost_class1_num) / (lost_class0_num+lost_class1_num)

    return lost_class0_num, lost_class1_num, ratio


def prune(neutral_zone_pts, X0, X1, min_num_pts = 5, ratio_thresh = 0.9):
    _DEBUG = False
    pts = np.copy(neutral_zone_pts)
    print('prunning neutral zone extreme points...')
    print('neutral zone extreme points before pruning:', pts.shape[0])
    prune_cnt = 0
    while pts.shape[0] > min_num_pts:
        if _DEBUG:
            print('current num of extreme points on neutral zone:', pts.shape[0])
        max_num_eliminated = -1
        max_idx = -1
        for idx in range(pts.shape[0]):
            lost_class0_num, lost_class1_num, ratio = get_outpost_info(pts, idx, X0, X1)
            num_eliminated = max(lost_class0_num, lost_class1_num)
            if ratio > ratio_thresh:
                if num_eliminated > max_num_eliminated:
                    max_num_eliminated = num_eliminated
                    max_idx = idx
            if _DEBUG:
                print(pts[idx])
                print(lost_class0_num, lost_class1_num)
                print(ratio)
                print()
        if max_idx == -1:
            break
        pts = np.delete(pts, max_idx, 0)
        prune_cnt += 1
    print('pruning complete.', prune_cnt, 'points pruned')
    print('new neutral zone extreme points number:', pts.shape[0])
    return pts

# uniform random generation of points inside of convex hull
# by generating convex hull's extreme points' convex combinations
def generate_random_convex_combination(n):
    nums = np.random.uniform(0, 1, n)
    nums = nums / np.sum(nums)
    return nums
