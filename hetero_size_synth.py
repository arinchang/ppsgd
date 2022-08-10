import argparse
import numpy as np
import os
import pickle
from opacus.accountants.rdp import RDPAccountant


class Dataset(object):
    def __init__(self, d=20, N=1000, user_feats=None, local_weight=0.01, seed=42):
        self.d = d
        self.N = N
        self.rd = np.random.RandomState(seed)
        mu0 = 10. * self.rd.randn(d)
        
        if user_feats is None:
            user_feats = (-5, None)
        mask = np.zeros(d)
        mask[user_feats[0]:user_feats[1]] = 1.
        self.theta0 = mu0[None,:]  + local_weight * self.rd.randn(N, d) * mask[None,:]
        self.var = np.array(1. / (1. + np.arange(d)))
        self.sigma = 0.01


    def get_batch(self, b=10):
        X = self.rd.randn(b, self.N, self.d) * np.sqrt(self.var)[None,None,:]
        y = np.sum(X * self.theta0[None,:,:], axis=2) + self.sigma * self.rd.randn(b, self.N)
        return X, y

    def get_mini_batch(self, user, b):
        """
        Generate a mini-batch of data for a user, mini batch sizes b were generated from a poisson distribution. 
        """
        x = self.rd.randn(b, self.d) * np.sqrt(self.var)[None,:] # x is bxd
        y = np.sum(x * self.theta0[None, user], axis=1) + self.sigma * self.rd.randn(b) # y is bx1
        return x, y

    def excess_risk(self, w, thetas):
        return np.mean(self.var[None,:] * ((w[None,:] + thetas) - self.theta0) ** 2)

    def excess_risk_user(self, w, thetas):
        """
        Return array containing excess risk per user. 
        """
        return np.mean(self.var[None,:] * ((w[None,:] + thetas) - self.theta0) ** 2, axis=1) #dim should be (N,)



def training_priv(ds, batch_sizes,
                  step=0.5, stepw=None, steptheta=None,
                  niter=300, C=10., noise_mult=0.05, sample_rate=1., delta=1e-4, num_users=1000):
    w = np.zeros(ds.d)
    thetas = np.zeros((ds.N, ds.d))

    priv = RDPAccountant() 

    if stepw is None:
        stepw = step
    if steptheta is None:
        steptheta = step

    ls = []
    epss = [] 
    for i in range(niter):
        # first just implement case where all clients are sampled every round. TODO implement sampling with some probability
        # experiment with both settings 
        w_grad_clients = []
        for client in range(num_users):
            batch_size = batch_sizes[client]
            x, y = ds.get_mini_batch(client, batch_size)
            pred = np.sum(x * (w[None,:] + thetas[client][None,:]), axis=1) # x is bxd. w[None,:] is 1xd. want pred to be bx1

            theta_grad = np.sum(x * (pred - y)[:,None], axis=0) / batch_size # bxd, bx1 -> theta_grad is dx1

            if steptheta > 0:
                thetas[client] -= steptheta * theta_grad

            theta_grad /= np.clip(np.linalg.norm(theta_grad) / C, a_min=1., a_max=None)
            w_grad_clients.append(theta_grad)

        w_grad = np.mean(w_grad_clients, axis=0)
        if stepw > 0:
            w -= stepw * (w_grad + noise_mult * C * np.random.randn(ds.d) / ds.N)
            priv.step(noise_multiplier=noise_mult, sample_rate=1.)

        risk_at_iter = ds.excess_risk_user(w, thetas)
        ls.append(risk_at_iter)
        eps, best_alpha = priv.get_privacy_spent(delta=delta)
        epss.append(eps)    
    return ls, epss


if __name__ == '__main__':
    parser = argparse.ArgumentParser('private FRL')
    # parser.add_argument('task_id', type=int)
    # parser.add_argument('num_tasks', type=int)
    parser.add_argument('--task_offset', type=int, default=0)
    parser.add_argument('--name', default='hetero')
    parser.add_argument('--d', type=int, default=20, help='dimension')
    parser.add_argument('--N', type=int, default=1000, help='num users')
    # parser.add_argument('--b', type=int, default=10, help='batch size')
    parser.add_argument('--niter', type=int, default=1000)
    parser.add_argument('--local_weight', type=float, default=0.1)
    parser.add_argument('--out_name', default='testing')
    args = parser.parse_args()

    ds = Dataset(d=args.d, N=args.N, local_weight=args.local_weight)

    os.makedirs(os.path.join('res', args.name), exist_ok=True)

    outfile = os.path.join('res', args.name, f'out{args.out_name}.pkl')

    grid = {
        'step': [0.1],
        'alpha': [0, 0.1, 0.3, 1., 3., -1],
        'noise_mult': [0, 1., 10.],
        'C': [10.],
        # 'step': [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1., 1.2, 1.5, 1.8],
        # 'alpha': [0, 0.1, 0.3, 1., 3., 10., 30., 100., -1],
        # 'noise_mult': [0, 0.1, 0.3, 1., 3., 10., 30., 100., 300., 1000.],
        # 'C': [0.1, 1., 10., 100.],
    }
    
    batch_sizes = np.random.poisson(lam=10, size=args.N) # generates mini-batch sizes m_i from poisson distribution

    results = []
    from itertools import product

    for i, vals in enumerate(product(*grid.values())):
        # if i % args.num_tasks != (args.task_id - 1):
        #     continue
        kv = dict(zip(grid.keys(), vals))
        print(kv, flush=True)

        if kv['alpha'] <= 1. and kv['alpha'] >= 0:
            ls, epss = training_priv(ds, batch_sizes, stepw=kv['alpha'] * kv['step'], steptheta=kv['step'], niter=args.niter, C=kv['C'], noise_mult=kv['noise_mult'], num_users=args.N)

        elif kv['alpha'] == -1: # -1 means alpha = infinity, global only
            ls, epss = training_priv(ds, batch_sizes, stepw=kv['step'], steptheta=0, niter=args.niter, C=kv['C'], noise_mult=kv['noise_mult'], num_users=args.N)

        else: # alpha >= 1
            ls, epss = training_priv(ds, batch_sizes, stepw=kv['step'], steptheta=kv['step'] / kv['alpha'], niter=args.niter, C=kv['C'], noise_mult=kv['noise_mult'], num_users=args.N)

        results.append((kv, ls, epss, batch_sizes))

    pickle.dump(results, open(outfile, 'wb'))
