"""Pure-numpy models: histogram GBM (L2 + pinball), random forest, ridge, kNN."""
import numpy as np


# ---------------- histogram gradient boosting ----------------
class HistGBM:
    def __init__(self, n_trees=150, lr=0.08, depth=3, min_leaf=25, n_bins=32,
                 loss='l2', q=0.5, subsample=0.9, colsample=0.9, seed=0):
        self.p = dict(n_trees=n_trees, lr=lr, depth=depth, min_leaf=min_leaf,
                      n_bins=n_bins, loss=loss, q=q, subsample=subsample,
                      colsample=colsample)
        self.rng = np.random.default_rng(seed)

    def _bin(self, X, fit):
        if fit:
            self.edges = [np.unique(np.nanquantile(X[:, j], np.linspace(0, 1, self.p['n_bins'] + 1))[1:-1])
                          for j in range(X.shape[1])]
        B = np.empty(X.shape, np.int8)
        for j in range(X.shape[1]):
            B[:, j] = np.searchsorted(self.edges[j], X[:, j], side='right')
        B[np.isnan(X)] = 0
        return B

    def _leaf_value(self, resid):
        if self.p['loss'] == 'l2':
            return resid.mean()
        return np.quantile(resid, self.p['q'])

    def _grow(self, B, resid, rows, feats):
        # returns list of (feat, cut, left_child, right_child) nodes + leaf values
        tree = []
        stack = [(rows, 0)]
        nodes = {}
        node_id = 0
        out_nodes = {}
        stack = [(0, rows, 0)]
        next_id = 1
        while stack:
            nid, r, d = stack.pop()
            res = resid[r]
            if d >= self.p['depth'] or len(r) < 2 * self.p['min_leaf']:
                out_nodes[nid] = ('leaf', self._leaf_value(res))
                continue
            best = None
            tot_sum, tot_cnt = res.sum(), len(r)
            for j in feats:
                b = B[r, j]
                s = np.bincount(b, weights=res, minlength=self.p['n_bins'] + 1)
                c = np.bincount(b, minlength=self.p['n_bins'] + 1)
                cs, cc = np.cumsum(s), np.cumsum(c)
                nl = cc[:-1]
                nr = tot_cnt - nl
                ok = (nl >= self.p['min_leaf']) & (nr >= self.p['min_leaf'])
                if not ok.any():
                    continue
                sl = cs[:-1]
                gain = np.where(ok, sl ** 2 / np.maximum(nl, 1) +
                                (tot_sum - sl) ** 2 / np.maximum(nr, 1), -np.inf)
                k = int(np.argmax(gain))
                if best is None or gain[k] > best[0]:
                    best = (gain[k], j, k)
            base = tot_sum ** 2 / tot_cnt
            if best is None or best[0] <= base + 1e-12:
                out_nodes[nid] = ('leaf', self._leaf_value(res))
                continue
            _, j, cut = best
            mask = B[r, j] <= cut
            lid, rid = next_id, next_id + 1
            next_id += 2
            out_nodes[nid] = ('split', j, cut, lid, rid)
            stack.append((lid, r[mask], d + 1))
            stack.append((rid, r[~mask], d + 1))
        return out_nodes

    def _predict_tree(self, nodes, B):
        out = np.empty(B.shape[0])
        idx = np.arange(B.shape[0])
        stack = [(0, idx)]
        while stack:
            nid, r = stack.pop()
            node = nodes[nid]
            if node[0] == 'leaf':
                out[r] = node[1]
                continue
            _, j, cut, lid, rid = node
            m = B[r, j] <= cut
            stack.append((lid, r[m]))
            stack.append((rid, r[~m]))
        return out

    def fit(self, X, y, sample_weight=None):
        B = self._bin(X, fit=True)
        n, m = X.shape
        self.f0 = np.mean(y) if self.p['loss'] == 'l2' else np.quantile(y, self.p['q'])
        F = np.full(n, self.f0)
        self.trees = []
        nf = max(3, int(self.p['colsample'] * m))
        ns = max(50, int(self.p['subsample'] * n))
        pw = None
        if sample_weight is not None:
            pw = np.asarray(sample_weight, float)
            pw = pw / pw.sum()
        for t in range(self.p['n_trees']):
            if self.p['loss'] == 'l2':
                g = y - F
            else:
                g = np.where(y > F, self.p['q'], self.p['q'] - 1.0)
            rows = self.rng.choice(n, ns, replace=pw is not None, p=pw)
            feats = self.rng.choice(m, nf, replace=False)
            nodes = self._grow(B, (y - F) if self.p['loss'] == 'l2' else g, rows, feats)
            # for pinball: recompute leaf values as quantile of residual y-F within leaf
            if self.p['loss'] != 'l2':
                assign = self._assign(nodes, B)
                for nid, node in nodes.items():
                    if node[0] == 'leaf':
                        r = np.flatnonzero(assign == nid)
                        if r.size:
                            nodes[nid] = ('leaf', np.quantile((y - F)[r], self.p['q']))
            step = self._predict_tree(nodes, B)
            F += self.p['lr'] * step
            self.trees.append(nodes)
        return self

    def _assign(self, nodes, B):
        out = np.empty(B.shape[0], int)
        stack = [(0, np.arange(B.shape[0]))]
        while stack:
            nid, r = stack.pop()
            node = nodes[nid]
            if node[0] == 'leaf':
                out[r] = nid
                continue
            _, j, cut, lid, rid = node
            m = B[r, j] <= cut
            stack.append((lid, r[m]))
            stack.append((rid, r[~m]))
        return out

    def predict(self, X):
        B = self._bin(X, fit=False)
        F = np.full(X.shape[0], self.f0)
        for nodes in self.trees:
            F += self.p['lr'] * self._predict_tree(nodes, B)
        return F

    def feature_importance(self, m):
        imp = np.zeros(m)
        for nodes in self.trees:
            for node in nodes.values():
                if node[0] == 'split':
                    imp[node[1]] += 1
        return imp / max(imp.sum(), 1)


class RandomForest:
    def __init__(self, n_trees=120, depth=7, min_leaf=15, colsample=0.5, seed=0):
        self.gbm_kw = dict(depth=depth, min_leaf=min_leaf, n_bins=32)
        self.n_trees = n_trees
        self.colsample = colsample
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        self.base = HistGBM(n_trees=1, **self.gbm_kw)
        self.B_edges = None
        b = HistGBM(**self.gbm_kw)
        self.binner = b
        Bfull = b._bin(X, fit=True)
        n, m = X.shape
        self.trees = []
        for t in range(self.n_trees):
            rows = self.rng.choice(n, n, replace=True)
            feats = self.rng.choice(m, max(3, int(self.colsample * m)), replace=False)
            b.p['loss'] = 'l2'
            nodes = b._grow(Bfull, y - 0.0, rows, feats)
            self.trees.append(nodes)
        return self

    def predict_all(self, X):
        B = self.binner._bin(X, fit=False)
        return np.stack([self.binner._predict_tree(nodes, B) for nodes in self.trees])

    def predict(self, X):
        return self.predict_all(X).mean(0)


class Ridge:
    def __init__(self, alpha=10.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.mu = np.nanmean(X, 0)
        self.sd = np.nanstd(X, 0) + 1e-9
        Xs = (np.where(np.isnan(X), self.mu, X) - self.mu) / self.sd
        A = np.c_[Xs, np.ones(len(Xs))]
        I = np.eye(A.shape[1])
        I[-1, -1] = 0
        self.w = np.linalg.solve(A.T @ A + self.alpha * I, A.T @ y)
        return self

    def predict(self, X):
        Xs = (np.where(np.isnan(X), self.mu, X) - self.mu) / self.sd
        return np.c_[Xs, np.ones(len(Xs))] @ self.w


class KNN:
    def __init__(self, k=15):
        self.k = k

    def fit(self, X, y):
        self.mu = np.nanmean(X, 0)
        self.sd = np.nanstd(X, 0) + 1e-9
        self.X = (np.where(np.isnan(X), self.mu, X) - self.mu) / self.sd
        self.y = y
        return self

    def predict(self, X):
        Xs = (np.where(np.isnan(X), self.mu, X) - self.mu) / self.sd
        out = np.empty(len(Xs))
        for i in range(len(Xs)):
            d = ((self.X - Xs[i]) ** 2).sum(1)
            out[i] = self.y[np.argpartition(d, self.k)[:self.k]].mean()
        return out


# ---------------- metrics ----------------
def mae(y, p):
    return np.mean(np.abs(y - p))


def rmse(y, p):
    return np.sqrt(np.mean((y - p) ** 2))


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def pinball(y, p, q):
    d = y - p
    return np.mean(np.maximum(q * d, (q - 1) * d))


def crps_from_quantiles(y, Q, qs):
    """approx CRPS = 2 * mean_k pinball(y, Q_k, q_k)"""
    return 2 * np.mean([pinball(y, Q[k], qs[k]) for k in range(len(qs))])


def coverage(y, lo, hi):
    return np.mean((y >= lo) & (y <= hi))
