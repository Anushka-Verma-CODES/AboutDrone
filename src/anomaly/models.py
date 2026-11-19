import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ---------------------------------------------------------------- classic

class IForestDetector:
    name = "isolation_forest"

    def __init__(self, seed=0):
        self.m = make_pipeline(StandardScaler(),
                               IsolationForest(n_estimators=400, max_samples=4096,
                                               random_state=seed, n_jobs=-1))

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):            # higher = more anomalous
        return -self.m.score_samples(X)


class OCSVMDetector:
    name = "one_class_svm"

    def __init__(self, nu=0.05, max_train=15000, seed=0):
        self.max_train = max_train
        self.seed = seed
        self.m = make_pipeline(StandardScaler(), OneClassSVM(nu=nu, gamma="scale"))

    def fit(self, X):
        if len(X) > self.max_train:
            idx = np.random.default_rng(self.seed).choice(len(X), self.max_train, replace=False)
            X = X[idx]
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.decision_function(X)


class SupervisedGBM:
    """Not an anomaly detector: uses labels. Kept as a reference for how much
    the unsupervised models leave on the table."""
    name = "supervised_gbm"

    def __init__(self, seed=0):
        self.m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                class_weight="balanced", random_state=seed)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def score(self, X):
        return self.m.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- LSTM AE

class LSTMAE(nn.Module):
    def __init__(self, n_in, hidden=64, latent=16, layers=2):
        super().__init__()
        self.enc = nn.LSTM(n_in, hidden, layers, batch_first=True, dropout=0.1)
        self.to_z = nn.Linear(hidden, latent)
        self.from_z = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=0.1)
        self.out = nn.Linear(hidden, n_in)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.to_z(h[-1])
        rep = self.from_z(z).unsqueeze(1).repeat(1, x.size(1), 1)
        y, _ = self.dec(rep)
        return self.out(y)


class LSTMAEDetector:
    name = "lstm_autoencoder"

    def __init__(self, epochs=40, batch=512, lr=2e-3, device=None, seed=0):
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.history = []

    def _norm(self, S):
        return (S - self.mu) / self.sd

    def fit(self, S, S_val=None, log=print):
        torch.manual_seed(self.seed)
        self.mu = S.reshape(-1, S.shape[-1]).mean(0)
        self.sd = S.reshape(-1, S.shape[-1]).std(0) + 1e-6
        X = torch.tensor(self._norm(S), dtype=torch.float32)
        self.net = LSTMAE(S.shape[-1]).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        best, best_state = np.inf, None
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(X))
            tot = 0.0
            for i in range(0, len(X), self.batch):
                xb = X[perm[i:i + self.batch]].to(self.device)
                loss = ((self.net(xb) - xb) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(xb)
            sched.step()
            tr = tot / len(X)
            va = float(self.score(S_val).mean()) if S_val is not None else tr
            self.history.append((ep + 1, tr, va))
            if va < best:
                best = va
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            if (ep + 1) % 5 == 0 or ep == 0:
                log(f"  epoch {ep + 1:3d}  train {tr:.4f}  val {va:.4f}")
        self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def score(self, S, per_step=False):
        self.net.eval()
        X = torch.tensor(self._norm(S), dtype=torch.float32)
        out = []
        for i in range(0, len(X), 4096):
            xb = X[i:i + 4096].to(self.device)
            err = ((self.net(xb) - xb) ** 2).mean(-1)
            out.append(err.cpu())
        err = torch.cat(out).numpy()
        return err if per_step else err.mean(1)

    def state(self):
        return {"mu": self.mu, "sd": self.sd, "net": self.net.state_dict(),
                "n_in": self.mu.shape[0]}

    @classmethod
    def from_state(cls, st, device=None):
        d = cls(device=device)
        d.mu, d.sd = st["mu"], st["sd"]
        d.net = LSTMAE(st["n_in"]).to(d.device)
        d.net.load_state_dict(st["net"])
        return d
