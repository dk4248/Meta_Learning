"""FOMAML (First-Order MAML) for Numin2 - stops gradients before outer update."""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import pandas as pd, numpy as np, random, argparse, json, os
from copy import deepcopy
from tqdm import tqdm
from scipy.stats import spearmanr                         # Fix 12: already at top


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class NuminDataset:
    def __init__(self, path, window_size=50, support_days=5):
        self.window_size = window_size
        self.support_days = support_days
        self.tasks = []
        df = pd.read_parquet(path)
        # Fix 4: tz_convert(None) instead of tz_localize(None)
        if df.index.tz:
            df.index = df.index.tz_convert(None)
        df['year'] = df.index.year
        df['month'] = df.index.month
        for (_, _), g in df.groupby(['year', 'month']):
            g = g.sort_index()
            r = g.drop(['year', 'month'], axis=1).values
            if len(r) < window_size + support_days + 1:
                continue
            S, T = [], []
            for i in range(window_size, len(r)):
                S.append(r[i - window_size:i])
                T.append(np.argsort(np.argsort(-r[i])))
            if len(S) >= support_days + 1:
                self.tasks.append({
                    'samples': np.array(S, dtype=np.float32),
                    'targets': np.array(T, dtype=np.int64),
                })
        # Fix 11: sort tasks chronologically (they were grouped by year/month)
        # groupby already iterates in sorted key order, but we sort explicitly
        # to be safe after any future changes.  Since each task corresponds to
        # a (year, month) group that is already in calendar order coming out of
        # groupby on a sorted index, the list is chronological.  We keep it as-is
        # (no shuffle!) so that the temporal split below is valid.
        print(f"Created {len(self.tasks)} tasks")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        t = self.tasks[idx]
        s = t['samples'].copy()
        tg = t['targets']
        # Fix 2: compute mean/std from support data only (avoid data leakage)
        si = list(range(min(self.support_days, len(s) - 1)))
        qi = list(range(len(si), len(s)))
        support_data = s[si]
        m = support_data.mean()
        st = support_data.std() + 1e-8
        s = (s - m) / st
        return {
            'support_samples': torch.tensor(s[si]),
            'support_targets': torch.tensor(tg[si]),
            'query_samples': torch.tensor(s[qi]),
            'query_targets': torch.tensor(tg[qi]),
        }


class Model(nn.Module):
    def __init__(self, ns=50, hd=256):
        super().__init__()
        self.lstm = nn.LSTM(ns, hd, 2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hd * 2)
        self.dec = nn.Sequential(
            nn.Linear(hd * 2, hd), nn.ReLU(), nn.LayerNorm(hd),
            nn.Linear(hd, ns * ns),
        )
        self.ns = ns

    def forward(self, x):
        # Fix 13: flatten_parameters for LSTM (avoids warning & speeds up on CUDA)
        self.lstm.flatten_parameters()
        o, _ = self.lstm(x)
        o = self.norm(o.mean(1))
        return self.dec(o).view(-1, self.ns, self.ns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='numin_sample.parquet')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--outer_lr', type=float, default=0.001)
    parser.add_argument('--inner_steps', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', default='checkpoints_numin_fomaml')
    # Fix 10: add --gpu argument for CUDA device selection
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index')
    args = parser.parse_args()
    set_seed(args.seed)

    # Fix 10: use --gpu argument
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    ds = NuminDataset(args.data_path)
    # Fix 3: chronological (temporal) split — no shuffling
    indices = list(range(len(ds)))
    train_size = int(0.7 * len(indices))
    val_size = int(0.15 * len(indices))
    tr = indices[:train_size]
    va = indices[train_size:train_size + val_size]
    te = indices[train_size + val_size:]

    ns = 50
    model = Model(ns, args.hidden_dim).to(device)
    outer_opt = optim.Adam(model.parameters(), lr=args.outer_lr)
    os.makedirs(args.save_dir, exist_ok=True)
    best_val = -1
    best_path = os.path.join(args.save_dir, 'best_model.pt')

    for ep in range(args.epochs):
        model.train()                                       # Fix 5 (also here for clarity)
        random.shuffle(tr)
        losses = []
        for i in tqdm(tr, desc=f"Epoch {ep+1}"):
            task = ds[i]
            ss = task['support_samples'].to(device)
            st = task['support_targets'].to(device)
            qs = task['query_samples'].to(device)
            qt = task['query_targets'].to(device)

            # Inner loop with deepcopy (no second order) — FOMAML
            fast = deepcopy(model)
            iopt = optim.SGD(fast.parameters(), lr=args.inner_lr)
            for _ in range(args.inner_steps):
                # Fix 1: compute forward pass once, vectorize the loss
                logits = fast(ss)                           # (B, ns, ns)
                lo = F.cross_entropy(logits.view(-1, ns), st.view(-1))
                iopt.zero_grad()
                lo.backward()
                iopt.step()

            # FOMAML: evaluate on QUERY set, update original model
            outer_opt.zero_grad()
            # Fix 1: vectorized query loss
            q_logits = fast(qs)                             # (B, ns, ns)
            ql = F.cross_entropy(q_logits.view(-1, ns), qt.view(-1))
            ql.backward()

            # Copy gradients from fast model to original model
            with torch.no_grad():
                for (n, p), (n2, fp) in zip(model.named_parameters(),
                                             fast.named_parameters()):
                    if fp.grad is not None:
                        p.grad = fp.grad.clone()

            # Fix 9: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            outer_opt.step()
            losses.append(ql.item())
            del fast

        print(f"Epoch {ep+1}: Loss={np.mean(losses):.4f}")

        if (ep + 1) % 5 == 0:
            corrs = []
            for i in tqdm(va, desc="Val"):
                task = ds[i]
                ss = task['support_samples'].to(device)
                st = task['support_targets'].to(device)
                qs = task['query_samples'].to(device)
                qt = task['query_targets'].to(device)
                # Fix 6: use p.data.clone() for safe weight backup
                ow = {n: p.data.clone() for n, p in model.named_parameters()}
                iopt = optim.SGD(model.parameters(), lr=args.inner_lr)
                model.train()
                for _ in range(args.inner_steps):
                    # Fix 1: vectorized support loss
                    logits = model(ss)
                    lo = F.cross_entropy(logits.view(-1, ns), st.view(-1))
                    iopt.zero_grad()
                    lo.backward()
                    iopt.step()
                model.eval()
                with torch.no_grad():
                    pred = model(qs).argmax(-1).cpu().numpy()
                    tgt = qt.cpu().numpy()
                    for j in range(len(pred)):
                        c, _ = spearmanr(pred[j], tgt[j])
                        if not np.isnan(c):
                            corrs.append(c)
                # Restore original weights
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        p.data = ow[n]
            mc = np.mean(corrs) if corrs else 0
            print(f"Validation: {mc:.4f}")
            if mc > best_val:
                best_val = mc
                torch.save(model.state_dict(), best_path)
            # Fix 5: restore training mode after validation
            model.train()

    # ---- Test ----
    # Fix 7: reload best model before test
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))

    corrs = []
    for i in tqdm(te, desc="Test"):
        task = ds[i]
        ss = task['support_samples'].to(device)
        st = task['support_targets'].to(device)
        qs = task['query_samples'].to(device)
        qt = task['query_targets'].to(device)
        # Fix 6: use p.data.clone()
        ow = {n: p.data.clone() for n, p in model.named_parameters()}
        iopt = optim.SGD(model.parameters(), lr=args.inner_lr)
        model.train()
        for _ in range(args.inner_steps):
            # Fix 1: vectorized support loss
            logits = model(ss)
            lo = F.cross_entropy(logits.view(-1, ns), st.view(-1))
            iopt.zero_grad()
            lo.backward()
            iopt.step()
        model.eval()
        with torch.no_grad():
            pred = model(qs).argmax(-1).cpu().numpy()
            tgt = qt.cpu().numpy()
            for j in range(len(pred)):
                c, _ = spearmanr(pred[j], tgt[j])
                if not np.isnan(c):
                    corrs.append(c)
        # Restore original weights
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data = ow[n]

    tc = np.mean(corrs) if corrs else 0
    print(f"Test: {tc:.4f}")

    # Fix 8: use context manager to avoid file handle leak
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump({
            'best_val_corr': best_val,
            'test_correlation': tc,
            'args': vars(args),
        }, f, indent=2)


if __name__ == '__main__':
    main()
