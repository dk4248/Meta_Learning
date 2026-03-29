"""Multi-Seed Ensemble for Numin2 - Average predictions from 4 Reptile models."""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import pandas as pd, numpy as np, random, json, os, argparse
from tqdm import tqdm
from scipy.stats import spearmanr

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class NuminDataset:
    def __init__(self, path, window_size=50, support_days=5):
        self.window_size=window_size; self.support_days=support_days; self.tasks=[]
        df=pd.read_parquet(path)
        if df.index.tz: df.index=df.index.tz_convert(None)
        df['year']=df.index.year; df['month']=df.index.month
        for (y,m),g in df.groupby(['year','month']):
            g=g.sort_index(); r=g.drop(['year','month'],axis=1).values
            if len(r)<window_size+support_days+1: continue
            S,T=[],[]
            for i in range(window_size,len(r)):
                S.append(r[i-window_size:i]); T.append(np.argsort(np.argsort(-r[i])))
            if len(S)>=support_days+1:
                self.tasks.append({'samples':np.array(S,dtype=np.float32),'targets':np.array(T,dtype=np.int64),'key':(y,m)})
        # Sort tasks chronologically
        self.tasks.sort(key=lambda t: t['key'])
        print(f"Created {len(self.tasks)} tasks")
    def __len__(self): return len(self.tasks)
    def __getitem__(self, idx):
        t=self.tasks[idx]; s=t['samples'].copy(); tg=t['targets']
        si=list(range(min(self.support_days,len(s)-1))); qi=list(range(len(si),len(s)))
        # Support-only normalization
        sup=s[si]; m,st=sup.mean(),sup.std()+1e-8; s=(s-m)/st
        return {'support_samples':torch.tensor(s[si]),'support_targets':torch.tensor(tg[si]),
                'query_samples':torch.tensor(s[qi]),'query_targets':torch.tensor(tg[qi])}

class Model(nn.Module):
    def __init__(self, num_stocks=50, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(num_stocks, hidden_dim, 2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_stocks * num_stocks)
        )
        self.num_stocks = num_stocks
    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        enc = self.norm(out.mean(dim=1))
        return self.decoder(enc).view(-1, self.num_stocks, self.num_stocks)

def adapt_and_predict(model, task, inner_lr=0.01, inner_steps=10, device='cuda'):
    """Adapt model on support set, return softmax predictions on query."""
    ss=task['support_samples'].to(device); st=task['support_targets'].to(device)
    qs=task['query_samples'].to(device)
    ow={n:p.clone() for n,p in model.named_parameters()}
    iopt=optim.SGD(model.parameters(),lr=inner_lr); model.train()
    ns=model.num_stocks
    for _ in range(inner_steps):
        logits=model(ss)
        lo=F.cross_entropy(logits.view(-1,ns),st.view(-1))
        iopt.zero_grad(); lo.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        iopt.step()
    model.eval()
    with torch.no_grad():
        logits=model(qs)  # (B, 50, 50)
        probs=F.softmax(logits, dim=-1)  # softmax over rank classes
    with torch.no_grad():
        for n,p in model.named_parameters(): p.data=ow[n]
    return probs

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_path',default='numin_sample.parquet')
    parser.add_argument('--gpu',type=int,default=0,help='GPU id to use')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--save_dir',type=str,default='checkpoints_numin_ensemble_seeds')
    args=parser.parse_args()

    set_seed(42)
    if torch.cuda.is_available() and args.gpu >= 0:
        device=torch.device(f'cuda:{args.gpu}')
    else:
        device=torch.device('cpu')
    ds=NuminDataset(args.data_path)

    # Temporal split (tasks already sorted chronologically)
    n=len(ds)
    tr_end=int(.7*n); va_end=int(.85*n)
    va=list(range(tr_end,va_end)); te=list(range(va_end,n))

    # Load all available Reptile models
    model_paths = []
    for d in ['checkpoints_numin_reptile', 'checkpoints_numin_aggressive',
              'checkpoints_numin_augmented', 'checkpoints_numin_fomaml']:
        p = os.path.join(d, 'best_model.pt')
        if os.path.exists(p):
            model_paths.append(p)
            print(f"Found: {p}")

    if not model_paths:
        print("No model checkpoints found. Training 3 models with different seeds...")
        from numin_reptile import NuminDataset as ReptileDS, Reptile as ReptileTrainer
        for seed_i, seed in enumerate([42, 123, 456]):
            print(f"\nTraining seed model {seed_i+1}/3 (seed={seed})...")
            set_seed(seed)
            m = Model(50, 256).to(device)
            # Quick Reptile-like training
            inner_opt = torch.optim.SGD(m.parameters(), lr=0.01)
            outer_lr = 0.1
            for epoch in range(30):
                random.shuffle(tr_list := list(range(tr_end)))
                for idx in tr_list:
                    task = ds[idx]
                    ss, st = task['support_samples'].to(device), task['support_targets'].to(device)
                    qs, qt = task['query_samples'].to(device), task['query_targets'].to(device)
                    ow = {n: p.data.clone() for n, p in m.named_parameters()}
                    for _ in range(5):
                        logits = m(ss)
                        B, S, C = logits.shape
                        loss = F.cross_entropy(logits.reshape(B*S, C), st.reshape(B*S))
                        inner_opt.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                        inner_opt.step()
                    with torch.no_grad():
                        for n, p in m.named_parameters():
                            p.data = ow[n] + outer_lr * (1 - epoch/30) * (p.data - ow[n])
            save_p = os.path.join(args.save_dir, f'seed_{seed}.pt')
            torch.save(m.state_dict(), save_p)
            model_paths.append(save_p)
            print(f"  Saved: {save_p}")

    print(f"\nEnsembling {len(model_paths)} models")

    # Load models
    models = []
    for mp in model_paths:
        m = Model(50, 256).to(device)
        try:
            m.load_state_dict(torch.load(mp, map_location=device, weights_only=True))
            models.append(m)
            print(f"  Loaded: {mp}")
        except RuntimeError as e:
            print(f"  Skipping {mp}: incompatible architecture ({e})")

    # Evaluate ensemble on val
    print("\nEvaluating on validation set...")
    val_corrs = []
    for i in tqdm(va, desc="Val"):
        task = ds[i]
        qt = task['query_targets']

        # Get predictions from each model
        all_probs = []
        for m in models:
            probs = adapt_and_predict(m, task, device=device)
            all_probs.append(probs)

        # Average probabilities
        avg_probs = torch.stack(all_probs).mean(0)
        preds = avg_probs.argmax(-1).cpu().numpy()
        tgt = qt.numpy()

        for j in range(len(preds)):
            c, _ = spearmanr(preds[j], tgt[j])
            if not np.isnan(c): val_corrs.append(c)

    val_mc = np.mean(val_corrs) if val_corrs else 0
    print(f"Validation ensemble correlation: {val_mc:.4f}")

    # Evaluate on test
    print("\nEvaluating on test set...")
    test_corrs = []
    for i in tqdm(te, desc="Test"):
        task = ds[i]
        qt = task['query_targets']

        all_probs = []
        for m in models:
            probs = adapt_and_predict(m, task, device=device)
            all_probs.append(probs)

        avg_probs = torch.stack(all_probs).mean(0)
        preds = avg_probs.argmax(-1).cpu().numpy()
        tgt = qt.numpy()

        for j in range(len(preds)):
            c, _ = spearmanr(preds[j], tgt[j])
            if not np.isnan(c): test_corrs.append(c)

    test_mc = np.mean(test_corrs) if test_corrs else 0
    print(f"Test ensemble correlation: {test_mc:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump({'best_val_corr': val_mc, 'test_correlation': test_mc,
                   'num_models': len(models), 'model_paths': model_paths}, f, indent=2)
    print(f"\nDone! Val={val_mc:.4f}, Test={test_mc:.4f}")

if __name__=='__main__': main()
