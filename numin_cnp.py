"""Conditional Neural Process (CNP) for Numin2 - Model-based meta-learning, NO inner loop."""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import pandas as pd, numpy as np, random, argparse, json, os
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

class CNPModel(nn.Module):
    def __init__(self, ns=50, hd=256):
        super().__init__()
        self.ns=ns
        # Support pair encoder: encodes (window, ranks) -> hidden
        self.window_enc=nn.LSTM(ns,hd,2,batch_first=True,bidirectional=True)
        self.rank_embed=nn.Embedding(ns,64)
        self.pair_enc=nn.Sequential(nn.Linear(hd*2+64*ns,hd),nn.ReLU(),nn.Linear(hd,hd))
        # Query encoder
        self.query_enc=nn.LSTM(ns,hd,2,batch_first=True,bidirectional=True)
        self.query_proj=nn.Linear(hd*2,hd)
        # Decoder: [query_enc, task_repr] -> prediction
        self.decoder=nn.Sequential(nn.Linear(hd*2,hd),nn.ReLU(),nn.LayerNorm(hd),
                                   nn.Linear(hd,hd),nn.ReLU(),nn.Linear(hd,ns*ns))
    def encode_support(self,samples,targets):
        # samples: (K, W, 50), targets: (K, 50)
        self.window_enc.flatten_parameters()
        K=samples.size(0)
        w_out,_=self.window_enc(samples) # (K,W,hd*2)
        w_pool=w_out.mean(1) # (K,hd*2)
        r_emb=self.rank_embed(targets).view(K,-1) # (K,64*50)
        pair=self.pair_enc(torch.cat([w_pool,r_emb],dim=-1)) # (K,hd)
        return pair.mean(0) # (hd,) - task representation
    def forward(self,sup_samples,sup_targets,query_samples):
        task_repr=self.encode_support(sup_samples,sup_targets) # (hd,)
        self.query_enc.flatten_parameters()
        q_out,_=self.query_enc(query_samples) # (B,W,hd*2)
        q_pool=self.query_proj(q_out.mean(1)) # (B,hd)
        task_repr_exp=task_repr.unsqueeze(0).expand(q_pool.size(0),-1) # (B,hd)
        combined=torch.cat([q_pool,task_repr_exp],dim=-1) # (B,hd*2)
        return self.decoder(combined).view(-1,self.ns,self.ns) # (B,50,50)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_path',default='numin_sample.parquet')
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--hidden_dim',type=int,default=256)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--save_dir',default='checkpoints_numin_cnp')
    parser.add_argument('--gpu',type=int,default=0,help='GPU id to use')
    args=parser.parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available() and args.gpu >= 0:
        device=torch.device(f'cuda:{args.gpu}')
    else:
        device=torch.device('cpu')
    ds=NuminDataset(args.data_path)
    # Temporal split (tasks already sorted chronologically)
    n=len(ds)
    tr_end=int(.7*n); va_end=int(.85*n)
    tr=list(range(tr_end)); va=list(range(tr_end,va_end)); te=list(range(va_end,n))
    model=CNPModel(50,args.hidden_dim).to(device)
    opt=optim.Adam(model.parameters(),lr=args.lr)
    os.makedirs(args.save_dir,exist_ok=True); best_val=-1
    for ep in range(args.epochs):
        random.shuffle(tr); losses=[]
        model.train()
        for i in tqdm(tr,desc=f"Epoch {ep+1}"):
            task=ds[i]; ss=task['support_samples'].to(device); st=task['support_targets'].to(device)
            qs=task['query_samples'].to(device); qt=task['query_targets'].to(device)
            logits=model(ss,st,qs)
            loss=F.cross_entropy(logits.view(-1,model.ns),qt.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); losses.append(loss.item())
        print(f"Epoch {ep+1}: Loss={np.mean(losses):.4f}")
        if (ep+1)%5==0:
            model.eval(); corrs=[]
            for i in va:
                task=ds[i]; ss=task['support_samples'].to(device); st=task['support_targets'].to(device)
                qs=task['query_samples'].to(device); qt=task['query_targets'].to(device)
                with torch.no_grad():
                    pred=model(ss,st,qs).argmax(-1).cpu().numpy(); tgt=qt.cpu().numpy()
                    for j in range(len(pred)):
                        c,_=spearmanr(pred[j],tgt[j])
                        if not np.isnan(c): corrs.append(c)
            mc=np.mean(corrs) if corrs else 0; print(f"Validation: {mc:.4f}")
            if mc>best_val: best_val=mc; torch.save(model.state_dict(),os.path.join(args.save_dir,'best_model.pt'))
    # Reload best model before test
    best_path=os.path.join(args.save_dir,'best_model.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path,map_location=device))
    model.eval(); corrs=[]
    for i in te:
        task=ds[i]; ss=task['support_samples'].to(device); st=task['support_targets'].to(device)
        qs=task['query_samples'].to(device); qt=task['query_targets'].to(device)
        with torch.no_grad():
            pred=model(ss,st,qs).argmax(-1).cpu().numpy(); tgt=qt.cpu().numpy()
            for j in range(len(pred)):
                c,_=spearmanr(pred[j],tgt[j])
                if not np.isnan(c): corrs.append(c)
    tc=np.mean(corrs) if corrs else 0; print(f"Test: {tc:.4f}")
    with open(os.path.join(args.save_dir,'results.json'),'w') as f:
        json.dump({'best_val_corr':best_val,'test_correlation':tc,'args':vars(args)},f,indent=2)
if __name__=='__main__': main()
