"""CNP (Conditional Neural Process) for 1D-ARC - Model-based, no inner loop."""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import json, os, glob, numpy as np, random, argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class ARC1DDataset(Dataset):
    def __init__(self, data_dir, max_seq_len=100):
        self.max_seq_len=max_seq_len; self.tasks=[]
        for td in glob.glob(os.path.join(data_dir,"1d_*")):
            if os.path.isdir(td): self.tasks.extend(glob.glob(os.path.join(td,"*.json")))
        print(f"Loaded {len(self.tasks)} tasks")
    def __len__(self): return len(self.tasks)
    def pad(self,seq,ml):
        return (seq[:ml] if len(seq)>=ml else seq+[10]*(ml-len(seq)))  # Fix A: padding_value=10
    def __getitem__(self,idx):
        with open(self.tasks[idx]) as f: d=json.load(f)
        si,so,sm=[],[],[]
        for ex in d['train']:
            inp=ex['input'][0] if isinstance(ex['input'][0],list) else ex['input']
            out=ex['output'][0] if isinstance(ex['output'][0],list) else ex['output']
            if isinstance(inp,list) and inp and isinstance(inp[0],list): inp=inp[0]
            if isinstance(out,list) and out and isinstance(out[0],list): out=out[0]
            ol=len(out)
            si.append(self.pad(inp,self.max_seq_len)); so.append(self.pad(out,self.max_seq_len))
            sm.append([1]*min(ol,self.max_seq_len)+[0]*max(0,self.max_seq_len-ol))
        te=d['test'][0]
        ti=te['input'][0] if isinstance(te['input'][0],list) else te['input']
        to_=te['output'][0] if isinstance(te['output'][0],list) else te['output']
        if isinstance(ti,list) and ti and isinstance(ti[0],list): ti=ti[0]
        if isinstance(to_,list) and to_ and isinstance(to_[0],list): to_=to_[0]
        tol=len(to_)
        return {'support_inputs':torch.tensor(si,dtype=torch.long),'support_outputs':torch.tensor(so,dtype=torch.long),
                'support_masks':torch.tensor(sm,dtype=torch.float),
                'query_input':torch.tensor(self.pad(ti,self.max_seq_len),dtype=torch.long),
                'query_output':torch.tensor(self.pad(to_,self.max_seq_len),dtype=torch.long),
                'query_mask':torch.tensor([1]*min(tol,self.max_seq_len)+[0]*max(0,self.max_seq_len-tol),dtype=torch.float)}

class CNPEncoder(nn.Module):
    def __init__(self,vs=11,ed=64,hd=128):  # Fix A: vocab_size=11
        super().__init__()
        self.emb=nn.Embedding(vs,ed,padding_idx=10)  # Fix A: padding_idx=10
        self.conv1=nn.Conv1d(ed,hd,3,padding=1); self.conv2=nn.Conv1d(hd,hd,3,padding=1)
        self.n1=nn.LayerNorm(hd); self.n2=nn.LayerNorm(hd)
    def forward(self,x):
        x=self.emb(x).transpose(1,2)
        x=F.relu(self.conv1(x)).transpose(1,2); x=self.n1(x).transpose(1,2)
        x=F.relu(self.conv2(x)).transpose(1,2); return self.n2(x)

class ARCCNP(nn.Module):
    def __init__(self,hd=128):
        super().__init__()
        self.enc=CNPEncoder(hd=hd)
        self.pair_proj=nn.Sequential(nn.Linear(hd*2,hd),nn.ReLU(),nn.Linear(hd,hd))
        self.decoder=nn.Sequential(nn.Linear(hd*2,hd),nn.ReLU(),nn.LayerNorm(hd),nn.Linear(hd,10))
    def forward(self,sup_in,sup_out,q_in,support_masks=None,query_mask=None):
        # Encode support pairs and aggregate
        K=sup_in.size(0); task_reprs=[]
        for i in range(K):
            ie=self.enc(sup_in[i:i+1])  # (1,L,hd)
            oe=self.enc(sup_out[i:i+1])  # (1,L,hd)
            # Fix B: apply support mask before mean pooling
            if support_masks is not None:
                m=support_masks[i:i+1].unsqueeze(-1)  # (1,L,1)
                ie=(ie*m).sum(dim=1)/m.sum(dim=1).clamp(min=1)  # (1,hd)
                oe=(oe*m).sum(dim=1)/m.sum(dim=1).clamp(min=1)  # (1,hd)
            else:
                ie=ie.mean(1)  # (1,hd)
                oe=oe.mean(1)  # (1,hd)
            task_reprs.append(self.pair_proj(torch.cat([ie,oe],dim=-1)))
        task_repr=torch.stack(task_reprs).mean(0)  # (1,hd)
        # Encode query
        if q_in.dim()==1: q_in=q_in.unsqueeze(0)
        q_enc=self.enc(q_in)  # (1,L,hd)
        tr_exp=task_repr.unsqueeze(1).expand(-1,q_enc.size(1),-1)  # (1,L,hd)
        combined=torch.cat([q_enc,tr_exp],dim=-1)  # (1,L,hd*2)
        return self.decoder(combined).squeeze(0)  # (L,10)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',default='1D-ARC/dataset')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--hidden_dim',type=int,default=128)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--save_dir',default='checkpoints_arc_cnp')
    parser.add_argument('--gpu',type=int,default=0,help='GPU device id (-1 for CPU)')  # Fix D
    args=parser.parse_args()
    set_seed(args.seed)
    # Fix D: GPU selection
    if args.gpu >= 0 and torch.cuda.is_available():
        device=torch.device(f'cuda:{args.gpu}')
    else:
        device=torch.device('cpu')
    print(f"Using device: {device}")
    ds=ARC1DDataset(args.data_dir)
    idx=list(range(len(ds))); random.shuffle(idx)
    tr=idx[:int(.7*len(idx))]; va=idx[int(.7*len(idx)):int(.85*len(idx))]; te=idx[int(.85*len(idx)):]
    model=ARCCNP(args.hidden_dim).to(device); opt=optim.Adam(model.parameters(),lr=args.lr)
    os.makedirs(args.save_dir,exist_ok=True); best_val=0
    for ep in range(args.epochs):
        random.shuffle(tr); losses=[]; model.train()
        for i in tqdm(tr,desc=f"Epoch {ep+1}"):
            t=ds[i]; si=t['support_inputs'].to(device); so=t['support_outputs'].to(device)
            sm=t['support_masks'].to(device); qi=t['query_input'].to(device)
            qo=t['query_output'].to(device); qm=t['query_mask'].to(device)
            opt.zero_grad()  # Fix: zero_grad before forward/backward
            logits=model(si,so,qi,support_masks=sm,query_mask=qm)
            loss=F.cross_entropy(logits,qo,reduction='none', ignore_index=10)
            loss=(loss*qm).sum()/qm.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)  # Fix E: gradient clipping
            opt.step(); losses.append(loss.item())
        print(f"Epoch {ep+1}: Loss={np.mean(losses):.4f}")
        if (ep+1)%5==0:
            model.eval(); accs=[]
            for i in va:
                t=ds[i]; si=t['support_inputs'].to(device); so=t['support_outputs'].to(device)
                sm=t['support_masks'].to(device)
                qi=t['query_input'].to(device); qo=t['query_output'].to(device); qm=t['query_mask'].to(device)
                with torch.no_grad():
                    pred=model(si,so,qi,support_masks=sm,query_mask=qm).argmax(-1); m=qm.bool()
                    accs.append((pred[m]==qo[m]).float().mean().item())
            ma=np.mean(accs); print(f"Validation: {ma:.4f}")
            if ma>best_val: best_val=ma; torch.save(model.state_dict(),os.path.join(args.save_dir,'best_model.pt'))
    # Fix C: Reload best model before test
    best_model_path=os.path.join(args.save_dir,'best_model.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path,map_location=device))
        print("Loaded best model for testing.")
    model.eval(); accs=[]
    for i in te:
        t=ds[i]; si=t['support_inputs'].to(device); so=t['support_outputs'].to(device)
        sm=t['support_masks'].to(device)
        qi=t['query_input'].to(device); qo=t['query_output'].to(device); qm=t['query_mask'].to(device)
        with torch.no_grad():
            pred=model(si,so,qi,support_masks=sm,query_mask=qm).argmax(-1); m=qm.bool()
            accs.append((pred[m]==qo[m]).float().mean().item())
    ta=np.mean(accs); print(f"Test: {ta:.4f}")
    # Fix F: use with open()
    with open(os.path.join(args.save_dir,'results.json'),'w') as f:
        json.dump({'best_val_acc':best_val,'test_accuracy':ta,'args':vars(args)},f,indent=2)
if __name__=='__main__': main()
