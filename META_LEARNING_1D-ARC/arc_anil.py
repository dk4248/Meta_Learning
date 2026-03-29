"""ANIL (Almost No Inner Loop) for 1D-ARC - Only adapt the decoder head."""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import json, os, glob, numpy as np, random, argparse
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.func import functional_call

# Disable efficient SDPA for double backward compatibility
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

PAD_VALUE = 10

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class ARC1DDataset(Dataset):
    def __init__(self, data_dir, max_seq_len=100):
        self.max_seq_len=max_seq_len; self.tasks=[]
        for td in glob.glob(os.path.join(data_dir,"1d_*")):
            if os.path.isdir(td): self.tasks.extend(glob.glob(os.path.join(td,"*.json")))
        print(f"Loaded {len(self.tasks)} tasks")
    def __len__(self): return len(self.tasks)
    def pad(self,s,m): return s[:m] if len(s)>=m else s+[PAD_VALUE]*(m-len(s))
    def __getitem__(self,idx):
        with open(self.tasks[idx]) as f: d=json.load(f)
        si,so,sm=[],[],[]
        for ex in d['train']:
            i=ex['input'][0] if isinstance(ex['input'][0],list) else ex['input']
            o=ex['output'][0] if isinstance(ex['output'][0],list) else ex['output']
            if isinstance(i,list) and i and isinstance(i[0],list): i=i[0]
            if isinstance(o,list) and o and isinstance(o[0],list): o=o[0]
            ol=len(o); si.append(self.pad(i,self.max_seq_len)); so.append(self.pad(o,self.max_seq_len))
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

class Enc(nn.Module):
    def __init__(self,vs=11,ed=64,hd=128):
        super().__init__()
        self.emb=nn.Embedding(vs,ed,padding_idx=PAD_VALUE)
        self.c1=nn.Conv1d(ed,hd,3,padding=1); self.c2=nn.Conv1d(hd,hd,3,padding=1); self.c3=nn.Conv1d(hd,hd,3,padding=1)
        self.n1=nn.LayerNorm(hd); self.n2=nn.LayerNorm(hd); self.n3=nn.LayerNorm(hd)
    def forward(self,x):
        x=self.emb(x).transpose(1,2)
        x=F.relu(self.c1(x)).transpose(1,2); x=self.n1(x).transpose(1,2)
        x=F.relu(self.c2(x)).transpose(1,2); x=self.n2(x).transpose(1,2)
        x=F.relu(self.c3(x)).transpose(1,2); return self.n3(x)

class ANILArcModel(nn.Module):
    def __init__(self,hd=128):
        super().__init__()
        # Body (frozen in inner loop)
        self.enc=Enc(hd=hd)
        self.ca=nn.MultiheadAttention(hd,4,batch_first=True)
        # Head (adapted in inner loop)
        self.head=nn.Sequential(nn.Linear(hd*2,hd),nn.ReLU(),nn.Linear(hd,10))
    def forward(self,si,so,qi):
        if qi.dim()==1: qi=qi.unsqueeze(0)
        ie=self.enc(si); oe=self.enc(so); ex=torch.cat([ie,oe],1).mean(0).unsqueeze(0)
        qe=self.enc(qi); att,_=self.ca(qe,ex,ex)
        return self.head(torch.cat([qe,att],-1)).squeeze(0)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',default='1D-ARC/dataset')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--inner_lr',type=float,default=0.01)
    parser.add_argument('--outer_lr',type=float,default=0.001)
    parser.add_argument('--inner_steps',type=int,default=10)
    parser.add_argument('--hidden_dim',type=int,default=128)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--save_dir',default='checkpoints_arc_anil')
    parser.add_argument('--gpu',type=int,default=0,help='GPU id to use')
    args=parser.parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        device=torch.device(f'cuda:{args.gpu}')
    else:
        device=torch.device('cpu')
    ds=ARC1DDataset(args.data_dir)
    idx=list(range(len(ds))); random.shuffle(idx)
    tr=idx[:int(.7*len(idx))]; va=idx[int(.7*len(idx)):int(.85*len(idx))]; te=idx[int(.85*len(idx)):]
    model=ANILArcModel(args.hidden_dim).to(device)
    outer_opt=optim.Adam(model.parameters(),lr=args.outer_lr)
    best_val=0
    os.makedirs(args.save_dir,exist_ok=True)
    # Identify head keys (decoder/head layers)
    head_keys=[k for k in dict(model.named_parameters()).keys() if 'head' in k]
    for ep in range(args.epochs):
        random.shuffle(tr); model.train(); losses=[]
        for i in tqdm(tr,desc=f"Epoch {ep+1}"):
            t=ds[i]; si=t['support_inputs'].to(device); so=t['support_outputs'].to(device)
            sm=t['support_masks'].to(device); qi=t['query_input'].to(device)
            qo=t['query_output'].to(device); qm=t['query_mask'].to(device)
            K=si.size(0)
            # Clone all params for functional_call
            fast_params={k: v.clone() for k, v in dict(model.named_parameters()).items()}
            # Inner loop: only adapt head params
            for _ in range(args.inner_steps):
                for j in range(K):
                    # Leave-one-out context
                    ctx_idx=[k for k in range(K) if k!=j]
                    if len(ctx_idx)>0:
                        ctx_in=si[ctx_idx]; ctx_out=so[ctx_idx]
                    else:
                        ctx_in=si[j:j+1]; ctx_out=torch.zeros_like(so[j:j+1])
                    logits=functional_call(model,fast_params,(ctx_in,ctx_out,si[j]))
                    ce=F.cross_entropy(logits,so[j],reduction='none', ignore_index=PAD_VALUE)
                    loss=(ce*sm[j]).sum()/sm[j].sum()
                    head_params_list=[fast_params[k] for k in head_keys]
                    grads=torch.autograd.grad(loss,head_params_list,create_graph=True)
                    for k,g in zip(head_keys,grads):
                        fast_params[k]=fast_params[k]-args.inner_lr*g
            # Outer: query loss with ALL adapted params
            query_logits=functional_call(model,fast_params,(si,so,qi))
            query_loss=F.cross_entropy(query_logits,qo,reduction='none', ignore_index=PAD_VALUE)
            query_loss=(query_loss*qm).sum()/qm.sum().clamp(min=1)
            outer_opt.zero_grad()
            query_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            outer_opt.step()
            losses.append(query_loss.item())
        print(f"Epoch {ep+1}: Loss={np.mean(losses):.4f}")
        if (ep+1)%5==0:
            model.eval()
            accs=[]
            for i in va:
                t=ds[i]; si=t['support_inputs'].to(device); so=t['support_outputs'].to(device)
                sm=t['support_masks'].to(device); qi=t['query_input'].to(device)
                qo=t['query_output'].to(device); qm=t['query_mask'].to(device)
                K=si.size(0)
                fast_params={k: v.clone() for k, v in dict(model.named_parameters()).items()}
                for _ in range(args.inner_steps):
                    for j in range(K):
                        ctx_idx=[k for k in range(K) if k!=j]
                        if len(ctx_idx)>0:
                            ctx_in=si[ctx_idx]; ctx_out=so[ctx_idx]
                        else:
                            ctx_in=si[j:j+1]; ctx_out=torch.zeros_like(so[j:j+1])
                        logits=functional_call(model,fast_params,(ctx_in,ctx_out,si[j]))
                        ce=F.cross_entropy(logits,so[j],reduction='none', ignore_index=PAD_VALUE)
                        loss=(ce*sm[j]).sum()/sm[j].sum()
                        head_params_list=[fast_params[k] for k in head_keys]
                        grads=torch.autograd.grad(loss,head_params_list)
                        for k,g in zip(head_keys,grads):
                            fast_params[k]=fast_params[k]-args.inner_lr*g
                with torch.no_grad():
                    pred=functional_call(model,fast_params,(si,so,qi)).argmax(-1); m=qm.bool()
                    accs.append((pred[m]==qo[m]).float().mean().item())
            ma=np.mean(accs); print(f"Validation: {ma:.4f}")
            if ma>best_val: best_val=ma; torch.save(model.state_dict(),os.path.join(args.save_dir,'best_model.pt'))
            model.train()
    # Test - reload best model
    best_path=os.path.join(args.save_dir,'best_model.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path,map_location=device))
        print("Loaded best model for test evaluation.")
    model.eval()
    accs=[]
    for i in tqdm(te,desc="Test"):
        t=ds[i]; si=t['support_inputs'].to(device); so=t['support_outputs'].to(device)
        sm=t['support_masks'].to(device); qi=t['query_input'].to(device)
        qo=t['query_output'].to(device); qm=t['query_mask'].to(device)
        K=si.size(0)
        fast_params={k: v.clone() for k, v in dict(model.named_parameters()).items()}
        for _ in range(args.inner_steps):
            for j in range(K):
                ctx_idx=[k for k in range(K) if k!=j]
                if len(ctx_idx)>0:
                    ctx_in=si[ctx_idx]; ctx_out=so[ctx_idx]
                else:
                    ctx_in=si[j:j+1]; ctx_out=torch.zeros_like(so[j:j+1])
                logits=functional_call(model,fast_params,(ctx_in,ctx_out,si[j]))
                ce=F.cross_entropy(logits,so[j],reduction='none', ignore_index=PAD_VALUE)
                loss=(ce*sm[j]).sum()/sm[j].sum()
                head_params_list=[fast_params[k] for k in head_keys]
                grads=torch.autograd.grad(loss,head_params_list)
                for k,g in zip(head_keys,grads):
                    fast_params[k]=fast_params[k]-args.inner_lr*g
        with torch.no_grad():
            pred=functional_call(model,fast_params,(si,so,qi)).argmax(-1); m=qm.bool()
            accs.append((pred[m]==qo[m]).float().mean().item())
    ta=np.mean(accs); print(f"Test: {ta:.4f}")
    json.dump({'best_val_acc':best_val,'test_accuracy':ta,'args':vars(args)},
              open(os.path.join(args.save_dir,'results.json'),'w'),indent=2)
if __name__=='__main__': main()
