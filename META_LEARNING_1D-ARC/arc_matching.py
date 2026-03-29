"""Matching Networks for 1D-ARC (Lecture 6) - per-position attention-weighted nearest neighbor."""
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
    def pad(self,s,m): return s[:m] if len(s)>=m else s+[10]*(m-len(s))  # Fix A: padding_value=10
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
    def __init__(self,vs=11,ed=64,hd=128):  # Fix A: vocab_size=11
        super().__init__()
        self.emb=nn.Embedding(vs,ed,padding_idx=10)  # Fix A: padding_idx=10
        self.c1=nn.Conv1d(ed,hd,3,padding=1); self.c2=nn.Conv1d(hd,hd,3,padding=1); self.c3=nn.Conv1d(hd,hd,3,padding=1)
        self.n1=nn.LayerNorm(hd); self.n2=nn.LayerNorm(hd); self.n3=nn.LayerNorm(hd)
    def forward(self,x):
        # x: (batch, L) or (L,) -> returns (batch, L, hd)
        if x.dim()==1: x=x.unsqueeze(0)
        x=self.emb(x).transpose(1,2)
        x=F.relu(self.c1(x)).transpose(1,2); x=self.n1(x).transpose(1,2)
        x=F.relu(self.c2(x)).transpose(1,2); x=self.n2(x).transpose(1,2)
        x=F.relu(self.c3(x)).transpose(1,2); return self.n3(x)

class MatchingNetwork(nn.Module):
    """Matching Networks with per-position attention over support positions."""
    def __init__(self,hd=128):
        super().__init__()
        self.query_enc=Enc(hd=hd)   # f(X_test)
        self.support_enc=Enc(hd=hd) # g(X_train) - separate encoder for support
        self.output_enc=nn.Embedding(11,hd,padding_idx=10)  # Fix A: vocab_size=11, padding_idx=10; h(Y) - value encoder for labels
        # Decoder after attention: input is query features + attended values = hd*2
        self.decoder=nn.Sequential(nn.Linear(hd*2,hd),nn.ReLU(),nn.LayerNorm(hd),nn.Linear(hd,10))

    def forward(self, support_inputs, support_outputs, query_input, support_masks=None, query_mask=None):
        if query_input.dim()==1: query_input=query_input.unsqueeze(0)

        # Encode query per position
        q_enc=self.query_enc(query_input)  # (1,L_q,hd)
        q_enc=q_enc.squeeze(0)  # (L_q,hd)

        # Encode support per position and collect keys/values
        K=support_inputs.size(0)
        all_support_features=[]
        all_output_embeds=[]
        for i in range(K):
            s_enc=self.support_enc(support_inputs[i:i+1]).squeeze(0)  # (L_s,hd)
            o_emb=self.output_enc(support_outputs[i])  # (L_s,hd)
            if support_masks is not None:
                mask=support_masks[i].bool()  # (L_s,)
                s_enc=s_enc[mask]  # (valid_pos, hd)
                o_emb=o_emb[mask]  # (valid_pos, hd)
            all_support_features.append(s_enc)
            all_output_embeds.append(o_emb)

        keys=torch.cat(all_support_features,dim=0)  # (total_pos,hd)
        values=torch.cat(all_output_embeds,dim=0)  # (total_pos,hd)

        # Attention: each query position attends to all support positions
        attn_scores=torch.matmul(q_enc,keys.t())/(q_enc.size(-1)**0.5)  # (L_q, total_pos)
        attn_weights=F.softmax(attn_scores,dim=-1)  # (L_q, total_pos)
        attended=torch.matmul(attn_weights,values)  # (L_q,hd)

        # Concatenate query features with attended values
        combined=torch.cat([q_enc,attended],dim=-1)  # (L_q, hd*2)
        logits=self.decoder(combined)  # (L_q,10)
        return logits

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',default='1D-ARC/dataset')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--hidden_dim',type=int,default=128)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--save_dir',default='checkpoints_arc_matching')
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
    model=MatchingNetwork(args.hidden_dim).to(device)
    opt=optim.Adam(model.parameters(),lr=args.lr)
    os.makedirs(args.save_dir,exist_ok=True); best_val=0
    for ep in range(args.epochs):
        random.shuffle(tr); losses=[]; model.train()
        for i in tqdm(tr,desc=f"Epoch {ep+1}"):
            t=ds[i]; si=t['support_inputs'].to(device); so=t['support_outputs'].to(device)
            sm=t['support_masks'].to(device)
            qm_=t['query_mask'].to(device); qi=t['query_input'].to(device); qo=t['query_output'].to(device)
            opt.zero_grad()
            logits=model(si,so,qi,support_masks=sm,query_mask=qm_)
            loss=F.cross_entropy(logits,qo,reduction='none', ignore_index=10)
            loss=(loss*qm_).sum()/qm_.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)  # Fix E: gradient clipping
            opt.step(); losses.append(loss.item())
        print(f"Epoch {ep+1}: Loss={np.mean(losses):.4f}")
        if (ep+1)%5==0:
            model.eval(); accs=[]
            for i in va:
                t=ds[i]; si=t['support_inputs'].to(device); so=t['support_outputs'].to(device)
                sm=t['support_masks'].to(device)
                qi=t['query_input'].to(device); qo=t['query_output'].to(device); qm_=t['query_mask'].to(device)
                with torch.no_grad():
                    pred=model(si,so,qi,support_masks=sm,query_mask=qm_).argmax(-1); m=qm_.bool()
                    accs.append((pred[m]==qo[m]).float().mean().item())
            ma=np.mean(accs); print(f"Val: {ma:.4f}")
            if ma>best_val: best_val=ma; torch.save(model.state_dict(),os.path.join(args.save_dir,'best_model.pt'))
    # Fix C: Reload best model before test
    best_model_path=os.path.join(args.save_dir,'best_model.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path,map_location=device))
        print("Loaded best model for testing.")
    model.eval(); accs=[]
    for i in tqdm(te,desc="Test"):
        t=ds[i]; si=t['support_inputs'].to(device); so=t['support_outputs'].to(device)
        sm=t['support_masks'].to(device)
        qi=t['query_input'].to(device); qo=t['query_output'].to(device); qm_=t['query_mask'].to(device)
        with torch.no_grad():
            pred=model(si,so,qi,support_masks=sm,query_mask=qm_).argmax(-1); m=qm_.bool()
            accs.append((pred[m]==qo[m]).float().mean().item())
    ta=np.mean(accs); print(f"Test: {ta:.4f}")
    # Fix F: use with open()
    with open(os.path.join(args.save_dir,'results.json'),'w') as f:
        json.dump({'best_val_acc':best_val,'test_accuracy':ta,'args':vars(args)},f,indent=2)
if __name__=='__main__': main()
