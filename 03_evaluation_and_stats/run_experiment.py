"""Configuration-driven experiments using the original MiniLM and account head."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import site
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiment_common import ROOT, read_config, save_npz, sha, write_json
from evaluate_stability import evaluate
from gradient_analysis import collect, supervised_projection
import train_twibot20_local as base
from encode_own_tweets import Encoder
from prepare_twibot20_local import AUDIT
from train_twibot20_resampling_factorial import IndexedCache, gather_views


class SharedHeads(nn.Module):
    def __init__(self,original,count):
        super().__init__()
        self.text=deepcopy(original.text)
        self.profile=deepcopy(original.profile)
        self.heads=nn.ModuleList([deepcopy(original.head)])
        for _ in range(count-1):
            head=deepcopy(original.head)
            for module in head.modules():
                if isinstance(module,nn.Linear): module.reset_parameters()
            self.heads.append(head)

    def pieces(self,view):
        x=self.text(view.tweets.float())
        pooled=(x*view.mask[...,None]).sum(-2)/view.mask.sum(-1).clamp_min(1)[...,None]
        profile=self.profile(view.profile)[None].expand_as(pooled)
        fused=torch.cat((pooled,profile),dim=-1)
        return torch.stack([h(fused).squeeze(-1) for h in self.heads]),pooled,profile,fused

    def forward(self,view):
        logits,*_=self.pieces(view)
        if len(self.heads)==1: return logits[0]
        return torch.logit(logits.sigmoid().mean(0).clamp(1e-7,1-1e-7))


def slots_for(counts,views,rng,budget):
    result=np.full((len(counts),views,budget),-1,dtype=np.int16)
    for i,c in enumerate(counts):
        for v in range(views):
            result[i,v,:min(c,budget)]=rng.choice(int(c),min(c,budget),replace=False)
    return result


def objective(logits,pooled,labels,cfg):
    y=labels[None,None].expand_as(logits)
    weight=cfg['_pos_weight']
    cls=F.binary_cross_entropy_with_logits(logits,y,pos_weight=weight)
    if logits.shape[1]<2 or cfg['consistency']=='none':
        return cls,cls*0
    if cfg['consistency']=='feature_cosine':
        cons=(1-F.cosine_similarity(pooled[0],pooled[1],dim=-1)).mean()
    else:
        t=cfg['temperature']
        a,b=(logits[:,i]/t for i in (0,1))
        cons=(.5*(a.sigmoid()-b.sigmoid())*(a-b)*t*t).mean()
    return cls,cons


@torch.inference_mode()
def predict_bank(layer,head,cache,indices,slots,profiles,batch,ablations=False):
    scores=np.empty((len(indices),slots.shape[1]),dtype=np.float32)
    per_head=np.empty((len(indices),len(head.heads),slots.shape[1]),dtype=np.float32)
    extra={k:np.empty_like(scores) for k in ('zero_tweet','zero_profile')} if ablations else {}
    for start in range(0,len(indices),batch):
        sel=slice(start,start+batch)
        table=cache.selected(layer,indices[sel],slots[sel])
        # One view per head invocation; no cross-view prediction fusion.
        for m in range(slots.shape[1]):
            view=gather_views(table,slots[sel,m:m+1],profiles.at(sel))
            z,*_=head.pieces(view)
            p=z.sigmoid().mean(0)
            combined=z[0] if len(head.heads)==1 else torch.logit(p.clamp(1e-7,1-1e-7))
            scores[sel,m]=combined.squeeze(0).cpu().numpy()
            per_head[sel,:,m]=z[:,0].T.cpu().numpy()
            if ablations:
                masked=base.Views(view.tweets,torch.zeros_like(view.mask),view.profile,view.labels,view.counts)
                removed=base.Views(view.tweets,view.mask,torch.zeros_like(view.profile),view.labels,view.counts)
                for name,data in (('zero_tweet',masked),('zero_profile',removed)):
                    extra[name][sel,m]=head(data).squeeze(0).cpu().numpy()
    return scores,per_head,extra


def stratified_limit(indices,y,limit,seed):
    if limit is None or len(indices)<=limit: return indices
    rng=np.random.default_rng(seed)
    groups=[indices[y[indices]==c] for c in (0,1)]
    sizes=[round(limit*len(g)/len(indices)) for g in groups]
    sizes[-1]=limit-sizes[0]
    return np.concatenate([rng.choice(g,n,replace=False) for g,n in zip(groups,sizes,strict=True)])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=['smoke','pilot','full'],default='smoke')
    args=parser.parse_args()
    cfg=read_config(args.config)
    if not cfg.get('runnable',True):
        raise ValueError(cfg['selection_status'])
    if cfg['optimizer']!='AdamW':
        raise ValueError('Only the configured existing AdamW optimizer is implemented')
    if (ROOT/cfg['data_path']).resolve()!=base.OUTPUT.resolve():
        raise ValueError('Data path must identify the audited existing data')
    stage=cfg['stages'][args.stage]
    cfg.update(stage)
    if cfg['unfreeze_layers'] not in (0,1):
        raise ValueError('Existing prefix cache supports K=0 or 1. Rebuild an earlier-layer prefix before using K>1.')
    if cfg['budget']!=8: raise ValueError('This study fixes exactly eight tweets in the primary cohort')
    if args.stage=='full' and not cfg['resource_reviewed']:
        raise ValueError('Full experiment requires the recorded resource estimate to be reviewed in the config')
    out=ROOT/cfg['output_root']/args.stage/cfg['name']
    out.mkdir(parents=True,exist_ok=True)
    resolved=dict(cfg)
    resolved['source_sha256']=sha(Path(__file__))
    resolved['source_files']={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'scripts').glob('*.py'))}
    config_hash=hashlib.sha256(json.dumps(resolved,sort_keys=True).encode()).hexdigest()
    existing=out/'resolved_config.json'
    if existing.exists() and json.loads(existing.read_text())!=resolved:
        raise ValueError('Existing run configuration differs; use a new output root')
    write_json(existing,resolved)
    assert torch.cuda.is_available() and not site.ENABLE_USER_SITE
    torch.set_num_threads(cfg['cpu_threads'])
    sys.path.append(str(AUDIT/'encoder_deps'))
    encoder=Encoder(AUDIT/'encoder',cfg['sentence_microbatch'],128,'cuda')
    torch.set_num_threads(cfg['cpu_threads'])
    template=deepcopy(encoder.model.encoder.layer[-1]).float().eval()
    del encoder
    sys.path.append(site.getusersitepackages())
    torch.set_float32_matmul_precision('highest')
    import train_twibot20_resampling_factorial as cache_code
    cache_code.SENTENCES=cfg['sentence_microbatch']
    raw,dev=base.load_data('train'),base.load_data('dev')
    original=json.loads((base.OUTPUT/'training_protocol.json').read_text())
    fit,cal=(np.asarray(original[k]) for k in ('fit_indices','cal_indices'))
    fit=fit[raw['count'][fit]>=8]
    fit=stratified_limit(fit,raw['labels'],cfg['fit_limit'],cfg['sampling_seed'])
    cal=cal[raw['count'][cal]>=8]
    cal=stratified_limit(cal,raw['labels'],cfg['cal_limit'],cfg['sampling_seed']+1)
    libpath=ROOT/cfg['sampling_library']
    with np.load(libpath) as saved: library=dict(saved)
    np.testing.assert_array_equal(library['ids'],dev['ids'])
    library_hash=sha(libpath)
    di=np.flatnonzero(library['primary'])
    di=stratified_limit(di,dev['labels'],cfg['eval_limit'],cfg['sampling_seed']+2)
    bank=library['slots'][di,:cfg['views']]
    assert (bank>=0).all() and bank.shape[-1]==8
    mu,sd=(np.asarray(original[k],dtype=np.float32) for k in ('meta_mean','meta_std'))
    profiles=[base.make_views(d,ids,mu,sd) for d,ids in ((raw,fit),(raw,cal),(dev,di))]
    caches=[IndexedCache(ROOT/cfg['prefix_cache'],split,d['ids'],d['count'],width)
            for split,d,width in (('train',raw,16),('dev',dev,32))]
    cal_slots=slots_for(raw['count'][cal],1,np.random.default_rng(cfg['sampling_seed']+3),8)
    scores,thresholds,seed_values,head_values=[],[],[],[]
    for seed in cfg['seeds']:
        pending=out/f'seed_{seed}.npz'
        if all(p.exists() for p in (pending,out/f'seed_{seed}.json',out/f'seed_{seed}.pt')):
            with np.load(pending) as cached:
                scores.append(cached['logits']);thresholds.append(float(cached['threshold']));head_values.append(cached['per_head'])
            seed_values.append(json.loads((out/f'seed_{seed}.json').read_text()))
            continue
        torch.manual_seed(seed)
        weights=torch.load(ROOT/cfg['initialization']/f'seed_{seed}_tuned_natural.pt',map_location='cuda',weights_only=True)
        layer=deepcopy(template).requires_grad_(cfg['unfreeze_layers']>0).eval()
        layer.load_state_dict(weights['last_layer'])
        anchor=base.AccountHead().cuda();anchor.load_state_dict(weights['head'])
        head=SharedHeads(anchor,cfg['heads']).cuda()
        nparams=sum(p.numel() for p in layer.parameters() if p.requires_grad)+sum(p.numel() for p in head.parameters())
        parameters=[p for p in list(layer.parameters())+list(head.parameters()) if p.requires_grad]
        groups=[{'params':head.parameters(),'lr':cfg['head_lr']}]
        if cfg['unfreeze_layers']:
            groups.append({'params':layer.parameters(),'lr':cfg['encoder_lr']})
        optimizer=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
        frozen=caches[0].all_valid(layer,fit) if not cfg['unfreeze_layers'] else None
        rng=np.random.default_rng(cfg['sampling_seed']+seed)
        fixed=slots_for(raw['count'][fit],cfg['train_views'],rng,8)
        pos=torch.tensor((raw['labels'][fit]==0).sum()/(raw['labels'][fit]==1).sum(),device='cuda',dtype=torch.float32)
        cfg['_pos_weight']=pos
        traces,diagnostics=[],[]
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        for epoch in range(cfg['epochs'] if cfg['train'] else 0):
            head.train()
            plan=slots_for(raw['count'][fit],cfg['train_views'],rng,8) if cfg['dynamic'] and epoch>0 else fixed
            order=np.random.default_rng(seed+epoch*1000).permutation(len(fit))
            losses=[]
            for start in range(0,len(order),cfg['batch']):
                idx=order[start:start+cfg['batch']]
                table=caches[0].selected(layer,fit[idx],plan[idx]) if cfg['unfreeze_layers'] else frozen[idx]
                view=gather_views(table,plan[idx],profiles[0].at(idx))
                if not cfg['unfreeze_layers'] and cfg['consistency']!='none':
                    view.tweets=view.tweets.detach().requires_grad_()
                z,pooled,profile,fused=head.pieces(view)
                cls,cons=objective(z,pooled,view.labels,cfg)
                if cfg['consistency']!='none' and start==0:
                    diagnostics.append({'epoch':epoch+1, 'groups':collect(cls,cons,{
                        'tweet_encoder':list(layer.parameters()),'tweet_input_representation':[view.tweets],
                        'tweet_pool':[pooled],'profile_branch':list(head.profile.parameters()),
                        'tweet_projection':list(head.text.parameters()),'fusion_representation':[fused],
                        'classification_head':list(head.heads.parameters())})})
                optimizer.zero_grad(set_to_none=True)
                if cfg['gradient_projection'] and cfg['consistency']!='none':
                    supervised_projection(cls,cons,parameters,cfg['lambda_cons'])
                else:
                    (cls+cfg['lambda_cons']*cons).backward()
                torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                optimizer.step()
                losses.append((float(cls.detach()),float(cons.detach())))
            traces.append({'epoch':epoch+1,'loss':np.mean(losses,axis=0).tolist(),'steps':len(losses),
                           'plan_sha256':hashlib.sha256(plan.tobytes()).hexdigest()})
        head.eval()
        train_seconds=time.perf_counter()-started
        evaluation_started=time.perf_counter()
        calibration,_,_=predict_bank(layer,head,caches[0],cal,cal_slots,profiles[1],cfg['batch'])
        cutoff=float(base.cutoff(raw['labels'][cal],calibration[:,0]))
        pred,individual,ablated=predict_bank(layer,head,caches[1],di,bank,profiles[2],cfg['batch'],True)
        result,values=evaluate(pred[:,None],dev['labels'][di],np.array([cutoff]))
        branch={k:evaluate(v[:,None],dev['labels'][di],np.array([cutoff]))[0]['mean'] for k,v in ablated.items()}
        save_npz(pending,ids=dev['ids'][di],logits=pred,per_head=individual,threshold=np.array(cutoff),
                 cal_ids=raw['ids'][cal],cal_y=raw['labels'][cal],cal_logits=calibration,**ablated)
        temp=out/f'seed_{seed}.part'
        torch.save({'layer':layer.state_dict(),'head':head.state_dict(),'config_sha256':config_hash},temp)
        temp.replace(out/f'seed_{seed}.pt')
        restored=torch.load(out/f'seed_{seed}.pt',map_location='cuda',weights_only=True)
        for key,value in head.state_dict().items():
            torch.testing.assert_close(value,restored['head'][key],rtol=0,atol=0)
        record={'seed':seed,'metrics':result['mean'],'ablations':branch,'traces':traces,'gradient':diagnostics,
                'train_seconds':train_seconds,'peak_gib':torch.cuda.max_memory_allocated()/2**30,
                'evaluation_seconds':time.perf_counter()-evaluation_started,
                'checkpoint_reload_verified':True,'trainable_params':nparams,'config_sha256':config_hash,'library_sha256':library_hash}
        write_json(out/f'seed_{seed}.json',record)
        write_json(out/'status.json',{'stage':args.stage,'method':cfg['name'],'completed_seeds':cfg['seeds'][:len(scores)+1],
                                     'remaining_seeds':cfg['seeds'][len(scores)+1:]})
        scores.append(pred);thresholds.append(cutoff);head_values.append(individual);seed_values.append(record)
        print(cfg['name'],seed,result['mean'],flush=True)
    save_npz(out/'predictions.npz',ids=dev['ids'][di],labels=dev['labels'][di],seeds=np.array(cfg['seeds']),
             logits=np.stack(scores,axis=1),per_head_logits=np.stack(head_values,axis=1),thresholds=np.array(thresholds),
             library_sha256=np.array(library_hash),method=np.array(cfg['name']),view_slots=bank)
    result,_=evaluate(np.stack(scores,axis=1),dev['labels'][di],np.array(thresholds))
    write_json(out/'summary.json',{'stage':args.stage,'config_sha256':config_hash,'results':result,
                                 'runtime':seed_values,'test_scored':False})


if __name__=='__main__': main()
