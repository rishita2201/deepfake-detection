"""Regenerates the data-driven paper figures from results/ and history/.
Run from anywhere:  python scripts/make_paper_figures.py
(Grad-CAM figures come from notebook 05 and are stored as final PNGs in figures/.)"""
import numpy as np, pandas as pd, json, random, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix

from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TAB=str(ROOT/'results'/'tables')+'/'; PRED=str(ROOT/'results'/'predictions')+'/'; HIST=str(ROOT/'history')+'/'; OUT=str(ROOT/'figures')+'/'
plt.rcParams.update({'font.family':'Liberation Sans','font.size':7,'axes.labelsize':7.5,'axes.titlesize':8,
  'legend.fontsize':6.5,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':0.6,'lines.linewidth':1.1,
  'savefig.dpi':300,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
M=['resnet18','resnet50','efficientnet_b0','xception']
NAME={'resnet18':'ResNet-18','resnet50':'ResNet-50','efficientnet_b0':'EfficientNet-B0','xception':'Xception'}
COL={'resnet18':'#1f77b4','resnet50':'#ff7f0e','efficientnet_b0':'#2ca02c','xception':'#d62728'}
W=3.4  # single-column width (in)

# ---- labels / probs (sample order reproduced from the notebook) ----
n={'val':(19787,19641),'test':(5413,5492)}
def order(N):
    L=list(range(N)); random.Random(42).shuffle(L); return np.array(L)
lab=lambda s: np.array([0]*n[s][0]+[1]*n[s][1])
yt=lab('test')[order(sum(n['test']))]; yv=lab('val')[order(sum(n['val']))]
PT={m:np.load(PRED+f'{m}_seed42_test.npy') for m in M}; PV={m:np.load(PRED+f'{m}_seed42_val.npy') for m in M}
H={m:json.load(open(HIST+f'{m}_history.json')) for m in M}

def save(fig,name): fig.savefig(OUT+name+'.png',bbox_inches='tight',pad_inches=0.03); plt.close(fig)

# 1 training curves (2 x 4 grid: loss row / acc row)
fig,ax=plt.subplots(2,4,figsize=(W,2.3),sharex=False)
for j,m in enumerate(M):
    h=H[m]; ep=np.arange(1,len(h['val_loss'])+1); b=int(np.argmin(h['val_loss']))+1
    ax[0,j].plot(ep,h['train_loss'],color='#444'); ax[0,j].plot(ep,h['val_loss'],color=COL[m],ls='--')
    ax[1,j].plot(ep,h['train_acc'],color='#444',label='Train'); ax[1,j].plot(ep,h['val_acc'],color=COL[m],ls='--',label='Validation')
    for r in (0,1): ax[r,j].axvline(b,color='gray',lw=0.5,ls=':'); ax[r,j].set_xticks(ep[::2] if len(ep)>5 else ep)
    ax[0,j].set_title(NAME[m].replace('EfficientNet-B0','EffNet-B0'),fontsize=6.5); ax[1,j].set_xlabel('Epoch')
ax[0,0].set_ylabel('Loss'); ax[1,0].set_ylabel('Accuracy')
from matplotlib.lines import Line2D
fig.legend([Line2D([0],[0],color='#444'),Line2D([0],[0],color='#555',ls='--')],['Train','Validation'],loc='upper center',ncol=2,frameon=False,fontsize=6,bbox_to_anchor=(0.5,1.06))
fig.tight_layout(pad=0.4,w_pad=0.5); save(fig,'f_curves')

# 2 metrics with bootstrap CI
B=pd.read_csv(TAB+'B1_bootstrap_ci.csv')
def parse(s):
    a=s.replace('[','').replace(']','').replace(',','').split(); return float(a[0]),float(a[1]),float(a[2])
fig,ax=plt.subplots(figsize=(W,1.9))
metrics=[('acc_CI','accuracy','Accuracy'),('f1_CI','f1','F1'),('auc_CI','roc_auc','ROC-AUC')]
x=np.arange(3); bw=0.19
for i,m in enumerate(M):
    r=B[B.model==m].iloc[0]; vals=[]; lo=[]; hi=[]
    for c,pt,_ in metrics:
        v=r[pt]; mean,l,h=parse(r[c]); vals.append(v); lo.append(v-l); hi.append(h-v)
    ax.bar(x+(i-1.5)*bw,vals,bw,yerr=[lo,hi],color=COL[m],label=NAME[m],error_kw={'lw':0.7,'capsize':1.5})
ax.set_xticks(x); ax.set_xticklabels([t for *_,t in metrics]); ax.set_ylim(0.82,1.0); ax.set_ylabel('Score (95% bootstrap CI)')
ax.legend(frameon=False,ncol=2,loc='upper left',fontsize=6); fig.tight_layout(pad=0.4); save(fig,'f_bars')

# 3 confusion matrices
fig,ax=plt.subplots(1,4,figsize=(W,1.0))
for j,m in enumerate(M):
    cm=confusion_matrix(yt,(PT[m]>=.5).astype(int)); ax[j].imshow(cm,cmap='Blues',vmin=0,vmax=6000)
    for (a,b_),v in np.ndenumerate(cm): ax[j].text(b_,a,f'{v}',ha='center',va='center',fontsize=6,color='white' if v>3000 else 'black')
    ax[j].set_xticks([0,1]); ax[j].set_yticks([0,1]); ax[j].set_xticklabels(['Real','Fake'],fontsize=5.5); ax[j].set_yticklabels(['Real','Fake'] if j==0 else ['',''],fontsize=5.5)
    ax[j].set_title(NAME[m].replace('EfficientNet-B0','EffNet-B0'),fontsize=6); ax[j].set_xlabel('Predicted',fontsize=5.5)
    for s in ax[j].spines.values(): s.set_visible(False)
ax[0].set_ylabel('True',fontsize=5.5); fig.tight_layout(pad=0.3,w_pad=0.2); save(fig,'f_cm')

# 4 ROC + PR
fig,ax=plt.subplots(1,2,figsize=(W,1.75))
AP={}
for m in M:
    f,t,_=roc_curve(yt,PT[m]); ax[0].plot(f,t,color=COL[m],label=f'{NAME[m]} ({roc_auc_score(yt,PT[m]):.3f})')
    p,r,_=precision_recall_curve(yt,PT[m]); AP[m]=average_precision_score(yt,PT[m]); ax[1].plot(r,p,color=COL[m],label=f'{NAME[m]} ({AP[m]:.3f})')
ax[0].plot([0,1],[0,1],color='gray',lw=0.5,ls=':'); ax[0].set_xlabel('False-positive rate'); ax[0].set_ylabel('True-positive rate')
ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision'); ax[1].set_ylim(0.6,1.01)
ax[0].legend(frameon=False,loc='lower right',fontsize=5.2,title='AUC',title_fontsize=5.5); ax[1].legend(frameon=False,loc='lower left',fontsize=5.2,title='AP',title_fontsize=5.5)
fig.tight_layout(pad=0.4,w_pad=0.8); save(fig,'f_rocpr'); json.dump(AP,open(TAB+'average_precision.json','w'))

# 5 validation vs test per-class accuracy (the gap)
G=pd.read_csv(TAB+'B3_val_vs_test_gap.csv')
fig,ax=plt.subplots(1,2,figsize=(W,1.7),sharey=True)
for k,(a,b_,t) in enumerate([('val_specificity','test_specificity','Real images (specificity)'),('val_recall','test_recall','Fake images (recall)')]):
    x=np.arange(4)
    ax[k].bar(x-0.19,G[a],0.38,color='#9ecae1',label='Validation'); ax[k].bar(x+0.19,G[b_],0.38,color='#08519c',label='Test')
    ax[k].set_xticks(x); ax[k].set_xticklabels(['R18','R50','EffB0','Xcep'],fontsize=6); ax[k].set_title(t,fontsize=6.5); ax[k].set_ylim(0.7,1.1)
ax[0].set_ylabel('Per-class accuracy'); ax[0].set_yticks([0.7,0.8,0.9,1.0]); ax[0].legend(frameon=False,loc='upper right',ncol=2,fontsize=5.5); fig.tight_layout(pad=0.4,w_pad=0.5); save(fig,'f_gap')

# 6 confidence histograms (correct vs incorrect)
fig,ax=plt.subplots(2,2,figsize=(W,2.2),sharex=True)
for k,m in enumerate(M):
    a=ax[k//2,k%2]; conf=np.where(PT[m]>=.5,PT[m],1-PT[m]); ok=((PT[m]>=.5)==yt)
    bins=np.linspace(0.5,1,11); a.hist(conf[ok],bins=bins,color='#6baed6',alpha=0.9,label='Correct'); a.hist(conf[~ok],bins=bins,color='#cb181d',alpha=0.9,label='Incorrect')
    a.set_yscale('log'); a.set_title(NAME[m],fontsize=6.5)
    if k%2==0: a.set_ylabel('Images')
    if k//2==1: a.set_xlabel('Predicted-class confidence')
ax[0,0].legend(frameon=False,fontsize=5.5,loc='upper left'); fig.tight_layout(pad=0.4,h_pad=0.4,w_pad=0.5); save(fig,'f_conf')

# 7 computational cost (params vs latency, batch 32 FP32 and batch 1)
T=pd.read_csv(TAB+'F1_timing.csv'); params={'resnet18':11.18,'resnet50':23.51,'efficientnet_b0':4.01,'xception':20.81}
fig,ax=plt.subplots(1,2,figsize=(W,1.6))
ax[0].bar([NAME[m].replace('EfficientNet-B0','EffNet-B0') for m in M],[params[m] for m in M],color=[COL[m] for m in M]); ax[0].set_ylabel('Parameters (M)'); ax[0].tick_params(axis='x',labelsize=5.5,rotation=20)
for i,bs in enumerate([1,32]):
    v=[T[(T.model==m)&(T.batch==bs)&(T.precision=='FP32')].ms_per_img_mean.iloc[0] for m in M]
    ax[1].bar(np.arange(4)+(i-0.5)*0.38,v,0.38,color=['#bbbbbb','#555555'][i],label=f'batch {bs}')
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(['R18','R50','EffB0','Xcep'],fontsize=5.5); ax[1].set_ylabel('Latency (ms/image, FP32)'); ax[1].legend(frameon=False,fontsize=5.5)
fig.tight_layout(pad=0.4,w_pad=0.8); save(fig,'f_cost')

# 8 robustness (full test set)
D=pd.read_csv(TAB+'D1_robustness_full_testset.csv'); degs=['JPEG','Blur','Noise','Resize']
fig,ax=plt.subplots(2,2,figsize=(W,2.5),sharey=False)
for k,dg in enumerate(degs):
    a=ax[k//2,k%2]; sub=D[D.degradation==dg]
    for m in M:
        s=sub[sub.model==m].sort_values('level'); a.plot([0]+list(s.level),[s.clean_f1.iloc[0]]+list(s.f1),marker='o',ms=2.2,color=COL[m],label=NAME[m])
    par={'JPEG':'JPEG quality 75 / 50 / 25','Blur':'Blur \u03c3 = 1 / 2 / 3','Noise':'Noise \u03c3 = 5 / 10 / 15','Resize':'Scale 0.8 / 0.65 / 0.5'}[dg]
    a.set_title(par,fontsize=6.3); a.set_xticks([0,1,2,3]); a.set_xticklabels(['clean','1','2','3'])
    if k%2==0: a.set_ylabel('F1-score')
    if k//2==1: a.set_xlabel('Severity level')
ax[0,0].legend(frameon=False,fontsize=5,loc='lower left'); fig.tight_layout(pad=0.4,h_pad=0.5,w_pad=0.5); save(fig,'f_robust')

# 9 Grad-CAM centre concentration
C=pd.read_csv(TAB+'G2_gradcam_by_class_and_correctness.csv')
fig,ax=plt.subplots(figsize=(W,1.7)); groups=[('fake',1,'Fake, correct'),('real',1,'Real, correct'),('real',0,'Real, wrongly\nflagged fake')]
for i,m in enumerate(M):
    v=[C[(C.model==m)&(C.true_class==tc)&(C.correct==cr)]['mean'].iloc[0] for tc,cr,_ in groups]
    ax.bar(np.arange(3)+(i-1.5)*0.19,v,0.19,color=COL[m],label=NAME[m])
ax.axhline(0.25,color='k',lw=0.6,ls='--'); ax.text(-0.45,0.275,'uniform map (0.25)',fontsize=5.2,ha='left'); ax.set_xticks(range(3)); ax.set_xticklabels([g[2] for g in groups],fontsize=6)
ax.set_ylabel('Centre-box Grad-CAM mass'); ax.set_ylim(0,1.25); ax.set_yticks([0,0.25,0.5,0.75,1.0]); ax.legend(frameon=False,ncol=4,fontsize=5.2,loc='upper center',columnspacing=0.8,handlelength=1.0); fig.tight_layout(pad=0.4); save(fig,'f_camconc')

# 10 near-duplicate / threshold figure: val-opt threshold effect (test FPR at 0.5 vs tuned)
Th=pd.read_csv(TAB+'B4_threshold_tuning.csv'); Th=Th[Th.rule!='val-Youden-optimal']
fig,ax=plt.subplots(figsize=(W,1.6)); x=np.arange(4)
a=[Th[(Th.model==m)&(Th.rule=='tau=0.5')].test_fpr.iloc[0] for m in M]; b_=[Th[(Th.model==m)&(Th.rule=='val-F1-optimal')].test_fpr.iloc[0] for m in M]
ax.bar(x-0.19,a,0.38,color='#9ecae1',label='\u03c4 = 0.5'); ax.bar(x+0.19,b_,0.38,color='#08519c',label='\u03c4 tuned on validation')
ax.set_xticks(x); ax.set_xticklabels(['R18','R50','EffB0','Xcep']); ax.set_ylabel('Test false-positive rate'); ax.legend(frameon=False,fontsize=5.8,loc='upper left'); fig.tight_layout(pad=0.4); save(fig,'f_thr')

print('ok', AP)
