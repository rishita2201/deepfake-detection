"""Draws the system-overview figure (Fig. 1)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'Liberation Sans','savefig.dpi':300})
W,H=3.4,5.15
fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,150); ax.axis('off')
def box(x,y,w,h,txt,fc,ec,fs=5.4,bold_first=True):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=2",fc=fc,ec=ec,lw=0.8))
    lines=txt.split('\n')
    n=len(lines); step=min(5.0,(h-2)/max(n,1)*1.0)
    y0=y+h/2+(n-1)*step/2
    for i,l in enumerate(lines):
        ax.text(x+w/2,y0-i*step,l,ha='center',va='center',fontsize=fs,fontweight='bold' if (i==0 and bold_first) else 'normal',linespacing=1.0)
def arrow(x,y0,y1): ax.annotate('',xy=(x,y1),xytext=(x,y0),arrowprops=dict(arrowstyle='-|>',lw=0.8,color='#333',mutation_scale=6))
blue=('#dbe8fb','#4472c4'); green=('#e2efd9','#548235'); amber=('#fff2cc','#bf9000'); red=('#fadbd8','#c00000'); grey=('#ebeef1','#595959')
box(8,134,84,13,"A. Dataset\nKaggle \u201cDeepfake and Real Images\u201d (OpenForensics subset, 256\u00d7256 JPEG)\nTrain 140,002 | Validation 39,428 | Test 10,905 (balanced)",*blue)
arrow(50,134,130)
box(8,114,84,16,"B. Pre-processing\nTrain: resize 256, RandomResizedCrop 224 (0.85\u20131.0), flip,\n\u00b110\u00b0 rotation, colour jitter 0.1.  Val/Test: resize 224\nImageNet mean/std normalisation (all models)",*blue)
arrow(50,114,110)
ws=[19.2]*4; xs=[8,29.5,51,72.5]
names=[("ResNet-18","11.18 M"),("ResNet-50","23.51 M"),("EfficientNet-B0","4.01 M"),("Xception","20.81 M")]
for x,(n,p) in zip(xs,names): box(x,90,19.5,17,f"{n}\nImageNet-\npretrained\n{p} params",*green,fs=4.8)
ax.text(50,109.1,'',fontsize=1)
for x in xs: arrow(x+9.75,90,86)
box(8,76,84,10,"C. Classification head\nGlobal average pooling \u2192 Linear(1) \u2192 logit z \u2192 sigmoid p = \u03c3(z)",*amber)
arrow(50,76,72)
box(8,56,84,16,"D. Training (identical for all four backbones)\nBCE-with-logits (unweighted) | Adam + L2 (1e-4), lr 1e-4, no scheduler\nbatch 32 | AMP | early stopping on validation loss (patience 2)\nbest-validation-loss checkpoint restored | seed 42",*amber)
arrow(50,56,52)
box(8,38,84,14,"E. Test-set evaluation (\u03c4 = 0.5)\nAccuracy, precision, recall, F1, specificity, FPR/FNR, ROC-AUC, AP\nconfusion matrices, 95% bootstrap CIs",*red)
arrow(50,38,34)
box(8,8,84,26,"F. Extended analysis\nBrier score and ECE | validation-vs-test gap | threshold tuning on validation\nduplicate check (pHash, MD5) | McNemar + Holm, disagreement, oracle\nrobustness: 4 degradations \u00d7 3 severities on the full test set\nGrad-CAM + centre-concentration | inference timing (batch 1 / 32)",*red)
box(8,0.5,84,5.5,"PyTorch 2.11 (CUDA 12.8), timm, scikit-learn, statsmodels | NVIDIA T4",*grey,fs=4.8,bold_first=False)
fig.savefig(str(__import__('pathlib').Path(__file__).resolve().parents[1]/'figures'/'f_arch.png'),pad_inches=0.02); print('ok')
