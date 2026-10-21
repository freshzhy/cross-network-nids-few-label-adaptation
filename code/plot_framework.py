#!/usr/bin/env python3
"""plot_framework.py — Fig. 1 (evaluation framework); --ga writes the graphical-abstract version
without internal RQ / figure / table references.
  python code/plot_framework.py [--ga] [--out results/figures]"""
import argparse
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
plt.rcParams['font.family']='DejaVu Sans'
ap=argparse.ArgumentParser(); ap.add_argument('--ga',action='store_true'); ap.add_argument('--out',default=str(Path(__file__).resolve().parent.parent/'results'/'figures'))
A=ap.parse_args(); OUTD=Path(A.out); OUTD.mkdir(parents=True,exist_ok=True)
def ref(t): return '' if A.ga else t
fig,ax=plt.subplots(figsize=(7.2,5.6)); ax.set_xlim(0,100); ax.set_ylim(0,82); ax.axis('off')
def box(x,y,w,h,title,body='',fc='#FFFFFF',ec='#333333',ts=7.8,bs=6.6,bold=True):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.25,rounding_size=1.2',fc=fc,ec=ec,lw=0.9))
    if body:
        ax.text(x+w/2,y+h-1.6,title,ha='center',va='top',fontsize=ts,weight='bold' if bold else 'normal')
        ax.text(x+w/2,y+h-5.0,body,ha='center',va='top',fontsize=bs,linespacing=1.25)
    else:
        ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=ts,weight='bold' if bold else 'normal',linespacing=1.25)
def arr(x1,y1,x2,y2,txt='',dx=0,dy=0,c='#333333',ls='-'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color=c,lw=0.9,ls=ls,shrinkA=0,shrinkB=0,mutation_scale=9))
    if txt: ax.text((x1+x2)/2+dx,(y1+y2)/2+dy,txt,fontsize=6.8,ha='center',va='center',style='italic',color='#333333')
# band labels
for y,t in [(68.5,'Source\nnetwork'),(47.5,'Target\nnetwork'),(25,'Variant\ncoverage'),(4.5,'Deploy-\nment')]:
    ax.text(0.5,y,t,fontsize=7.0,weight='bold',va='center',ha='left',color='#555555')
# Row 1: source knowledge
G='#EEF2F8'
box(13,61,25,15,'Attack signatures','Per-class SHAP attributions;\nstable within a network,\ndivergent across networks\n'+ref('RQ1 · Figs. 2–3 · Tables III–IV'),fc=G)
box(43,61,24,15,'Feature subsets','Source Top-10 by SHAP,\nmutual information or gain\n\n'+ref('RQ2 · Figs. 4–5 · Table V'),fc=G)
box(72,61,26,15,'Source-trained model','XGBoost classifier trained\non the source network\n\n'+ref('RQ3 · Fig. 6'),fc='#F4EFF8')
# grouping of the source-side artifacts
from matplotlib.patches import FancyBboxPatch as _F
ax.add_patch(_F((11.6,59.6),57.8,18.4,boxstyle='round,pad=0.2,rounding_size=1.5',fc='none',ec='#7A8CA8',lw=0.8,ls=(0,(4,2))))
ax.add_patch(_F((70.6,59.6),28.8,18.4,boxstyle='round,pad=0.2,rounding_size=1.5',fc='none',ec='#9A7FB0',lw=0.8,ls=(0,(4,2))))
ax.text(40.5,78.9,'Feature knowledge (explain → select)',ha='center',va='center',fontsize=6.8,style='italic',color='#4A5D7E')
ax.text(85,78.9,'Trained decision function',ha='center',va='center',fontsize=6.8,style='italic',color='#6B4F86')
# Row 2: target
box(13,40,43,15,'Few-label target adaptation','Retrain on b labeled target attack flows\n'+('(b = 1, 2, 5, 10, 25, …, 1000, All)' if A.ga else '(b = 1*, 2*, 5*, 10, 25, …, 1000, All)')+'\nFull-40 · Source-10 · Target-10 · Random-10\n'+ref('RQ4 · Fig. 7 · Table VI'),fc='#FFFFFF')
box(62,40,36,15,'Zero-shot deployment','Source model applied unchanged\nmean F1 0.08;\nnear-chance low-FPR discrimination\n'+ref('RQ3 · Fig. 6'),fc='#F7F7F7')
arr(52,59.4,40,55.4,'retrain on target',dx=10,dy=-0.6)
arr(38.6,68.5,42.4,68.5)
arr(85,59.4,85,55.4,'deploy unchanged',dx=8.5,dy=-0.2)
# decision diamond
cx,cy=34.5,27.5
ax.add_patch(Polygon([(cx-17,cy),(cx,cy+7.5),(cx+17,cy),(cx,cy-7.5)],closed=True,fc='#FFF8E6',ec='#333333',lw=0.9))
ax.text(cx,cy,'Are the expected attack\nvariants covered by the\ntarget labels?'+ref(' (Fig. 8)'),ha='center',va='center',fontsize=6.6,weight='bold',linespacing=1.2)
arr(34.5,40,34.5,35.1)
# outcomes
box(58,30.5,40,7,'Covered variant:\nrapid recovery','',fc='#EAF5EA',ts=7.0)
ax.text(78,29.3,'',fontsize=6.5)
box(58,21.5,40,7,'Unseen tool, same family:\npartial transfer','',fc='#FBF3DE',ts=7.0)
box(58,12.5,40,7,'Unseen attack family:\nlittle benefit in IoT networks','',fc='#F8E5E5',ts=7.0)
arr(51.5,27.6,58,34,'yes',dx=-1.5,dy=1.4); arr(51.5,27.4,58,25,'partly',dy=1.6); arr(49,23.8,58,16,'no',dx=1.8,dy=1.2)
ax.text(31,15.0,'Standard same-class configurations:\n10 labeled flows → mean F1 0.81\n(zero-shot 0.08)',ha='center',va='center',fontsize=6.9,linespacing=1.25,color='#222222')
# validation
box(13,1,85,7,'Validate on the target network at realistic prevalence and low false-positive rates\n(PR-AUC, pAUC@1%FPR, TPR@1%FPR)','',fc='#FFFFFF',ts=7.0)
arr(78,12.5,78,8.4)
arr(31,11.2,31,8.4)
stem='Graphical_abstract' if A.ga else 'Fig1_framework'
for e in ('png','pdf'): fig.savefig(OUTD/f'{stem}.{e}',dpi=300,bbox_inches='tight',pad_inches=0.05)
print('→',OUTD/stem)
