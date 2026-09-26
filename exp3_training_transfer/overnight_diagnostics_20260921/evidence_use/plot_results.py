#!/usr/bin/env python3
"""Plot only the completed frozen diagnostic; individual retained seeds stay visible."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def main():
    summary=json.loads((ROOT/'analysis/summary.json').read_text())
    if not summary['complete']:
        raise SystemExit('Refusing to render a final figure from an incomplete checkpoint roster.')
    oracle=json.loads((ROOT/'analysis/known_mechanism_baseline.json').read_text())
    oracle={r['group']:r for r in oracle['summaries']}
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    arms=['base','causal_family','population_prior','oracle']
    labels=['Base','Matched (2 seeds)','Population prior (2 seeds)','Known-mechanism reference']
    colors=['#7a7a7a','#1768ac','#d97723','#4c956c']
    width=.18
    for col,disclosure in enumerate(['disclosed','undisclosed']):
        for row,(metric,ylabel) in enumerate([('change_mae','Response-change MAE (lower is better)'),('tracking_slope','Gain-change tracking slope')]):
            ax=axes[row,col]
            for ai,(arm,label,color) in enumerate(zip(arms,labels,colors)):
                xs=[];ys=[];dots=[]
                for gi,group in enumerate(['train','test']):
                    values=[oracle[group][metric]] if arm=='oracle' else [r[metric] for r in summary['gain_summary'] if r['disclosure']==disclosure and r['arm']==arm and r['group']==group]
                    x=gi+(ai-1.5)*width
                    xs.append(x);ys.append(np.mean(values));dots.append(values)
                ax.bar(xs,ys,width=width*.88,color=color,label=label,alpha=.8)
                for x,values in zip(xs,dots):
                    if len(values)>1:ax.scatter([x-.025,x+.025],values,s=18,c='black',zorder=5)
            if metric=='change_mae':
                for gi,group in enumerate(['train','test']):
                    ax.hlines(oracle[group]['no_change_mae'],gi-.42,gi+.42,color='black',linestyle='--',lw=1)
            else:
                ax.axhline(0,color='black',ls='--',lw=1)
                ax.axhline(1,color='black',ls=':',lw=1)
                ax.set_ylim(-.08,1.12)
            ax.set_xticks([0,1],['Seen mechanisms','Heldout mechanisms'])
            ax.set_ylabel(ylabel)
            if row==0:ax.set_title('Structure described' if disclosure=='disclosed' else 'Structure not described')
            ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
    fig.text(.5,.01,'Black dots: individual training seeds 45 and 46. Dashed lines: no-change prediction.\nKnown-mechanism reference receives extra information; only 16 pairs per mechanism. Gain pairs use k=9.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,.91])
    for suffix in ['png','pdf']:
        fig.savefig(ROOT/f'analysis/evidence_use.{suffix}',dpi=180,bbox_inches='tight')
    print(ROOT/'analysis/evidence_use.png')

if __name__=='__main__':main()
