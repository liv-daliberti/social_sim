"""Generate a reviewable ICLR appendix draft only from completed, audited results."""
import json,math
from pathlib import Path
from .freeze import check_artifact,artifact
from . import finish
from .. import run_local as c

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/fresh_evaluation_v1'
DRAFT=ROOT/'runs/fresh_evaluation_v1/paper_draft'
NAMES={'base':'Base','names':'Renamed','paraphrase':'Paraphrased','resample':'Same-text resample'}


def number(x,d=2):return '--' if x is None else f'{x:.{d}f}'

def interval(metric,scale=1):
    ci=metric['ci95'];point=metric['estimate']
    return number(None if point is None else scale*point,1)+((' ['+', '.join(number(scale*x,1) for x in ci)+']') if ci is not None else ' [--]')

def count(metric):return f"{metric['n_success']}/{metric['n_planned']}"

def paired(metric):return count(metric)+' ['+', '.join(number(100*x,1) for x in metric['ci95'])+']' if metric['ci95'] is not None else count(metric)+' [--]'

def table(headers,rows,align):
    return '\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'+'\n'.join(' & '.join(row)+' \\\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'

def block(caption,label,filename):
    return '\n\\begin{center}\n\\begin{minipage}{\\linewidth}\n\\centering\n\\captionof{table}{'+caption+'}\n\\label{'+label+'}\n\\small\n\\input{tables/'+filename+'}\n\\end{minipage}\n\\end{center}\n'


def write():
    audit=json.loads((RESULTS/'technical_audit.json').read_text());status=json.loads((RESULTS/'status.json').read_text())
    if audit.get('status')!='passed' or audit.get('frontier_status')!='complete' or status.get('status')!='complete':raise ValueError('All local and frontier results must be complete and audited before writing the paper')
    if audit['audit_code']['sha256']!=c.file_sha256(Path(finish.__file__)):raise ValueError('Audit code changed since results')
    for spec in audit['summary_artifacts'].values():check_artifact(spec)
    data={name:json.loads(check_artifact(spec).read_text()) for name,spec in audit['summary_artifacts'].items()}
    off,on=data['disabled.json'],data['enabled.json'];gpt=data['frontier.json'];contrast=data['reasoning_comparison.json']
    for s,n in [(off,80),(on,80),(gpt,24)]:
        if not s['all_planned_received'] or s['parent_instances']!=n:raise ValueError('Wrong cohort or incomplete summary')
        for v,m in s['by_variant'].items():
            assert m['paired_reversal']['n_planned']==n and m['sign_accuracy']['n_planned']==2*n
    DRAFT.mkdir(parents=True,exist_ok=True)
    def put(name,text):c.atomic_write(DRAFT/name,text)
    rows=[]
    for label,s in [('Off',off),('On',on)]:
        for v in NAMES:
            m=s['by_variant'][v];b=m['broken_absolute_pp']
            rows.append([label,NAMES[v],count(m['sign_accuracy']),paired(m['paired_reversal']),interval(b),str(b['n_observed'])+'/80'])
    put('exp1_fresh_performance.tex',table(['Thinking','Wording','Signs','Pairs [95\\% CI]','Broken $|\\Delta|$, pp','Observed'],rows,'llrrrr'))
    rows=[]
    for label,s in [('Off',off),('On',on)]:
        for v in NAMES:
            m=s['by_variant'][v]
            rows.append([label,NAMES[v],number(m['controls_absolute_pp']['no_news']['estimate']),number(m['controls_absolute_pp']['repeated_news']['estimate']),number(m['baseline_oracle_error_pp']['estimate']),number(m['new_oracle_error_pp']['estimate'])])
    put('exp1_fresh_controls.tex',table(['Thinking','Wording','No news','Repeated','Baseline error','Updated error'],rows,'llrrrr'))
    rows=[]
    for v in NAMES:
        interaction=contrast['interaction_relative_to_base_pp'].get(v)
        rows.append([NAMES[v],interval(contrast['paired_reversal_on_minus_off_pp'][v]),'--' if interaction is None else interval(interaction)])
    put('exp1_fresh_reasoning.tex',table(['Wording','On minus off, pp','Contrast minus base, pp'],rows,'lrr'))
    rows=[]
    for label,s in [('Off',off),('On',on)]:
        for v in ('names','paraphrase','resample'):
            m=s['edits'][v];rows.append([label,NAMES[v],interval(m['absolute_update_discrepancy_pp']),interval(m['excess_over_resampling_pp']),count(m['joint_pair_correct'])])
    put('exp1_fresh_edits.tex',table(['Thinking','Edit','Discrepancy, pp','Excess over resample, pp','Both pairs correct'],rows,'llrrr'))
    rows=[]
    for label,s in [('Qwen3 off',data['disabled_frontier_subset.json']),('Qwen3 on',data['enabled_frontier_subset.json']),('GPT-5.6',gpt)]:
        for v in ('base','paraphrase'):
            m=s['by_variant'][v];rows.append([label,NAMES[v],count(m['sign_accuracy']),paired(m['paired_reversal']),interval(m['broken_absolute_pp']),str(m['broken_absolute_pp']['n_observed'])+'/24'])
    put('exp1_fresh_frontier.tex',table(['Deployment','Wording','Signs','Pairs [95\\% CI]','Broken $|\\Delta|$, pp','Observed'],rows,'llrrrr'))
    rows=[]
    for label,s in [('Off',off),('On',on)]:
        for mechanism,m in s['by_variant']['base']['by_mechanism'].items():rows.append([label,mechanism.replace('_',' '),f"{int(m['success'])}/{m['planned']}"])
    put('exp1_fresh_mechanisms.tex',table(['Thinking','Mechanism class','Base pairs correct'],rows,'llr'))
    coverage='; '.join(f"thinking {label}: {s['coverage'].get('ok',0):,}/{s['coverage']['planned']:,}" for label,s in [('off',off),('on',on)])
    base_effect=contrast['paired_reversal_on_minus_off_pp']['base'];loo=list(contrast['leave_one_mechanism_out_base_difference_pp'].values())
    stable={label:[NAMES[v] for v,m in s['by_variant'].items() if m['broken_stability']['status']=='criterion_met'] for label,s in [('off',off),('on',on)]}
    stability=' '.join('Thinking '+label+': '+(', '.join(vs)+' meet the stability criterion.' if vs else 'no wording variant meets the stability criterion.') for label,vs in stable.items())
    text=r'''\subsection{Fresh finite-rule context reversal: reasoning and wording}
\label{app:exp1-fresh-factorial}

The development studies above confound material version with the reasoning
setting. We therefore froze a new diagnostic before collecting any target-model
responses: 80 new finite-rule instances, ten in each of eight mechanism classes,
with the same name, wording, and resampling controls in both Qwen3-32B thinking
settings. The earlier qualitative 80-concept blueprint was replaced by these
fully specified rules. They are parameterized instances of eight shared classes
and one outcome wrapper, not 80 independent mechanisms or a representative
natural-news sample. None of the original development outputs enters this cohort.

\paragraph{Explicit mechanisms and residual uncertainty.}
The classes cover detector calibration, prerequisite timing, component quorums,
procedural waivers, proportional brokerage, directed reachability, bounded
feedback, and dependence among stored reports. Each of three independently
specified channels has private variables and primitive draws, including a binary
state $X$. An audit revises $P(X=1)$ from $1/4$ to $3/4$ or conversely, without
revealing a realized state. Other conditional rules remain fixed. In calibration
and report-dependence cases, the stated $X$ probabilities already condition on
stored observations; the observation is not applied twice.

The project completes if a final inspection succeeds and either a reserve team
succeeds or the ordinary channel fires without the cancellation channel firing.
The independent inspection and reserve probabilities are $4/5$ and $1/5$.
Writing $a,b$ for the ordinary and cancellation firing probabilities gives
\[
 P(\mathrm{complete})=\tfrac45\{\tfrac15+\tfrac45a(1-b)\}.
\]
The archive channel has neither an outcome pathway nor an information link to
other channels. Thus identical news can raise, lower, or leave the forecast
unchanged solely through actor-to-channel bindings. All channel descriptions
appear in every context, with identical word and punctuation inventories.
Half the questions ask about noncompletion, and rising/falling audits are
balanced. Exact enumeration or Bayes calculations, cross-checked in a second
implementation, establish every sign and retain both outcomes after every update.
The question probability stays between $.16$ and $.84$; no normative forecast is
forced to an endpoint. Component probabilities are supplied, but the model's
initial forecast is elicited separately in each context and never replaced by
the oracle probability.

For example, in the first calibration instance the same audit changes a named
channel's $P(X=1)$ from $1/4$ to $3/4$. Its oracle baseline is $.3079$ in each
context. The new probabilities are $.3253$, $.2773$, and $.3079$ when that actor
operates the ordinary, cancellation, or archive channel, respectively. These
oracle values are analysis quantities, not supplied model forecasts.

\paragraph{Screening and its limits.}
No new human validation was obtained. Separate Llama-3.1-8B and Qwen3-14B
checkpoints screened all 720 distinct packets (80 instances, three wording
versions, three contexts), without intended labels, sibling contexts, or target
outputs. Same-text resamples reuse an identical packet. The final screens had
poor directional agreement: 224/720 and 182/720 planned judgments matched the
rule-derived labels; 44 Llama completions were invalid. These screens do not
validate the labels. Every valid endpoint judgment retained both outcomes.
Author adjudication repaired two actual text defects in earlier drafts: an
over-broad independence statement and reuse of ``inspection'' for two different
events. The final version explicitly distinguishes private variables and a local
approval coin from the final inspection. All versions and judgments are retained.
No instance was removed because a reviewer answered incorrectly. Label authority
is the disclosed formal specification and author cross-check, not model consensus
or independent human judgment. Qwen3-14B also shares a model family with the target.

\paragraph{Frozen paired collection.}
Each instance has base, name-only, paraphrase, and same-text resample variants.
Each variant contains three relational contexts and one own-context baseline
with three independent new-news, no-news, and already-known-information branches.
This gives 3,840 planned records per local reasoning mode. Qwen3-32B uses the same
checkpoint, temperature $.7$, top-$p=1$, request seeds, unconstrained decoding,
8,192-output-token allowance, and 16,384-token context in both modes; its thinking
chat prefix is the intended manipulation. A final probability object may have
one Markdown fence in either mode; enabled reasoning must close first. All
received invalid/truncated responses and durably recorded interrupted requests
are terminal failures, without resampling. This is a comparison of reasoning
settings under these deployments, not a general causal decomposition of capability.

Paired reversal requires both signed updates to be strictly correct; zero
movement and all invalid or missing responses fail the binary endpoint. Magnitudes
use observed values with missingness shown. The prespecified broken-link criterion
requires a signed-mean 90\% interval strictly inside $\pm2$ points and an upper
95\% mean-absolute bound below 2 points; any missing planned broken measurement
makes it indeterminate. Intervals use 2,000 whole-instance bootstrap draws,
keeping contexts, wordings, and modes paired. At all-correct or all-zero cells,
these empirical intervals collapse; this does not establish population certainty.
They are descriptive for the authored corpus; shared mechanism/template dependence limits generalization. Numerical
calculation and rounding remain possible sources of error even when the rules
do not force endpoint forecasts. The released scoring also reports accuracy in prespecified
oracle-effect bins below 1, 1--2, and at least 2 percentage points.
'''
    text+='\nValid final probabilities: '+coverage+'.\n'
    text+=block('Fresh local evaluation. Signs are correct updates out of 160 planned signed contexts per wording; pairs are correct reversals out of 80, followed by a 95\\% interval in percent. Broken movement is mean absolute change with its 95\\% interval, in percentage points; the final column reports observed broken measurements.','tab:exp1-fresh-performance','exp1_fresh_performance')
    text+='\nThe base-wording thinking contrast is '+interval(base_effect)+' percentage points. The leave-one-mechanism-out contrasts range from '+number(min(loo),1)+' to '+number(max(loo),1)+' points. '+stability+'\n'
    text+=block('Matched thinking contrasts in paired reversal. The final column subtracts the base-wording contrast; all contexts and modes remain paired by parent instance. Brackets are descriptive 95\\% intervals.','tab:exp1-fresh-reasoning','exp1_fresh_reasoning')
    text+=block('Fresh controls and oracle discrepancy, all in percentage points. No-news and repeated-information columns give mean absolute movement across the three contexts. Baseline and updated error are mean absolute deviations from the exact rule-based probabilities, not calibration against realized real-world outcomes.','tab:exp1-fresh-controls','exp1_fresh_controls')
    text+=block('Wording discrepancies relative to base, across all three contexts. Excess subtracts the identical-text resampling discrepancy. Joint correctness requires both the base and edited reversal pair to be correct out of 80. Intervals crossing zero do not establish equivalence; agreement alone does not establish correctness.','tab:exp1-fresh-edits','exp1_fresh_edits')
    text+=r'''
\paragraph{One frozen frontier evaluation.}
A prospectively fixed balanced subset contains three instances per mechanism
class (24 parents), with base and paraphrase wording. It has 576 planned records,
not 48 independent families. GPT-5.6 Sol uses low reasoning, a 2,048-output-token
allowance, strict probability JSON, no tools, and zero automatic retries. Batch
baselines precede a separate update batch using only their own saved priors.
The complete cohort was reserved within the user's budget before generation;
provider token-count receipts and the persistent ledger are archived. No released
reservation financed additional target cases. Local rows below use exactly this
same subset. Frontier/local contrasts remain descriptive because deployments and
reasoning allowances differ.
'''
    text+='\nGPT-5.6 returned '+str(gpt['coverage'].get('ok',0))+'/576 valid final probabilities.\n'
    text+=block('Matched 24-parent subset, with all planned failures retained. Signs are out of 48; paired reversals out of 24. Pair intervals are percentages; broken movement and its interval are percentage points.','tab:exp1-fresh-frontier','exp1_fresh_frontier')
    text+=block('Base-wording paired reversal by mechanism class. The equal-class macro average equals the overall rate because all classes contain ten instances. These repeated parameterizations do not supply 80 independent mechanism replications.','tab:exp1-fresh-mechanisms','exp1_fresh_mechanisms')
    # Triple-level scoring and unconditional accounting, computed in process from
    # the frozen plan and the saved records so they cannot go stale against the
    # tables they produce.
    from exp1_prospective.context_reversal import analyze_triples as tri, analyze_accounting as acc
    units=tri.load_units();triples=tri.build_triples(units)
    shortcuts=tri.score_shortcuts(triples)
    arms=[a for a in (tri.score_arm(label,pats,units,triples) for _,label,pats in tri.ARMS) if a['items_valid']]
    books=[b for b in (acc.score(label,pats,plan) for label,pats,plan in acc.ARMS) if b['categories'].get('ok')]
    labels={'no_change':'Predict no change','evidence_direction':'Follow the evidence direction','always_up':'Always revise upward','best_constant_with_oracle':'Best constant, given every oracle'}
    put('exp1_fresh_bound.tex',table(['Predictor','Items, direction','Triples, all three'],
        [[labels[k],f"{v['item_sign_correct']}/{v['items']}",f"{v['triple_sign_all_correct']}/{v['triples']}"] for k,v in shortcuts.items()]
        +[[r'\emph{Analytic bound}',r'$\leq 1/3$','$0$']],'lrr'))
    def mean_pp(a,metric):return f"{next(x for x in a['errors'] if x['metric']==metric)['mean_pp']:.3f}"
    put('exp1_fresh_triples.tex',table(['System','Triples','Items, direction','Items, exact','Posterior error, pp'],
        [[a['arm'],f"{a['triple_sign_unconditional']}/{a['triples_planned']}",f"{a['item_sign_correct']}/{a['items_planned']}",
          f"{a['item_exact_correct']}/{a['items_planned']}",mean_pp(a,'update absolute error')] for a in arms],'lrrrr'))
    put('exp1_fresh_accounting.tex',table(['System','Planned','Valid','Model','Instrument','Cascade'],
        [[b['arm'],str(b['planned_records']),str(b['categories'].get('ok',0)),str(b['by_attribution'].get('model',0)),
          str(b['by_attribution'].get('instrument',0)),str(b['by_attribution'].get('cascade',0))] for b in books],'lrrrrr'))
    text+=r'''
\paragraph{What a relation-blind predictor can reach.}
The unit of evidence is the triple: one parent instance and wording, with the
evidence text, the prior and the named actor identical across three stated
relations. All '''+str(len(triples))+r''' triples were verified to hold those
three constant, to cover the three relations, and to have three distinct correct
posteriors. The last property bounds every shortcut: a predictor reading only the
evidence, the prior and the actor is constant within a triple, so it matches at
most one of three items and never a whole triple. The bound is arithmetic, not
empirical, and the predictors below only exhibit it -- including an adversary
allowed to choose the best constant revision per triple knowing every oracle.
'''
    text+=block('Predictors that read only the evidence, the prior and the named actor. Each is constant within a triple, so none can exceed one third of items or win a single triple.','tab:exp1-fresh-bound','exp1_fresh_bound')
    text+=r'''
\paragraph{Exact agreement and the reasoning contrast.}
Scoring exact posterior agreement rather than revision direction separates
systems that a binary criterion cannot: a guess supplies a sign, not a value.
Denominators are unconditional, counting refusals, truncations and updates
blocked by their own failed baseline as failures.
'''
    text+=block('Triple and item accuracy against the exact oracles, with unconditional denominators. Exact agreement is to four decimals, the granularity at which the oracles are stated.','tab:exp1-fresh-triples','exp1_fresh_triples')
    text+=block('Every planned call, classified by attribution. A refusal or unparseable answer is the system\'s; a truncation at the output budget or a request killed mid-batch is ours; a record blocked by its own failed baseline is neither.','tab:exp1-fresh-accounting','exp1_fresh_accounting')
    text+=r'''
\paragraph{Interpretive boundary and audit.}
These data address the original relevance shortcut under explicit finite rules:
news-only, full-word, and word-bigram classifiers all achieve $1/3$ direction
accuracy and $.50$ balanced relevance accuracy in five-fold evaluation holding
out entire parent instances and all their variants. This does not eliminate every
possible structural heuristic. The new diagnostic does not retroactively validate
the original packet relevance comparison, establish natural-news coverage, or
replace the limitations of model screening. Raw completions, context-specific
priors, effective chat hashes, token counts, failure denominators, and primary
statistics were checked before these tables were generated. The artifact root is
\path{exp1_prospective/context_reversal/results/fresh_evaluation_v1}; the immutable
design is under \path{exp1_prospective/context_reversal/data/fresh_evaluation_v1/frozen_v1}.
'''
    put('experiment1_fresh_context_appendix.tex',text)
    manifest={'status':'audited_draft','created_at':c.utc_now(),'technical_audit':artifact(RESULTS/'technical_audit.json'),'inputs':audit['summary_artifacts'],'files':{p.name:c.file_sha256(p) for p in DRAFT.glob('*.tex')},'writer':artifact(Path(__file__))};put('draft_manifest.json',json.dumps(manifest,indent=2)+'\n');return manifest

if __name__=='__main__':print(json.dumps(write(),indent=2))
