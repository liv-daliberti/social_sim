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
    coverage=('Thinking disabled yields '+f"{off['coverage'].get('ok',0):,}"+' valid probabilities out of '+f"{off['coverage']['planned']:,}"+' observations; thinking enabled yields '+f"{on['coverage'].get('ok',0):,}"+' out of '+f"{on['coverage']['planned']:,}"+'.')
    base_effect=contrast['paired_reversal_on_minus_off_pp']['base'];loo=list(contrast['leave_one_mechanism_out_base_difference_pp'].values())
    stability_parts=[]
    for label,summary in [('disabled',off),('enabled',on)]:
        groups={status:[NAMES[v].lower() for v,m in summary['by_variant'].items() if m['broken_stability']['status']==status]
                for status in ('criterion_met','criterion_not_met','indeterminate')}
        if len(groups['criterion_not_met'])==len(NAMES):
            stability_parts.append('With thinking '+label+', no wording meets the stability criterion.')
        else:
            if groups['criterion_met']:
                stability_parts.append('With thinking '+label+', stability is established for the '+', '.join(groups['criterion_met'])+' wording.')
            if groups['criterion_not_met']:
                stability_parts.append('The criterion is not met for '+', '.join(groups['criterion_not_met'])+' wording.')
            if groups['indeterminate']:
                stability_parts.append('Stability remains indeterminate for the other variants because each has missing broken-link observations.')
    stability=' '.join(stability_parts)
    text=r'''\subsection{Finite-rule context reversal: reasoning and wording}
\label{app:exp1-fresh-factorial}

This task crosses reasoning mode with changes in names and wording on 80
scenarios governed by explicit probabilistic rules. Ten scenarios instantiate
each of eight mechanism classes. Both Qwen3-32B reasoning modes receive the
same scenarios and wording variants, allowing their performance to be compared
within each condition. The scenarios share eight mechanisms and a common
outcome rule; their number does not imply 80 independent mechanisms or
representative coverage of real-world news.

\paragraph{Mechanisms and outcome probabilities.}
The mechanism classes concern detector calibration, prerequisite timing,
component quorums, procedural waivers, proportional brokerage, directed
reachability, bounded feedback, and dependence among stored reports. Each
scenario describes three independent channels, each with its own latent variables and random
draws, including a binary state $X$. A report revises $P(X=1)$ from $1/4$ to
$3/4$, or conversely, without revealing the realized state. All other conditional
rules remain fixed. In calibration and report-dependence scenarios, these
probabilities already incorporate stored observations, which must not be counted
a second time.

The project completes if a final inspection succeeds and either a reserve team
succeeds or the ordinary channel activates without the cancellation channel
activating. Inspection and reserve success are independent, with probabilities
$4/5$ and $1/5$. If $a$ and $b$ are the activation probabilities of the ordinary
and cancellation channels, respectively, then
\[
 P(\mathrm{complete})=\tfrac45\left\{\tfrac15+\tfrac45a(1-b)\right\}.
\]
The archive channel neither affects the outcome nor provides information about
the other channels. Assigning the actor named in the report to different channels
therefore makes identical news increase, decrease, or leave unchanged the outcome
probability. All channel descriptions appear in every context, with identical
word and punctuation inventories. Half the questions ask about noncompletion;
reports that raise and lower $P(X=1)$ are balanced.

Exact enumeration or Bayesian calculation determines the correct probabilities,
with a second implementation used to check them. Probabilities remain between
$0.16$ and $0.84$, so both outcomes remain possible. For one calibration scenario,
a report raising $P(X=1)$ from $1/4$ to $3/4$ changes the outcome probability
from $0.3079$ to $0.3253$, $0.2773$, or $0.3079$, depending on whether the actor
operates the ordinary, cancellation, or archive channel. These exact values are
scoring references. Models receive the component probabilities and produce their
own initial forecast separately in each context.

\paragraph{Model review and label validity.}
Llama-3.1-8B and Qwen3-14B reviewed all 720 distinct packets: 80 scenarios,
three wording variants, and three contexts. Each reviewer saw a packet without
its intended label, alternative contexts, or target-model responses. Their
directional judgments matched the rule-derived labels in 224/720 and 182/720
cases, respectively; 44 Llama responses were invalid. Every valid judgment of
whether both outcomes remained possible agreed that they did. No scenarios
were excluded on the basis of reviewer disagreement.

The low directional agreement provides little support for interpretability.
Correct labels derive from the formal rules and the authors' computational
cross-check, rather than reviewer consensus. The text distinguishes each
channel's private variables and local approval event from the independent final
inspection. The materials have not undergone independent human validation,
and Qwen3-14B shares a model family with the evaluated Qwen3-32B.

\paragraph{Paired reasoning conditions.}
Each scenario has a base description, a version with renamed entities, a
paraphrase, and an identical-text resample. Each version has three relational
contexts, each with its own baseline and three independent update branches:
new news, no news, and repeated prior information. This yields 3,840 baseline
and update observations per reasoning mode. Both modes use the same Qwen3-32B
checkpoint, matched seeds, temperature $0.7$, top-$p=1$, unconstrained decoding,
an 8,192-token output limit, and a 16,384-token context limit. The prompt prefix
enables or disables thinking. Both modes accept a final probability JSON object
with at most one enclosing Markdown fence; any reasoning segment must be complete.
Invalid, truncated, and interrupted requests count as failures and are not
resampled. The comparison concerns these inference settings on this scenario set.

\clearpage
\paragraph{Scoring and uncertainty.}
Paired reversal requires both signed updates to move in the correct direction.
Zero movement, invalid responses, and missing responses count as failures.
Magnitude estimates use observed values, with coverage reported explicitly.
Broken-link stability requires a 90\% interval for mean signed change strictly
within $\pm2$~pp and a 95\% upper confidence bound for mean absolute change below
2~pp. Any missing broken-link observation makes the criterion indeterminate.
Intervals use 2,000 whole-scenario bootstrap draws, keeping contexts, wordings,
and reasoning modes paired. When every observation has the same value, empirical
intervals can collapse to a point; this does not imply population certainty.
Shared mechanisms and templates further limit generalization beyond these
scenarios. Calculation and rounding can also cause errors, so the analysis
separately reports accuracy for exact probability changes below 1~pp, from 1 to
less than 2~pp, and at least 2~pp.
'''
    text+='\n'+coverage+'\n'
    text+=block(r'\textbf{Qwen3-32B context reversal under explicit probabilistic rules.} Signs count correct directions out of 160 signed contexts per wording; pairs count successful reversals out of 80, with descriptive 95\% intervals in percent. Broken-link movement is mean absolute change with its 95\% interval, in percentage points. Observed counts show the available broken-link measurements.','tab:exp1-fresh-performance','exp1_fresh_performance')
    text+='\nEnabling thinking increases paired reversal under the base wording by '+interval(base_effect)+' percentage points (95\\% interval in brackets). Omitting one mechanism class at a time gives contrasts from '+number(min(loo),1)+' to '+number(max(loo),1)+' points. '+stability+'\n'
    text+=block(r'\textbf{Effect of reasoning mode on paired reversal.} Differences are in percentage points. The final column compares each reasoning contrast with the base-wording contrast. All contexts and modes remain paired by scenario; brackets give descriptive 95\% intervals.','tab:exp1-fresh-reasoning','exp1_fresh_reasoning')
    text+=block(r'\textbf{Control responses and error relative to exact probabilities.} All values are in percentage points and use observed responses. No-news and repeated-information columns give mean absolute movement across the three contexts. Baseline and updated error are mean absolute deviations from the exact rule-based probabilities; they do not measure calibration against realized outcomes.','tab:exp1-fresh-controls','exp1_fresh_controls')
    text+='\n\\clearpage\n'
    text+=block(r'\textbf{Sensitivity of probability revisions to names and wording.} Discrepancy is the mean absolute difference between the new-news revision for each variant and the base revision across the three contexts. Excess subtracts the corresponding identical-text resampling discrepancy. Both are in percentage points, with descriptive 95\% intervals. Joint correctness requires successful signed reversal in both versions, out of 80 scenarios. Intervals containing zero do not establish equivalence.','tab:exp1-fresh-edits','exp1_fresh_edits')
    text+=r'''
\paragraph{GPT-5.6 comparison on matched scenarios.}
A balanced subset contains 24 scenarios, three per mechanism class, with base
and paraphrased wording. Each wording has three contexts, each with a baseline
and three update branches, yielding 576 observations. GPT-5.6 uses low
reasoning effort, a 2,048-token output limit, structured probability JSON, no
tools, and no automatic retries. Updates receive only their own context's
baseline. The Qwen3 rows in Table~\ref{tab:exp1-fresh-frontier} use the same
subset. Differences in inference settings limit attribution of performance
differences to model identity alone.
'''
    text+='\nGPT-5.6 returns '+str(gpt['coverage'].get('ok',0))+'/576 valid probabilities.\n'
    text+=block(r'\textbf{Context reversal on the same 24 scenarios.} Each wording contributes 48 signed updates and 24 reversal pairs. All trials remain in the binary denominators. Pair intervals are percentages; broken-link movement and its interval are in percentage points. Observed counts report available broken-link measurements.','tab:exp1-fresh-frontier','exp1_fresh_frontier')
    text+=block(r'\textbf{Qwen3-32B paired reversal by mechanism class under base wording.} Each class contains ten scenarios, so the unweighted average of class accuracies equals overall accuracy. The 80 scenarios instantiate eight recurring mechanisms.','tab:exp1-fresh-mechanisms','exp1_fresh_mechanisms')
    from exp1_prospective.context_reversal import analyze_triples as tri, analyze_accounting as acc
    units=tri.load_units();triples=tri.build_triples(units)
    shortcuts=tri.score_shortcuts(triples)
    arms=[a for a in (tri.score_arm(label,pats,units,triples) for _,label,pats in tri.ARMS) if a['items_valid']]
    books=[b for b in (acc.score(label,pats,plan) for label,pats,plan in acc.ARMS) if b['categories'].get('ok')]
    labels={'no_change':'Predict no change','evidence_direction':'Follow the evidence direction','always_up':'Always revise upward','best_constant_with_oracle':'Best oracle-derived revision'}
    tiny_change=tri.EXACT/2
    tiny_correct=sum(sum(abs(tiny_change)<=tri.EXACT if m['expected_sign']==0 else tiny_change*m['expected_sign']>0 for m in members) for members in triples.values())
    gaps=[abs(tri.frac(a['oracle_update'])-tri.frac(b['oracle_update'])) for members in triples.values() for i,a in enumerate(members) for b in members[i+1:]]
    if min(gaps)<=2*tri.EXACT:raise ValueError('Exact-match tolerance intervals overlap; revise the analytic bound')
    put('exp1_fresh_bound.tex',table(['Constant-revision predictor','Items, direction','Triples, all three'],
        [[labels[k],f"{v['item_sign_correct']}/{v['items']}",f"{v['triple_sign_all_correct']}/{v['triples']}"] for k,v in shortcuts.items()]
        +[[r'Revise upward by $5\times10^{-5}$',f'{tiny_correct}/{3*len(triples)}',f'0/{len(triples)}'],
          [r'\emph{Directional upper bound}',r'$2/3$','$0$']],'lrr'))
    def mean_pp(a,metric):return f"{next(x for x in a['errors'] if x['metric']==metric)['mean_pp']:.3f}"
    put('exp1_fresh_triples.tex',table(['System','Triples','Items, direction','Items, exact','Posterior error, pp'],
        [[a['arm'],f"{a['triple_sign_unconditional']}/{a['triples_planned']}",f"{a['item_sign_correct']}/{a['items_planned']}",
          f"{a['item_exact_correct']}/{a['items_planned']}",mean_pp(a,'update absolute error')] for a in arms],'lrrrr'))
    put('exp1_fresh_accounting.tex',table(['System','Expected','Valid','Response','Limit','Dependent'],
        [[b['arm'],str(b['planned_records']),str(b['categories'].get('ok',0)),str(b['by_attribution'].get('model',0)),
          str(b['by_attribution'].get('instrument',0)),str(b['by_attribution'].get('cascade',0))] for b in books],'lrrrrr'))
    text+=r'''
\clearpage
\paragraph{Bounds for predictors that ignore the relation.}
A triple comprises the three contexts for one scenario and wording. Across all
'''+str(len(triples))+r''' triples, the news, named actor, and exact baseline
probability are identical within each triple. Models' elicited baselines may differ; the shared reference baseline is rule-derived.
A predictor that returns a constant revision within a triple cannot satisfy both
the positive and negative targets, so it cannot answer an entire triple correctly.

The directional scoring rule accepts any correctly signed change and treats
$|\Delta p|\leq10^{-4}$ as unchanged in the irrelevant context. A tiny positive
revision therefore satisfies both the positive and irrelevant targets, giving a
sharp directional bound of $2/3$, rather than $1/3$. For posterior agreement
within $10^{-4}$, the three reference probabilities are separated by more than
$2\times10^{-4}$; a constant posterior can match at most one, giving a separate
$1/3$ bound. Both bounds concern constant outputs within a triple, not arbitrary
predictors that exploit additional context.
'''
    text+=block(r'\textbf{Direction scores for constant revisions within each triple.} The oracle-derived predictor selects among the three reference revisions. A small positive revision attains the directional upper bound because the unchanged tolerance overlaps the positive target. No constant revision satisfies all three directions.','tab:exp1-fresh-bound','exp1_fresh_bound')
    text+=r'''
\paragraph{Direction, posterior accuracy, and missing responses.}
Table~\ref{tab:exp1-fresh-triples} separates successful direction judgments from
posteriors within $10^{-4}$ of the exact probability (0.01~pp). Triple success
requires all three directions to be correct. Refusals, invalid answers, and
updates unavailable because their baseline failed remain in these denominators;
mean posterior error uses only valid baseline--update pairs. Claude Opus~5 is
also evaluated on the 24-scenario subset, with low effort, a 1,024-token output
limit, and structured probability output. Its incomplete coverage is reported
explicitly in Table~\ref{tab:exp1-fresh-accounting}.
'''
    text+=block(r'\textbf{Direction and posterior accuracy.} Counts include all trials; posterior agreement uses an absolute tolerance of $10^{-4}$. Mean posterior error uses observed pairs, so low error among valid responses can coexist with low overall success. Hosted models use 48 triples; local models use 320.','tab:exp1-fresh-triples','exp1_fresh_triples')
    text+=block(r'\textbf{Response completeness and failure categories.} Response denotes refusals or invalid answers; Limit denotes output-limit truncation or interrupted requests; Dependent denotes unavailable updates or undispatched requests. Every expected observation remains in the binary denominators.','tab:exp1-fresh-accounting','exp1_fresh_accounting')
    text+=r'''
\paragraph{Scope of inference.}
News-only, full-word, and word-bigram classifiers achieve $1/3$ direction
accuracy and $.50$ balanced relevance accuracy when five-fold evaluation holds
out whole scenarios and their variants. Other shortcuts remain possible; shared
templates and the absence of independent human validation limit generalization
to natural news.
'''
    put('experiment1_fresh_context_appendix.tex',text)
    manifest={'status':'audited_draft','created_at':c.utc_now(),'technical_audit':artifact(RESULTS/'technical_audit.json'),'inputs':audit['summary_artifacts'],'files':{p.name:c.file_sha256(p) for p in DRAFT.glob('*.tex')},'writer':artifact(Path(__file__))};put('draft_manifest.json',json.dumps(manifest,indent=2)+'\n');return manifest

if __name__=='__main__':print(json.dumps(write(),indent=2))
