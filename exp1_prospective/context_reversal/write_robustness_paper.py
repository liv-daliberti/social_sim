#!/usr/bin/env python3
"""Write a separate, audited three-model robustness appendix draft; never the paper.

Refuses output until all three summaries are complete and their independent
technical audits pass with exact source hashes. No inference or LaTeX build.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import run_local as common

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]
RUN = SOURCE / 'runs/robustness_development_v2'
MODELS = {'qwen2_5_72b_instruct': 'Qwen2.5-72B', 'llama3_1_70b_instruct': 'Llama-3.1-70B', 'qwen3_32b': 'Qwen3-32B'}
VARIANTS = {'repaired_base': 'Repaired base', 'name_only': 'Renamed', 'paraphrase': 'Paraphrased', 'resample': 'Same-text resample'}
BOOTSTRAP = {'unit': 'parent_family_id', 'draws': 2000, 'seed': 20260921, 'variants_are_not_independent_families': True}
FILES = ('exp1_context_robustness_v2.tex', 'exp1_context_robustness_v2_performance.tex',
         'exp1_context_robustness_v2_edits.tex', 'exp1_context_robustness_v2_controls.tex', 'draft_manifest.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def artifact(path):
    return {'path': str(path.resolve()), 'sha256': common.file_sha256(path)}


def metric(value, n, *, binary=False):
    require(value['n_planned'] == n and value['n_observed'] + value['n_missing'] == n, 'Metric denominator/missingness changed')
    require(value['bootstrap_draws_requested'] == 2000 and value['interval_status'] in ('descriptive_family_bootstrap', 'insufficient_observed_families', 'undefined_resamples_due_to_missingness'), 'Metric bootstrap definition changed')
    require(value['missing_policy'] == ('counts_as_failure' if binary else 'observed_only_reported_explicitly'), 'Metric missingness policy changed')
    if binary:
        require(type(value['n_success']) is int and 0 <= value['n_success'] <= n
                and math.isclose(value['estimate'], value['n_success'] / n, abs_tol=1e-12), 'Binary count/estimate mismatch')
    if value['estimate'] is not None:
        require(math.isfinite(value['estimate']), 'Nonfinite metric')
    ci = value['ci95']
    require(ci is None or (len(ci) == 2 and all(math.isfinite(v) for v in ci) and ci[0] <= ci[1]), 'Invalid interval')
    return value


def validate_summary(summary, model):
    require(summary.get('schema_version') == 'controlled_robustness_analysis_v1' and summary.get('status') == 'exploratory_development', 'Unexpected summary schema/status')
    require(summary.get('model_key') == model and summary.get('record_complete') is True, f'Incomplete model summary: {model}')
    require(summary.get('n_sampling_families') == 20 and summary.get('n_variants') == 4 and summary.get('n_units') == 320, 'Wrong family/variant/unit coverage')
    require(summary.get('bootstrap') == BOOTSTRAP and summary.get('tolerance_pp') == 2.0, 'Frozen bootstrap/tolerance differs')
    require(summary.get('ignored_other_model_records') == {'probability': 0, 'direction': 0}, 'Unexpected other-model responses')
    for task, n in (('probability', 1280), ('direction', 960)):
        c = summary['coverage'][task]
        require(c['planned'] == c['present'] == n and c['missing'] == 0 and c['valid'] + c['invalid'] == n, 'Summary does not retain every planned response')
    require(set(summary['by_variant']) == set(VARIANTS) and set(summary['invariance_vs_repaired_base']) == set(VARIANTS) - {'repaired_base'}, 'Summary variant coverage changed')
    for value in summary['by_variant'].values():
        for key, n in (('numeric_sign_correct', 40), ('numeric_paired_reversal', 20), ('direction_paired_reversal', 20), ('direction_broken_unchanged', 20)):
            metric(value[key], n, binary=True)
        metric(value['broken_movement']['mean_absolute_pp'], 20)
        metric(value['broken_movement']['signed_mean_pp'], 20)
        require(value['broken_stability']['status'] in ('criterion_met', 'criterion_not_met', 'indeterminate'), 'Unknown stability status')
        require(set(value['controls_by_condition']) == {'no_news', 'repeated_news'}, 'Controls were pooled or omitted')
        for control in value['controls_by_condition'].values():
            metric(control['numeric_within_2pp'], 80, binary=True)
            metric(control['numeric_movement']['mean_absolute_pp'], 80)
            metric(control['direction_unchanged'], 80, binary=True)
    for comparison in summary['invariance_vs_repaired_base'].values():
        metric(comparison['baseline']['change_difference']['mean_absolute_pp'], 80)
        value = comparison['new_news']
        for item in (value['change_difference']['mean_absolute_pp'], value['absolute_difference_minus_resample_pp']):
            metric(item, 80)
        metric(value['within_2pp'], 80, binary=True)
        metric(value['direction_agreement'], 80, binary=True)
        metric(value['both_direction_judgments_correct'], 60, binary=True)


def inputs(submission_path):
    submission = load(submission_path)
    require(set(submission['groups']) == set(MODELS), 'Exactly the three original local checkpoints are required')
    protocol = Path(submission['protocol_path'])
    require(common.file_sha256(protocol) == submission['protocol_sha256'], 'Protocol differs from its submission freeze')
    summaries, audits, sources = {}, {}, {str(submission_path.resolve()): common.file_sha256(submission_path), str(protocol.resolve()): common.file_sha256(protocol)}
    for model, job in submission['groups'].items():
        path = Path(job['results_dir']) / 'summary.json'
        audit_path = path.with_name('independent_audit.json')
        require(path.is_file() and audit_path.is_file(), f'Completed summary and independent audit required for {model}')
        summary, audit = load(path), load(audit_path)
        validate_summary(summary, model)
        require(audit.get('schema_version') == 'independent_robustness_audit_v1' and audit.get('audit_pass') is True
                and audit.get('model_key') == model and audit.get('errors') == {}, f'Independent technical audit has not passed for {model}')
        parents = sorted({r['parent_family_id'] for r in summary['trial_rows']})
        require(len(parents) == 20 and audit.get('sampling_families') == parents, 'Audit must cover all 20 parent families')
        require(audit.get('audit_code_sha256') == common.file_sha256(SOURCE / 'audit_robustness_results.py'), 'Independent audit code changed')
        sources[str((SOURCE / 'audit_robustness_results.py').resolve())] = audit['audit_code_sha256']
        frozen_sources = audit['sources_sha256']
        require(frozen_sources.get(str(path.resolve())) == common.file_sha256(path), 'Audit does not cover the exact summary')
        for name, digest in frozen_sources.items():
            source = Path(name)
            require(source.is_file() and common.file_sha256(source) == digest, f'Audited source changed: {source.name}')
            sources[str(source.resolve())] = digest
        for key, spec in summary['provenance'].items():
            source = Path(spec['path'])
            require(source.is_file() and common.file_sha256(source) == spec['sha256'], f'Summary provenance changed: {key}')
            if key == 'analysis_code':
                require(spec['sha256'] == submission['code_sha256']['analyze_robustness.py'], 'Analysis code differs from frozen submission')
                sources[str(source.resolve())] = spec['sha256']
            else:
                require(frozen_sources.get(str(source.resolve())) == spec['sha256'], f'Audit omits summary source: {key}')
            if key in ('probability_responses', 'direction_responses'):
                require(source.resolve() == Path(job[key]).resolve(), 'Submission and summary response paths differ')
        summaries[model], audits[model] = summary, audit
        sources[str(audit_path.resolve())] = common.file_sha256(audit_path)
    lexical_path = Path(next(iter(submission['groups'].values()))['results_dir']).parent / 'material_audit/lexical_audit.json'
    lexical = load(lexical_path)
    require(lexical.get('schema_version') == 'controlled_robustness_lexical_audit_v1' and lexical.get('n_parent_families') == 20
            and lexical.get('n_variants') == 4 and lexical.get('split_unit') == 'parent_family_id_all_variants_together', 'Lexical audit split/coverage differs')
    design = lexical['provenance']['design']
    require(common.file_sha256(Path(design['path'])) == design['sha256'], 'Lexical audit design changed')
    require(all(s['provenance']['direction_design'] == design for s in summaries.values()), 'Lexical and inference designs differ')
    require(lexical['provenance']['code_sha256'] == submission['code_sha256']['audit_robustness_lexical.py'], 'Lexical audit code differs from frozen submission')
    require(set(lexical['held_out_baselines']) == {'news_unigram', 'full_unigram', 'full_bigram', 'full_character'}, 'Lexical baselines omitted')
    sources[str(lexical_path.resolve())] = common.file_sha256(lexical_path)
    return summaries, lexical, sources


def count(value):
    return f"{value['n_success']}/{value['n_planned']}"


def magnitude(value, *, interval=True):
    if value['estimate'] is None:
        text = '--'
    else:
        v = 0.0 if abs(value['estimate']) < .005 else value['estimate']
        text = f'{v:.2f}'
        if interval and value['ci95'] is not None:
            text += f" $[{value['ci95'][0]:.2f},\\,{value['ci95'][1]:.2f}]$"
    if value['n_observed'] != value['n_planned']:
        text += f" ({value['n_observed']}/{value['n_planned']} obs.)"
    return text


def table(columns, rows):
    return '\n'.join([r'\begin{tabular}{@{}ll' + 'c' * (len(columns) - 2) + '@{}}', r'\toprule',
                      ' & '.join(columns) + r' \\', r'\midrule', *[' & '.join(row) + r' \\' for row in rows],
                      r'\bottomrule', r'\end{tabular}', ''])


def render(summaries, lexical):
    performance, edits, controls = [], [], []
    for model, label in MODELS.items():
        summary = summaries[model]
        for variant, name in VARIANTS.items():
            value = summary['by_variant'][variant]
            performance.append([label, name, count(value['numeric_sign_correct']), count(value['numeric_paired_reversal']),
                magnitude(value['broken_movement']['mean_absolute_pp']), count(value['direction_paired_reversal']), count(value['direction_broken_unchanged'])])
            control_row = [label, name]
            for condition in ('no_news', 'repeated_news'):
                c = value['controls_by_condition'][condition]
                control_row += [count(c['numeric_within_2pp']), magnitude(c['numeric_movement']['mean_absolute_pp'], interval=False), count(c['direction_unchanged'])]
            controls.append(control_row)
            if variant != 'repaired_base':
                compare = summary['invariance_vs_repaired_base'][variant]
                value = compare['new_news']
                edits.append([label, name, magnitude(compare['baseline']['change_difference']['mean_absolute_pp'], interval=False),
                    magnitude(value['change_difference']['mean_absolute_pp'], interval=False), magnitude(value['absolute_difference_minus_resample_pp']),
                    count(value['within_2pp']), count(value['direction_agreement']), count(value['both_direction_judgments_correct'])])
    require((len(performance), len(edits), len(controls)) == (12, 9, 12), 'Draft tables lost a model or variant')
    tables = {
        FILES[1]: table(['Model', 'Version', 'Numeric sign', 'Numeric pair', r'Broken $|\Delta p|$ (pp)', 'Direction pair', 'Broken unchanged'], performance),
        FILES[2]: table(['Model', 'Vs. repaired base', r'Baseline $D_0$', r'Update $D_\Delta$', r'Excess $E_\Delta$ [95\% CI]', r'$D_\Delta\leq2$ pp', 'Direction agree', 'Both correct'], edits),
        FILES[3]: table(['Model', 'Version', r'No news: $\leq2$ pp', r'$|\Delta p|$', 'Unchanged', r'Repeated: $\leq2$ pp', r'$|\Delta p|$', 'Unchanged'], controls),
    }
    coverage = '; '.join(f"{MODELS[m]}: {s['coverage']['probability']['valid']}/1,280 numeric and {s['coverage']['direction']['valid']}/960 categorical" for m,s in summaries.items())
    lexical_order = ('news_unigram', 'full_unigram', 'full_bigram', 'full_character')
    direction_scores = ', '.join(f"{100*lexical['held_out_baselines'][k]['direction_accuracy']:.1f}\\%" for k in lexical_order)
    relevance_scores = ', '.join(f"{100*lexical['held_out_baselines'][k]['relevance_balanced_accuracy']:.1f}\\%" for k in lexical_order)
    stability = '; '.join(MODELS[m] + ': ' + ', '.join(f"{VARIANTS[v]} {s['by_variant'][v]['broken_stability']['status'].replace('_',' ')}" for v in VARIANTS) for m,s in summaries.items())
    base = [summaries[model]['by_variant']['repaired_base'] for model in MODELS]
    base_pairs = ', '.join(count(value['numeric_paired_reversal']) for value in base)
    base_movement = ', '.join(magnitude(value['broken_movement']['mean_absolute_pp'], interval=False) for value in base)
    status_labels = {'criterion_met': 'met', 'criterion_not_met': 'not met', 'indeterminate': 'indeterminate'}
    base_stability = ', '.join(status_labels[value['broken_stability']['status']] for value in base)
    base_results = ('For Qwen2.5-72B, Llama-3.1-70B, and Qwen3-32B, respectively,\n'
                    + 'repaired-base numerical paired reversal is ' + base_pairs + ', and broken-link\n'
                    + 'mean absolute movement is ' + base_movement + ' points. The corresponding\n'
                    + 'interval-based stability statuses are ' + base_stability + '.\n')
    text = r'''\paragraph{Controlled robustness development on the original families.}
A separately versioned development run retains all 20 original parent families,
including previous successes and failures. Each family has a repaired base,
consistent entity renaming, a paraphrase, and a resample with identical visible
text but different request IDs and seeds. Explicit transfer destinations repair
underspecified routes; signed and broken contexts share the same inventory of
positive, negative, and side-channel roles, with actor assignments changed and
row order counterbalanced. These revisions do not replace the original results.
Repair changes the supplied premises and can change elicited priors, so an
old-versus-repaired contrast is not a causal estimate of clarification.

All four versions retain positive, negative, broken, and masked contexts.
Each numerical context receives its own baseline and three independent branches:
new news, no news, and information already present at baseline. The separate
categorical task presents no numerical prior. The three original local
checkpoints use constrained JSON, temperature $0.7$, top-$p=1$, and a 128-token
output limit; Qwen3 thinking is disabled. Each model has all 1,280 numerical and
960 categorical records. Valid outputs are COVERAGE.
Invalid responses remain scored failures, and magnitude estimates use observed
measurements with missingness shown explicitly. The independent audit verifies
execution, provenance, and scoring; it is not independent human validation of
these revised materials.

Table~\ref{tab:exp1-robustness-v2-performance} reports every model and version.
Zero numerical movement fails a signed target. Paired reversal requires both
signed contexts to be correct; categorical unclear fails a determinate target.
All intervals are descriptive, using 2,000 whole-parent-family bootstrap draws
(seed 20260921), carrying contexts, branches, and versions together. There are
20 sampling families; the four versions do not create 80 independent families.

\begin{center}
\begin{minipage}{\linewidth}
\centering
\captionof{table}{Controlled robustness development. Binary entries are
correct/planned counts: 40 signed trials, 20 reversal pairs, and 20 broken-link
judgments per version. Broken movement is mean absolute numerical revision in
percentage points, with its descriptive 95\% family-bootstrap interval.}
\label{tab:exp1-robustness-v2-performance}
\small
\resizebox{\linewidth}{!}{\input{tables/exp1_context_robustness_v2_performance}}
\end{minipage}
\end{center}

BASE_RESULTS
The numerical broken-link stability diagnostic retains the original criterion:
the signed-mean 90\% interval must lie strictly within $\pm2$ points and the
upper 95\% bound on mean absolute movement must be below 2 points, with all
planned measurements observed. Detailed statuses are retained in the source
summaries; a small point estimate alone does not establish stability.

\paragraph{Editing and stochastic variation.}
Let $\Delta p_v$ be a version's new-news update relative to its own context's
baseline. Table~\ref{tab:exp1-robustness-v2-edits} separates mean absolute
baseline discrepancy $D_0$ from mean absolute update discrepancy
$D_\Delta=|\Delta p_v-\Delta p_{\mathrm{base}}|$, each averaged across the
80 planned family--context pairs. The within-2-point count describes individual paired observations.
The excess column averages
$E_\Delta=|\Delta p_v-\Delta p_{\mathrm{base}}|
-|\Delta p_{\mathrm{resample}}-\Delta p_{\mathrm{base}}|$
on common observed measurements. The same-text resample is a stochastic
reference at the same temperature; its excess is zero by construction.
Renaming and paraphrase discrepancies must therefore be read alongside this
reference. A single resample does not isolate a causal effect of editing,
and an interval including zero does not establish equivalence or invariance.

\begin{center}
\begin{minipage}{\linewidth}
\centering
\captionof{table}{Each edited version or same-text resample compared with repaired
base. Numerical discrepancies are in percentage points; brackets give 95\%
parent-family-bootstrap intervals for excess over resampling. Agreement uses
80 planned categorical pairs, including masked new news. Joint correctness uses
60 determinate pairs, excluding masked new news. Stable wrong answers count as
agreement but fail joint correctness. Missing/invalid binary pairs remain failures;
magnitudes are observed only.}
\label{tab:exp1-robustness-v2-edits}
\small
\resizebox{\linewidth}{!}{\input{tables/exp1_context_robustness_v2_edits}}
\end{minipage}
\end{center}

\paragraph{Controls and lexical checks.}
No-news and repeated-information controls remain separate
(Table~\ref{tab:exp1-robustness-v2-controls}); the repeated information was already
present at baseline and is not a second presentation of genuinely new evidence.
Each control has 80 planned contexts per model and version. Their performance
does not substitute for directional or broken-link performance.

\begin{center}
\begin{minipage}{\linewidth}
\centering
\captionof{table}{Separate no-news and repeated-information controls. Within-2-point
and categorical unchanged entries use all 80 planned contexts. Mean absolute
numerical movement is in percentage points; any missing magnitude measurements
are shown explicitly. Full descriptive family-bootstrap intervals are retained
in the source summaries.}
\label{tab:exp1-robustness-v2-controls}
\small
\resizebox{\linewidth}{!}{\input{tables/exp1_context_robustness_v2_controls}}
\end{minipage}
\end{center}

Whole-parent-family-held-out news-unigram, full-text unigram, full-text bigram,
and character-ngram baselines obtain directional accuracies of LEXICAL_DIRECTION,
and balanced relevance accuracies of LEXICAL_RELEVANCE, respectively. The chance
references are $1/3$ and $1/2$. These measurements concern the tested lexical
baselines; they do not establish that all shortcuts are removed. The revised
materials and all three checkpoints remain development evidence. No independent
fresh-cohort result is asserted here.
'''
    text = text.replace('BASE_RESULTS', base_results).replace('COVERAGE', coverage).replace('LEXICAL_DIRECTION', direction_scores).replace('LEXICAL_RELEVANCE', relevance_scores)
    return {FILES[0]: text, **tables}, {'performance_rows': 12, 'comparison_rows': 9, 'control_rows': 12,
                                      'broken_stability_source_statuses': stability,
                                      'all_planned_binary_denominators_retained': True, 'missing_magnitudes_imputed': False}


def write(submission_path, output):
    output = output.resolve()
    require(output.is_relative_to(RUN.resolve() / 'paper_draft'), 'Draft output must remain under runs/robustness_development_v2/paper_draft; paper files are never overwritten')
    summaries, lexical, sources = inputs(submission_path.resolve())
    files, notes = render(summaries, lexical)
    generated = {('tables/' if name in FILES[1:4] else '') + name: common.text_sha256(text) for name, text in files.items()}
    manifest = {'schema_version': 'robustness_paper_draft_v1', 'status': 'complete_audited_draft_for_root_review',
                'sampling_families': 20, 'bootstrap': BOOTSTRAP, 'source_sha256': sources,
                'writer': artifact(Path(__file__)), 'generated_sha256': generated, 'tables': notes, 'paper_files_modified': False, 'latex_build_performed': False}
    files[FILES[-1]] = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n'
    for name, text in files.items():
        path = output / ('tables' if name in FILES[1:4] else '') / name
        require(not path.exists() or path.read_text() == text, 'Refusing to overwrite a different reviewed draft; choose a new draft subdirectory')
    for name, text in files.items():
        path = output / ('tables' if name in FILES[1:4] else '') / name
        if not path.exists():
            common.atomic_write(path, text)
    return {'status': manifest['status'], 'output': str(output), 'tables': notes, 'paper_files_modified': False,
            'draft_manifest_sha256': common.file_sha256(output / FILES[-1]), 'generated_sha256': generated}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', type=Path, default=RUN / 'submission.json')
    parser.add_argument('--output', type=Path, default=RUN / 'paper_draft')
    args = parser.parse_args()
    print(json.dumps(write(args.submission, args.output), indent=2))


if __name__ == '__main__':
    main()
