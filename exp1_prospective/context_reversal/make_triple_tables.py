#!/usr/bin/env python3
"""Render the triple analysis as appendix tables and a results summary."""
from __future__ import annotations

import json
from pathlib import Path

from exp1_prospective.context_reversal import run_local as common

HERE = Path(__file__).resolve().parent
REPORT = HERE / 'results/fresh_evaluation_v1/triple_analysis.json'
ACCOUNTING = HERE / 'results/fresh_evaluation_v1/unconditional_accounting.json'
TABLES = HERE.parents[1] / 'paper/tables'
SUMMARY = HERE / 'results/fresh_evaluation_v1/triple_analysis.md'
LABELS = {'no_change': 'Predict no change', 'evidence_direction': 'Follow the evidence direction',
          'always_up': 'Always revise upward', 'best_constant_with_oracle': 'Best constant, given every oracle'}


def latex(path, body):
    common.atomic_write(path, body.rstrip() + '\n')
    return path.name


def main():
    r = json.loads(REPORT.read_text())
    arms = [a for a in r['arms'] if a['items_valid']]
    written = []

    rows = '\n'.join(
        f"{LABELS[k]} & {v['item_sign_correct']}/{v['items']} & {v['triple_sign_all_correct']}/{v['triples']} \\\\"
        for k, v in r['shortcuts'].items())
    written.append(latex(TABLES / 'exp1_context_triple_bound.tex', f"""\\begin{{tabular}}{{lcc}}
\\toprule
Predictor & Items, direction & Triples, all three \\\\
\\midrule
{rows}
\\midrule
\\emph{{Analytic bound}} & $\\leq 1/3$ & $0$ \\\\
\\bottomrule
\\end{{tabular}}"""))

    rows = '\n'.join(
        f"{a['arm']} & {a['triple_sign_unconditional']}/{a['triples_planned']} & "
        f"{a['triple_sign_complete_case']}/{a['triples_complete']} & "
        f"{a['item_sign_correct']}/{a['items_planned']} & {a['item_exact_correct']}/{a['items_planned']} \\\\"
        for a in arms)
    written.append(latex(TABLES / 'exp1_context_triple_accuracy.tex', f"""\\begin{{tabular}}{{lcccc}}
\\toprule
& \\multicolumn{{2}}{{c}}{{Triples, all three correct}} & \\multicolumn{{2}}{{c}}{{Items}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
System & Unconditional & Complete case & Direction & Exact \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}"""))

    def err(a, metric):
        e = next(x for x in a['errors'] if x['metric'] == metric)
        return f"{e['mean_pp']:.3f} & {e['median_pp']:.3f} & {e['max_pp']:.3f}"

    rows = '\n'.join(f"{a['arm']} & {err(a, 'update absolute error')} & {err(a, 'revision-size error')} \\\\" for a in arms)
    written.append(latex(TABLES / 'exp1_context_triple_error.tex', f"""\\begin{{tabular}}{{lcccccc}}
\\toprule
& \\multicolumn{{3}}{{c}}{{Posterior absolute error (pp)}} & \\multicolumn{{3}}{{c}}{{Revision-size error (pp)}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}}
System & Mean & Median & Max & Mean & Median & Max \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}"""))

    rows = '\n'.join(
        f"{a['arm']} & {a['inert']['exact_zero']}/{a['inert']['n']} & {a['inert']['within_stability']}/{a['inert']['n']} & "
        f"{a['inert']['movement'].get('mean_pp', 0):.3f} & {a['relevant_movement'].get('mean_pp', 0):.3f} \\\\"
        for a in arms)
    written.append(latex(TABLES / 'exp1_context_triple_inert.tex', f"""\\begin{{tabular}}{{lcccc}}
\\toprule
& \\multicolumn{{2}}{{c}}{{Inert items}} & \\multicolumn{{2}}{{c}}{{Mean movement (pp)}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
System & Exactly zero & Within 2\\,pp & Inert & Relevant \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}"""))

    lines = [f"# Triple-level analysis of the frozen context-reversal cohort\n",
             f"Generated {r['created_at']} from `{Path(r['plan']['path']).name}` "
             f"(sha256 `{r['plan']['sha256'][:16]}`). Offline; no model was called.\n",
             f"## Design\n",
             f"{r['triples']} triples over {r['units']} units. Within a triple the evidence text, the prior and the "
             f"named entity are identical and only the stated relation differs; all five invariants were verified: "
             f"{', '.join(r['design_invariants_verified'])}.\n",
             f"## The bound\n", f"{r['impossibility_bound']['statement']} "
             f"Maximum item accuracy {r['impossibility_bound']['max_item_accuracy']}, "
             f"maximum triple accuracy {r['impossibility_bound']['max_triple_accuracy']}.\n",
             "| Predictor | Items, direction | Triples, all three |", "|---|---:|---:|"]
    lines += [f"| {LABELS[k]} | {v['item_sign_correct']}/{v['items']} | {v['triple_sign_all_correct']}/{v['triples']} |"
              for k, v in r['shortcuts'].items()]
    lines += ["\n## Systems\n",
              "| System | Triples (uncond.) | Triples (complete) | Items direction | Items exact | Posterior err. mean (pp) | Inert exactly zero |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for a in arms:
        e = next(x for x in a['errors'] if x['metric'] == 'update absolute error')
        lines.append(f"| {a['arm']} | {a['triple_sign_unconditional']}/{a['triples_planned']} | "
                     f"{a['triple_sign_complete_case']}/{a['triples_complete']} | "
                     f"{a['item_sign_correct']}/{a['items_planned']} | {a['item_exact_correct']}/{a['items_planned']} | "
                     f"{e['mean_pp']:.3f} | {a['inert']['exact_zero']}/{a['inert']['n']} |")
    acc = json.loads(ACCOUNTING.read_text())
    cols = ['ok', 'model_refusal', 'model_unparseable', 'model_framing',
            'instrument_truncation', 'instrument_harness_kill', 'cascade_blocked', 'not_attempted', 'pending']
    live = [a for a in acc['arms'] if any(a['categories'].get(c) for c in cols)]
    rows = '\n'.join(
        f"{a['arm']} & {a['planned_records']} & " + ' & '.join(str(a['categories'].get(c, 0)) for c in cols[:-1])
        + (f" & {a['categories'].get('pending', 0)}" if any(x['in_progress'] for x in live) else '') + ' \\\\'
        for a in live)
    pending_col = ' & Pending' if any(a['in_progress'] for a in live) else ''
    spec = 'l' + 'c' * (9 if any(a['in_progress'] for a in live) else 8)
    written.append(latex(TABLES / 'exp1_context_accounting.tex', f"""\\begin{{tabular}}{{{spec}}}
\\toprule
& & & \\multicolumn{{3}}{{c}}{{Model failure}} & \\multicolumn{{2}}{{c}}{{Instrument failure}} & \\multicolumn{{2}}{{c}}{{Cascade}} \\\\
\\cmidrule(lr){{4-6}} \\cmidrule(lr){{7-8}} \\cmidrule(lr){{9-10}}
System & Planned & Valid & Refused & Unparseable & Frame & Truncated & Killed & Blocked & Unsent{pending_col} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}"""))

    lines += ["\n## Unconditional accounting\n", acc['rule'], "",
              "| System | Planned | Valid | Model failure | Instrument failure | Cascade | Pending |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for a in live:
        att = a['by_attribution']
        lines.append(f"| {a['arm']} | {a['planned_records']} | {a['categories'].get('ok', 0)} | "
                     f"{att.get('model', 0)} | {att.get('instrument', 0)} | {att.get('cascade', 0)} | {att.get('pending', 0)} |")
    lines.append("\nUnconditional denominators count refusals, truncations and blocked updates as failures. "
                 "Exact agreement is to four decimals, the granularity the oracles are stated at.\n")
    common.atomic_write(SUMMARY, '\n'.join(lines))
    print('tables:', ', '.join(written))
    print('summary:', SUMMARY)


if __name__ == '__main__':
    main()
