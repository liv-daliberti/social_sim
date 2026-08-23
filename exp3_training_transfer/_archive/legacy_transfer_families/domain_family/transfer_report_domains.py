#!/usr/bin/env python3
"""Turn exp3d eval dumps into the three transfer metrics, per held-out domain and arm.

dag_family/transfer_report.py cannot be reused: it keys on structure names from the DAG catalog,
assumes poll points, and normalises against a Kalman prior-blind filter. Here the graph is fixed and
the DOMAIN varies, so every quantity lives in that domain's displayed units.

Metrics, per (arm, domain, prefix k), matching the structure arm's definitions:

    pi    forecast error        mean_i |pred_i - target_i|          (displayed units)
    ghat  implied sensitivity   slope of the four forecasts vs the four probe shocks
    eps   recovery error        mean_city |ghat - g_true|           (displayed units)
    eps~  normalised recovery   eps / eps_blind                     (<1 beats not inferring)
    rho   informativeness       corr(ghat, g_true) across episodes

The blind floor is the best a model can do WITHOUT inferring the per-episode sensitivity: freeze the
canonical gain at its prior midpoint 0.55 and map it through the domain skin, giving
g_blind = 0.55 * (obs_scale / in_scale). Because the probed structure responds immediately (h* = 1),
the response IS the gain and no Kalman filter is needed -- the closed form is exact.

pi and eps are NOT comparable across domains (a crate is not a poll point). rho and eps~ are
dimensionless and are the quantities to pool. The report refuses to print a pooled pi.

    python transfer_report_domains.py                       # completed seeds at step 300
    python transfer_report_domains.py --step 300 --latex ../../paper/ICLR/tables/exp3d_transfer.tex
    python transfer_report_domains.py --interim-latex ../../paper/ICLR/tables/exp3d_interim.tex
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "dag_family")]

from domains import BY_NAME, HELD_OUT  # noqa: E402
from prompt_domain import parse_forecasts  # noqa: E402

PRIOR_MEAN_CANONICAL = 0.55       # midpoint of the canonical gain range; the "do not infer" policy
ARMS = ("d1", "d2", "d3", "structureless")
ARM_LABEL = {"d1": "1 domain", "d2": "2 domains", "d3": "3 domains",
             "structureless": "structureless"}
ARM_TRAINING = {"d1": "CoinCity", "d2": "$+$ CoinFishing",
                "d3": "$+$ CoinFishing, CoinFarm",
                "structureless": "driver decoupled"}


def find_dumps(arm: str) -> dict[int, Path]:
    """step -> newest eval dump for that step, across all runs of this arm."""
    out: dict[int, tuple[float, Path]] = {}
    for d in (ROOT / "reports").glob(f"exp3d_{arm}_*/"):
        for f in d.glob("debug_*/eval_results/*.json"):
            try:
                step = int(f.stem)
            except ValueError:
                continue
            mt = f.stat().st_mtime
            if step not in out or mt > out[step][0]:
                out[step] = (mt, f)
    return {k: v[1] for k, v in sorted(out.items())}


def job_registry() -> dict[str, tuple[str, int]]:
    """job id -> (arm, training seed), from the immutable launch ledgers."""
    out = {}
    for ledger in sorted((ROOT / "runs").glob("exp3d_*.json")):
        try:
            payload = json.loads(ledger.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for sub in payload.get("submissions", []):
            job = str(sub.get("job_id", ""))
            arm = sub.get("arm")
            seed = sub.get("seed")
            if job and arm in ARMS and isinstance(seed, int):
                out[job] = (arm, seed)
    return out


def completed_seed_dumps(arm: str, step: int) -> dict[int, Path]:
    """Newest registered endpoint dump for every completed training seed.

    A seed can have several Slurm attempts. We key reports through the launch
    ledgers and retain the newest attempt that actually emitted the requested
    endpoint, so failed/restarted jobs cannot be mistaken for independent seeds.
    """
    registry = job_registry()
    out: dict[int, tuple[float, Path]] = {}
    for path in (ROOT / "reports").glob(
            f"exp3d_{arm}_*_j*/debug_*/eval_results/{step}.json"):
        match = re.search(r"_j(\d+)", str(path))
        if not match:
            continue
        registered = registry.get(match.group(1))
        if registered is None or registered[0] != arm:
            continue
        seed = registered[1]
        mtime = path.stat().st_mtime
        if seed not in out or mtime > out[seed][0]:
            out[seed] = (mtime, path)
    return {seed: item[1] for seed, item in sorted(out.items())}


def score_dump(path: Path) -> list[dict]:
    """One row per eval record: domain, k, true gain, implied gain, forecast error."""
    rows = []
    for rec in json.loads(path.read_text()):
        ref = json.loads(rec["reference"])
        dom = BY_NAME.get(ref.get("domain"))
        if dom is None:                      # not a domain dump (e.g. a structure run)
            continue
        gen = rec["output"][0] if rec.get("output") else ""
        preds = parse_forecasts(gen, dom)
        targets = np.asarray(ref["targets"], float)
        shocks = np.asarray(ref["shocks"], float)
        if preds is None:                    # unparseable reply: counted, but cannot be scored
            rows.append({"domain": dom.name, "k": ref["k"], "seed": ref["seed"],
                         "formatted": False, "pi": np.nan, "ghat": np.nan,
                         "g_true": ref["true_gain"]})
            continue
        p = np.asarray(preds, float)
        ghat = float(np.cov(shocks, p, bias=True)[0, 1] / np.var(shocks))
        rows.append({"domain": dom.name, "k": ref["k"], "seed": ref["seed"], "formatted": True,
                     "pi": float(np.mean(np.abs(p - targets))), "ghat": ghat,
                     "g_true": float(ref["true_gain"])})
    return rows


def summarise(rows: list[dict]) -> dict[str, dict]:
    """Per-domain pi / eps / eps~ / rho, pooling over episodes and prefixes."""
    out = {}
    for dom in HELD_OUT:
        r = [x for x in rows if x["domain"] == dom.name]
        if not r:
            continue
        fmt = float(np.mean([x["formatted"] for x in r]))
        ok = [x for x in r if x["formatted"]]
        if not ok:
            out[dom.name] = {"format_rate": fmt, "n": len(r), "pi": np.nan,
                             "eps": np.nan, "eps_norm": np.nan, "rho": np.nan}
            continue
        g_true = np.asarray([x["g_true"] for x in ok])
        ghat = np.asarray([x["ghat"] for x in ok])
        g_blind = PRIOR_MEAN_CANONICAL * dom.gain_ratio
        eps = float(np.mean(np.abs(ghat - g_true)))
        eps_blind = float(np.mean(np.abs(g_blind - g_true)))
        rho = (float(np.corrcoef(ghat, g_true)[0, 1])
               if np.std(ghat) > 1e-12 and np.std(g_true) > 1e-12 else float("nan"))
        out[dom.name] = {
            "format_rate": fmt, "n": len(r),
            "pi": float(np.mean([x["pi"] for x in ok])),
            "eps": eps, "eps_blind": eps_blind,
            "eps_norm": eps / eps_blind if eps_blind > 0 else float("nan"),
            "rho": rho,
        }
    return out


def latex(table: dict[str, dict[str, dict]], base: dict[str, dict] | None) -> str:
    """Held-out domains as row blocks; arms as columns. pi stays inside a domain block because
    forecast error is in that domain's own units and must never be pooled across domains."""
    cols = [a for a in ARMS if a in table]
    L = ["% generated by domain_family/transfer_report_domains.py -- do not edit",
         r"\begin{tabular}{l l " + " ".join("r" for _ in range(len(cols) + (1 if base else 0))) + "}",
         r"\toprule",
         "Held-out domain & Metric & " +
         (" base &" if base else "") + " & ".join(ARM_LABEL[a] for a in cols) + r" \\",
         r"\midrule"]
    for dom in HELD_OUT:
        label = "\\texttt{" + dom.label + "}"
        metrics = (("$\\pi\\downarrow$", "pi", "{:.2f}"),
                   ("$\\tilde\\varepsilon\\downarrow$", "eps_norm", "{:.2f}"),
                   ("$\\rho\\uparrow$", "rho", "{:.3f}"))
        for row_i, (metric, key, fmt) in enumerate(metrics):
            cells = []
            if base:
                b = base.get(dom.name, {}).get(key, float("nan"))
                cells.append(fmt.format(b) if b == b else "--")
            for a in cols:
                v = table[a].get(dom.name, {}).get(key, float("nan"))
                cells.append(fmt.format(v) if v == v else "--")
            first = label if row_i == 0 else ""      # domain name spans its metric block
            L.append(first + " & " + metric + " & " + " & ".join(cells) + r" \\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def interim_latex(table: dict[str, dict[str, dict]], base: dict[str, dict],
                  completed: dict[str, int]) -> str:
    """Main-text rows with seed-pooled metrics and explicit completeness."""
    domains = [dom.name for dom in HELD_OUT]

    def metrics(summary: dict[str, dict]) -> list[str]:
        cells = []
        for domain in domains:
            values = summary.get(domain, {})
            for key, fmt in (("pi", "{:.2f}"), ("eps_norm", "{:.2f}"),
                             ("rho", "{:+.3f}")):
                value = values.get(key, float("nan"))
                cells.append("$" + fmt.format(value) + "$" if value == value else "--")
        return cells

    lines = ["% generated by domain_family/transfer_report_domains.py -- do not edit",
             r"\begin{tabular}{l l c rrr rrr}",
             r"\toprule",
             r"& & & \multicolumn{3}{c}{CoinBasketball} & \multicolumn{3}{c}{CoinClinic} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"Arm & Training domains & complete / 3 &",
             r"$\pi\downarrow$ & $\tilde\varepsilon\downarrow$ & $\rho\uparrow$ &",
             r"$\pi\downarrow$ & $\tilde\varepsilon\downarrow$ & $\rho\uparrow$ \\",
             r"\midrule",
             "Untrained base & --- & --- & " + " & ".join(metrics(base)) + r" \\",
             r"\midrule"]
    for arm in ARMS:
        label = {"d1": "$D_1$", "d2": "$D_2$", "d3": "$D_3$"}.get(
            arm, "structureless")
        n = completed.get(arm, 0)
        row = f"{label} & {ARM_TRAINING[arm]} & {n}/3"
        if arm in table:
            row += " & " + " & ".join(metrics(table[arm])) + r" \\"
        else:
            row += r" & \multicolumn{6}{c}{\emph{endpoint not yet available}} \\"
        lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=300, help="training step to report (endpoint)")
    ap.add_argument("--base-step", type=int, default=0,
                    help="step whose eval is the untrained baseline (OAT evaluates at step 0)")
    ap.add_argument("--latex", type=Path)
    ap.add_argument("--interim-latex", type=Path,
                    help="write the seed-pooled main-text interim table")
    args = ap.parse_args()

    table, base, completed = {}, None, {}
    for arm in ARMS:
        dumps = completed_seed_dumps(arm, args.step)
        completed[arm] = len(dumps)
        if not dumps:
            print(f"[{arm:<13}] 0/3 completed seeds at step {args.step}")
            continue
        rows = []
        for seed, dump in dumps.items():
            rows.extend(score_dump(dump))
            print(f"[{arm:<13}] seed {seed}, step {args.step}: {dump}")
        table[arm] = summarise(rows)
        if base is None:
            first = next(iter(dumps.values()))
            base_path = first.with_name(f"{args.base_step}.json")
            if base_path.exists():
                # Step zero is a fixed, deterministic checkpoint; duplicate
                # copies from training jobs are not independent seed results.
                base = summarise(score_dump(base_path))
        print(f"[{arm:<13}] {len(dumps)}/3 completed seeds")
        for dom, m in table[arm].items():
            print(f"    {dom:<18} pi={m['pi']:.2f}  eps~={m['eps_norm']:.2f}  "
                  f"rho={m['rho']:+.3f}  parsed={m['format_rate']:.0%}  n={m['n']}")

    if not table:
        print("\nnothing to report yet -- no arm has produced endpoint eval dumps")
        return
    print("\nNOTE: pi is in each domain's own units and is never pooled across domains; "
          "eps~ and rho are dimensionless.")
    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(latex(table, base) + "\n", encoding="utf-8")
        print(f"latex -> {args.latex}")
    if args.interim_latex:
        if base is None:
            raise SystemExit("cannot write interim table: no step-zero baseline found")
        args.interim_latex.parent.mkdir(parents=True, exist_ok=True)
        args.interim_latex.write_text(
            interim_latex(table, base, completed) + "\n", encoding="utf-8")
        print(f"interim latex -> {args.interim_latex}")


if __name__ == "__main__":
    main()
