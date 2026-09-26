#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from matplotlib.axes import Axes


PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))

import generate_exp4_scale_figure as figure  # noqa: E402


def test_registered_scale_report_and_table_are_exact() -> None:
    report = figure.load_report()
    llama = figure.load_llama_report()
    qwen32 = figure.load_qwen32_report()

    assert report["n"] == 318
    assert tuple(figure.MODEL_KEYS) == ("qwen3_4b", "qwen3_8b", "qwen3_14b")
    assert report["decision_flags"]["qwen3_14b_beats_qwen3_8b_conclusive"]
    table = figure.render_table(report, llama, qwen32=qwen32)
    for checkpoint in (
        "Qwen3-4B-Instruct-2507",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-32B",
        "Llama-3.1-8B",
    ):
        assert checkpoint in table
    for value in ("0.1366", "0.1286", "0.1270", "0.1265", "0.1264"):
        assert value in table
    assert llama["models"]["base"]["draw_parse_coverage"] == pytest.approx(1581 / 1590)
    assert all(
        llama["models"][f"seed_{seed}"]["draw_parse_coverage"] == 1
        for seed in (42, 43, 44)
    )


def test_qwen32_extension_is_registered_sealed_and_exact() -> None:
    report = figure.load_qwen32_report()

    assert report["model"] == "Qwen/Qwen3-32B"
    assert report["holdout"]["n"] == 318
    assert report["models"]["base"]["brier"] == pytest.approx(
        0.126592910345912
    )
    trained = sum(
        report["models"][f"seed_{seed}"]["brier"] for seed in (42, 43, 44)
    ) / 3
    assert trained == pytest.approx(0.1265183279454927)
    effect = report["comparisons_brier"]["trained_seed_mean_minus_base"]
    assert effect["estimate"] == pytest.approx(-0.00007458240041928731)
    assert effect["ci95_low"] == pytest.approx(-0.00045125965079365147)
    assert effect["ci95_high"] == pytest.approx(0.0002891352089407185)


def test_scale_generator_fails_closed_on_report_drift(tmp_path: Path) -> None:
    report = json.loads(figure.SUMMARY.read_text(encoding="utf-8"))
    report["decision_flags"]["qwen3_14b_beats_qwen3_8b_conclusive"] = False
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="decision flags drifted"):
        figure.load_report(drifted)


def test_llama_generator_fails_closed_on_report_drift(tmp_path: Path) -> None:
    report = json.loads(figure.LLAMA_SUMMARY.read_text(encoding="utf-8"))
    report["model"] = "wrong/model"
    drifted = tmp_path / "llama_drifted.json"
    drifted.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="model drifted"):
        figure.load_llama_report(drifted)


def test_scale_figure_renders_pdf_and_png(tmp_path: Path) -> None:
    report = figure.load_report()
    llama = figure.load_llama_report()
    qwen32 = figure.load_qwen32_report()
    pdf = tmp_path / "scale.pdf"
    png = tmp_path / "scale.png"

    hosted = figure.load_hosted_reports()
    figure.draw_figure(
        report,
        llama,
        pdf=pdf,
        png=png,
        hosted=hosted,
        qwen32=qwen32,
    )

    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.read_bytes().startswith(b"\x89PNG")
    assert pdf.stat().st_size > 10_000
    assert png.stat().st_size > 50_000
    extracted = subprocess.run(
        ["pdftotext", str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "Polymarket" in extracted
    assert "Polymarket users" not in extracted
    assert "Parameters (billions; log scale)" in extracted
    assert "32" in extracted
    assert "Qwen3 checkpoints" in extracted
    assert "Llama family" not in extracted
    assert "Hosted frontier references" in extracted
    for label in ("V4-Pro", "Opus 4.8", "GPT-5.6", "K3"):
        assert label in extracted
    assert "K3*" not in extracted
    assert "Opus 5" not in extracted
    assert "Platt" not in extracted
    assert "controlled 14B" not in extracted


def synthetic_extension(report: dict, llama: dict) -> dict:
    qwen = report["models"]["qwen3_4b"]
    qwen_small = {
        "base_brier": qwen["base_brier"],
        "trained_seed_mean_brier": qwen["trained_seed_mean_brier"],
        "trained_seed_brier": qwen["trained_seed_brier"],
        "models": {
            f"seed_{seed}": {"complete_task_parse_coverage": 1.0}
            for seed in (42, 43, 44)
        },
    }
    llama_seeds = {
        str(seed): llama["models"][f"seed_{seed}"]["brier"] for seed in (42, 43, 44)
    }
    llama_small = {
        "base_brier": llama["models"]["base"]["brier"],
        "trained_seed_mean_brier": sum(llama_seeds.values()) / 3,
        "trained_seed_brier": llama_seeds,
        "models": {
            f"seed_{seed}": {"complete_task_parse_coverage": 1.0}
            for seed in (42, 43, 44)
        },
    }
    return {
        "market_brier": report["baselines"]["market"]["brier"],
        "models": {
            "qwen3_1_7b": qwen_small,
            "llama3_2_3b": llama_small,
        },
    }


def test_optional_extension_splits_qwen_and_llama_parameter_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = figure.load_report()
    llama = figure.load_llama_report()
    qwen32 = figure.load_qwen32_report()
    extension = synthetic_extension(report, llama)
    pdf = tmp_path / "extended.pdf"
    png = tmp_path / "extended.png"
    llama_pdf = tmp_path / "llama.pdf"
    llama_png = tmp_path / "llama.png"
    plot_x: list[tuple[float, ...]] = []
    original_plot = Axes.plot

    def record_plot(self, *args, **kwargs):
        if args:
            values = tuple(float(value) for value in args[0])
            plot_x.append(values)
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", record_plot)
    figure.draw_figure(
        report,
        llama,
        pdf=pdf,
        png=png,
        extension=extension,
        qwen32=qwen32,
    )

    extracted = subprocess.run(
        ["pdftotext", str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for label in ("1.7", "8", "Parameters (billions; log scale)"):
        assert label in extracted
    assert "Llama" not in extracted
    assert "Platt" not in extracted
    assert (1.7, 4.0, 8.0, 14.0, 32.0) in plot_x
    assert (3.0, 8.0) not in plot_x

    plot_x.clear()
    figure.draw_llama_figure(
        llama,
        extension,
        pdf=llama_pdf,
        png=llama_png,
    )
    llama_extracted = subprocess.run(
        ["pdftotext", str(llama_pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "Llama replication" in llama_extracted
    assert "Parameters (billions; log scale)" in llama_extracted
    assert (3.0, 8.0) in plot_x
    assert llama_pdf.read_bytes().startswith(b"%PDF")
    assert llama_png.read_bytes().startswith(b"\x89PNG")

    table = figure.render_table(report, llama, extension, qwen32)
    assert "Qwen3-1.7B" in table
    assert "Qwen3-32B" in table
    assert "Llama-3.2-3B" in table
    assert table.index("Qwen3-1.7B") < table.index("Qwen3-4B")
    assert "Platt" not in table


def test_hosted_loader_uses_separate_categorical_reference_roster() -> None:
    hosted = figure.load_hosted_reports()

    assert [point["model"] for point in hosted] == [
        "DeepSeek-V4-Pro",
        "claude-opus-4-8",
        "gpt-5.6-sol",
        "FW-Kimi-K3",
    ]
    assert [point["version"] for point in hosted] == [
        "V4-Pro",
        "Opus 4.8",
        "GPT-5.6",
        "K3",
    ]
    assert all(Path(point["icon"]).is_file() for point in hosted)
    assert all("post_hoc_repair" not in point for point in hosted)
    assert hosted[3]["brier"] == pytest.approx(0.12681767704402525)
    assert all(point["brier"] > 0 for point in hosted)
    assert all(
        point["ci_low"] <= point["brier"] <= point["ci_high"] for point in hosted
    )


def test_finalized_architecture_extension_values_are_locked() -> None:
    extension = figure.load_architecture_extension()

    assert extension is not None
    assert extension["status"] == "pass"
    assert extension["market_brier"] == pytest.approx(0.12654794182389942)
    qwen = extension["models"]["qwen3_1_7b"]
    llama = extension["models"]["llama3_2_3b"]
    assert qwen["trained_seed_mean_brier"] == pytest.approx(0.1247933455241122)
    assert qwen["base_brier"] == pytest.approx(0.22531262264184918)
    assert llama["trained_seed_mean_brier"] == pytest.approx(0.1367088449132422)
    assert llama["base_brier"] == pytest.approx(0.21886725305341212)
    assert llama["models"]["base"]["complete_task_parse_coverage"] == 1.0
    assert llama["models"]["base"]["draw_parse_coverage"] == 1.0
    assert all(
        llama["models"][f"seed_{seed}"]["complete_task_parse_coverage"] == 1
        for seed in (42, 43, 44)
    )


def test_extension_loader_fails_closed_before_artifacts_are_complete(
    tmp_path: Path,
) -> None:
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(
        json.dumps({"status": "running"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not a pass"):
        figure.load_architecture_extension(incomplete)
