"""Evaluate the local translation pipeline on the frozen held-out corpus."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from accurate_translator import AccurateTranslator, DEFAULT_CONFIG
from translation_eval import (
    compare_reports,
    evaluate_predictions,
    load_eval_cases,
    write_report,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_DIR / "eval" / "translation_eval.jsonl"


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--direction", choices=("zh-en", "en-zh", "both"), default="both")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="test")
    parser.add_argument("--reviewer", choices=("on", "off"), default="off")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "eval" / "reports" / "latest.json")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--comet-model", default="")
    return parser.parse_args()


def _score_comet(cases, predictions, model_name: str) -> dict[str, object]:
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as error:
        raise RuntimeError("COMET 未安装；请安装可选依赖 unbabel-comet") from error
    checkpoint = download_model(model_name)
    model = load_from_checkpoint(checkpoint)
    data = [
        {"src": case.source, "mt": prediction, "ref": case.reference}
        for case, prediction in zip(cases, predictions)
    ]
    result = model.predict(data, batch_size=4, gpus=1)
    return {"model": model_name, "system_score": float(result.system_score), "scores": list(map(float, result.scores))}


def main() -> int:
    args = parse_args()
    directions = ("zh-en", "en-zh") if args.direction == "both" else (args.direction,)
    split = None if args.split == "all" else args.split
    reports: dict[str, object] = {}
    for direction in directions:
        cases = load_eval_cases(args.dataset, direction=direction, split=split)
        translator = AccurateTranslator(
            args.config,
            direction=direction,
            reviewer_enabled=args.reviewer == "on",
        )
        try:
            predictions = translator.translate_many_texts([case.source for case in cases])
            report = evaluate_predictions(cases, predictions)
            report["review"] = {
                "enabled": args.reviewer == "on",
                "summary": translator.review_summary,
                "stats": vars(translator.review_stats),
            }
            if args.comet_model:
                report["comet"] = _score_comet(cases, predictions, args.comet_model)
            reports[direction] = report
        finally:
            translator.release()

    output: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": _portable_path(args.dataset),
        "split": args.split,
        "reviewer": args.reviewer,
        "directions": reports,
    }
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        comparisons = {}
        for direction, report in reports.items():
            comparisons[direction] = compare_reports(
                report, baseline["directions"][direction]
            )
        output["comparison"] = comparisons
        output["accepted"] = all(value["accepted"] for value in comparisons.values())
    write_report(args.output, output)
    for direction, report in reports.items():
        print(
            f"{direction}: chrF++={report['chrf']:.2f} | "
            f"terms={report['term_accuracy']:.1%} | numbers={report['number_accuracy']:.1%} | "
            f"critical={report['critical_errors']}"
        )
    print(f"报告：{args.output.resolve()}")
    return 0 if output.get("accepted", True) else 4


if __name__ == "__main__":
    raise SystemExit(main())
