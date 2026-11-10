"""Run the matcher over labelled resume/JD pairs and compare RAG vs full-resume context.

Usage: python eval/run_eval.py [--runs 3] [--configs rag:v1,full:v1,rag:v2]
       python eval/run_eval.py --rescore   # re-score the saved results.json against the current labels, no LLM calls
       python eval/run_eval.py --rescore results-partial   # same for the last non-default run
Another dataset (e.g. private real resumes, .txt or .pdf):
       python eval/run_eval.py --cases eval/private/cases.csv --resumes resumes --jds eval/private/jds --out eval/private
Needs Ollama running with the models the backend uses (except for --rescore).
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT.parent / "backend"
sys.path.insert(0, str(BACKEND))


def compute_metrics(rows: list[dict]) -> dict:
    """rows: one per case, with expected_score, expected_recommendation, scores, recommendations."""
    from ai_logic import recommendation_for

    failures = sum(r["failures"] for r in rows)
    rows = [r for r in rows if r["scores"]]  # failed runs aren't scores; a case with none is skipped
    if len(rows) < 2:  # not enough scored cases for any metric, e.g. every call failed
        nan = float("nan")
        return {
            "mae": nan,
            "bias": nan,
            "correlation": nan,
            "rec_accuracy": nan,
            "derived_rec_accuracy": nan,
            "score_stdev": nan,
            "failures": failures,
            "context_chars": nan,
        }
    means = [statistics.mean(r["scores"]) for r in rows]
    expected = [r["expected_score"] for r in rows]
    recs = [(rec, r["expected_recommendation"]) for r in rows for rec in r["recommendations"]]
    derived = [(recommendation_for(s), r["expected_recommendation"]) for r in rows for s in r["scores"]]
    return {
        "mae": statistics.mean(abs(m - e) for m, e in zip(means, expected, strict=True)),
        "bias": statistics.mean(m - e for m, e in zip(means, expected, strict=True)),  # > 0: over-scores
        # correlation is undefined when every mean score is identical
        "correlation": statistics.correlation(means, expected) if len(set(means)) > 1 else float("nan"),
        "rec_accuracy": sum(got == want for got, want in recs) / len(recs),  # the model's own recommendation
        "derived_rec_accuracy": sum(got == want for got, want in derived) / len(derived),  # what the API returns
        "score_stdev": statistics.mean(statistics.pstdev(r["scores"]) for r in rows),
        "failures": failures,
        "context_chars": statistics.mean(r["context_chars"] for r in rows),
    }


def load_texts(cases: list[dict], resumes_dir: Path, jds_dir: Path) -> dict[str, str]:
    """Every resume and JD the cases use, read once. PDFs go through the app's own parser (local pypdf only)."""
    import parser as pdf_parser

    texts = {}
    for name in {c["resume"] for c in cases}:
        txt = resumes_dir / f"{name}.txt"
        if txt.exists():
            texts[name] = txt.read_text(encoding="utf-8")
        else:
            if pdf_parser.PDF_PARSER != "pypdf":  # never send eval resumes to a cloud parser
                raise ValueError("PDF resumes are only read with PDF_PARSER=pypdf")
            texts[name] = pdf_parser.parse_resume(str(resumes_dir / f"{name}.pdf")) or ""
    unreadable = sorted(n for n, t in texts.items() if not t.strip())
    if unreadable:  # the app rejects these with a 400 before any LLM call, so they can't be scored
        raise ValueError(f"no text extracted from {unreadable}; drop those cases")
    for name in {c["jd"] for c in cases}:
        texts[f"jd:{name}"] = (jds_dir / f"{name}.txt").read_text(encoding="utf-8")
    return texts


def run(
    mode: str, prompt_version: str, cases: list[dict], texts: dict[str, str], runs: int
) -> tuple[list[dict], float]:
    """Raw results only; labels are joined in at report time so --rescore picks up label edits."""
    from ai_logic import FALLBACK_RESULT, analyze_match
    from main import get_context

    rows, llm_seconds = [], 0.0
    for case in cases:
        resume, jd = texts[case["resume"]], texts[f"jd:{case['jd']}"]
        # retrieval is deterministic, so it runs once per case; the runs measure LLM variance
        context = get_context(resume, jd, mode)
        row = {
            "resume": case["resume"],
            "jd": case["jd"],
            "context_chars": len(context),
            "scores": [],
            "recommendations": [],
            "failures": 0,
        }
        for _ in range(runs):
            start = time.perf_counter()  # times the model call only, so rag and full are comparable
            result = analyze_match(context, jd, prompt_version)  # raw model recommendation, not the API's derived one
            llm_seconds += time.perf_counter() - start
            if result is FALLBACK_RESULT:
                row["failures"] += 1
                continue
            row["scores"].append(result.match_percentage)
            row["recommendations"].append(result.recommendation)
        rows.append(row)
        print(f"[{mode}:{prompt_version}] {case['resume']} x {case['jd']}: {row['scores']} {row['recommendations']}")
    return rows, llm_seconds / (len(cases) * runs)


def report(saved: dict, cases: list[dict]) -> str:
    for config_name in saved["configs"]:
        saved_cases = {(r["resume"], r["jd"]) for r in saved["rows"][config_name]}
        missing = sorted({(c["resume"], c["jd"]) for c in cases} - saved_cases)
        if missing:
            raise ValueError(f"no saved {config_name} scores for {missing}; new cases need a full run, not --rescore")
    summary = [
        "| Config | Avg context chars | Score bias | MAE vs label | Correlation | Rec. accuracy (model) "
        "| Rec. accuracy (from score) | Run-to-run stdev | Failed calls | Sec per LLM call |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    detail = [
        "| Resume | JD | Label | " + " | ".join(saved["configs"]) + " |",
        "|---|---|---|" + "---|" * len(saved["configs"]),
    ]
    labelled = {}
    for config_name in saved["configs"]:
        by_case = {(r["resume"], r["jd"]): r for r in saved["rows"][config_name]}
        labelled[config_name] = [
            {
                **by_case[(c["resume"], c["jd"])],
                "expected_score": int(c["expected_score"]),
                "expected_recommendation": c["expected_recommendation"],
            }
            for c in cases
        ]
        m = compute_metrics(labelled[config_name])
        latency = saved["latency"][config_name]  # None: measured before timing covered only the LLM call
        summary.append(
            f"| {config_name} | {m['context_chars']:.0f} | {m['bias']:+.1f} | {m['mae']:.1f} "
            f"| {m['correlation']:.2f} | {m['rec_accuracy']:.0%} | {m['derived_rec_accuracy']:.0%} "
            f"| {m['score_stdev']:.1f} | {m['failures']} | {'n/a' if latency is None else f'{latency:.1f}'} |"
        )

    for i, case in enumerate(cases):
        cells = [
            f"{statistics.mean(r['scores']):.0f} {r['recommendations'][0]}" if r["scores"] else "all runs failed"
            for r in (rows[i] for rows in labelled.values())
        ]
        detail.append(
            f"| {case['resume']} | {case['jd']} | {case['expected_score']} {case['expected_recommendation']} | "
            + " | ".join(cells)
            + " |"
        )

    return (
        f"# Eval results\n\n{len(cases)} cases x {saved['runs']} runs per config. {saved.get('setup', 'Settings not recorded (older results file)')}.\n\n"
        + "\n".join(summary)
        + "\n\n## Per case (mean score, model's first-run recommendation)\n\n"
        + "\n".join(detail)
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--configs", default="rag:v1,full:v1,rag:v2", help="context_mode:prompt_version pairs")
    parser.add_argument(
        "--rescore",
        nargs="?",
        const="results",
        choices=["results", "results-partial"],
        help="re-score a saved run (results.json by default) against the current labels",
    )
    parser.add_argument("--cases", type=Path, default=ROOT / "data/cases.csv")
    parser.add_argument("--resumes", type=Path, default=ROOT / "data/resumes", help=".txt or .pdf per resume")
    parser.add_argument("--jds", type=Path, default=ROOT / "data/jds")
    parser.add_argument("--out", type=Path, default=ROOT, help="folder for results.json and results.md")
    args = parser.parse_args()

    from ai_logic import LLM_MODEL, PROMPT_VERSIONS
    from main import CONTEXT_MODES, RETRIEVE_FRACTION
    from vector_db import CHUNK_CHARS, EMBED_MODEL

    with open(args.cases, newline="", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    setup = (
        f"LLM `{LLM_MODEL}`, embeddings `{EMBED_MODEL}`, {CHUNK_CHARS}-char chunks, "
        f"rag keeps {RETRIEVE_FRACTION:.0%} of chunks"
    )

    if args.rescore:
        if args.configs != parser.get_default("configs") or args.runs != parser.get_default("runs"):
            parser.error("--rescore re-scores a saved run; it can't take --configs or --runs")
        stem = args.rescore
        saved = json.loads((args.out / f"{stem}.json").read_text(encoding="utf-8"))
    else:
        configs = args.configs.split(",")
        pairs = [tuple(c.split(":")) for c in configs]
        if len(set(configs)) != len(configs) or not all(
            len(p) == 2 and p[0] in CONTEXT_MODES and p[1] in PROMPT_VERSIONS for p in pairs
        ):
            parser.error(f"--configs must be distinct mode:prompt pairs from {CONTEXT_MODES} x {PROMPT_VERSIONS}")
        try:
            texts = load_texts(cases, args.resumes, args.jds)
        except ValueError as e:
            parser.error(str(e))
        # setup is saved with the scores, so a later --rescore reports the settings the run actually used
        saved = {"runs": args.runs, "configs": configs, "setup": setup, "rows": {}, "latency": {}}
        for config_name, (mode, prompt_version) in zip(configs, pairs, strict=True):
            saved["rows"][config_name], saved["latency"][config_name] = run(
                mode, prompt_version, cases, texts, args.runs
            )
        # only the full default run replaces the results the README cites; other runs keep their raw scores too,
        # so their numbers can be re-scored (--rescore results-partial) instead of living only in a Markdown file
        full_run = args.configs == parser.get_default("configs") and args.runs == parser.get_default("runs")
        stem = "results" if full_run else "results-partial"
        (args.out / f"{stem}.json").write_text(json.dumps(saved, indent=1), encoding="utf-8")

    try:
        text = report(saved, cases)
    except ValueError as e:
        parser.error(str(e))
    (args.out / f"{stem}.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
