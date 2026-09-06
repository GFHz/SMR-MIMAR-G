"""Same-protocol local LLM-IPP-style versus frozen SMR-MIMAR-G comparison."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from repro import local_llm
from repro.evaluators.formal.metrics import evaluate_prepared
from repro.evaluators.formal.protocol import PROTOCOL_VERSION, prepare_path
from repro.evaluators.formal.title_resolver import TitleResolver
from repro.evaluators.sasrec.checkpoint_loader import load_checkpoint
from repro.evaluators.sasrec.data_adapter import DataAdapter
from repro.evaluators.sasrec.evaluator import SASRecEvaluator
from repro.experiments.pilot.evaluate_pilot_formal import CHECKPOINT, EXPECTED_CHECKPOINT_SHA, sha
from repro.movie_path_parser import parse_movie_path, truncation_diagnostic
from repro.utils import ROOT, read_json, write_json

USERS = [419, 5021, 2677, 3113, 2249]
OUT = ROOT / "repro/results/llmipp_vs_smr_mimar_g_same_protocol"
SMR = ROOT / "repro/results/smr_mimar_g/controlled_positive_5users"
FORMAT_PROMPT = (
    "Output the influence path in the format of python list object. "
    "Each element must be the exact movie title in chronological order."
)
SYSTEM_PROMPT = (
    "You are a recommender system. Given the user profile and chronological historical data, "
    "analyze the user's interests and construct an influence path from that history toward the "
    "predefined target movie. Include the predefined target movie as the final item of the intended "
    "path. Do not recommend only the target: include at least five intermediate movies before it. "
    "Any adjacent movies should have a strong relation with each other. The generated movies must "
    "not be included in the historical data and should be MovieLens-compatible movies. Think step "
    "by step, then provide an ordered movie sequence."
)


def digest_tree(folder: Path) -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def prompt(user: dict) -> dict:
    d = user["demographics"]
    history = "".join(f"{x['title']} Genre:{'|'.join(x['genres'])}\n" for x in user["history"])
    target = user["target"]
    user_text = (f"Gender:{d['gender']}\nAge:{d['age']}\nOccupation:{d['occupation']}\n\n"
                 f"Historical data (chronological):\n{history}\nTarget movie: {target['title']} "
                 f"Genre:{'|'.join(target['genres'])}\n")
    return {"system_prompt": SYSTEM_PROMPT, "user_prompt": user_text,
            "formatting_user_prompt": FORMAT_PROMPT}


def raw_saver(folder: Path):
    counter = [0]
    def save(record):
        counter[0] += 1
        path = folder / f"transport_{counter[0]:02d}.json"
        write_json(path, record)
        return str(path)
    return save


def generate(user: dict, index: int) -> dict:
    bundle = prompt(user)
    folder = OUT / "raw_api" / str(user["user_id"]) / f"path_{index}"
    old = local_llm.save_record
    try:
        local_llm.save_record = raw_saver(folder)
        plan = local_llm.generate(bundle["system_prompt"], bundle["user_prompt"])
        messages = [{"role": "system", "content": bundle["system_prompt"]},
                    {"role": "user", "content": bundle["user_prompt"]},
                    {"role": "assistant", "content": plan["content"]},
                    {"role": "user", "content": bundle["formatting_user_prompt"]}]
        formatted = local_llm.generate_messages(messages)
    finally:
        local_llm.save_record = old
    parsed = parse_movie_path(formatted["content"])
    return {"method": "LLM-IPP-style same-protocol", "user_id": user["user_id"],
            "path_index": index, "target": user["target"], "prompt": bundle,
            "raw_plan": plan["content"], "raw_formatting_response": formatted["content"],
            "plan_response_metadata": plan["response"],
            "format_response_metadata": formatted["response"],
            "plan_truncation": truncation_diagnostic(plan["response"], 2048),
            "format_truncation": truncation_diagnostic(formatted["response"], 2048),
            "parser": parsed, "RAW_LLM_IPP_PATH": parsed["path"],
            "EVALUATION_NORMALIZED_PATH": None,
            "normalization_note": "No normalization or repair was applied; Formal-Evaluation-v1 receives the raw parsed path.",
            "elapsed_seconds": plan["elapsed_seconds"] + formatted["elapsed_seconds"]}


def raw_diagnostic(record: dict, user: dict, resolver: TitleResolver) -> dict:
    path = record.get("RAW_LLM_IPP_PATH", record.get("parsed_path", []))
    parse_ok = record.get("parser", {}).get("parse_success", record.get("parse_success", False))
    items = [resolver.resolve(x) for x in path] if parse_ok else []
    target_title = user["target"]["title"]
    target_id = user["target"]["movie_id"]
    history_ids = {x["movie_id"] for x in user["history"]}
    raw_target_positions = [i for i, x in enumerate(path, 1) if x == target_title]
    resolved_target_positions = [i for i, x in enumerate(items, 1) if x["raw_item_id"] == target_id]
    non_target = [(title, item) for title, item in zip(path, items) if item["raw_item_id"] != target_id]
    reused = [title for title, item in non_target if item["raw_item_id"] in history_ids]
    keys = [("id", item["raw_item_id"]) if item["raw_item_id"] is not None else ("raw", title)
            for title, item in non_target]
    counts = Counter(keys)
    dup_count = sum(n - 1 for n in counts.values())
    dup_titles = []
    for key, n in counts.items():
        if n > 1:
            dup_titles.append(next(title for title, item in non_target
                if (("id", item["raw_item_id"]) if item["raw_item_id"] is not None else ("raw", title)) == key))
    unresolved = [x["raw_title"] for x in items if x["resolution_status"] != "resolved"]
    return {"RAW_GENERATED_ITEM_COUNT": len(path),
            "EXACTLY_RESOLVED_ITEM_COUNT": sum(x["resolution_status"] == "resolved" and x["resolution_type"] == "exact" for x in items),
            "RESOLVED_ITEM_COUNT": sum(x["resolution_status"] == "resolved" for x in items),
            "UNRESOLVED_ITEM_COUNT": len(unresolved), "UNRESOLVED_TITLES": unresolved,
            "RESOLVED_PATH": [x["resolved_title"] for x in items],
            "HISTORY_REUSE_COUNT": len(reused),
            "HISTORY_REUSE_RATE": len(reused) / max(len(non_target), 1),
            "HISTORY_REUSED_TITLES": reused,
            "RAW_TARGET_PRESENT": bool(raw_target_positions),
            "RESOLVED_TARGET_PRESENT": bool(resolved_target_positions),
            "RAW_TARGET_LAST": bool(path) and path[-1] == target_title,
            "RESOLVED_TARGET_LAST": bool(items) and items[-1]["raw_item_id"] == target_id,
            "INTRA_PATH_DUPLICATE_COUNT": dup_count,
            "INTRA_PATH_DUPLICATE_RATE": dup_count / max(len(non_target), 1),
            "DUPLICATED_TITLES": dup_titles}


def evaluate(records: list[dict], users: list[dict], resolver: TitleResolver) -> list[dict]:
    model, dataset, _, _ = load_checkpoint()
    evaluator, adapter = SASRecEvaluator(model), DataAdapter(dataset)
    by_user = {x["user_id"]: x for x in users}
    rows = []
    for record in records:
        user = by_user[record["user_id"]]
        path = record["RAW_LLM_IPP_PATH"] if "RAW_LLM_IPP_PATH" in record else record["parsed_path"]
        parse_ok = record.get("parser", {}).get("parse_success", record.get("parse_success", False))
        target = resolver.resolve(user["target"]["title"])
        history = [x["movie_id"] for x in user["history"]]
        prepared = prepare_path({"path": path, "parse_success": parse_ok}, resolver, target, set(history), set())
        scored, _ = evaluate_prepared(prepared, target, adapter.history_raw_ids_to_internal(history), adapter, evaluator)
        diag = raw_diagnostic(record, user, resolver)
        m = scored["metrics"]
        rows.append({"Method": record["method"], "USER": record["user_id"], "PATH_ID": record["path_index"],
            "TARGET": user["target"]["title"], "RAW_PATH": path, **diag,
            "IoI": m["IoI"], "IoR": m["IoR"], "Proxy": m["ProxyAcceptability"],
            "Coherence": m["Coherence"], "EVALUATOR_VALID": scored["FORMAL_METRIC_STATUS"] == "valid",
            "FORMAL_METRIC_STATUS": scored["FORMAL_METRIC_STATUS"]})
    return rows


def mean(values):
    values = [x for x in values if x is not None]
    return statistics.mean(values) if values else None


def aggregate(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["EVALUATOR_VALID"]]
    n = len(rows)
    return {"Valid Paths": len(valid), "Total Paths": n,
        "IoI": mean(r["IoI"] for r in valid), "IoR": mean(r["IoR"] for r in valid),
        "Proxy": mean(r["Proxy"] for r in valid), "Coherence": mean(r["Coherence"] for r in valid),
        "HistoryReuseRate": mean(r["HISTORY_REUSE_RATE"] for r in rows),
        "TargetPresenceRate": sum(r["RAW_TARGET_PRESENT"] for r in rows) / n,
        "TargetLastRate": sum(r["RAW_TARGET_LAST"] for r in rows) / n,
        "IntraPathDuplicateRate": mean(r["INTRA_PATH_DUPLICATE_RATE"] for r in rows),
        "PathsWithHistoryReuse": sum(r["HISTORY_REUSE_COUNT"] > 0 for r in rows),
        "PathsTargetMissing": sum(not r["RAW_TARGET_PRESENT"] for r in rows),
        "PathsTargetPresentButNotLast": sum(r["RAW_TARGET_PRESENT"] and not r["RAW_TARGET_LAST"] for r in rows),
        "PathsWithIntraPathDuplicates": sum(r["INTRA_PATH_DUPLICATE_COUNT"] > 0 for r in rows)}


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row.get(k), list) else row.get(k) for k in columns})


def rel(base, new):
    return (new - base) / abs(base) * 100 if base is not None and base > 0 else None


def main():
    resume = OUT.exists() and (OUT / "llmipp_paths.json").is_file() and (OUT / "smr_mimar_g_paths.json").is_file()
    if OUT.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite incomplete/unrecognized output {OUT}")
    if PROTOCOL_VERSION != "Formal-Evaluation-v1" or sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Frozen evaluator/checkpoint mismatch")
    before = {"smr_method": digest_tree(ROOT / "repro/methods/smr_mimar_g"),
              "formal": digest_tree(ROOT / "repro/evaluators/formal"),
              "smr_results": digest_tree(SMR)}
    manifest = read_json(ROOT / "repro/results/pilot/pilot_manifest.json")
    by_user = {u["user_id"]: u for u in manifest["users"]}
    users = [by_user[uid] for uid in USERS]
    _, version = local_llm.request("/api/version", timeout=15)
    _, tags = local_llm.request("/api/tags", timeout=15)
    cfg = read_json(local_llm.CONFIG)
    installed = next((m for m in tags["models"] if m["name"] == cfg["model"]), None)
    if installed is None or installed["digest"] != cfg["model_digest"]:
        raise RuntimeError("Frozen local model/digest mismatch")
    if [u["user_id"] for u in users] != USERS:
        raise RuntimeError("Frozen user order mismatch")
    expected_targets = {419:"Clean Slate (Coup de Torchon) (1981)",5021:"Drunken Master (Zui quan) (1979)",
        2677:"Jingle All the Way (1996)",3113:"League of Their Own, A (1992)",2249:"Timecop (1994)"}
    if any(u["target"]["title"] != expected_targets[u["user_id"]] for u in users):
        raise RuntimeError("Frozen target mismatch")
    if resume:
        llmipp = read_json(OUT / "llmipp_paths.json")
        smr_records = read_json(OUT / "smr_mimar_g_paths.json")
        print("resuming evaluation from 10 saved LLM-IPP paths; no LLM generation", flush=True)
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        llmipp = []
        for user in users:
            for index in (1, 2):
                started = time.perf_counter()
                record = generate(user, index)
                record["wall_elapsed_seconds"] = time.perf_counter() - started
                llmipp.append(record)
                print(f"generated {len(llmipp)}/10 user={user['user_id']} path={index} parse={record['parser']['parse_success']}", flush=True)
        smr_records = []
        for user in users:
            for index in (1, 2):
                source = read_json(SMR / "generation" / str(user["user_id"]) / f"path_{index}.json")
                smr_records.append({"method":"SMR-MIMAR-G", "user_id":user["user_id"], "path_index":index,
                    "target":source["target"], "parsed_path":source["parsed_path"], "parse_success":source["parse_success"],
                    "target_appended_by_normalization":source.get("target_appended_by_normalization"),
                    "source_file":str(SMR / "generation" / str(user["user_id"]) / f"path_{index}.json")})
        write_json(OUT / "llmipp_paths.json", llmipp)
        write_json(OUT / "smr_mimar_g_paths.json", smr_records)
    resolver = TitleResolver.from_movies_dat(ROOT / "dataset/ml-1m/movies.dat")
    llm_rows = evaluate(llmipp, users, resolver)
    smr_rows = evaluate(smr_records, users, resolver)
    llm_agg, smr_agg = aggregate(llm_rows), aggregate(smr_rows)
    write_csv(OUT / "llmipp_path_validity.csv", llm_rows, list(llm_rows[0]))
    write_csv(OUT / "smr_path_validity.csv", smr_rows, list(smr_rows[0]))
    per = []
    for left, right in zip(llm_rows, smr_rows):
        per.append({"USER":left["USER"],"PATH_ID":left["PATH_ID"],
            **{f"LLMIPP_{k}":left[k] for k in ("IoI","IoR","Proxy","Coherence","HISTORY_REUSE_RATE","RAW_TARGET_PRESENT","RAW_TARGET_LAST","INTRA_PATH_DUPLICATE_RATE","EVALUATOR_VALID")},
            **{f"SMR_{k}":right[k] for k in ("IoI","IoR","Proxy","Coherence","HISTORY_REUSE_RATE","RAW_TARGET_PRESENT","RAW_TARGET_LAST","INTRA_PATH_DUPLICATE_RATE","EVALUATOR_VALID")}})
    write_csv(OUT / "per_path_comparison.csv", per, list(per[0]))
    table = [{"Method":"LLM-IPP-style same-protocol", **llm_agg}, {"Method":"SMR-MIMAR-G", **smr_agg}]
    write_csv(OUT / "comparison_table.csv", table, list(table[0]))
    deltas = {f"DELTA_{k.upper()}": smr_agg[k]-llm_agg[k] for k in ("IoI","IoR","Proxy","Coherence")}
    changes = {"RELATIVE_IOI_CHANGE_PERCENT":rel(llm_agg["IoI"],smr_agg["IoI"]),
               "RELATIVE_IOR_CHANGE_PERCENT":rel(llm_agg["IoR"],smr_agg["IoR"]),
               "HISTORY_REUSE_REDUCTION":llm_agg["HistoryReuseRate"]-smr_agg["HistoryReuseRate"],
               "TARGET_PRESENCE_GAIN":smr_agg["TargetPresenceRate"]-llm_agg["TargetPresenceRate"],
               "TARGET_LAST_GAIN":smr_agg["TargetLastRate"]-llm_agg["TargetLastRate"],
               "DUPLICATE_REDUCTION":llm_agg["IntraPathDuplicateRate"]-smr_agg["IntraPathDuplicateRate"]}
    after = {"smr_method": digest_tree(ROOT / "repro/methods/smr_mimar_g"),
             "formal": digest_tree(ROOT / "repro/evaluators/formal"), "smr_results": digest_tree(SMR)}
    if after != before:
        raise RuntimeError("Protected SMR/evaluator files changed")
    summary = {"completed":True,"users":USERS,"paths_per_user":2,
        "baseline_label":"faithful local LLM-IPP-style same-protocol reproduction; not original GPT paper result",
        "baseline_protocol_disclosure":"Independent two-turn local implementation frozen before generation; differs from released source prompt by the task-required five-intermediate and explicit target-final constraints.",
        "title_resolution":"Formal-Evaluation-v1 exact match, then unique The/A/An article normalization only; no semantic repair.",
        "smr_endpoint_disclosure":"SMR-MIMAR-G appends the predefined target by protocol normalization; target presence/last are guaranteed by protocol, not learned success.",
        "LLM-IPP-style same-protocol":llm_agg,"SMR-MIMAR-G":smr_agg,"deltas":{**deltas,**changes},
        "limitations":["5 users and 2 paths per user only","fixed seed can yield duplicated paired outputs",
            "local Qwen baseline is not the original GPT-based LLM-IPP result","no significance or general-superiority claim"],
        "protection":{"FORMAL_EVALUATOR_MODIFIED":False,"SMR_MIMAR_G_MODIFIED":False,"LLMIPP_TUNED_AFTER_RESULTS":False}}
    write_json(OUT / "comparison_summary.json", summary)
    report = "# LLM-IPP-style vs SMR-MIMAR-G: same-protocol comparison\n\n"
    report += "This is a five-user local Qwen case comparison, not a reproduction of the original GPT-based paper result.\n\n"
    report += "| Method | Valid | IoI | IoR | Proxy | Coherence | History reuse | Target present | Target last | Duplicate rate |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for row in table:
        report += f"| {row['Method']} | {row['Valid Paths']}/10 | {row['IoI']} | {row['IoR']} | {row['Proxy']} | {row['Coherence']} | {row['HistoryReuseRate']} | {row['TargetPresenceRate']} | {row['TargetLastRate']} | {row['IntraPathDuplicateRate']} |\n"
    report += "\nSMR-MIMAR-G target presence/last are guaranteed by endpoint normalization. Formal metrics use only strict evaluator-valid paths. No significance or general superiority is claimed.\n"
    (OUT / "comparison_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
