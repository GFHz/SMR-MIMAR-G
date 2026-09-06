"""Audit unresolved pilot titles without changing paths, resolver, or formal metrics."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from repro.experiments.pilot.evaluate_pilot_formal import (
    CHECKPOINT, EXPECTED_CHECKPOINT_SHA, EXPECTED_MANIFEST_SHA, GENERATION, MANIFEST,
    canonical_manifest_sha, directory_freeze, protected_snapshot, read, sha,
)

ROOT = Path(__file__).resolve().parents[3]
FORMAL = ROOT / "repro/results/pilot/formal_evaluation_v1"
OUT = ROOT / "repro/results/pilot/catalog_grounding_audit"
EXPECTED_GENERATION_SHA = "f35a442065a34e10b790411d0e3f9ce315a3df89ccd80fbc1da0e1355b72ef56"

# Curated existence evidence is intentionally separate from deterministic catalog checks.
# No entry is used to alter or rescore a generated path.
EVIDENCE = {
    "My Stepmother Is an Alien (1988)": ("A_REAL_OFF_CATALOG", "My Stepmother Is an Alien", 1988, True, "https://en.wikipedia.org/wiki/My_Stepmother_Is_an_Alien"),
    "The Best Man (1996)": ("C_LIKELY_HALLUCINATION_OR_METADATA_ERROR", "The Best Man", 1999, True, "https://en.wikipedia.org/wiki/The_Best_Man_(1999_film)"),
    "The Best Man Holiday (2008)": ("C_LIKELY_HALLUCINATION_OR_METADATA_ERROR", "The Best Man Holiday", 2013, True, "https://en.wikipedia.org/wiki/The_Best_Man_Holiday"),
    "The Breakfast Club 2 (1992)": ("C_LIKELY_HALLUCINATION_OR_METADATA_ERROR", None, None, False, "https://www.slashfilm.com/1434121/the-breakfast-club-2-is-it-happening/"),
    "The Cabinet of Dr. Caligari (1920)": ("A_REAL_OFF_CATALOG", "The Cabinet of Dr. Caligari", 1920, True, "https://en.wikipedia.org/wiki/The_Cabinet_of_Dr._Caligari"),
    "The Chronicles of Narnia: The Lion, the Witch and the Wardrobe (2005)": ("A_REAL_OFF_CATALOG", "The Chronicles of Narnia: The Lion, the Witch and the Wardrobe", 2005, True, "https://en.wikipedia.org/wiki/The_Chronicles_of_Narnia:_The_Lion,_the_Witch_and_the_Wardrobe"),
    "The Departed (2006)": ("A_REAL_OFF_CATALOG", "The Departed", 2006, True, "https://en.wikipedia.org/wiki/The_Departed"),
    "The Faculty's Prequel: The Faculty 2 (1999)": ("C_LIKELY_HALLUCINATION_OR_METADATA_ERROR", "The Faculty 2", 2007, "uncertain", "https://www.imdb.com/title/tt1473424/"),
    "The Good, the Bad and the Ugly (1966)": ("B_NORMALIZATION_RESOLVABLE", "The Good, the Bad and the Ugly", 1966, True, "https://en.wikipedia.org/wiki/The_Good,_the_Bad_and_the_Ugly"),
    "The Incredibles (2004)": ("A_REAL_OFF_CATALOG", "The Incredibles", 2004, True, "https://en.wikipedia.org/wiki/The_Incredibles"),
    "The Last Samurai (2003)": ("A_REAL_OFF_CATALOG", "The Last Samurai", 2003, True, "https://en.wikipedia.org/wiki/The_Last_Samurai"),
    "The Man in the High Castle (1990)": ("C_LIKELY_HALLUCINATION_OR_METADATA_ERROR", "The Man in the High Castle", 2015, True, "https://en.wikipedia.org/wiki/The_Man_in_the_High_Castle_(TV_series)"),
    "The Matrix Reloaded (2003)": ("A_REAL_OFF_CATALOG", "The Matrix Reloaded", 2003, True, "https://en.wikipedia.org/wiki/The_Matrix_Reloaded"),
    "The Secret Life of Pets (2016)": ("A_REAL_OFF_CATALOG", "The Secret Life of Pets", 2016, True, "https://en.wikipedia.org/wiki/The_Secret_Life_of_Pets"),
    "The Secret Life of Walter Mitty (2013)": ("A_REAL_OFF_CATALOG", "The Secret Life of Walter Mitty", 2013, True, "https://en.wikipedia.org/wiki/The_Secret_Life_of_Walter_Mitty_(2013_film)"),
    "The Texas Chain Saw Massacre (1974)": ("C_LIKELY_HALLUCINATION_OR_METADATA_ERROR", "The Texas Chain Saw Massacre", 1974, True, "https://en.wikipedia.org/wiki/The_Texas_Chain_Saw_Massacre"),
    "The Wicker Man (1973)": ("A_REAL_OFF_CATALOG", "The Wicker Man", 1973, True, "https://en.wikipedia.org/wiki/The_Wicker_Man"),
}


def load_catalog():
    rows = []
    with (ROOT / "dataset/ml-1m/movies.dat").open(encoding="latin-1") as stream:
        for line in stream:
            movie_id, title, genres = line.rstrip("\r\n").split("::")
            rows.append({"movie_id": int(movie_id), "title": title, "genres": genres.split("|")})
    return rows


def allowed_key(title: str):
    """Audit-only comparison: article movement, whitespace, and case only."""
    title = " ".join(title.split())
    match = re.fullmatch(r"(.+) \((\d{4})\)", title)
    if not match:
        return None
    body, year = match.groups()
    lead = re.match(r"^(The|A|An) (.+)$", body, re.I)
    trail = re.fullmatch(r"(.+), (The|A|An)", body, re.I)
    article, stem = None, body
    if lead:
        article, stem = lead.group(1), lead.group(2)
    if trail:
        article, stem = trail.group(2), trail.group(1)
    return (" ".join(stem.split()).casefold(), article.casefold() if article else None, year)


def tree_hash(folder: Path):
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def save(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run():
    if OUT.exists():
        raise FileExistsError("catalog_grounding_audit exists; refusing overwrite")
    manifest = read(MANIFEST)
    if canonical_manifest_sha(manifest) != EXPECTED_MANIFEST_SHA or manifest["manifest_sha256"] != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("manifest hash mismatch")
    generation_before = directory_freeze(GENERATION)
    if generation_before["generation_v1_sha256"] != EXPECTED_GENERATION_SHA:
        raise RuntimeError("generation hash mismatch")
    formal_before, protected_before = tree_hash(FORMAL), protected_snapshot()
    if sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("checkpoint hash mismatch")

    rows = read(FORMAL / "path_level_metrics.json")
    occurrences = defaultdict(list)
    for row in rows:
        for title in row["unresolved_items"]:
            occurrences[title].append({"user_id": row["user_id"], "method": row["method"], "path_index": row["path_index"]})
    if set(occurrences) != set(EVIDENCE) or len(occurrences) != 17 or sum(map(len, occurrences.values())) != 30:
        raise RuntimeError("Frozen unresolved inventory differs from audited 17-title scope")

    catalog = load_catalog()
    by_key = defaultdict(list)
    for movie in catalog:
        by_key[allowed_key(movie["title"])].append(movie)
    audit = []
    for title in sorted(occurrences):
        category, canonical, year, exists, source = EVIDENCE[title]
        matches = by_key.get(allowed_key(title), [])
        if category == "B_NORMALIZATION_RESOLVABLE" and len(matches) != 1:
            raise RuntimeError(f"B title is not uniquely normalization-resolvable: {title}")
        audit.append({"raw_title": title, "occurrence_count": len(occurrences[title]),
                      "occurrences": occurrences[title], "classification": category,
                      "exists_as_real_film": exists, "canonical_real_title": canonical,
                      "canonical_year": year, "verification_source": source,
                      "movieLens_exact_match": any(x["title"] == title for x in catalog),
                      "audit_normalization_matches": matches,
                      "formal_result_changed": False})

    method_summary = {}
    for method, label in (("baseline", "Baseline"), ("mi_bridge", "MI-Bridge")):
        group = [x for x in rows if x["method"] == method]
        invalid = [x for x in group if x["formal_metric_status"] != "valid"]
        titles = [t for x in group for t in x["unresolved_items"]]
        categories = Counter(EVIDENCE[t][0] for t in titles)
        unique_categories = Counter(EVIDENCE[t][0] for t in set(titles))
        method_summary[label] = {"total_paths": len(group), "valid_paths": len(group) - len(invalid),
            "invalid_paths": len(invalid), "unresolved_occurrences": len(titles),
            "unique_unresolved_titles": len(set(titles)), "CatalogResolvablePathRate": (len(group)-len(invalid))/len(group),
            "classification_occurrence_counts": {k: categories.get(k, 0) for k in ("A_REAL_OFF_CATALOG", "B_NORMALIZATION_RESOLVABLE", "C_LIKELY_HALLUCINATION_OR_METADATA_ERROR", "D_UNVERIFIED")},
            "classification_unique_title_counts": {k: unique_categories.get(k, 0) for k in ("A_REAL_OFF_CATALOG", "B_NORMALIZATION_RESOLVABLE", "C_LIKELY_HALLUCINATION_OR_METADATA_ERROR", "D_UNVERIFIED")}}

    records = [read(p) for p in sorted((GENERATION / "users").glob("*/*.json"))]
    baseline_grounded = any(r["method"] == "baseline" and r["bridge_candidates"] for r in records)
    mi_grounded = all(r["method"] != "mi_bridge" or bool(r["bridge_candidates"]) for r in records)
    prompt = {"BASELINE_HAS_CATALOG_GROUNDED_CANDIDATES": baseline_grounded,
              "MI_BRIDGE_HAS_CATALOG_GROUNDED_CANDIDATES": mi_grounded,
              "CATALOG_GROUNDING_ASYMMETRY": baseline_grounded != mi_grounded,
              "evidence": "Frozen Baseline records contain history+target and null bridge_candidates; frozen MI-Bridge records contain manifest-derived MovieLens bridge_candidates.",
              "prompts_modified": False}
    fairness = {"FAIRNESS_RISK": "HIGH", "PAIRED_VALID_ANALYSIS_STILL_USABLE": True,
                "RECOMMENDED_EXPERIMENT_ACTION": "RUN_SMALL_CATALOG_GROUNDED_ABLATION",
                "RECOMMENDED_FAIRNESS_FIX": "Use the same 10 users, targets, model and settings; give both methods the same catalog-grounding constraint/candidate access, leaving bridge context as MI-Bridge's only extra information.",
                "basis": "Valid-rate gap is 55% vs 95%; Baseline has 26 unresolved occurrences across 9 paths versus MI-Bridge 4 across 1 path. Paired-valid analysis reduces but cannot remove selection/denominator risk.",
                "formal_metrics_recomputed": False, "significance_test_performed": False}

    if directory_freeze(GENERATION) != generation_before or tree_hash(FORMAL) != formal_before or protected_snapshot() != protected_before:
        raise RuntimeError("protected input changed during audit")
    OUT.mkdir(parents=True)
    save("unresolved_title_audit.json", {"unresolved_path_count": sum(bool(x["unresolved_items"]) for x in rows),
         "unresolved_occurrence_count": sum(map(len, occurrences.values())), "unique_unresolved_title_count": len(occurrences),
         "titles": audit, "classification_is_diagnostic_only": True})
    save("method_grounding_summary.json", method_summary)
    save("prompt_grounding_comparison.json", prompt)
    save("fairness_assessment.json", fairness)
    print(json.dumps({"unresolved_paths": sum(bool(x["unresolved_items"]) for x in rows), "method_summary": method_summary,
                      "prompt": prompt, "fairness": fairness}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
