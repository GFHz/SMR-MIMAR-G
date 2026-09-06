"""Dataset-level genre co-occurrence feasibility."""
import json
import math
from pathlib import Path


class GenreCooccurrence:
    def __init__(self, movies):
        self.vocabulary = sorted({g for movie in movies for g in movie["genres"]})
        self.counts = {g: 0 for g in self.vocabulary}
        self.joint = {(a, b): 0 for a in self.vocabulary for b in self.vocabulary}
        for movie in movies:
            genres = set(movie["genres"])
            for a in genres:
                self.counts[a] += 1
                for b in genres:
                    self.joint[(a, b)] += 1

    def detail(self, interest, target_genre):
        ci, cg = self.counts.get(interest, 0), self.counts.get(target_genre, 0)
        both = self.joint.get((interest, target_genre), 0)
        score = both / math.sqrt(ci * cg) if ci and cg else 0.0
        return {"interest": interest, "target_genre": target_genre,
                "count_interest": ci, "count_target_genre": cg,
                "count_joint": both, "normalized_cooccurrence": score}

    def score(self, interest, target_genre):
        return self.detail(interest, target_genre)["normalized_cooccurrence"]

    def save_cache_new(self, path):
        path = Path(path)
        payload = {"vocabulary": self.vocabulary, "counts": self.counts,
                   "matrix": {a: {b: self.score(a, b) for b in self.vocabulary}
                              for a in self.vocabulary}}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
