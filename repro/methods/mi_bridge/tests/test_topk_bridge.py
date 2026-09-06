import copy
import unittest

from repro.methods.mi_bridge.select_bridge import BridgeRanking, select_bridge
from repro.methods.mi_bridge.topk_bridge import METHOD_VERSION, select_topk_bridges


def movie(mid, genres):
    return {"id": mid, "title": str(mid), "genres": genres}


def pair(rank, interest, target="Romance", score=1.0):
    return {"rank": rank, "user_existing_interest": interest, "target_related_interest": target,
            "bridge_score": score, "user_frequency": 0.5, "target_need": 0.5}


class TopKBridgeTests(unittest.TestCase):
    def ranking(self, pairs, movies=None):
        return BridgeRanking(pairs, movies or [], [], movie(999, ["Romance"]))

    def test_deterministic_and_distinct_preference(self):
        pairs = [pair(1, "Children's"), pair(2, "Animation"), pair(3, "Children's", "Drama"),
                 pair(4, "Animation", "Drama"), pair(5, "Comedy")]
        movies = [movie(1, ["Children's", "Romance"]), movie(2, ["Animation", "Romance"]),
                  movie(3, ["Children's", "Drama"]), movie(4, ["Animation", "Drama"]),
                  movie(5, ["Comedy", "Romance"])]
        ranking = self.ranking(pairs, movies)
        expected = ["Children's", "Animation", "Comedy"]
        self.assertEqual([x["existing_interest"] for x in select_topk_bridges(ranking)], expected)
        self.assertEqual(select_topk_bridges(ranking), select_topk_bridges(ranking))

    def test_fill_when_distinct_interests_are_insufficient(self):
        pairs = [pair(1, "A", "G1"), pair(2, "A", "G2"), pair(3, "B", "G1")]
        movies = [movie(1, ["A", "G1"]), movie(2, ["A", "G2"]), movie(3, ["B", "G1"])]
        selected = select_topk_bridges(self.ranking(pairs, movies), 3)
        self.assertEqual([(x["existing_interest"], x["target_interest"]) for x in selected],
                         [("A", "G1"), ("B", "G1"), ("A", "G2")])

    def test_k_larger_than_feasible_and_k_one(self):
        pairs = [pair(1, "A"), pair(2, "B")]
        movies = [movie(1, ["A", "Romance"]), movie(2, ["B", "Romance"])]
        ranking = self.ranking(pairs, movies)
        self.assertEqual(len(select_topk_bridges(ranking, 9)), 2)
        top1 = select_topk_bridges(ranking, 1)[0]
        v1 = select_bridge(ranking)
        self.assertEqual((top1["existing_interest"], top1["target_interest"]),
                         (v1["existing_interest"], v1["target_interest"]))

    def test_metadata_complete_and_input_not_mutated(self):
        pairs = [pair(1, "A")]
        ranking = self.ranking(pairs, [movie(1, ["A", "Romance"])])
        before = copy.deepcopy(list(ranking))
        selected = select_topk_bridges(ranking, 1)
        required = {"ranking_rank", "existing_interest", "target_interest", "bridge_score",
                    "user_frequency", "target_need", "mode", "candidate_count", "candidates",
                    "fallback_intermediate_genre", "source_bridge_pair"}
        self.assertTrue(required <= set(selected[0]))
        self.assertEqual(list(ranking), before)
        self.assertEqual(METHOD_VERSION, "MI-Bridge v2 Top-K Multi-Bridge Context")

    def test_unavailable_is_excluded_and_fallback_is_reused(self):
        pairs = [pair(1, "X"), pair(2, "A")]
        movies = [movie(1, ["A", "M"]), movie(2, ["M", "Romance"])]
        selected = select_topk_bridges(self.ranking(pairs, movies), 3)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["mode"], "multi_hop")
        self.assertEqual(selected[0]["fallback_intermediate_genre"], "M")

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            select_topk_bridges(self.ranking([], []), 0)


if __name__ == "__main__":
    unittest.main()
