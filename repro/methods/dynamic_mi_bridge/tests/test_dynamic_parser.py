import unittest

from repro.methods.dynamic_mi_bridge.planner import parse_one_item


class DynamicParserRegressionTests(unittest.TestCase):
    def setUp(self):
        self.allowed = {
            "Something to Talk About (1995)",
            "I.Q. (1994)",
            "French Twist (Gazon maudit) (1995)",
            "French Kiss (1995)",
        }

    def test_accept_exact_bare_movielens_titles(self):
        for title in (
            "Something to Talk About (1995)",
            "I.Q. (1994)",
            "French Twist (Gazon maudit) (1995)",
        ):
            with self.subTest(title=title):
                self.assertEqual(parse_one_item(title, self.allowed),
                                 {"success": True, "title": title, "error": None})

    def test_accept_singleton_string_list(self):
        result = parse_one_item('["Something to Talk About (1995)"]', self.allowed)
        self.assertTrue(result["success"])
        self.assertEqual(result["title"], "Something to Talk About (1995)")

    def test_reject_explanation_numbering_and_bullet(self):
        rejected = (
            "Something to Talk About (1995) because it is a good choice.",
            "1. Something to Talk About (1995)",
            "- Something to Talk About (1995)",
        )
        for raw in rejected:
            with self.subTest(raw=raw):
                self.assertFalse(parse_one_item(raw, self.allowed)["success"])

    def test_reject_multiple_items_and_unknown_title(self):
        self.assertFalse(parse_one_item(
            '["Something to Talk About (1995)", "French Kiss (1995)"]', self.allowed)["success"])
        self.assertFalse(parse_one_item("Unknown Movie (1995)", self.allowed)["success"])

    def test_reject_malformed_containers(self):
        for raw in ('["Something to Talk About (1995)"',
                    'Something to Talk About (1995)]',
                    '(Something to Talk About (1995)'):
            with self.subTest(raw=raw):
                self.assertFalse(parse_one_item(raw, self.allowed)["success"])

    def test_no_fuzzy_matching(self):
        self.assertFalse(parse_one_item("something to talk about (1995)", self.allowed)["success"])


if __name__ == "__main__":
    unittest.main()
