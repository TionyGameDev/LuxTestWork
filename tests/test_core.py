import unittest
from datetime import date

from src.aggregate import split_weeks, summarize, trend
from src.classify import RateLimitError, call_with_backoff, extract_json, validate
from src.load import Review, make_id
from src.report import render, split_messages


def review(day, sku="A", rating=3, text="текст"):
    d = date(2026, 10, day)
    return Review(make_id(d, sku, text + str(day)), d, sku, rating, text + str(day))


class ValidateTests(unittest.TestCase):
    def test_accepts_good_answer(self):
        answer = {"results": [{"id": "a", "labels": [{"topic": "noise", "sentiment": "neg"}]}]}
        good, errors = validate(answer, {"a"})
        self.assertEqual(good, {"a": [["noise", "neg"]]})
        self.assertEqual(errors, [])

    def test_rejects_unknown_theme_and_missing_id(self):
        answer = {"results": [{"id": "a", "labels": [{"topic": "color", "sentiment": "neg"}]}]}
        good, errors = validate(answer, {"a", "b"})
        self.assertEqual(good, {})
        self.assertEqual(len(errors), 2)

    def test_rejects_duplicate_theme(self):
        labels = [{"topic": "noise", "sentiment": "neg"}, {"topic": "noise", "sentiment": "pos"}]
        good, errors = validate({"results": [{"id": "a", "labels": labels}]}, {"a"})
        self.assertEqual(good, {})
        self.assertTrue(errors)

    def test_extract_json_strips_markdown_fence(self):
        self.assertEqual(extract_json('```json\n{"results": []}\n```'), {"results": []})


class BackoffTests(unittest.TestCase):
    def test_retries_rate_limit_then_succeeds(self):
        attempts = []

        def call(system, user):
            attempts.append(1)
            if len(attempts) < 3:
                raise RateLimitError("429")
            return "ok"

        self.assertEqual(call_with_backoff(call, "s", "u", log=lambda m: None, waits=(0, 0)), "ok")
        self.assertEqual(len(attempts), 3)

    def test_gives_up_after_waits_are_exhausted(self):
        def call(system, user):
            raise RateLimitError("429")

        with self.assertRaises(RateLimitError):
            call_with_backoff(call, "s", "u", log=lambda m: None, waits=(0,))


class AggregateTests(unittest.TestCase):
    def test_weeks_are_seven_days_each(self):
        reviews = [review(d) for d in (4, 5, 11)]
        start, end, cur, prev = split_weeks(reviews)
        self.assertEqual((start, end), (date(2026, 10, 5), date(2026, 10, 11)))
        self.assertEqual(len(cur), 2)
        self.assertEqual(len(prev), 1)

    def test_trend_needs_minimum_signal(self):
        self.assertIsNone(trend(2, 0))
        self.assertEqual(trend(3, 0), "new")
        self.assertEqual(trend(5, 2), "up")
        self.assertEqual(trend(1, 4), "down")

    def test_summary_counts_complaints_and_flags_mismatch(self):
        reviews = [review(5, rating=1), review(6, rating=5), review(7, rating=2)]
        labels = {
            reviews[0].id: [["noise", "neg"]],
            reviews[1].id: [["noise", "neg"]],
            reviews[2].id: [["performance", "pos"]],
        }
        s = summarize(reviews, labels)
        self.assertEqual(s["top"][0]["theme"], "noise")
        self.assertEqual(s["top"][0]["count"], 2)
        self.assertEqual(len(s["mismatch"]), 2)

    def test_same_review_is_not_example_for_two_themes(self):
        both = review(5, rating=1)
        only_noise = review(6, rating=2)
        only_support = review(7, rating=3)
        labels = {both.id: [["noise", "neg"], ["support", "neg"]],
                  only_noise.id: [["noise", "neg"]],
                  only_support.id: [["support", "neg"]]}
        top = summarize([both, only_noise, only_support], labels)["top"]
        examples = {t["theme"]: [r.id for r in t["examples"]] for t in top}
        self.assertEqual(examples["noise"], [both.id, only_noise.id])
        self.assertEqual(examples["support"], [only_support.id])

    def test_unlabeled_reviews_are_reported_as_pending(self):
        reviews = [review(5), review(6)]
        s = summarize(reviews, {reviews[0].id: [["noise", "neg"]]})
        self.assertEqual(s["pending"], 1)


class ReportTests(unittest.TestCase):
    def test_messages_respect_limit(self):
        parts = split_messages(["x" * 60] * 10, limit=150)
        self.assertTrue(all(len(p) <= 150 for p in parts))
        self.assertEqual(sum(p.count("x") for p in parts), 600)

    def test_quotes_are_escaped(self):
        r = review(5, text="<b>hack</b> ")
        s = summarize([r], {r.id: [["noise", "neg"]]})
        self.assertNotIn("<b>hack", "".join(render(s)))


if __name__ == "__main__":
    unittest.main()
