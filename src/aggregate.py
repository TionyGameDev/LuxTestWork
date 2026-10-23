from collections import Counter
from datetime import timedelta

from .config import MIN_SIGNAL


def split_weeks(reviews, end=None):
    """Текущая неделя: 7 дней, заканчивающихся в `end` (по умолчанию последний день в данных)."""
    end = end or max(r.day for r in reviews)
    start = end - timedelta(days=6)
    prev_start = start - timedelta(days=7)
    cur = [r for r in reviews if start <= r.day <= end]
    prev = [r for r in reviews if prev_start <= r.day < start]
    return start, end, cur, prev


def _topics(labels, rid, sentiment):
    return {t for t, s in labels.get(rid, []) if s == sentiment}


def _avg(rs):
    return sum(r.rating for r in rs) / len(rs) if rs else None


def complaint_counts(reviews, labels):
    c = Counter()
    for r in reviews:
        c.update(t for t in _topics(labels, r.id, "neg") if t != "other")
    return c


def trend(cur, prev):
    if cur >= MIN_SIGNAL and prev == 0:
        return "new"
    if cur >= MIN_SIGNAL and cur - prev >= 2:
        return "up"
    if prev >= MIN_SIGNAL and prev - cur >= 2:
        return "down"
    return None


def summarize(reviews, labels, end=None):
    """Все числа считает код. LLM только ставит метки, цитаты берутся из исходных текстов."""
    start, end, cur_all, prev_all = split_weeks(reviews, end)
    cur = [r for r in cur_all if r.id in labels]
    prev = [r for r in prev_all if r.id in labels]
    pending = len(cur_all) - len(cur)

    themes = {}
    for r in cur:
        for t, s in labels[r.id]:
            row = themes.setdefault(t, {"reviews": 0, "neg": 0, "pos": 0})
            row["reviews"] += 1
            row["neg"] += s == "neg"
            row["pos"] += s == "pos"

    cur_neg, prev_neg = complaint_counts(cur, labels), complaint_counts(prev, labels)
    top = []
    for t, n in cur_neg.items():
        rs = sorted((r for r in cur if t in _topics(labels, r.id, "neg")), key=lambda r: (r.rating, r.text))
        top.append({"theme": t, "count": n, "prev": prev_neg.get(t, 0), "avg": _avg(rs), "reviews": rs})
    top.sort(key=lambda x: (-x["count"], x["avg"]))
    top = top[:3]
    shown = set()  # один отзыв не показываем примером сразу для нескольких тем
    for item in top:
        fresh = [r for r in item["reviews"] if r.id not in shown] or item["reviews"]
        item["examples"] = fresh[:2]
        shown.update(r.id for r in item["examples"])

    changes = []
    for t in sorted(set(cur_neg) | set(prev_neg)):
        kind = trend(cur_neg.get(t, 0), prev_neg.get(t, 0))
        if kind:
            changes.append({"theme": t, "kind": kind, "cur": cur_neg.get(t, 0), "prev": prev_neg.get(t, 0)})

    skus = []
    for sku in sorted({r.sku for r in cur}):
        c = [r for r in cur if r.sku == sku]
        p = [r for r in prev if r.sku == sku]
        skus.append({"sku": sku, "cur": _avg(c), "prev": _avg(p), "n": len(c), "prev_n": len(p)})

    mismatch = []
    for r in cur:
        neg, pos = _topics(labels, r.id, "neg"), _topics(labels, r.id, "pos")
        if (r.rating >= 4 and neg and not pos) or (r.rating <= 2 and pos and not neg):
            mismatch.append(r)

    def with_complaint(rs):
        return sum(1 for r in rs if _topics(labels, r.id, "neg") - {"other"})

    return {
        "start": start, "end": end,
        "n": len(cur), "prev_n": len(prev), "pending": pending,
        "avg": _avg(cur), "prev_avg": _avg(prev),
        "complaint_reviews": with_complaint(cur), "prev_complaint_reviews": with_complaint(prev),
        "themes": themes, "top": top, "changes": changes,
        "skus": skus, "mismatch": mismatch,
    }
