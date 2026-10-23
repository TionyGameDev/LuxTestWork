"""Сравнение разметки с эталоном data/gold.csv.

Эталон разбит на dev (по нему правят промпт) и test (его не трогают при настройке).
"""
import argparse
import csv
import sys

from .classify import LLMError, classify
from .config import GOLD_PATH, REVIEWS_PATH, RESULTS_DIR, THEMES, load_env
from .load import load_reviews


def load_gold(path):
    gold = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pairs = {tuple(x.split(":")) for x in row["labels"].split("|") if x}
            gold[row["id"]] = (pairs, row["split"])
    return gold


def ratio(a, b):
    return a / b if b else 0.0


def evaluate(reviews, gold, predicted):
    tp = fp = fn = 0
    topic_tp = topic_fp = topic_fn = 0
    sentiment_ok = sentiment_total = exact = 0
    per_theme = {t: [0, 0, 0] for t in THEMES}  # в эталоне, угадано, лишних
    mismatches = []
    for r in reviews:
        g = gold[r.id][0]
        p = {tuple(x) for x in predicted.get(r.id, [])}
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        gt, pt = {t for t, _ in g}, {t for t, _ in p}
        topic_tp += len(gt & pt)
        topic_fp += len(pt - gt)
        topic_fn += len(gt - pt)
        gd, pd = dict(g), dict(p)
        for t in gt & pt:
            sentiment_ok += gd[t] == pd[t]
            sentiment_total += 1
        for t in gt:
            per_theme[t][0] += 1
            per_theme[t][1] += t in pt
        for t in pt - gt:
            per_theme[t][2] += 1
        if g == p:
            exact += 1
        else:
            mismatches.append((r, g, p))
    prec, rec = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {
        "n": len(reviews), "exact": exact, "precision": prec, "recall": rec,
        "f1": ratio(2 * prec * rec, prec + rec),
        "topic_f1": ratio(2 * topic_tp, 2 * topic_tp + topic_fp + topic_fn),
        "sentiment_acc": ratio(sentiment_ok, sentiment_total),
        "per_theme": per_theme, "mismatches": mismatches,
    }


def fmt(pairs):
    return ", ".join(f"{t}:{s}" for t, s in sorted(pairs)) or "—"


def report(m, title):
    lines = [title,
             f"Отзывов: {m['n']}, полное совпадение: {m['exact']}/{m['n']}",
             f"Метки тема+тон: precision {m['precision']:.0%}, recall {m['recall']:.0%}, F1 {m['f1']:.0%}",
             f"Только темы: F1 {m['topic_f1']:.0%}; тон среди угаданных тем: {m['sentiment_acc']:.0%}",
             "", "По темам (в эталоне / угадано / лишних):"]
    for t, (g, ok, extra) in m["per_theme"].items():
        if g or extra:
            lines.append(f"  {t}: {g} / {ok} / {extra}")
    lines += ["", f"Расхождения ({len(m['mismatches'])}):"]
    for r, g, p in m["mismatches"]:
        lines.append(f"  [{r.sku}, {r.rating}★] {r.text}\n      эталон: {fmt(g)}\n      модель: {fmt(p)}")
    return "\n".join(lines)


def main(argv=None):
    load_env()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Оценка точности разметки по эталону")
    p.add_argument("--provider", choices=["llm", "openai", "keywords"], default="llm")
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--force", action="store_true", help="игнорировать кэш разметки")
    args = p.parse_args(argv)

    try:
        reviews, _ = load_reviews(REVIEWS_PATH)
        gold = load_gold(GOLD_PATH)
        subset = [r for r in reviews if r.id in gold and args.split in ("all", gold[r.id][1])]
        predicted, pending = classify(subset, args.provider, args.force, log=lambda m: print(m, file=sys.stderr))
    except (OSError, ValueError, LLMError) as e:
        print(f"ошибка: {e}", file=sys.stderr)
        return 1

    text = report(evaluate(subset, gold, predicted), f"Провайдер: {args.provider}, выборка: {args.split}")
    if pending:
        text += f"\n\nНе разобрано моделью: {len(pending)} (считаются как пустые ответы)"
    print(text)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"eval_{args.provider}_{args.split}.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
