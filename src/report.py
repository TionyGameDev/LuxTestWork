from html import escape

from .config import TELEGRAM_LIMIT, THEMES

SMALL_SAMPLE = 30


def _name(theme):
    return THEMES[theme][0]


def _quote(review, limit=140):
    text = review.text if len(review.text) <= limit else review.text[:limit].rsplit(" ", 1)[0] + "…"
    return f"«{escape(text)}» ({review.sku}, {review.rating}★)"


def _num(x):
    return "—" if x is None else f"{x:.1f}"


def render(s):
    """Собирает HTML-сводку и режет её на сообщения не длиннее лимита Telegram."""
    out = [f"📊 <b>Отзывы Luxhommè: {s['start']:%d.%m}–{s['end']:%d.%m}</b>\n"
           f"Отзывов: {s['n']} (неделей ранее {s['prev_n']}), "
           f"средняя оценка {_num(s['avg'])} (было {_num(s['prev_avg'])})\n"
           f"С жалобами: {s['complaint_reviews']} из {s['n']} "
           f"(было {s['prev_complaint_reviews']} из {s['prev_n']})"]

    rows = sorted(s["themes"].items(), key=lambda kv: -kv[1]["reviews"])
    out.append("📋 <b>Темы</b> (отзывов / жалоб / похвал)\n" + "\n".join(
        f"• {_name(t)}: {v['reviews']} / {v['neg']} / {v['pos']}" for t, v in rows))

    if s["top"]:
        lines = ["🔥 <b>На что жалуются</b>"]
        for i, item in enumerate(s["top"], 1):
            lines.append(f"{i}. {_name(item['theme'])}: {item['count']} (неделей ранее {item['prev']})")
            lines += [f"   {_quote(r)}" for r in item["examples"]]
        out.append("\n".join(lines))

    if s["changes"]:
        icon = {"new": "🆕 новое", "up": "📈 растёт", "down": "📉 снижается"}
        out.append("🔄 <b>Что изменилось к прошлой неделе</b>\n" + "\n".join(
            f"{icon[c['kind']]}: {_name(c['theme'])}, жалоб {c['prev']} → {c['cur']}" for c in s["changes"]))

    sku_lines = [f"• {k['sku']}: {_num(k['prev'])} → {_num(k['cur'])} ({k['prev_n']} → {k['n']} отзывов)"
                 for k in s["skus"]]
    out.append("🏷 <b>Средняя оценка по товарам</b>\n" + "\n".join(sku_lines))

    if s["mismatch"]:
        out.append(f"🧐 <b>Оценка и текст расходятся: {len(s['mismatch'])}</b>\n" +
                   "\n".join(_quote(r) for r in s["mismatch"][:2]))
    if s["pending"]:
        out.append(f"⚠️ Не разобрано из-за сбоя модели: {s['pending']}")
    if s["n"] < SMALL_SAMPLE:
        out.append("<i>Выборка небольшая, выводы предварительные.</i>")
    return split_messages(out)


def split_messages(blocks, limit=TELEGRAM_LIMIT):
    messages, current = [], ""
    for block in blocks:
        if current and len(current) + len(block) + 2 > limit:
            messages.append(current)
            current = ""
        current = f"{current}\n\n{block}" if current else block
    if current:
        messages.append(current)
    return messages
