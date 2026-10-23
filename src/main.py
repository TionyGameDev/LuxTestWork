import argparse
import os
import sys
from datetime import datetime

from .aggregate import summarize
from .classify import LLMError, classify
from .config import REVIEWS_PATH, load_env
from .load import DATE_FORMAT, load_reviews
from .report import render
from .telegram import TelegramError, send_message


def main(argv=None):
    load_env()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Еженедельная сводка по отзывам")
    p.add_argument("--csv", default=str(REVIEWS_PATH), help="файл с отзывами (DATE,SKU,RATING,TEXT)")
    p.add_argument("--provider", choices=["llm", "openai", "keywords"], default="llm",
                   help="llm: Anthropic; openai: OpenAI-совместимый API (OpenRouter и др.); keywords: без ключей")
    p.add_argument("--end", help="последний день недели, ДД.ММ.ГГГГ (по умолчанию последний день в данных)")
    p.add_argument("--force", action="store_true", help="игнорировать кэш разметки")
    p.add_argument("--send", action="store_true", help="отправить в Telegram (иначе только печать)")
    args = p.parse_args(argv)

    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if args.send and not (token and chat):
        p.error("для --send нужны TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID (см. .env.example)")

    try:
        reviews, skipped = load_reviews(args.csv)
        for msg in skipped:
            print(f"пропущено: {msg}", file=sys.stderr)
        if not reviews:
            print("нет подходящих отзывов", file=sys.stderr)
            return 1
        end = datetime.strptime(args.end, DATE_FORMAT).date() if args.end else None
        labels, _ = classify(reviews, args.provider, args.force, log=lambda m: print(m, file=sys.stderr))
        messages = render(summarize(reviews, labels, end))
    except (OSError, ValueError, LLMError) as e:
        print(f"ошибка: {e}", file=sys.stderr)
        return 1

    for text in messages:
        print(text, end="\n\n")
    if args.send:
        try:
            for text in messages:
                send_message(token, chat, text)
        except TelegramError as e:
            print(f"ошибка отправки: {e}", file=sys.stderr)
            return 1
        print(f"отправлено сообщений: {len(messages)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
