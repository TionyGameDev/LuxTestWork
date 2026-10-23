import csv
import hashlib
from dataclasses import dataclass
from datetime import date, datetime

DATE_FORMAT = "%d.%m.%Y"
REQUIRED = ("DATE", "SKU", "RATING", "TEXT")


@dataclass(frozen=True)
class Review:
    id: str
    day: date
    sku: str
    rating: int
    text: str


def make_id(day, sku, text):
    raw = f"{day.isoformat()}|{sku}|{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:8]


def load_reviews(path):
    """Читает CSV. Битые строки не роняют запуск, а попадают в список skipped с причиной."""
    reviews, skipped, seen = [], [], set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"в {path} нет колонок: {', '.join(missing)}")
        for n, row in enumerate(reader, start=2):
            try:
                day = datetime.strptime(row["DATE"].strip(), DATE_FORMAT).date()
                rating = int(row["RATING"])
                text = row["TEXT"].strip()
                sku = row["SKU"].strip()
            except (ValueError, AttributeError) as e:
                skipped.append(f"строка {n}: {e}")
                continue
            if not 1 <= rating <= 5:
                skipped.append(f"строка {n}: рейтинг {rating} вне 1-5")
                continue
            if not text or not sku:
                skipped.append(f"строка {n}: пустой текст или SKU")
                continue
            rid = make_id(day, sku, text)
            if rid in seen:
                skipped.append(f"строка {n}: дубликат")
                continue
            seen.add(rid)
            reviews.append(Review(rid, day, sku, rating, text))
    return reviews, skipped
