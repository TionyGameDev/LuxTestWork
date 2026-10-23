import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = ROOT / "prompts" / "classify.md"
CACHE_PATH = ROOT / "data" / "cache" / "labels.json"
REVIEWS_PATH = ROOT / "data" / "reviews.csv"
GOLD_PATH = ROOT / "data" / "gold.csv"
RESULTS_DIR = ROOT / "results"

# Единственное место, где перечислены темы: промпт, оценка и сводка берут их отсюда.
# id -> (название для сводки, подсказка для модели)
THEMES = {
    "performance": ("Работа и результат", "как техника справляется с задачей: убирает, готовит, варит, отпаривает"),
    "durability": ("Надёжность и качество", "поломки, брак, течь, не включается, скрип, неприятный запах пластика"),
    "noise": ("Шум", "громкость работы"),
    "delivery": ("Доставка", "сроки доставки и работа курьера"),
    "packaging": ("Упаковка", "состояние коробки при получении"),
    "usability": ("Удобство", "мытьё, вес, размер, шнур, приложение и подключение"),
    "price": ("Цена", "соотношение цены и ценности"),
    "support": ("Поддержка и возврат", "ответы продавца, возврат денег, замена, гарантия"),
    "other": ("Другое", "отзыв не про свойства товара: пустое «супер», ошибка продавца, не тот цвет"),
}
SENTIMENTS = ("pos", "neg", "neu")

BATCH_SIZE = 10
RETRIES = 3
MIN_SIGNAL = 3          # меньше стольких жалоб не называем это трендом
TELEGRAM_LIMIT = 4000   # лимит Telegram 4096, оставляем запас
DEFAULT_MODEL = "claude-haiku-5-5"
# OpenAI-совместимый провайдер; по умолчанию OpenRouter с бесплатной моделью (список меняется,
# актуальные бесплатные модели: https://openrouter.ai/models?max_price=0)
DEFAULT_OPENAI_BASE = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


def load_env(path=None):
    """Минимальный разбор .env, чтобы не тянуть зависимость. Уже заданные переменные не трогает."""
    path = Path(path) if path else ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
