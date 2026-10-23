import hashlib
import json
import os
import time
import urllib.error
import urllib.request

from .config import (BATCH_SIZE, CACHE_PATH, DEFAULT_MODEL, PROMPT_PATH, RETRIES,
                     SENTIMENTS, THEMES)
from .keywords import classify_keywords

API_URL = "https://api.anthropic.com/v1/messages"


class LLMError(RuntimeError):
    pass


class FatalLLMError(LLMError):
    """Ошибка, которую повтор не исправит: неверный ключ, нет денег на счёте, неверный запрос."""


def prompt_text():
    themes = "\n".join(f"- {tid}: {name}. {hint}" for tid, (name, hint) in THEMES.items())
    return PROMPT_PATH.read_text(encoding="utf-8").replace("{{THEMES}}", themes)


def prompt_version():
    return hashlib.sha1(prompt_text().encode("utf-8")).hexdigest()[:8]


def call_llm(system, user, model, key):
    body = json.dumps({
        "model": model,
        "max_tokens": 4096,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, headers={
        "content-type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}"
        raise (FatalLLMError if e.code in (400, 401, 403, 404) else LLMError)(detail) from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise LLMError(f"сеть: {e.__class__.__name__}") from None
    return "".join(b.get("text", "") for b in data.get("content", []))


def extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("в ответе нет JSON")
    return json.loads(text[start:end + 1])


def _check_labels(labels):
    if not isinstance(labels, list) or not labels:
        return None, "labels пуст или не список"
    out, topics = [], set()
    for item in labels:
        if not isinstance(item, dict):
            return None, "элемент labels не объект"
        topic, sentiment = item.get("topic"), item.get("sentiment")
        if topic not in THEMES:
            return None, f"неизвестная тема {topic!r}"
        if sentiment not in SENTIMENTS:
            return None, f"неизвестный тон {sentiment!r}"
        if topic in topics:
            return None, f"тема {topic} повторена"
        topics.add(topic)
        out.append([topic, sentiment])
    return out, None


def validate(answer, ids):
    """Возвращает (корректные разметки по id, список ошибок). Битые отзывы не принимаются."""
    good, errors = {}, []
    items = answer.get("results") if isinstance(answer, dict) else None
    if not isinstance(items, list):
        return good, ["в ответе нет списка results"]
    bad = set()
    for item in items:
        rid = item.get("id") if isinstance(item, dict) else None
        if rid not in ids:
            errors.append(f"неизвестный id {rid!r}")
        elif rid in good or rid in bad:
            errors.append(f"id {rid} повторён")
        else:
            labels, err = _check_labels(item.get("labels"))
            if err:
                errors.append(f"{rid}: {err}")
                bad.add(rid)
            else:
                good[rid] = labels
    errors += [f"нет ответа для id {rid}" for rid in ids if rid not in good and rid not in bad]
    return good, errors


def classify_batches(reviews, model, key, log=print):
    """Размечает пачками. При ошибке валидации повторяет запрос только по проблемным отзывам
    и передаёт модели текст ошибок. Что не удалось за RETRIES попыток, возвращается как pending."""
    system = prompt_text()
    labels, todo, errors = {}, list(reviews), []
    for attempt in range(1, RETRIES + 1):
        if not todo:
            break
        errors = []
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            payload = {"reviews": [{"id": r.id, "text": r.text} for r in batch]}
            if attempt > 1:
                payload["previous_errors"] = prev_errors
            try:
                answer = extract_json(call_llm(system, json.dumps(payload, ensure_ascii=False), model, key))
            except FatalLLMError:
                raise
            except (LLMError, ValueError) as e:
                errors.append(str(e))
                log(f"попытка {attempt}: {e}")
                time.sleep(2 ** attempt)
                continue
            good, errs = validate(answer, {r.id for r in batch})
            labels.update(good)
            errors += errs
            for e in errs:
                log(f"попытка {attempt}: {e}")
        prev_errors = errors[:10]
        todo = [r for r in todo if r.id not in labels]
    return labels, todo


def _load_cache():
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def classify_llm(reviews, force=False, log=print):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("нет ANTHROPIC_API_KEY (см. .env.example) или запустите с --provider keywords")
    model = os.environ.get("REVIEW_LLM_MODEL", DEFAULT_MODEL)
    version = prompt_version()
    cache = {} if force else _load_cache()
    k = lambda r: f"{r.id}:{version}:{model}"
    labels = {r.id: cache[k(r)] for r in reviews if k(r) in cache}
    todo = [r for r in reviews if r.id not in labels]
    pending = []
    if todo:
        fresh, pending = classify_batches(todo, model, key, log)
        labels.update(fresh)
        cache.update({k(r): fresh[r.id] for r in todo if r.id in fresh})
        _save_cache(cache)
    return labels, pending


def classify(reviews, provider="llm", force=False, log=print):
    """Единая точка входа. Тихого перехода с LLM на ключевые слова нет: провайдер выбирает пользователь."""
    if provider == "keywords":
        return classify_keywords(reviews), []
    if provider == "llm":
        return classify_llm(reviews, force, log)
    raise ValueError(f"неизвестный провайдер {provider!r}")
