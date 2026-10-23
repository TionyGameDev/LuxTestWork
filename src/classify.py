import hashlib
import json
import os
import time
import urllib.error
import urllib.request

from .config import (BATCH_SIZE, CACHE_PATH, DEFAULT_MODEL, DEFAULT_OPENAI_BASE,
                     DEFAULT_OPENAI_MODEL, PROMPT_PATH, RETRIES, SENTIMENTS, THEMES)
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


def _post_json(url, headers, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        detail = f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}"
        raise (FatalLLMError if e.code in (400, 401, 403, 404) else LLMError)(detail) from None
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise LLMError(f"сеть или формат ответа: {e.__class__.__name__}") from None


def call_anthropic(system, user, model, key):
    data = _post_json(API_URL, {"x-api-key": key, "anthropic-version": "2023-06-01"}, {
        "model": model, "max_tokens": 4096, "system": system,
        "messages": [{"role": "user", "content": user}],
    })
    return "".join(b.get("text", "") for b in data.get("content", []))


def call_openai_compatible(system, user, model, key):
    """Любой сервис с OpenAI-совместимым chat/completions: OpenRouter, Groq, Together и др."""
    base = os.environ.get("LLM_BASE_URL") or DEFAULT_OPENAI_BASE
    url = f"{base.rstrip('/')}/chat/completions"
    headers = {"authorization": f"Bearer {key}"}
    body = {"model": model, "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    try:
        data = _post_json(url, headers, body)
    except FatalLLMError as e:
        # Некоторые модели (в том числе Gemma у части провайдеров) не принимают системный промпт.
        if not any(w in str(e).lower() for w in ("instruction", "system")):
            raise
        body["messages"] = [{"role": "user", "content": f"{system}\n\n---\n\n{user}"}]
        data = _post_json(url, headers, body)
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise LLMError("неожиданный формат ответа API") from None


PROVIDERS = {
    "llm": (call_anthropic, "ANTHROPIC_API_KEY", DEFAULT_MODEL),
    "openai": (call_openai_compatible, "LLM_API_KEY", DEFAULT_OPENAI_MODEL),
}


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


def classify_batches(reviews, call, log=print):
    """Размечает пачками. При ошибке валидации повторяет запрос только по проблемным отзывам
    и передаёт модели текст ошибок. Что не удалось за RETRIES попыток, возвращается как pending.
    `call(system, user)` возвращает текст ответа модели."""
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
                answer = extract_json(call(system, json.dumps(payload, ensure_ascii=False)))
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


def classify_llm(reviews, provider="llm", force=False, log=print):
    call_api, key_var, default_model = PROVIDERS[provider]
    key = os.environ.get(key_var)
    if not key:
        raise LLMError(f"нет {key_var} (см. .env.example) или запустите с --provider keywords")
    model = os.environ.get("REVIEW_LLM_MODEL") or default_model
    version = prompt_version()
    cache = {} if force else _load_cache()
    k = lambda r: f"{r.id}:{version}:{model}"
    labels = {r.id: cache[k(r)] for r in reviews if k(r) in cache}
    todo = [r for r in reviews if r.id not in labels]
    pending = []
    if todo:
        fresh, pending = classify_batches(todo, lambda s, u: call_api(s, u, model, key), log)
        labels.update(fresh)
        cache.update({k(r): fresh[r.id] for r in todo if r.id in fresh})
        _save_cache(cache)
    return labels, pending


def classify(reviews, provider="llm", force=False, log=print):
    """Единая точка входа. Тихого перехода с LLM на ключевые слова нет: провайдер выбирает пользователь."""
    if provider == "keywords":
        return classify_keywords(reviews), []
    if provider in PROVIDERS:
        return classify_llm(reviews, provider, force, log)
    raise ValueError(f"неизвестный провайдер {provider!r}")
