import json
import time
import urllib.error
import urllib.request


class TelegramError(RuntimeError):
    pass


def send_message(token, chat_id, text, attempts=3):
    """Отправляет одно сообщение. Повторяет при 429 и 5xx. Токен в тексты ошибок не попадает."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                       "disable_web_page_preview": True}).encode("utf-8")
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if json.load(resp).get("ok"):
                    return
                raise TelegramError("Telegram вернул ok=false")
        except urllib.error.HTTPError as e:
            detail = e.read()[:200].decode("utf-8", "replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < attempts:
                time.sleep(2 * attempt)
                continue
            raise TelegramError(f"Telegram HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < attempts:
                time.sleep(2 * attempt)
                continue
            raise TelegramError(f"сеть: {e.__class__.__name__}") from None
