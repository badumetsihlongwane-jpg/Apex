from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

BRIDGE_DIR = "/root/apex/bridge"
BRIDGE_DIR = os.getenv("BRIDGE_DIR", BRIDGE_DIR)
LOCK_TIMEOUT_SECONDS = float(os.getenv("MT5_BRIDGE_LOCK_TIMEOUT", "5"))
RETRY_DELAY_SECONDS = float(os.getenv("MT5_BRIDGE_RETRY_DELAY", "0.2"))
MAX_RETRIES = int(os.getenv("MT5_BRIDGE_MAX_RETRIES", "25"))

POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
TRADE_ACTION_DEAL = 1
ORDER_TIME_GTC = 0
ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2
TRADE_RETCODE_DONE = 10009
TIMEFRAME_M5 = 5
TIMEFRAME_M15 = 15
TIMEFRAME_H1 = 60
TIMEFRAME_H4 = 240

_logger = logging.getLogger("mt5_bridge")
if not _logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
_logger.setLevel(logging.INFO)
_logger.propagate = False

_last_error: tuple[int, str] = (0, "OK")
_initialized = False


def _log(level: int, event: str, **fields: Any) -> None:
    payload = {"event": event, "ts": time.time(), **fields}
    _logger.log(level, json.dumps(payload, default=str))


def _set_last_error(code: int, message: str) -> None:
    global _last_error
    _last_error = (code, message)


def _paths() -> Dict[str, Path]:
    base = Path(BRIDGE_DIR)
    return {
        "base": base,
        "orders": base / "orders.json",
        "status": base / "status.json",
        "positions": base / "positions.json",
        "market": base / "market",
    }


def _ensure_bridge_files() -> None:
    p = _paths()
    p["base"].mkdir(parents=True, exist_ok=True)
    p["market"].mkdir(parents=True, exist_ok=True)
    _ensure_json_file(p["orders"], [])
    _ensure_json_file(p["status"], {})
    _ensure_json_file(p["positions"], [])


def _ensure_json_file(path: Path, default: Any) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(default), encoding="utf-8")


@contextmanager
def _locked_open(path: Path, mode: str, exclusive: bool):
    deadline = time.time() + LOCK_TIMEOUT_SECONDS
    path.parent.mkdir(parents=True, exist_ok=True)
    if "r" in mode and "+" not in mode and not path.exists():
        _ensure_json_file(path, {} if path.name == "status.json" else [])

    while True:
        fh = open(path, mode, encoding="utf-8")
        try:
            lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            try:
                fcntl.flock(fh.fileno(), lock_type | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                if time.time() >= deadline:
                    raise TimeoutError(f"Lock timeout for {path}")
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            yield fh
            break
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            fh.close()


def _retry(action: str, fn):
    attempts = 0
    while True:
        attempts += 1
        try:
            return fn()
        except Exception as exc:
            _set_last_error(1, f"{action} failed: {exc}")
            if attempts >= MAX_RETRIES:
                _log(logging.ERROR, "bridge_action_failed", action=action, attempts=attempts, error=str(exc))
                raise
            _log(logging.WARNING, "bridge_action_retry", action=action, attempts=attempts, error=str(exc))
            time.sleep(RETRY_DELAY_SECONDS)


def _read_json(path: Path, default: Any) -> Any:
    _ensure_json_file(path, default)

    def _op():
        with _locked_open(path, "r", exclusive=False) as fh:
            text = fh.read().strip()
            return default if not text else json.loads(text)

    return _retry(f"read {path.name}", _op)


def _write_json(path: Path, value: Any) -> None:
    def _op():
        with _locked_open(path, "r+", exclusive=True) as fh:
            fh.seek(0)
            fh.truncate()
            json.dump(value, fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())

    _retry(f"write {path.name}", _op)


def _to_ns(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_ns(v) for v in value]
    return value


def initialize(*args, **kwargs) -> bool:
    global _initialized
    try:
        _ensure_bridge_files()
        _initialized = True
        _set_last_error(0, "OK")
        _log(logging.INFO, "initialize", bridge_dir=str(_paths()["base"]))
        return True
    except Exception as exc:
        _set_last_error(1, str(exc))
        _log(logging.ERROR, "initialize_failed", error=str(exc))
        return False


def shutdown() -> None:
    global _initialized
    _initialized = False
    _set_last_error(0, "OK")
    _log(logging.INFO, "shutdown")


def account_info():
    try:
        data = _read_json(_paths()["status"], {})
        if not isinstance(data, dict):
            raise ValueError("status.json must be a JSON object")
        _set_last_error(0, "OK")
        return _to_ns(data)
    except Exception as exc:
        _set_last_error(1, str(exc))
        _log(logging.ERROR, "account_info_failed", error=str(exc))
        return None


def positions_get(symbol: Optional[str] = None):
    try:
        data = _read_json(_paths()["positions"], [])
        if isinstance(data, dict):
            data = data.get("positions", [])
        if not isinstance(data, list):
            raise ValueError("positions.json must be a JSON array")
        if symbol:
            data = [p for p in data if str(p.get("symbol", "")) == symbol]
        _set_last_error(0, "OK")
        return [_to_ns(p) for p in data]
    except Exception as exc:
        _set_last_error(1, str(exc))
        _log(logging.ERROR, "positions_get_failed", symbol=symbol, error=str(exc))
        return None


def _load_market_json(candidates: List[Path], default: Any) -> Any:
    for path in candidates:
        if path.exists():
            return _read_json(path, default)
    return default


def symbol_info(symbol: str):
    market = _paths()["market"]
    data = _load_market_json(
        [market / f"{symbol}_info.json", market / symbol / "info.json"],
        {"filling_mode": ORDER_FILLING_IOC},
    )
    if not isinstance(data, dict):
        data = {"filling_mode": ORDER_FILLING_IOC}
    data.setdefault("filling_mode", ORDER_FILLING_IOC)
    _set_last_error(0, "OK")
    return _to_ns(data)


def symbol_info_tick(symbol: str):
    market = _paths()["market"]
    data = _load_market_json(
        [
            market / f"{symbol}_tick.json",
            market / symbol / "tick.json",
            market / symbol / "latest_tick.json",
        ],
        None,
    )
    if not isinstance(data, dict):
        _set_last_error(1, f"No tick data for {symbol}")
        return None
    _set_last_error(0, "OK")
    return _to_ns(data)


def _timeframe_names(timeframe: int) -> List[str]:
    tf = str(timeframe).lower()
    aliases = {
        str(TIMEFRAME_M5): ["m5", "5m"],
        str(TIMEFRAME_M15): ["m15", "15m"],
        str(TIMEFRAME_H1): ["h1", "1h", "60m"],
        str(TIMEFRAME_H4): ["h4", "4h", "240m"],
    }
    names = [tf]
    for key, vals in aliases.items():
        if tf == key or tf in vals:
            names.extend([key, *vals])
            break
    return list(dict.fromkeys(names))


def copy_rates_from_pos(symbol: str, timeframe: int, start: int, count: int):
    market = _paths()["market"]
    tf_names = _timeframe_names(timeframe)
    candidates = []
    for tf in tf_names:
        candidates.extend(
            [
                market / f"{symbol}_{tf}.json",
                market / symbol / f"{tf}.json",
                market / symbol / f"rates_{tf}.json",
            ]
        )
    data = _load_market_json(candidates, [])
    if isinstance(data, dict):
        data = data.get("rates", data.get("candles", []))
    if not isinstance(data, list):
        _set_last_error(1, f"Invalid candle data for {symbol} {timeframe}")
        return None
    end = max(0, len(data) - max(0, int(start)))
    begin = max(0, end - max(0, int(count)))
    sliced = data[begin:end]
    _set_last_error(0, "OK")
    return sliced


def order_send(request: Dict[str, Any]):
    orders_path = _paths()["orders"]
    try:
        orders = _read_json(orders_path, [])
        if isinstance(orders, dict):
            orders = orders.get("orders", [])
        if not isinstance(orders, list):
            orders = []

        payload = {
            "request": request,
            "created_at": time.time(),
        }
        orders.append(payload)
        _write_json(orders_path, orders)
        _set_last_error(0, "OK")
        _log(logging.INFO, "order_queued", symbol=request.get("symbol"), order_type=request.get("type"))
        return _to_ns({"retcode": TRADE_RETCODE_DONE, "comment": "QUEUED", "request_id": len(orders)})
    except Exception as exc:
        _set_last_error(1, str(exc))
        _log(logging.ERROR, "order_send_failed", error=str(exc))
        return None


def symbol_select(symbol: str, enable: bool = True) -> bool:
    _set_last_error(0, "OK")
    _log(logging.INFO, "symbol_select", symbol=symbol, enable=enable)
    return True


def last_error():
    return _last_error
