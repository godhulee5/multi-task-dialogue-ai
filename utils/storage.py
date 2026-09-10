
from pathlib import Path
import json
from datetime import datetime

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "conversations.json"

def get_records():
    if not DATA_FILE.exists():
        return []
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_record(record):
    records = get_records()
    record = dict(record)
    record.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    records.append(record)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

def clear_records():
    DATA_FILE.write_text("[]", encoding="utf-8")
