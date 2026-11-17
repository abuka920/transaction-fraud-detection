import json, uuid, heapq
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

PENDING = "PENDING"
CONFIRMED = "CONFIRMED_FRAUD"
FALSE_POS = "FALSE_POSITIVE"
REVIEWED = "REVIEWED"


@dataclass
class FraudAlert:
    alert_id: str
    transaction_id: str
    amount: float
    fraud_probability: float
    estimated_loss: float
    top_reason: str
    explanation: str
    timestamp: str
    status: str = PENDING
    reviewed_at: Optional[str] = None
    analyst_note: Optional[str] = None

    def __lt__(self, other):
        return self.estimated_loss > other.estimated_loss


class AnalystQueue:
    def __init__(self, persist_path="alerts.json"):
        self.path = Path(persist_path)
        self._alerts: dict[str, FraudAlert] = {}
        self._heap: list = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                for item in json.loads(self.path.read_text()):
                    a = FraudAlert(**item)
                    self._alerts[a.alert_id] = a
                    if a.status == PENDING:
                        heapq.heappush(self._heap, (-a.estimated_loss, a.alert_id))
            except Exception:
                pass

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(a) for a in self._alerts.values()], indent=2))

    def add_alert(self, transaction_id, amount, fraud_probability, top_reason, explanation, timestamp):
        loss = round(amount * fraud_probability, 2)
        a = FraudAlert(str(uuid.uuid4()), transaction_id, round(amount,2),
                       round(fraud_probability,4), loss, top_reason, explanation, timestamp)
        self._alerts[a.alert_id] = a
        heapq.heappush(self._heap, (-loss, a.alert_id))
        self._save()
        return a

    def update_status(self, alert_id, new_status, analyst_note=""):
        a = self._alerts.get(alert_id)
        if not a:
            return None
        a.status = new_status
        a.reviewed_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        if analyst_note:
            a.analyst_note = analyst_note
        self._save()
        return a

    def get_all_alerts(self, status=None):
        alerts = list(self._alerts.values())
        if status:
            alerts = [a for a in alerts if a.status == status]
        return sorted(alerts, key=lambda a: -a.estimated_loss)

    def get_queue_stats(self):
        from collections import Counter
        c = Counter(a.status for a in self._alerts.values())
        total_loss = sum(a.estimated_loss for a in self._alerts.values() if a.status == PENDING)
        return {"pending": c[PENDING], "confirmed_fraud": c[CONFIRMED],
                "false_positive": c[FALSE_POS], "reviewed": c[REVIEWED],
                "total_alerts": len(self._alerts),
                "pending_estimated_loss": round(total_loss, 2)}
