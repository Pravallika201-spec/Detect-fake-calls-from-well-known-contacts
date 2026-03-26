"""Simple pub-sub alert dispatcher (console + websocket placeholder)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, List

from backend.services.risk_engine import RiskBreakdown


@dataclass
class Alert:
    title: str
    message: str
    risk: RiskBreakdown


class AlertService:
    def __init__(self):
        self.subscribers: List[Callable[[Alert], None]] = []

    def subscribe(self, callback: Callable[[Alert], None]) -> None:
        self.subscribers.append(callback)

    def notify(self, risk: RiskBreakdown, contact_name: str) -> Alert:
        title = f"Call risk: {risk.tier} ({risk.overall_score:.1f})"
        message = json.dumps(risk.notes)
        alert = Alert(title=title, message=message, risk=risk)
        for callback in self.subscribers:
            callback(alert)
        print(f"[ALERT] {title} for {contact_name} :: {message}")
        return alert
