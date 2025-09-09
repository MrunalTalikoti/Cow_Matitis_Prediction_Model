
from dataclasses import dataclass
@dataclass
class RiskThresholds:
    urgent: float
    high: float
    low: float
def map_risk(p, t: 'RiskThresholds'):
    if p>=t.urgent: return "URGENT"
    if p>=t.high: return "HIGH"
    if p>=t.low: return "MEDIUM"
    return "LOW"
