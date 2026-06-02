"""
Blue-ian vs Red-ian Election DAG simulator.

Nodes are sampled in topological order. Each node specifies either a `prior`
(root node) or a `cpt` list of conditional rules. The first rule whose
conditions are a subset of the actual parent states wins; ties broken by
the number of matching conditions (more specific wins).
"""

import random
import math
from typing import Dict, Optional, Any

# ── Topological order ─────────────────────────────────────────────────────────
TOPO_ORDER = [
    "A1", "A2", "A3",
    "B1", "B2", "B3",
    "C1", "C2", "C3", "C4",
    "E1", "E2", "E3", "E4",
    "D1", "D2", "D3", "D4",
    "G1", "G2", "G3",
    "I1", "I2",
]

# ── Group metadata (for UI) ───────────────────────────────────────────────────
GROUP_INFO = {
    "A": {"name": "Fundamentals",       "color": "#bfdbfe", "text": "#1e40af"},  # blue
    "B": {"name": "Campaign State",     "color": "#fed7aa", "text": "#9a3412"},  # orange
    "C": {"name": "External Events",    "color": "#fed7aa", "text": "#9a3412"},  # orange
    "D": {"name": "Public Opinion",     "color": "#bbf7d0", "text": "#14532d"},  # green
    "E": {"name": "News Layer",         "color": "#e9d5ff", "text": "#6b21a8"},  # purple
    "G": {"name": "Election Mechanics", "color": "#fecdd3", "text": "#9f1239"},  # pink/red
    "I": {"name": "Final Outcome",      "color": "#fecdd3", "text": "#9f1239"},  # pink/red
}

# ── DAG edges for the UI (group level) ───────────────────────────────────────
DAG_EDGES = [
    ("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("A", "G"),
    ("B", "D"), ("B", "E"), ("B", "G"),
    ("C", "E"), ("C", "D"), ("C", "G"),
    ("E", "D"),
    ("D", "G"),
    ("G", "I"),
]

# ── Node definitions ──────────────────────────────────────────────────────────
NODES: Dict[str, Any] = {

    # ── A: Fundamentals (root nodes) ─────────────────────────────────────────
    "A1": {
        "label": "Economy",
        "group": "A",
        "states": ["Strong", "Neutral", "Weak"],
        "parents": [],
        "prior": {"Strong": 0.30, "Neutral": 0.45, "Weak": 0.25},
        "desc": "Macroeconomic conditions heading into the election",
    },
    "A2": {
        "label": "Institutional Trust",
        "group": "A",
        "states": ["High", "Medium", "Low"],
        "parents": [],
        "prior": {"High": 0.25, "Medium": 0.50, "Low": 0.25},
        "desc": "Public trust in government and media institutions",
    },
    "A3": {
        "label": "Partisan Baseline",
        "group": "A",
        "states": ["Blue-leaning", "Toss-up", "Red-leaning"],
        "parents": [],
        "prior": {"Blue-leaning": 0.35, "Toss-up": 0.30, "Red-leaning": 0.35},
        "desc": "Structural partisan lean of the electorate",
    },

    # ── B: Campaign State ─────────────────────────────────────────────────────
    "B1": {
        "label": "Blue Candidate Quality",
        "group": "B",
        "states": ["Strong", "Average", "Weak"],
        "parents": ["A3"],
        "cpt": [
            {"cond": {"A3": "Blue-leaning"}, "dist": {"Strong": 0.40, "Average": 0.45, "Weak": 0.15}},
            {"cond": {"A3": "Toss-up"},      "dist": {"Strong": 0.30, "Average": 0.50, "Weak": 0.20}},
            {"cond": {"A3": "Red-leaning"},  "dist": {"Strong": 0.25, "Average": 0.50, "Weak": 0.25}},
        ],
        "desc": "Blue-ian candidate charisma, competence, and ground organisation",
    },
    "B2": {
        "label": "Red Candidate Quality",
        "group": "B",
        "states": ["Strong", "Average", "Weak"],
        "parents": ["A3"],
        "cpt": [
            {"cond": {"A3": "Blue-leaning"}, "dist": {"Strong": 0.25, "Average": 0.50, "Weak": 0.25}},
            {"cond": {"A3": "Toss-up"},      "dist": {"Strong": 0.30, "Average": 0.50, "Weak": 0.20}},
            {"cond": {"A3": "Red-leaning"},  "dist": {"Strong": 0.40, "Average": 0.45, "Weak": 0.15}},
        ],
        "desc": "Red-ian candidate charisma, competence, and ground organisation",
    },
    "B3": {
        "label": "Ground Game",
        "group": "B",
        "states": ["Blue advantage", "Even", "Red advantage"],
        "parents": ["A3", "A2"],
        "cpt": [
            {"cond": {"A3": "Blue-leaning", "A2": "High"},   "dist": {"Blue advantage": 0.55, "Even": 0.35, "Red advantage": 0.10}},
            {"cond": {"A3": "Blue-leaning", "A2": "Medium"}, "dist": {"Blue advantage": 0.45, "Even": 0.40, "Red advantage": 0.15}},
            {"cond": {"A3": "Blue-leaning", "A2": "Low"},    "dist": {"Blue advantage": 0.35, "Even": 0.40, "Red advantage": 0.25}},
            {"cond": {"A3": "Toss-up",      "A2": "High"},   "dist": {"Blue advantage": 0.35, "Even": 0.45, "Red advantage": 0.20}},
            {"cond": {"A3": "Toss-up",      "A2": "Medium"}, "dist": {"Blue advantage": 0.30, "Even": 0.40, "Red advantage": 0.30}},
            {"cond": {"A3": "Toss-up",      "A2": "Low"},    "dist": {"Blue advantage": 0.20, "Even": 0.45, "Red advantage": 0.35}},
            {"cond": {"A3": "Red-leaning",  "A2": "High"},   "dist": {"Blue advantage": 0.20, "Even": 0.45, "Red advantage": 0.35}},
            {"cond": {"A3": "Red-leaning",  "A2": "Medium"}, "dist": {"Blue advantage": 0.15, "Even": 0.40, "Red advantage": 0.45}},
            {"cond": {"A3": "Red-leaning",  "A2": "Low"},    "dist": {"Blue advantage": 0.10, "Even": 0.35, "Red advantage": 0.55}},
        ],
        "desc": "GOTV infrastructure, volunteer network, and organisational advantage",
    },

    # ── C: External Events ────────────────────────────────────────────────────
    "C1": {
        "label": "Major Event Occurs",
        "group": "C",
        "states": ["Yes", "No"],
        "parents": ["A1", "A2"],
        "cpt": [
            {"cond": {"A1": "Strong",  "A2": "High"},   "dist": {"Yes": 0.20, "No": 0.80}},
            {"cond": {"A1": "Strong",  "A2": "Medium"}, "dist": {"Yes": 0.25, "No": 0.75}},
            {"cond": {"A1": "Strong",  "A2": "Low"},    "dist": {"Yes": 0.35, "No": 0.65}},
            {"cond": {"A1": "Neutral", "A2": "High"},   "dist": {"Yes": 0.30, "No": 0.70}},
            {"cond": {"A1": "Neutral", "A2": "Medium"}, "dist": {"Yes": 0.40, "No": 0.60}},
            {"cond": {"A1": "Neutral", "A2": "Low"},    "dist": {"Yes": 0.50, "No": 0.50}},
            {"cond": {"A1": "Weak",    "A2": "High"},   "dist": {"Yes": 0.45, "No": 0.55}},
            {"cond": {"A1": "Weak",    "A2": "Medium"}, "dist": {"Yes": 0.55, "No": 0.45}},
            {"cond": {"A1": "Weak",    "A2": "Low"},    "dist": {"Yes": 0.70, "No": 0.30}},
        ],
        "desc": "Whether a significant exogenous shock occurs during the campaign",
    },
    "C2": {
        "label": "Event Type",
        "group": "C",
        "states": ["Policy shock", "Geopolitical shock", "Cultural event", "Weather event", "Scandal", "None"],
        "parents": ["C1"],
        "cpt": [
            {"cond": {"C1": "Yes"}, "dist": {"Policy shock": 0.25, "Geopolitical shock": 0.20, "Cultural event": 0.20, "Weather event": 0.15, "Scandal": 0.20, "None": 0.00}},
            {"cond": {"C1": "No"},  "dist": {"Policy shock": 0.00, "Geopolitical shock": 0.00, "Cultural event": 0.00, "Weather event": 0.00, "Scandal": 0.00, "None": 1.00}},
        ],
        "desc": "Nature of the shock: policy, geopolitical, cultural, weather, or scandal",
    },
    "C3": {
        "label": "Event Target",
        "group": "C",
        "states": ["Helps Blue", "Ambiguous", "Helps Red"],
        "parents": ["C2"],
        "cpt": [
            {"cond": {"C2": "Policy shock"},       "dist": {"Helps Blue": 0.30, "Ambiguous": 0.40, "Helps Red": 0.30}},
            {"cond": {"C2": "Geopolitical shock"}, "dist": {"Helps Blue": 0.25, "Ambiguous": 0.50, "Helps Red": 0.25}},
            {"cond": {"C2": "Cultural event"},     "dist": {"Helps Blue": 0.35, "Ambiguous": 0.30, "Helps Red": 0.35}},
            {"cond": {"C2": "Weather event"},      "dist": {"Helps Blue": 0.35, "Ambiguous": 0.30, "Helps Red": 0.35}},
            {"cond": {"C2": "Scandal"},            "dist": {"Helps Blue": 0.40, "Ambiguous": 0.25, "Helps Red": 0.35}},
            {"cond": {"C2": "None"},               "dist": {"Helps Blue": 0.00, "Ambiguous": 1.00, "Helps Red": 0.00}},
        ],
        "desc": "Which party benefits from the event",
    },
    "C4": {
        "label": "Event Severity",
        "group": "C",
        "states": ["Low", "Medium", "High"],
        "parents": ["C2"],
        "cpt": [
            {"cond": {"C2": "Policy shock"},       "dist": {"Low": 0.35, "Medium": 0.45, "High": 0.20}},
            {"cond": {"C2": "Geopolitical shock"}, "dist": {"Low": 0.20, "Medium": 0.45, "High": 0.35}},
            {"cond": {"C2": "Cultural event"},     "dist": {"Low": 0.40, "Medium": 0.40, "High": 0.20}},
            {"cond": {"C2": "Weather event"},      "dist": {"Low": 0.30, "Medium": 0.45, "High": 0.25}},
            {"cond": {"C2": "Scandal"},            "dist": {"Low": 0.25, "Medium": 0.45, "High": 0.30}},
            {"cond": {"C2": "None"},               "dist": {"Low": 1.00, "Medium": 0.00, "High": 0.00}},
        ],
        "desc": "How consequential the event is",
    },

    # ── E: News Layer ─────────────────────────────────────────────────────────
    "E1": {
        "label": "News Type",
        "group": "E",
        "states": ["Economy", "Scandal", "Campaign", "Policy", "Foreign crisis", "Weather/turnout"],
        "parents": ["C2", "A1"],
        "cpt": [
            # Event-driven rows — event type dominates regardless of A1
            {"cond": {"C2": "Policy shock"},       "dist": {"Economy": 0.20, "Scandal": 0.05, "Campaign": 0.10, "Policy": 0.55, "Foreign crisis": 0.05, "Weather/turnout": 0.05}},
            {"cond": {"C2": "Geopolitical shock"}, "dist": {"Economy": 0.10, "Scandal": 0.05, "Campaign": 0.05, "Policy": 0.10, "Foreign crisis": 0.65, "Weather/turnout": 0.05}},
            {"cond": {"C2": "Cultural event"},     "dist": {"Economy": 0.05, "Scandal": 0.10, "Campaign": 0.30, "Policy": 0.15, "Foreign crisis": 0.05, "Weather/turnout": 0.35}},
            {"cond": {"C2": "Weather event"},      "dist": {"Economy": 0.10, "Scandal": 0.05, "Campaign": 0.05, "Policy": 0.05, "Foreign crisis": 0.10, "Weather/turnout": 0.65}},
            {"cond": {"C2": "Scandal"},            "dist": {"Economy": 0.05, "Scandal": 0.70, "Campaign": 0.10, "Policy": 0.05, "Foreign crisis": 0.05, "Weather/turnout": 0.05}},
            # No event: A1 drives the dominant news agenda
            {"cond": {"C2": "None", "A1": "Weak"},    "dist": {"Economy": 0.45, "Scandal": 0.10, "Campaign": 0.20, "Policy": 0.15, "Foreign crisis": 0.05, "Weather/turnout": 0.05}},
            {"cond": {"C2": "None", "A1": "Neutral"}, "dist": {"Economy": 0.25, "Scandal": 0.10, "Campaign": 0.35, "Policy": 0.15, "Foreign crisis": 0.05, "Weather/turnout": 0.10}},
            {"cond": {"C2": "None", "A1": "Strong"},  "dist": {"Economy": 0.15, "Scandal": 0.10, "Campaign": 0.40, "Policy": 0.15, "Foreign crisis": 0.05, "Weather/turnout": 0.15}},
            {"cond": {"C2": "None"},                  "dist": {"Economy": 0.25, "Scandal": 0.10, "Campaign": 0.35, "Policy": 0.15, "Foreign crisis": 0.05, "Weather/turnout": 0.10}},
        ],
        "desc": "The dominant type of news coverage during the campaign period",
    },
    "E2": {
        "label": "News Reliability",
        "group": "E",
        "states": ["Rumor", "Reported", "Confirmed", "Official"],
        "parents": ["C4", "A2"],
        "cpt": [
            {"cond": {"C4": "Low",    "A2": "High"},   "dist": {"Rumor": 0.20, "Reported": 0.40, "Confirmed": 0.30, "Official": 0.10}},
            {"cond": {"C4": "Low",    "A2": "Medium"}, "dist": {"Rumor": 0.30, "Reported": 0.40, "Confirmed": 0.25, "Official": 0.05}},
            {"cond": {"C4": "Low",    "A2": "Low"},    "dist": {"Rumor": 0.45, "Reported": 0.35, "Confirmed": 0.15, "Official": 0.05}},
            {"cond": {"C4": "Medium", "A2": "High"},   "dist": {"Rumor": 0.10, "Reported": 0.35, "Confirmed": 0.40, "Official": 0.15}},
            {"cond": {"C4": "Medium", "A2": "Medium"}, "dist": {"Rumor": 0.20, "Reported": 0.40, "Confirmed": 0.30, "Official": 0.10}},
            {"cond": {"C4": "Medium", "A2": "Low"},    "dist": {"Rumor": 0.35, "Reported": 0.40, "Confirmed": 0.20, "Official": 0.05}},
            {"cond": {"C4": "High",   "A2": "High"},   "dist": {"Rumor": 0.05, "Reported": 0.20, "Confirmed": 0.45, "Official": 0.30}},
            {"cond": {"C4": "High",   "A2": "Medium"}, "dist": {"Rumor": 0.10, "Reported": 0.30, "Confirmed": 0.45, "Official": 0.15}},
            {"cond": {"C4": "High",   "A2": "Low"},    "dist": {"Rumor": 0.25, "Reported": 0.40, "Confirmed": 0.25, "Official": 0.10}},
        ],
        "desc": "How verifiable and credible the prevailing news is",
    },
    "E3": {
        "label": "News Tone",
        "group": "E",
        "states": ["Blue-favorable", "Neutral/mixed", "Red-favorable"],
        "parents": ["C3", "E2", "B1", "B2", "A3"],
        "cpt": [
            # C3 = Helps Blue: event direction dominates; strong Blue campaign amplifies
            {"cond": {"C3": "Helps Blue", "B1": "Strong", "E2": "Official"},  "dist": {"Blue-favorable": 0.85, "Neutral/mixed": 0.13, "Red-favorable": 0.02}},
            {"cond": {"C3": "Helps Blue", "B1": "Strong", "E2": "Confirmed"}, "dist": {"Blue-favorable": 0.75, "Neutral/mixed": 0.22, "Red-favorable": 0.03}},
            {"cond": {"C3": "Helps Blue", "E2": "Official"},  "dist": {"Blue-favorable": 0.80, "Neutral/mixed": 0.17, "Red-favorable": 0.03}},
            {"cond": {"C3": "Helps Blue", "E2": "Confirmed"}, "dist": {"Blue-favorable": 0.70, "Neutral/mixed": 0.25, "Red-favorable": 0.05}},
            {"cond": {"C3": "Helps Blue", "E2": "Reported"},  "dist": {"Blue-favorable": 0.60, "Neutral/mixed": 0.30, "Red-favorable": 0.10}},
            {"cond": {"C3": "Helps Blue", "E2": "Rumor"},     "dist": {"Blue-favorable": 0.45, "Neutral/mixed": 0.40, "Red-favorable": 0.15}},
            # C3 = Helps Red: event direction dominates; strong Red campaign amplifies
            {"cond": {"C3": "Helps Red", "B2": "Strong", "E2": "Official"},  "dist": {"Blue-favorable": 0.02, "Neutral/mixed": 0.13, "Red-favorable": 0.85}},
            {"cond": {"C3": "Helps Red", "B2": "Strong", "E2": "Confirmed"}, "dist": {"Blue-favorable": 0.03, "Neutral/mixed": 0.22, "Red-favorable": 0.75}},
            {"cond": {"C3": "Helps Red",  "E2": "Official"},  "dist": {"Blue-favorable": 0.03, "Neutral/mixed": 0.17, "Red-favorable": 0.80}},
            {"cond": {"C3": "Helps Red",  "E2": "Confirmed"}, "dist": {"Blue-favorable": 0.05, "Neutral/mixed": 0.25, "Red-favorable": 0.70}},
            {"cond": {"C3": "Helps Red",  "E2": "Reported"},  "dist": {"Blue-favorable": 0.10, "Neutral/mixed": 0.30, "Red-favorable": 0.60}},
            {"cond": {"C3": "Helps Red",  "E2": "Rumor"},     "dist": {"Blue-favorable": 0.15, "Neutral/mixed": 0.40, "Red-favorable": 0.45}},
            # C3 = Ambiguous: campaign quality drives tone (score=3, wins over A3 rows)
            {"cond": {"C3": "Ambiguous", "B1": "Strong", "B2": "Weak"},    "dist": {"Blue-favorable": 0.60, "Neutral/mixed": 0.30, "Red-favorable": 0.10}},
            {"cond": {"C3": "Ambiguous", "B1": "Weak",   "B2": "Strong"},  "dist": {"Blue-favorable": 0.10, "Neutral/mixed": 0.30, "Red-favorable": 0.60}},
            {"cond": {"C3": "Ambiguous", "B1": "Strong", "B2": "Average"}, "dist": {"Blue-favorable": 0.48, "Neutral/mixed": 0.38, "Red-favorable": 0.14}},
            {"cond": {"C3": "Ambiguous", "B1": "Average","B2": "Strong"},  "dist": {"Blue-favorable": 0.14, "Neutral/mixed": 0.38, "Red-favorable": 0.48}},
            {"cond": {"C3": "Ambiguous", "B1": "Average","B2": "Weak"},    "dist": {"Blue-favorable": 0.44, "Neutral/mixed": 0.40, "Red-favorable": 0.16}},
            {"cond": {"C3": "Ambiguous", "B1": "Weak",   "B2": "Average"}, "dist": {"Blue-favorable": 0.16, "Neutral/mixed": 0.40, "Red-favorable": 0.44}},
            {"cond": {"C3": "Ambiguous", "B1": "Strong", "B2": "Strong"},  "dist": {"Blue-favorable": 0.28, "Neutral/mixed": 0.44, "Red-favorable": 0.28}},
            {"cond": {"C3": "Ambiguous", "B1": "Weak",   "B2": "Weak"},    "dist": {"Blue-favorable": 0.28, "Neutral/mixed": 0.44, "Red-favorable": 0.28}},
            # C3 = Ambiguous: partisan baseline adds a lean when campaigns are roughly even (score=2)
            {"cond": {"C3": "Ambiguous", "A3": "Blue-leaning"}, "dist": {"Blue-favorable": 0.42, "Neutral/mixed": 0.40, "Red-favorable": 0.18}},
            {"cond": {"C3": "Ambiguous", "A3": "Red-leaning"},  "dist": {"Blue-favorable": 0.18, "Neutral/mixed": 0.40, "Red-favorable": 0.42}},
            {"cond": {"C3": "Ambiguous"},                        "dist": {"Blue-favorable": 0.28, "Neutral/mixed": 0.44, "Red-favorable": 0.28}},
        ],
        "desc": "Overall partisan tilt of news coverage",
    },
    "E4": {
        "label": "News Volume",
        "group": "E",
        "states": ["Low", "Medium", "High"],
        "parents": ["C4", "B3"],
        "cpt": [
            # C4=High: major event drives wall-to-wall coverage regardless of ground game
            {"cond": {"C4": "High"},                   "dist": {"Low": 0.05, "Medium": 0.25, "High": 0.70}},
            # C4=Medium: active ground game (either side) adds modest volume boost
            {"cond": {"C4": "Medium", "B3": "Even"},   "dist": {"Low": 0.20, "Medium": 0.55, "High": 0.25}},
            {"cond": {"C4": "Medium"},                  "dist": {"Low": 0.15, "Medium": 0.52, "High": 0.33}},
            # C4=Low: background campaign activity becomes the primary volume driver
            {"cond": {"C4": "Low",    "B3": "Even"},   "dist": {"Low": 0.65, "Medium": 0.25, "High": 0.10}},
            {"cond": {"C4": "Low"},                     "dist": {"Low": 0.50, "Medium": 0.37, "High": 0.13}},
        ],
        "desc": "Volume of election-related media coverage",
    },

    # ── D: Public Opinion ─────────────────────────────────────────────────────
    "D1": {
        "label": "Blue Momentum",
        "group": "D",
        "states": ["Rising", "Stable", "Falling"],
        "parents": ["B1", "B2", "C3", "E3"],
        "cpt": [
            # Strong Blue vs Weak Red
            {"cond": {"B1": "Strong", "B2": "Weak", "E3": "Blue-favorable"}, "dist": {"Rising": 0.85, "Stable": 0.12, "Falling": 0.03}},
            {"cond": {"B1": "Strong", "B2": "Weak", "E3": "Neutral/mixed"},  "dist": {"Rising": 0.70, "Stable": 0.25, "Falling": 0.05}},
            {"cond": {"B1": "Strong", "B2": "Weak", "E3": "Red-favorable"},  "dist": {"Rising": 0.55, "Stable": 0.35, "Falling": 0.10}},
            # Strong Blue vs Average Red
            {"cond": {"B1": "Strong", "B2": "Average", "C3": "Helps Blue"},  "dist": {"Rising": 0.70, "Stable": 0.25, "Falling": 0.05}},
            {"cond": {"B1": "Strong", "B2": "Average", "E3": "Blue-favorable"}, "dist": {"Rising": 0.60, "Stable": 0.32, "Falling": 0.08}},
            {"cond": {"B1": "Strong", "B2": "Average", "E3": "Neutral/mixed"},  "dist": {"Rising": 0.48, "Stable": 0.38, "Falling": 0.14}},
            # Average vs Average
            {"cond": {"B1": "Average", "B2": "Average", "C3": "Ambiguous", "E3": "Blue-favorable"}, "dist": {"Rising": 0.45, "Stable": 0.40, "Falling": 0.15}},
            {"cond": {"B1": "Average", "B2": "Average", "C3": "Ambiguous", "E3": "Neutral/mixed"},  "dist": {"Rising": 0.25, "Stable": 0.50, "Falling": 0.25}},
            {"cond": {"B1": "Average", "B2": "Average", "C3": "Ambiguous", "E3": "Red-favorable"},  "dist": {"Rising": 0.15, "Stable": 0.40, "Falling": 0.45}},
            {"cond": {"B1": "Average", "B2": "Average", "C3": "Helps Blue"},  "dist": {"Rising": 0.50, "Stable": 0.35, "Falling": 0.15}},
            {"cond": {"B1": "Average", "B2": "Average", "C3": "Helps Red"},   "dist": {"Rising": 0.15, "Stable": 0.35, "Falling": 0.50}},
            # Weak Blue vs Strong Red
            {"cond": {"B1": "Weak", "B2": "Strong", "E3": "Red-favorable"},  "dist": {"Rising": 0.03, "Stable": 0.17, "Falling": 0.80}},
            {"cond": {"B1": "Weak", "B2": "Strong", "E3": "Neutral/mixed"},  "dist": {"Rising": 0.10, "Stable": 0.30, "Falling": 0.60}},
            {"cond": {"B1": "Weak", "B2": "Strong", "E3": "Blue-favorable"}, "dist": {"Rising": 0.20, "Stable": 0.45, "Falling": 0.35}},
            {"cond": {"B1": "Weak", "B2": "Average", "C3": "Helps Red"},     "dist": {"Rising": 0.15, "Stable": 0.45, "Falling": 0.40}},
            # Event-driven (fallbacks when quality match is weak)
            {"cond": {"C3": "Helps Blue", "E3": "Blue-favorable"}, "dist": {"Rising": 0.65, "Stable": 0.28, "Falling": 0.07}},
            {"cond": {"C3": "Helps Blue", "E3": "Neutral/mixed"},  "dist": {"Rising": 0.50, "Stable": 0.35, "Falling": 0.15}},
            {"cond": {"C3": "Helps Red",  "E3": "Red-favorable"},  "dist": {"Rising": 0.07, "Stable": 0.28, "Falling": 0.65}},
            {"cond": {"C3": "Helps Red",  "E3": "Neutral/mixed"},  "dist": {"Rising": 0.15, "Stable": 0.35, "Falling": 0.50}},
            {"cond": {"E3": "Blue-favorable"}, "dist": {"Rising": 0.55, "Stable": 0.30, "Falling": 0.15}},
            {"cond": {"E3": "Red-favorable"},  "dist": {"Rising": 0.15, "Stable": 0.30, "Falling": 0.55}},
            {"cond": {},                        "dist": {"Rising": 0.30, "Stable": 0.40, "Falling": 0.30}},
        ],
        "desc": "Trajectory of Blue-ian support in the electorate",
    },
    "D2": {
        "label": "Red Momentum",
        "group": "D",
        "states": ["Rising", "Stable", "Falling"],
        "parents": ["B2", "B1", "C3", "E3"],
        "cpt": [
            # Strong Red vs Weak Blue
            {"cond": {"B2": "Strong", "B1": "Weak", "E3": "Red-favorable"},  "dist": {"Rising": 0.85, "Stable": 0.12, "Falling": 0.03}},
            {"cond": {"B2": "Strong", "B1": "Weak", "E3": "Neutral/mixed"},  "dist": {"Rising": 0.70, "Stable": 0.25, "Falling": 0.05}},
            {"cond": {"B2": "Strong", "B1": "Weak", "E3": "Blue-favorable"}, "dist": {"Rising": 0.55, "Stable": 0.35, "Falling": 0.10}},
            {"cond": {"B2": "Strong", "B1": "Average", "C3": "Helps Red"},   "dist": {"Rising": 0.70, "Stable": 0.25, "Falling": 0.05}},
            {"cond": {"B2": "Strong", "B1": "Average", "E3": "Red-favorable"},  "dist": {"Rising": 0.60, "Stable": 0.32, "Falling": 0.08}},
            {"cond": {"B2": "Strong", "B1": "Average", "E3": "Neutral/mixed"},  "dist": {"Rising": 0.48, "Stable": 0.38, "Falling": 0.14}},
            # Average vs Average
            {"cond": {"B2": "Average", "B1": "Average", "C3": "Ambiguous", "E3": "Red-favorable"},  "dist": {"Rising": 0.45, "Stable": 0.40, "Falling": 0.15}},
            {"cond": {"B2": "Average", "B1": "Average", "C3": "Ambiguous", "E3": "Neutral/mixed"},  "dist": {"Rising": 0.25, "Stable": 0.50, "Falling": 0.25}},
            {"cond": {"B2": "Average", "B1": "Average", "C3": "Ambiguous", "E3": "Blue-favorable"}, "dist": {"Rising": 0.15, "Stable": 0.40, "Falling": 0.45}},
            {"cond": {"B2": "Average", "B1": "Average", "C3": "Helps Red"},  "dist": {"Rising": 0.50, "Stable": 0.35, "Falling": 0.15}},
            {"cond": {"B2": "Average", "B1": "Average", "C3": "Helps Blue"}, "dist": {"Rising": 0.15, "Stable": 0.35, "Falling": 0.50}},
            # Weak Red vs Strong Blue
            {"cond": {"B2": "Weak", "B1": "Strong", "E3": "Blue-favorable"}, "dist": {"Rising": 0.03, "Stable": 0.17, "Falling": 0.80}},
            {"cond": {"B2": "Weak", "B1": "Strong", "E3": "Neutral/mixed"},  "dist": {"Rising": 0.10, "Stable": 0.30, "Falling": 0.60}},
            {"cond": {"B2": "Weak", "B1": "Strong", "E3": "Red-favorable"},  "dist": {"Rising": 0.20, "Stable": 0.45, "Falling": 0.35}},
            {"cond": {"B2": "Weak", "B1": "Average", "C3": "Helps Blue"},    "dist": {"Rising": 0.15, "Stable": 0.45, "Falling": 0.40}},
            # Event-driven fallbacks
            {"cond": {"C3": "Helps Red",  "E3": "Red-favorable"},  "dist": {"Rising": 0.65, "Stable": 0.28, "Falling": 0.07}},
            {"cond": {"C3": "Helps Red",  "E3": "Neutral/mixed"},  "dist": {"Rising": 0.50, "Stable": 0.35, "Falling": 0.15}},
            {"cond": {"C3": "Helps Blue", "E3": "Blue-favorable"}, "dist": {"Rising": 0.07, "Stable": 0.28, "Falling": 0.65}},
            {"cond": {"C3": "Helps Blue", "E3": "Neutral/mixed"},  "dist": {"Rising": 0.15, "Stable": 0.35, "Falling": 0.50}},
            {"cond": {"E3": "Red-favorable"},  "dist": {"Rising": 0.55, "Stable": 0.30, "Falling": 0.15}},
            {"cond": {"E3": "Blue-favorable"}, "dist": {"Rising": 0.15, "Stable": 0.30, "Falling": 0.55}},
            {"cond": {},                        "dist": {"Rising": 0.30, "Stable": 0.40, "Falling": 0.30}},
        ],
        "desc": "Trajectory of Red-ian support in the electorate",
    },
    "D3": {
        "label": "Voter Uncertainty",
        "group": "D",
        "states": ["High", "Medium", "Low"],
        "parents": ["A2", "E2", "E4"],
        "cpt": [
            {"cond": {"A2": "High",   "E2": "Official",  "E4": "Low"},  "dist": {"High": 0.05, "Medium": 0.25, "Low": 0.70}},
            {"cond": {"A2": "High",   "E2": "Official",  "E4": "High"}, "dist": {"High": 0.10, "Medium": 0.35, "Low": 0.55}},
            {"cond": {"A2": "High",   "E2": "Confirmed"},               "dist": {"High": 0.10, "Medium": 0.35, "Low": 0.55}},
            {"cond": {"A2": "Medium", "E2": "Reported"},                "dist": {"High": 0.25, "Medium": 0.50, "Low": 0.25}},
            {"cond": {"A2": "Medium", "E2": "Rumor",     "E4": "High"}, "dist": {"High": 0.50, "Medium": 0.40, "Low": 0.10}},
            {"cond": {"A2": "Medium", "E2": "Rumor"},                   "dist": {"High": 0.40, "Medium": 0.45, "Low": 0.15}},
            {"cond": {"A2": "Low",    "E2": "Confirmed"},               "dist": {"High": 0.35, "Medium": 0.45, "Low": 0.20}},
            {"cond": {"A2": "Low",    "E2": "Rumor",     "E4": "High"}, "dist": {"High": 0.75, "Medium": 0.20, "Low": 0.05}},
            {"cond": {"A2": "Low",    "E2": "Rumor"},                   "dist": {"High": 0.60, "Medium": 0.30, "Low": 0.10}},
            {"cond": {"A2": "Low"},                                      "dist": {"High": 0.45, "Medium": 0.40, "Low": 0.15}},
            {"cond": {"E2": "Official"},                                 "dist": {"High": 0.08, "Medium": 0.30, "Low": 0.62}},
            {"cond": {"E2": "Rumor"},                                    "dist": {"High": 0.50, "Medium": 0.35, "Low": 0.15}},
            {"cond": {},                                                  "dist": {"High": 0.25, "Medium": 0.50, "Low": 0.25}},
        ],
        "desc": "How uncertain voters are about the candidates and likely outcome",
    },
    "D4": {
        "label": "Issue Salience",
        "group": "D",
        "states": ["Economy", "Trust/corruption", "Security", "Culture", "Climate/weather"],
        "parents": ["A1", "C2", "E1"],
        "cpt": [
            {"cond": {"C2": "Geopolitical shock", "E1": "Foreign crisis"}, "dist": {"Economy": 0.10, "Trust/corruption": 0.10, "Security": 0.65, "Culture": 0.10, "Climate/weather": 0.05}},
            {"cond": {"C2": "Geopolitical shock"},                          "dist": {"Economy": 0.15, "Trust/corruption": 0.10, "Security": 0.55, "Culture": 0.10, "Climate/weather": 0.10}},
            {"cond": {"C2": "Scandal", "E1": "Scandal"},                    "dist": {"Economy": 0.10, "Trust/corruption": 0.70, "Security": 0.05, "Culture": 0.10, "Climate/weather": 0.05}},
            {"cond": {"C2": "Scandal"},                                      "dist": {"Economy": 0.15, "Trust/corruption": 0.60, "Security": 0.05, "Culture": 0.15, "Climate/weather": 0.05}},
            {"cond": {"C2": "Cultural event"},                               "dist": {"Economy": 0.15, "Trust/corruption": 0.15, "Security": 0.10, "Culture": 0.50, "Climate/weather": 0.10}},
            {"cond": {"C2": "Weather event"},                                "dist": {"Economy": 0.20, "Trust/corruption": 0.10, "Security": 0.10, "Culture": 0.10, "Climate/weather": 0.50}},
            {"cond": {"C2": "Policy shock"},                                 "dist": {"Economy": 0.35, "Trust/corruption": 0.20, "Security": 0.15, "Culture": 0.15, "Climate/weather": 0.15}},
            {"cond": {"A1": "Weak",    "C2": "None"},                       "dist": {"Economy": 0.55, "Trust/corruption": 0.15, "Security": 0.10, "Culture": 0.10, "Climate/weather": 0.10}},
            {"cond": {"A1": "Neutral", "C2": "None"},                       "dist": {"Economy": 0.35, "Trust/corruption": 0.20, "Security": 0.15, "Culture": 0.15, "Climate/weather": 0.15}},
            {"cond": {"A1": "Strong",  "C2": "None"},                       "dist": {"Economy": 0.25, "Trust/corruption": 0.25, "Security": 0.15, "Culture": 0.20, "Climate/weather": 0.15}},
            {"cond": {},                                                      "dist": {"Economy": 0.30, "Trust/corruption": 0.20, "Security": 0.15, "Culture": 0.20, "Climate/weather": 0.15}},
        ],
        "desc": "Which policy dimension dominates voter decision-making",
    },


    # ── G: Election Mechanics ─────────────────────────────────────────────────
    "G1": {
        "label": "Blue Turnout",
        "group": "G",
        "states": ["High", "Normal", "Low"],
        "parents": ["B3", "C2", "D1"],
        "cpt": [
            {"cond": {"B3": "Blue advantage", "C2": "Weather event", "D1": "Rising"},  "dist": {"High": 0.45, "Normal": 0.40, "Low": 0.15}},
            {"cond": {"B3": "Blue advantage", "C2": "Weather event"},                  "dist": {"High": 0.30, "Normal": 0.45, "Low": 0.25}},
            {"cond": {"B3": "Blue advantage", "D1": "Rising"},                         "dist": {"High": 0.65, "Normal": 0.30, "Low": 0.05}},
            {"cond": {"B3": "Blue advantage", "D1": "Stable"},                         "dist": {"High": 0.45, "Normal": 0.45, "Low": 0.10}},
            {"cond": {"B3": "Blue advantage", "D1": "Falling"},                        "dist": {"High": 0.30, "Normal": 0.50, "Low": 0.20}},
            {"cond": {"B3": "Even",           "C2": "Weather event"},                  "dist": {"High": 0.15, "Normal": 0.50, "Low": 0.35}},
            {"cond": {"B3": "Even",           "D1": "Rising"},                         "dist": {"High": 0.40, "Normal": 0.45, "Low": 0.15}},
            {"cond": {"B3": "Even",           "D1": "Stable"},                         "dist": {"High": 0.25, "Normal": 0.55, "Low": 0.20}},
            {"cond": {"B3": "Even",           "D1": "Falling"},                        "dist": {"High": 0.10, "Normal": 0.50, "Low": 0.40}},
            {"cond": {"B3": "Red advantage",  "D1": "Rising"},                         "dist": {"High": 0.25, "Normal": 0.50, "Low": 0.25}},
            {"cond": {"B3": "Red advantage",  "D1": "Stable"},                         "dist": {"High": 0.10, "Normal": 0.45, "Low": 0.45}},
            {"cond": {"B3": "Red advantage",  "D1": "Falling"},                        "dist": {"High": 0.05, "Normal": 0.35, "Low": 0.60}},
            {"cond": {},                                                                 "dist": {"High": 0.25, "Normal": 0.50, "Low": 0.25}},
        ],
        "desc": "Blue-ian voter turnout relative to expectations",
    },
    "G2": {
        "label": "Red Turnout",
        "group": "G",
        "states": ["High", "Normal", "Low"],
        "parents": ["B3", "C2", "D2"],
        "cpt": [
            {"cond": {"B3": "Red advantage",  "C2": "Weather event", "D2": "Rising"},  "dist": {"High": 0.45, "Normal": 0.40, "Low": 0.15}},
            {"cond": {"B3": "Red advantage",  "C2": "Weather event"},                  "dist": {"High": 0.30, "Normal": 0.45, "Low": 0.25}},
            {"cond": {"B3": "Red advantage",  "D2": "Rising"},                         "dist": {"High": 0.65, "Normal": 0.30, "Low": 0.05}},
            {"cond": {"B3": "Red advantage",  "D2": "Stable"},                         "dist": {"High": 0.45, "Normal": 0.45, "Low": 0.10}},
            {"cond": {"B3": "Red advantage",  "D2": "Falling"},                        "dist": {"High": 0.30, "Normal": 0.50, "Low": 0.20}},
            {"cond": {"B3": "Even",           "C2": "Weather event"},                  "dist": {"High": 0.15, "Normal": 0.50, "Low": 0.35}},
            {"cond": {"B3": "Even",           "D2": "Rising"},                         "dist": {"High": 0.40, "Normal": 0.45, "Low": 0.15}},
            {"cond": {"B3": "Even",           "D2": "Stable"},                         "dist": {"High": 0.25, "Normal": 0.55, "Low": 0.20}},
            {"cond": {"B3": "Even",           "D2": "Falling"},                        "dist": {"High": 0.10, "Normal": 0.50, "Low": 0.40}},
            {"cond": {"B3": "Blue advantage", "D2": "Rising"},                         "dist": {"High": 0.25, "Normal": 0.50, "Low": 0.25}},
            {"cond": {"B3": "Blue advantage", "D2": "Stable"},                         "dist": {"High": 0.10, "Normal": 0.45, "Low": 0.45}},
            {"cond": {"B3": "Blue advantage", "D2": "Falling"},                        "dist": {"High": 0.05, "Normal": 0.35, "Low": 0.60}},
            {"cond": {},                                                                 "dist": {"High": 0.25, "Normal": 0.50, "Low": 0.25}},
        ],
        "desc": "Red-ian voter turnout relative to expectations",
    },
    "G3": {
        "label": "Independent Split",
        "group": "G",
        "states": ["Blue +5", "Near-even", "Red +5"],
        "parents": ["A3", "D4", "D1", "D2"],
        "cpt": [
            # Toss-up baseline
            {"cond": {"A3": "Toss-up", "D1": "Rising",  "D2": "Falling"},              "dist": {"Blue +5": 0.60, "Near-even": 0.30, "Red +5": 0.10}},
            {"cond": {"A3": "Toss-up", "D1": "Stable",  "D2": "Stable"},               "dist": {"Blue +5": 0.25, "Near-even": 0.50, "Red +5": 0.25}},
            {"cond": {"A3": "Toss-up", "D1": "Falling", "D2": "Rising"},               "dist": {"Blue +5": 0.10, "Near-even": 0.30, "Red +5": 0.60}},
            {"cond": {"A3": "Toss-up", "D4": "Economy",          "D1": "Rising"},      "dist": {"Blue +5": 0.55, "Near-even": 0.33, "Red +5": 0.12}},
            {"cond": {"A3": "Toss-up", "D4": "Economy",          "D2": "Rising"},      "dist": {"Blue +5": 0.12, "Near-even": 0.33, "Red +5": 0.55}},
            {"cond": {"A3": "Toss-up", "D4": "Trust/corruption", "D1": "Rising"},      "dist": {"Blue +5": 0.55, "Near-even": 0.35, "Red +5": 0.10}},
            {"cond": {"A3": "Toss-up", "D4": "Trust/corruption", "D2": "Rising"},      "dist": {"Blue +5": 0.10, "Near-even": 0.35, "Red +5": 0.55}},
            {"cond": {"A3": "Toss-up", "D4": "Security",         "D2": "Rising"},      "dist": {"Blue +5": 0.15, "Near-even": 0.35, "Red +5": 0.50}},
            # Blue-leaning baseline
            {"cond": {"A3": "Blue-leaning", "D1": "Rising"},                            "dist": {"Blue +5": 0.55, "Near-even": 0.35, "Red +5": 0.10}},
            {"cond": {"A3": "Blue-leaning", "D1": "Stable"},                            "dist": {"Blue +5": 0.45, "Near-even": 0.40, "Red +5": 0.15}},
            {"cond": {"A3": "Blue-leaning", "D2": "Rising"},                            "dist": {"Blue +5": 0.30, "Near-even": 0.45, "Red +5": 0.25}},
            {"cond": {"A3": "Blue-leaning"},                                             "dist": {"Blue +5": 0.48, "Near-even": 0.38, "Red +5": 0.14}},
            # Red-leaning baseline
            {"cond": {"A3": "Red-leaning",  "D2": "Rising"},                            "dist": {"Blue +5": 0.10, "Near-even": 0.35, "Red +5": 0.55}},
            {"cond": {"A3": "Red-leaning",  "D2": "Stable"},                            "dist": {"Blue +5": 0.15, "Near-even": 0.40, "Red +5": 0.45}},
            {"cond": {"A3": "Red-leaning",  "D1": "Rising"},                            "dist": {"Blue +5": 0.25, "Near-even": 0.45, "Red +5": 0.30}},
            {"cond": {"A3": "Red-leaning"},                                              "dist": {"Blue +5": 0.14, "Near-even": 0.38, "Red +5": 0.48}},
            {"cond": {},                                                                  "dist": {"Blue +5": 0.30, "Near-even": 0.40, "Red +5": 0.30}},
        ],
        "desc": "How independent voters break — net margin among swing voters",
    },


    # ── I: Final Outcome ──────────────────────────────────────────────────────
    # I1 is fully deterministic given (G1, G2, G3).
    # Scoring: G1 High=+1/Normal=0/Low=-1, G2 Low=+1/Normal=0/High=-1,
    #          G3 Blue+5=+1/Near-even=0/Red+5=-1.
    # Score≥2→Blue landslide, 1→Blue narrow, 0→Recount, -1→Red narrow, ≤-2→Red landslide.
    # Covers all 27 combinations (4+6+7+6+4=27).
    "I1": {
        "label": "Vote Share Category",
        "group": "I",
        "states": ["Blue landslide", "Blue narrow win", "Red narrow win", "Red landslide", "Recount/disputed"],
        "parents": ["G1", "G2", "G3"],
        "cpt": [
            # Score ≥ 2 → Blue landslide
            {"cond": {"G1": "High",   "G2": "Low",    "G3": "Blue +5"},   "dist": {"Blue landslide": 1.0}},
            {"cond": {"G1": "High",   "G2": "Low",    "G3": "Near-even"}, "dist": {"Blue landslide": 1.0}},
            {"cond": {"G1": "High",   "G2": "Normal", "G3": "Blue +5"},   "dist": {"Blue landslide": 1.0}},
            {"cond": {"G1": "Normal", "G2": "Low",    "G3": "Blue +5"},   "dist": {"Blue landslide": 1.0}},
            # Score = 1 → Blue narrow win
            {"cond": {"G1": "High",   "G2": "Low",    "G3": "Red +5"},    "dist": {"Blue narrow win": 1.0}},
            {"cond": {"G1": "High",   "G2": "Normal", "G3": "Near-even"}, "dist": {"Blue narrow win": 1.0}},
            {"cond": {"G1": "High",   "G2": "High",   "G3": "Blue +5"},   "dist": {"Blue narrow win": 1.0}},
            {"cond": {"G1": "Normal", "G2": "Low",    "G3": "Near-even"}, "dist": {"Blue narrow win": 1.0}},
            {"cond": {"G1": "Normal", "G2": "Normal", "G3": "Blue +5"},   "dist": {"Blue narrow win": 1.0}},
            {"cond": {"G1": "Low",    "G2": "Low",    "G3": "Blue +5"},   "dist": {"Blue narrow win": 1.0}},
            # Score = 0 → Recount/disputed
            {"cond": {"G1": "High",   "G2": "Normal", "G3": "Red +5"},    "dist": {"Recount/disputed": 1.0}},
            {"cond": {"G1": "High",   "G2": "High",   "G3": "Near-even"}, "dist": {"Recount/disputed": 1.0}},
            {"cond": {"G1": "Normal", "G2": "Low",    "G3": "Red +5"},    "dist": {"Recount/disputed": 1.0}},
            {"cond": {"G1": "Normal", "G2": "Normal", "G3": "Near-even"}, "dist": {"Recount/disputed": 1.0}},
            {"cond": {"G1": "Normal", "G2": "High",   "G3": "Blue +5"},   "dist": {"Recount/disputed": 1.0}},
            {"cond": {"G1": "Low",    "G2": "Low",    "G3": "Near-even"}, "dist": {"Recount/disputed": 1.0}},
            {"cond": {"G1": "Low",    "G2": "Normal", "G3": "Blue +5"},   "dist": {"Recount/disputed": 1.0}},
            # Score = -1 → Red narrow win
            {"cond": {"G1": "High",   "G2": "High",   "G3": "Red +5"},    "dist": {"Red narrow win": 1.0}},
            {"cond": {"G1": "Normal", "G2": "Normal", "G3": "Red +5"},    "dist": {"Red narrow win": 1.0}},
            {"cond": {"G1": "Normal", "G2": "High",   "G3": "Near-even"}, "dist": {"Red narrow win": 1.0}},
            {"cond": {"G1": "Low",    "G2": "Low",    "G3": "Red +5"},    "dist": {"Red narrow win": 1.0}},
            {"cond": {"G1": "Low",    "G2": "Normal", "G3": "Near-even"}, "dist": {"Red narrow win": 1.0}},
            {"cond": {"G1": "Low",    "G2": "High",   "G3": "Blue +5"},   "dist": {"Red narrow win": 1.0}},
            # Score ≤ -2 → Red landslide
            {"cond": {"G1": "Normal", "G2": "High",   "G3": "Red +5"},    "dist": {"Red landslide": 1.0}},
            {"cond": {"G1": "Low",    "G2": "Normal", "G3": "Red +5"},    "dist": {"Red landslide": 1.0}},
            {"cond": {"G1": "Low",    "G2": "High",   "G3": "Near-even"}, "dist": {"Red landslide": 1.0}},
            {"cond": {"G1": "Low",    "G2": "High",   "G3": "Red +5"},    "dist": {"Red landslide": 1.0}},
        ],
        "desc": "Vote share determined by turnout + independent split",
    },
    "I2": {
        "label": "Winner",
        "group": "I",
        "states": ["Blue wins", "Red wins"],
        "parents": ["I1"],
        "cpt": [
            {"cond": {"I1": "Blue landslide"},   "dist": {"Blue wins": 1.0, "Red wins": 0.0}},
            {"cond": {"I1": "Blue narrow win"},  "dist": {"Blue wins": 1.0, "Red wins": 0.0}},
            {"cond": {"I1": "Red narrow win"},   "dist": {"Blue wins": 0.0, "Red wins": 1.0}},
            {"cond": {"I1": "Red landslide"},    "dist": {"Blue wins": 0.0, "Red wins": 1.0}},
            {"cond": {"I1": "Recount/disputed"}, "dist": {"Blue wins": 0.5, "Red wins": 0.5}},
        ],
        "desc": "Declared winner — deterministic except for a genuine recount",
    },
}


# ── Sampling logic ────────────────────────────────────────────────────────────

def _find_dist(node: Dict, parent_states: Dict[str, str]) -> Dict[str, float]:
    """Return the best-matching CPT row's distribution."""
    if "prior" in node:
        return node["prior"]

    best_dist = None
    best_score = -1

    for rule in node["cpt"]:
        cond = rule["cond"]
        score = 0
        ok = True
        for k, v in cond.items():
            if parent_states.get(k) == v:
                score += 1
            else:
                ok = False
                break
        if ok and score > best_score:
            best_score = score
            best_dist = rule["dist"]

    if best_dist is None:
        states = node["states"]
        return {s: 1.0 / len(states) for s in states}
    return best_dist


def _sample(dist: Dict[str, float]) -> str:
    states = list(dist.keys())
    weights = list(dist.values())
    return random.choices(states, weights=weights)[0]


def simulate(overrides: Optional[Dict[str, str]] = None,
             seed: Optional[int] = None) -> Dict:
    """
    Run one complete election simulation.

    Parameters
    ----------
    overrides : dict mapping node_id → forced state (bypasses sampling)
    seed      : random seed for reproducibility

    Returns
    -------
    dict with keys:
        states    : {node_id: sampled_state}
        probs     : {node_id: p(sampled_state | parents)}
        surprise  : {node_id: bits of surprise = -log2(p)}
        narrative : list of phase dicts for the UI
    """
    if seed is not None:
        random.seed(seed)
    if overrides is None:
        overrides = {}

    states:        Dict[str, str]             = {}
    probs:         Dict[str, float]           = {}
    distributions: Dict[str, Dict[str, float]] = {}  # full CPT lookup per node

    for node_id in TOPO_ORDER:
        node = NODES[node_id]
        parent_states = {p: states[p] for p in node.get("parents", [])}
        dist = _find_dist(node, parent_states)
        distributions[node_id] = dist

        if node_id in overrides:
            forced = overrides[node_id]
            states[node_id] = forced
            probs[node_id] = dist.get(forced, 0.0)
        else:
            state = _sample(dist)
            states[node_id] = state
            probs[node_id] = dist.get(state, 0.0)

    surprise = {
        nid: -math.log2(max(p, 1e-9))
        for nid, p in probs.items()
    }

    return {
        "states":        states,
        "probs":         probs,
        "surprise":      surprise,
        "distributions": distributions,
        "narrative":     _build_narrative(states, probs, surprise, distributions),
    }


def _pct(p: float) -> str:
    return f"{p:.0%}"


def _build_narrative(states: Dict, probs: Dict, surprise: Dict,
                     distributions: Dict) -> list:
    """Structured causal narrative for the UI."""

    def item(node_id: str, label_override: str = None) -> dict:
        label = label_override or NODES[node_id]["label"]
        node  = NODES[node_id]
        return {
            "node":         node_id,
            "label":        label,
            "state":        states[node_id],
            "prob":         probs[node_id],
            "surprise":     surprise[node_id],
            "distribution": distributions[node_id],        # full CPT conditional
            "all_states":   node["states"],                # ordered state list
            "parents":      {p: states[p]                  # parent context
                             for p in node.get("parents", [])},
        }

    event_occurred = states["C1"] == "Yes"

    phases = [
        {
            "phase": "A — Fundamentals",
            "group": "A",
            "items": [item("A1"), item("A2"), item("A3")],
        },
        {
            "phase": "B — Campaign",
            "group": "B",
            "items": [item("B1"), item("B2"), item("B3")],
        },
        {
            "phase": "C — External Event",
            "group": "C",
            "items": (
                [item("C1"), item("C2"), item("C3"), item("C4")]
                if event_occurred
                else [item("C1")]
            ),
            "highlight": event_occurred,
        },
        {
            "phase": "E — News Layer",
            "group": "E",
            "items": [item("E1"), item("E2"), item("E3"), item("E4")],
        },
        {
            "phase": "D — Public Opinion",
            "group": "D",
            "items": [item("D1"), item("D2"), item("D3"), item("D4")],
        },
        {
            "phase": "G — Election Mechanics",
            "group": "G",
            "items": [item("G1"), item("G2"), item("G3")],
        },
        {
            "phase": "I — Final Outcome",
            "group": "I",
            "items": [item("I1"), item("I2")],
            "highlight": True,
        },
    ]
    return phases


def _compute_edge_influence(src_group: str, dst_group: str) -> float:
    """
    Causal influence of src_group on dst_group: average TVD between child
    distributions when a parent in src_group takes different values.
    TVD in [0,1]: 0 = parent irrelevant, 1 = parent fully determines child.
    """
    scores = []
    for dst_nid, dst_node in NODES.items():
        if dst_node["group"] != dst_group or "prior" in dst_node:
            continue
        relevant_parents = [p for p in dst_node.get("parents", [])
                            if NODES[p]["group"] == src_group]
        for par_nid in relevant_parents:
            # Bucket CPT rows by this parent's value
            by_val: Dict[str, list] = {}
            for rule in dst_node["cpt"]:
                val = rule["cond"].get(par_nid)
                if val is not None:
                    by_val.setdefault(val, []).append(rule["dist"])
            if len(by_val) < 2:
                continue
            # Average the distributions within each bucket
            avg: Dict[str, Dict[str, float]] = {}
            for val, dists in by_val.items():
                merged: Dict[str, float] = {}
                for d in dists:
                    for s, p in d.items():
                        merged[s] = merged.get(s, 0.0) + p
                n = len(dists)
                avg[val] = {s: p / n for s, p in merged.items()}
            # Pairwise TVD across buckets
            vals = list(avg.keys())
            tvds = []
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    d1, d2 = avg[vals[i]], avg[vals[j]]
                    all_s = set(d1) | set(d2)
                    tvd = 0.5 * sum(abs(d1.get(s, 0.0) - d2.get(s, 0.0)) for s in all_s)
                    tvds.append(tvd)
            if tvds:
                scores.append(sum(tvds) / len(tvds))
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def get_structure() -> dict:
    """Return DAG structure for the UI (nodes + edges + group info)."""
    nodes_out = {}
    for nid, n in NODES.items():
        entry: Dict[str, Any] = {
            "label":   n["label"],
            "group":   n["group"],
            "states":  n["states"],
            "parents": n.get("parents", []),
            "desc":    n.get("desc", ""),
        }
        if "prior" in n:
            entry["prior"] = n["prior"]
        else:
            entry["cpt"] = n.get("cpt", [])
        nodes_out[nid] = entry
    # Raw TVD per edge
    raw: Dict[str, float] = {
        f"{src}-{dst}": _compute_edge_influence(src, dst)
        for src, dst in DAG_EDGES
    }

    # Normalize per destination: incoming edges to each group sum to 1.0.
    # Edges with 0 raw TVD stay at 0 (no direct CPT dependency detected).
    # Non-zero incoming edges are re-scaled so their shares sum to 1.
    incoming: Dict[str, list] = {}
    for src, dst in DAG_EDGES:
        incoming.setdefault(dst, []).append(src)

    edge_influence: Dict[str, float] = {}
    for dst, srcs in incoming.items():
        pos = {src: raw[f"{src}-{dst}"] for src in srcs if raw[f"{src}-{dst}"] > 0}
        total = sum(pos.values())
        for src in srcs:
            key = f"{src}-{dst}"
            if total > 0 and raw[key] > 0:
                edge_influence[key] = round(raw[key] / total, 3)
            else:
                edge_influence[key] = 0.0  # no direct CPT dependency

    return {
        "nodes":          nodes_out,
        "edges":          DAG_EDGES,
        "groups":         GROUP_INFO,
        "topo_order":     TOPO_ORDER,
        "edge_influence": edge_influence,
    }
