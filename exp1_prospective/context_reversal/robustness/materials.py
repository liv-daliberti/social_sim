"""Controlled variants of all 20 parent families, with shared role inventories.

The paraphrase variant rewrites background/mechanism/relations while keeping the
question, resolution, prior and message text fixed. Name-only variants apply an
explicit simultaneous entity map. Resample variants change IDs only. Variants
are repeated measurements within their 20 original parent families.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

from exp1_prospective.context_reversal import materials as parent_materials
from .flow_specs import SPECS as FLOW_SPECS
from .nonflow_specs import SPECS as NONFLOW_SPECS

HERE = Path(__file__).resolve().parent
PARENT_PATH = HERE.parent / "runs/development_v1/development_families.jsonl"
VARIANTS = ("repaired_base", "name_only", "paraphrase", "resample")
RELATIONAL_CONTEXTS = ("positive", "negative", "broken")
VISIBLE_FIELDS = ("question", "resolution", "background", "prior_information", "evidence", "repeat_news")
PROTOCOL = "exp1_context_reversal_robustness_development_v2"
SPECS = {**FLOW_SPECS, **NONFLOW_SPECS}


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def rename(text: str, mapping: dict[str, str]) -> str:
    """Apply whole-phrase replacements simultaneously; replacement text is not rescanned."""
    if not mapping or len(set(mapping.values())) != len(mapping):
        raise ValueError("Rename maps must be nonempty and one-to-one")
    expression = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(name) for name in sorted(mapping, key=len, reverse=True)) + r")(?!\w)")
    return expression.sub(lambda match: mapping[match.group()], text)


def visible_payload(family: dict) -> dict:
    return {**{field: family[field] for field in VISIBLE_FIELDS}, "contexts": family["contexts"]}


def lexical_inventory(text: str) -> Counter:
    return Counter(re.findall(r"\w+|[^\w\s]", text.casefold()))


def relation_schedule(spec: dict, family_number: int, trend: str, context: str, *, paraphrase: bool = False) -> tuple[str, dict]:
    # Roles 0/1/2 respectively improve/harm/leave unchanged the target when activity rises.
    # Falling evidence reverses which active role gives the positive update.
    if trend not in {"rising", "falling"}:
        raise ValueError("Unsupported parent evidence trend")
    offset = 2 if context == "broken" else (0 if context == "positive" else 1)
    if trend == "falling" and context != "broken":
        offset = 1 - offset
    actor_for_role = {((actor_index + offset) % 3): actor_index for actor_index in range(3)}
    # The active actor moves across all three sentence positions across contexts.
    # The role ordering itself is rotated across parent families.
    role_order = [(family_number - 1 + position) % 3 for position in range(3)]
    roles = spec["roles_paraphrase" if paraphrase else "roles"]
    frame = spec["assignment_paraphrase" if paraphrase else "assignment_format"]
    sentences = [frame.format(actor=spec["actors"][actor_for_role[role]], role=roles[role]) for role in role_order]
    return " ".join(sentences), {
        "actor_for_role": {str(role): spec["actors"][actor_for_role[role]] for role in range(3)},
        "role_order": role_order,
        "evidence_actor_sentence_position": role_order.index(offset),
        "evidence_actor_role": offset,
    }


def _make_family(parent: dict, spec: dict, variant: str) -> dict:
    number = int(parent["family_id"].split("_")[-1])
    paraphrase = variant == "paraphrase"
    common = spec["common_paraphrase" if paraphrase else "common"]
    clauses, permutations = {}, {}
    for context in RELATIONAL_CONTEXTS:
        clauses[context], permutations[context] = relation_schedule(spec, number, parent["private_metadata"]["evidence_trend"], context, paraphrase=paraphrase)
    row = {
        "family_id": f"rv2_{parent['family_id']}_{variant}",
        "parent_family_id": parent["family_id"], "sampling_family_id": parent["family_id"],
        "variant": variant, "protocol": PROTOCOL,
        "split": "development", "provenance": parent_materials.PROVENANCE, "domain": parent["domain"],
        **{field: parent[field] for field in VISIBLE_FIELDS if field != "background"},
        "background": spec["background_paraphrase" if paraphrase else "background"],
        "contexts": {context: common + " " + clauses[context] for context in RELATIONAL_CONTEXTS},
        "private_metadata": {
            "author": "OpenAI Codex assistant", "independently_validated": False,
            "human_edited": False, "development_only": True,
            "expected_direction": {"positive": 1, "negative": -1, "broken": 0},
            "evidence_trend": parent["private_metadata"]["evidence_trend"],
            "mechanism_class": parent["private_metadata"]["mechanism_class"],
            "parent_material_sha256": sha(canonical(parent)),
            "common_context": common, "decisive_clauses": clauses,
            "relation_permutations": permutations,
            "actors": list(spec["actors"]),
            "roles_when_activity_increases": list(spec["roles_paraphrase" if paraphrase else "roles"]),
            "rename_map": {}, "repair_notes": list(spec["repair_notes"]),
            "sampling_note": "Four within-family variants; this row is not an independent sampling family.",
            "paraphrase_scope": ["background", "common_mechanism", "relation_schedule"],
            "fixed_across_base_paraphrase": ["question", "resolution", "prior_information", "evidence", "repeat_news"],
        },
    }
    if variant == "name_only":
        mapping = dict(spec["names"])
        for field in VISIBLE_FIELDS:
            row[field] = rename(row[field], mapping)
        row["contexts"] = {context: rename(text, mapping) for context, text in row["contexts"].items()}
        metadata = row["private_metadata"]
        metadata["common_context"] = rename(metadata["common_context"], mapping)
        metadata["decisive_clauses"] = {context: rename(text, mapping) for context, text in metadata["decisive_clauses"].items()}
        metadata["actors"] = [rename(actor, mapping) for actor in metadata["actors"]]
        metadata["roles_when_activity_increases"] = [rename(role, mapping) for role in metadata["roles_when_activity_increases"]]
        for permutation in metadata["relation_permutations"].values():
            permutation["actor_for_role"] = {role: rename(actor, mapping) for role, actor in permutation["actor_for_role"].items()}
        metadata["rename_map"] = mapping
    return row


def build_families(parents: list[dict] | None = None) -> list[dict]:
    parents = parent_materials.load_families(PARENT_PATH) if parents is None else parents
    parent_materials.validate_families(parents)
    expected = {f"dev_{number:02d}" for number in range(1, 21)}
    if {p["family_id"] for p in parents} != expected or len(parents) != 20 or set(SPECS) != set(range(1, 21)):
        raise ValueError("Robustness v2 must retain exactly all 20 original parent families")
    rows = [_make_family(parent, SPECS[int(parent["family_id"].split("_")[-1])], variant)
            for parent in parents for variant in VARIANTS]
    validate_families(rows)
    return rows


def validate_families(rows: list[dict]) -> None:
    parent_materials.validate_families(rows)
    if len(rows) != 80:
        raise ValueError("Expected 20 parents times four variants")
    grouped = {}
    for row in rows:
        parent = row.get("parent_family_id")
        if parent != row.get("sampling_family_id") or row.get("variant") not in VARIANTS:
            raise ValueError("Missing or invalid parent/variant sampling metadata")
        if row["family_id"] != f"rv2_{parent}_{row['variant']}":
            raise ValueError("Material identity does not match parent and variant")
        if row["variant"] in grouped.setdefault(parent, {}):
            raise ValueError("Duplicate parent/variant")
        grouped[parent][row["variant"]] = row
        contexts = list(row["contexts"].values())
        if any(lexical_inventory(contexts[0]) != lexical_inventory(text) for text in contexts[1:]):
            raise ValueError("Relational contexts must have the same lexical inventory")
        metadata = row["private_metadata"]
        positions = [p["evidence_actor_sentence_position"] for p in metadata["relation_permutations"].values()]
        if sorted(positions) != [0, 1, 2]:
            raise ValueError("Evidence-linked actor must occupy every position across contexts")
        for text in contexts:
            if any(text.count(actor) < 1 for actor in metadata["actors"]):
                raise ValueError("Every relational context must include all three actors")
        for context, clause in metadata["decisive_clauses"].items():
            expected_counter = Counter(metadata["actors"])
            if Counter({actor: clause.count(actor) for actor in metadata["actors"]}) != expected_counter:
                raise ValueError("Every schedule must assign each actor exactly once")
    if set(grouped) != {f"dev_{number:02d}" for number in range(1, 21)}:
        raise ValueError("All 20 original parents must remain sampling families")
    for parent, variants in grouped.items():
        if set(variants) != set(VARIANTS):
            raise ValueError("Each parent needs all four variants")
        base, names, paraphrase, resample = (variants[v] for v in VARIANTS)
        if visible_payload(base) != visible_payload(resample):
            raise ValueError("Resample must be byte-identical in every visible material field")
        mapping = names["private_metadata"]["rename_map"]
        expected = {field: rename(base[field], mapping) for field in VISIBLE_FIELDS}
        expected["contexts"] = {context: rename(text, mapping) for context, text in base["contexts"].items()}
        if visible_payload(names) != expected:
            raise ValueError("Name-only variant changed more than its explicit entity map")
        inverse = {value: key for key, value in mapping.items()}
        restored = {field: rename(names[field], inverse) for field in VISIBLE_FIELDS}
        restored["contexts"] = {context: rename(text, inverse) for context, text in names["contexts"].items()}
        if restored != visible_payload(base):
            raise ValueError("Name map must be exactly reversible on visible text")
        for field in ("question", "resolution", "prior_information", "evidence", "repeat_news"):
            if paraphrase[field] != base[field]:
                raise ValueError("Paraphrase changed a fixed question/prior/message field")
        if paraphrase["background"] == base["background"] or any(paraphrase["contexts"][c] == base["contexts"][c] for c in RELATIONAL_CONTEXTS):
            raise ValueError("Paraphrase must actually rewrite background and relations")


def repair_log(parents: list[dict], rows: list[dict]) -> list[dict]:
    by_parent = {p["family_id"]: p for p in parents}
    bases = [row for row in rows if row["variant"] == "repaired_base"]
    return [{"parent_family_id": base["parent_family_id"], "sampling_family_id": base["parent_family_id"],
             "domain": base["domain"], "parent_material_sha256": sha(canonical(by_parent[base["parent_family_id"]])),
             "repaired_base_sha256": sha(canonical(visible_payload(base))),
             "repairs": base["private_metadata"]["repair_notes"],
             "shared_repairs": ["All three actor-to-role assignments appear in each relational context; only the assignment changes.",
                                "Role rows are ordered by a parent-specific rotation, and the evidence actor occupies all three row positions across contexts.",
                                "Sidechannel facts and vocabulary appear in every relational context.",
                                "Resample duplicates repaired-base visible text with a different request identity."],
             "fixed_original_fields": [field for field in VISIBLE_FIELDS if field != "background"],
             "original_background": by_parent[base["parent_family_id"]]["background"],
             "repaired_background": base["background"],
             "original_contexts": by_parent[base["parent_family_id"]]["contexts"],
             "repaired_contexts": base["contexts"],
             "relation_permutations": base["private_metadata"]["relation_permutations"],
             "variants": [row["family_id"] for row in rows if row["parent_family_id"] == base["parent_family_id"]]}
            for base in bases]
