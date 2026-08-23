"""Connected components over event, series, and normalized-template identities."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def family_keys(row: dict[str, Any]) -> set[str]:
    keys = {f"event:{row['event_id']}"}
    if row.get("template_key"):
        keys.add(f"template:{row['template_key']}")
    if row.get("series_id"):
        keys.add(f"series:{row['series_id']}")
    return keys


def connected_index_groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    owner: dict[str, int] = {}
    for index, row in enumerate(rows):
        for key in family_keys(row):
            if key in owner:
                union(index, owner[key])
            else:
                owner[key] = index

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    return list(groups.values())
