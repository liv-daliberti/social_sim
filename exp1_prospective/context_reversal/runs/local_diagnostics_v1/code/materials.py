"""Synthetic development materials for the paired context-reversal pilot.

These assistant-authored candidates have NOT received independent human validation.
They are for exploratory protocol development only; they cannot constitute a locked
confirmatory sample. All causal relations are stipulated properties of fictional
settings. ``private_metadata`` and condition keys must never enter model prompts.

Regenerate the inspectable JSONL with ``python -m
exp1_prospective.context_reversal.materials``. No archived modules are imported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_MATERIALS_PATH = Path(__file__).parent / "data" / "development_families.jsonl"
PROVENANCE = "authored_development_unvalidated"
CONTEXTS = ("positive", "negative", "broken")


def _family(
    number: int, domain: str, question: str, resolution: str, background: str,
    baseline: str, repeat: str, evidence: str, common: str,
    clauses: tuple[str, str, str], chain: str, evidence_trend: str,
    mechanism_class: str,
) -> dict[str, Any]:
    return {
        "family_id": f"dev_{number:02d}",
        "split": "development",
        "provenance": PROVENANCE,
        "domain": domain,
        "question": question,
        "resolution": resolution,
        "background": background,
        "prior_information": f"{baseline} {repeat}",
        "repeat_news": repeat,
        "evidence": evidence,
        "contexts": dict(zip(CONTEXTS, (f"{common} {clause}" for clause in clauses))),
        "private_metadata": {
            "author": "OpenAI Codex assistant",
            "independently_validated": False,
            "human_edited": False,
            "expected_direction": {"positive": 1, "negative": -1, "broken": 0},
            "chain": chain,
            "evidence_trend": evidence_trend,
            "mechanism_class": mechanism_class,
            "common_context": common,
            "decisive_clauses": dict(zip(CONTEXTS, clauses)),
            "development_only": True,
        },
    }


def build_development_families() -> list[dict[str, Any]]:
    """Return 20 fictional candidates authored without inspecting model responses."""
    rows = [
        _family(
            1, "elections",
            "Will Mira Vale win the Westmere mayoral election on 14 October?",
            "YES if the certified Westmere mayoral tally names Mira Vale as winner; otherwise NO.",
            "Mira Vale and Oren Pike are the two Westmere mayoral candidates. Tessa Wynn runs for council in a separate town whose electorate does not overlap Westmere. The mayoral contest is competitive.",
            "Brightbridge's planning sheet lists 320 phone-bank hours for September.",
            "Brightbridge's planning meeting is scheduled for 2 September.",
            "Brightbridge reports 640 committed phone-bank hours for September; its previous planning sheet listed 320.",
            "Brightbridge's phone bank recruits canvassers. Additional phone-bank hours yield additional canvassers, whose door visits shift votes toward the candidate they serve. The teams and budgets for these three campaigns are separate.",
            ("Brightbridge's canvassers are assigned exclusively to Mira Vale.",
             "Brightbridge's canvassers are assigned exclusively to Oren Pike.",
             "Brightbridge's canvassers are assigned exclusively to Tessa Wynn."),
            "Phone-bank hours -> recruited canvassers -> votes for the assigned candidate -> mayoral result.",
            "rising", "competitive_allocation",
        ),
        _family(
            2, "transport_policy",
            "Will the Eastbank council select the Canal bus-lane proposal at its November meeting?",
            "YES if Canal receives the council's single transport-capital allocation in November; otherwise NO.",
            "Eastbank will choose exactly one of Canal bus lanes and Ridge tram upgrades. Grove footpaths belongs to a separate district and budget. Both Eastbank proposals are still under technical review.",
            "Arc Design has scheduled 180 engineering hours for the final review period.",
            "Arc Design's coordination session is booked for 6 October.",
            "Arc Design's revised schedule lists 90 engineering hours for the final review period; the earlier schedule listed 180.",
            "Arc Design uses engineering hours to resolve a proposal's construction uncertainties. Resolving these uncertainties lowers its contingency budget, and the council favors the proposal requiring the smaller budget. Work on one proposal does not service the others.",
            ("Arc Design's contract covers only the Ridge proposal.",
             "Arc Design's contract covers only the Canal proposal.",
             "Arc Design's contract covers only the Grove proposal."),
            "Engineering hours -> resolved construction uncertainties -> contingency budget -> choice between Canal and Ridge.",
            "falling", "competitive_allocation",
        ),
        _family(
            3, "business_procurement",
            "Will Northstar Furnishings receive the hotel desk contract on 20 November?",
            "YES if the hotel awards its desk contract to Northstar Furnishings on or before 20 November; otherwise NO.",
            "Northstar Furnishings and Solway Furnishings are the remaining hotel bidders. Pineway Furnishings serves a separate market and is not bidding. The hotel weighs both quoted price and quality.",
            "Cedar Components' production plan assumes a usable-panel yield of 70 panels per 100 blanks.",
            "Cedar Components' next stock count is scheduled for 9 October.",
            "Cedar Components' production audit records 90 usable panels per 100 blanks; the production plan assumed 70.",
            "Cedar Components passes savings from usable-panel yield to its sole contracted furniture firm. A higher yield reduces that firm's materials bill and permits a cheaper hotel bid without reducing quality. The other firms buy from independent suppliers.",
            ("Cedar Components' sole furniture contract is with Northstar Furnishings.",
             "Cedar Components' sole furniture contract is with Solway Furnishings.",
             "Cedar Components' sole furniture contract is with Pineway Furnishings."),
            "Panel yield -> contracted firm's material cost -> bid price -> hotel contract allocation.",
            "rising", "competitive_allocation",
        ),
        _family(
            4, "science_competition",
            "Will the Alton group be the first to submit a complete L-7 genome assembly by 30 November?",
            "YES if Alton submits a qualifying L-7 assembly by 30 November before its sole rival Bracken; otherwise NO.",
            "Alton and Bracken are assembling L-7 genomes for a first-submission award. Corven studies an unrelated species outside the award. All groups still require additional sequencing and assembly work.",
            "Mosaic Core's instrument schedule provides 160 sequencing batches during October.",
            "Mosaic Core's routine safety briefing is scheduled for 4 October.",
            "Mosaic Core's revised instrument schedule provides 80 sequencing batches during October; the previous schedule provided 160.",
            "Mosaic Core supplies sequence reads exclusively to one group. Additional batches fill gaps in that group's draft genome, reducing the assembly work needed before submission. Its allocation does not change the instruments available to other groups.",
            ("Mosaic Core's sequencing allocation belongs entirely to Bracken.",
             "Mosaic Core's sequencing allocation belongs entirely to Alton.",
             "Mosaic Core's sequencing allocation belongs entirely to Corven."),
            "Sequencing batches -> draft gaps filled -> assembly completion time -> order of L-7 submissions.",
            "falling", "competitive_allocation",
        ),
        _family(
            5, "sports",
            "Will the Marlin club win its 18 November match against the Falcon club?",
            "YES if Marlin is recorded as the winner after any required tiebreak on 18 November; otherwise NO.",
            "Marlin and Falcon are evenly matched volleyball clubs. The Heron club plays in an unrelated league with no shared players. Fitness and in-match execution remain uncertain.",
            "Seabright Court's November plan provides 24 specialist practice hours.",
            "Seabright Court's equipment inventory is booked for 1 November.",
            "Seabright Court's confirmed November bookings provide 48 specialist practice hours; the earlier plan provided 24.",
            "Seabright's specialist sessions improve serve placement through repeated drills. Better serve placement produces more difficult returns and additional scoring opportunities. The booked court time is exclusive to one club and does not displace the others' usual practices.",
            ("Seabright's specialist sessions are reserved for the Marlin club.",
             "Seabright's specialist sessions are reserved for the Falcon club.",
             "Seabright's specialist sessions are reserved for the Heron club."),
            "Practice hours -> serve placement -> scoring opportunities -> Marlin/Falcon match result.",
            "rising", "competitive_allocation",
        ),
        _family(
            6, "warehouse_logistics",
            "Will the Fenwick depot dispatch at least 900 orders during the first week of November?",
            "YES if Fenwick records at least 900 dispatched orders from 1 through 7 November; otherwise NO.",
            "Fenwick's packing line draws empty cartons from the Cedar buffer. The Birch buffer belongs to an independent depot. Fenwick's order demand is sufficient, but packaging interruptions and other work may affect dispatches.",
            "Loader K's work plan assumes 60 carton transfers per hour.",
            "Loader K's maintenance log is filed every Thursday.",
            "Loader K's acceptance test records 30 carton transfers per hour; its work plan assumed 60.",
            "Cartons available in Cedar prevent stoppages at Fenwick's packing line, allowing more completed parcels to reach dispatch. Cedar is below its storage limit throughout the planned range, and all transfers mentioned here are feasible. Birch has no operational connection to Fenwick.",
            ("Loader K transfers cartons out of the Cedar buffer.",
             "Loader K transfers cartons into the Cedar buffer.",
             "Loader K transfers cartons inside the Birch buffer."),
            "Loader transfer rate -> Cedar carton stock -> packing interruptions -> dispatched orders.",
            "falling", "buffer_flow",
        ),
        _family(
            7, "energy",
            "Will Harrow's hydro station supply at least 500 megawatt-hours during November?",
            "YES if the station's meter records at least 500 megawatt-hours generated during November; otherwise NO.",
            "Harrow's turbines draw water from Lake Alder. Lake Birch is on a separate catchment and grid. Alder remains well below its spill threshold; future rainfall and turbine maintenance are uncertain.",
            "Canal K's operating plan specifies a flow of 12 cubic metres per second.",
            "Canal K's inspection report is issued every Monday.",
            "Canal K's flow test records 18 cubic metres per second; its operating plan specified 12.",
            "Additional usable water in Alder sustains turbine operation through dry intervals, adding generation hours. The planned flows stay within channel and turbine operating limits. Water moving through Birch cannot reach Alder or change Harrow's power demand.",
            ("Canal K carries water into Lake Alder's usable storage.",
             "Canal K carries water from Lake Alder's usable storage.",
             "Canal K carries water within Lake Birch's isolated storage."),
            "Canal flow -> Alder stored water -> turbine operating hours -> generated energy.",
            "rising", "buffer_flow",
        ),
        _family(
            8, "laboratory_biology",
            "Will strain R's fluorescence assay exceed 400 units at the end of the October batch?",
            "YES if the final calibrated fluorescence reading from strain R exceeds 400 units; otherwise NO.",
            "Strain R grows in a controlled culture. Its fluorescence is proportional to its living cell count. Its final biomass remains uncertain because several inputs are variable.",
            "Strain K's feed schedule supplies 20 grams of substrate per day.",
            "Strain K's weighing log is checked each Wednesday.",
            "Strain K's confirmed feed schedule supplies 10 grams of substrate per day; the previous schedule supplied 20.",
            "Extra substrate grows more strain K cells, producing additional secretion that enters R's culture. Within the stated range the secretion's effect is stable, and K neither competes with R for other inputs nor changes the fluorescence instrument.",
            ("Strain K's secretion acts solely as a growth inhibitor for R.",
             "Strain K's secretion acts solely as a growth nutrient for R.",
             "Strain K's secretion acts solely as an inert tracer for R."),
            "K substrate -> K biomass and secretion -> R growth -> R fluorescence.",
            "falling", "biological_interaction",
        ),
        _family(
            9, "software_operations",
            "Will the Larch service finish its release validation by 17 November?",
            "YES if every required release-validation check finishes by the end of 17 November; otherwise NO.",
            "Larch's validation pipeline shares a worker pool with the Maple queue. A separate Cypress queue uses isolated workers. Check runtimes and incoming production traffic remain uncertain.",
            "Worker K's capacity plan lists 12 record operations per second.",
            "Worker K's version manifest is archived each Tuesday.",
            "Worker K's benchmark records 24 record operations per second; its capacity plan listed 12.",
            "Pending records in Maple consume the workers needed for release validation. Clearing those records frees worker time for the required checks. Worker K has a separate processor, all described operations are useful, and Cypress consumes none of Larch's resources.",
            ("Worker K removes pending records from the Maple queue.",
             "Worker K inserts pending records into the Maple queue.",
             "Worker K mirrors existing records within the Cypress queue."),
            "K operations -> Maple backlog -> worker time available for checks -> release validation completion.",
            "rising", "queue_load",
        ),
        _family(
            10, "manufacturing",
            "Will the Brindle factory complete at least 4,000 moulded cases in November?",
            "YES if the factory's accepted-output count reaches 4,000 cases during November; otherwise NO.",
            "Brindle's moulding line uses resin from hopper A. Hopper B serves an independent factory. The line has sufficient orders, but maintenance and resin availability can interrupt production.",
            "Conveyor K's schedule specifies 40 kilograms of resin moved per minute.",
            "Conveyor K's certification sheet is reviewed on 5 November.",
            "Conveyor K's load test records 20 kilograms of resin moved per minute; its schedule specified 40.",
            "Resin available in hopper A prevents idle cycles at Brindle's moulding line. Fewer idle cycles yield additional accepted cases. The hoppers remain within operating limits throughout the scheduled range, and the two factories have no shared material supply.",
            ("Conveyor K carries resin out of hopper A.",
             "Conveyor K carries resin into hopper A.",
             "Conveyor K carries resin within hopper B."),
            "Conveyor flow -> A resin availability -> moulding idle cycles -> accepted cases.",
            "falling", "buffer_flow",
        ),
        _family(
            11, "agriculture",
            "Will the Willow farm harvest at least 70 tonnes of carrots in November?",
            "YES if Willow's November weighbridge records at least 70 tonnes of harvested carrots; otherwise NO.",
            "Willow's carrot plots are drier than their preferred range. Adjacent Juniper plots belong to a separate water system. Rainfall and pests may still affect the harvest.",
            "Channel K's irrigation schedule specifies 8 litres per second.",
            "Channel K's staff rota is posted each Friday.",
            "Channel K's flow test records 12 litres per second; its irrigation schedule specified 8.",
            "Within the scheduled water range, moisture retained in Willow's plots supports leaf growth, which supplies energy for heavier carrot roots. Neither flooding nor nutrient runoff occurs over this range. Juniper's water remains outside Willow's root zone.",
            ("Channel K carries water into the Willow plots.",
             "Channel K carries water from the Willow plots.",
             "Channel K carries water within the Juniper plots."),
            "Channel flow -> Willow soil moisture -> leaf growth -> carrot root mass and harvest.",
            "rising", "buffer_flow",
        ),
        _family(
            12, "education",
            "Will Elm school receive the district's single robotics-finals place in November?",
            "YES if the district awards its robotics-finals place to Elm school in November; otherwise NO.",
            "Elm and Ash are the two schools competing for one finals place. Yew competes in another district with no shared participants. Judges score both robot reliability and task completion.",
            "The Beacon mentors' schedule contains 40 workshop hours before district trials.",
            "The Beacon mentors' coordination call is booked for 3 October.",
            "The Beacon mentors' confirmed schedule contains 20 workshop hours before district trials; the earlier schedule contained 40.",
            "Beacon workshops teach teams to diagnose mechanical faults. Diagnosing faults before trials reduces breakdowns, yielding more completed tasks under the judges' rules. The mentors serve one team, and their hours do not change the other schools' resources.",
            ("Beacon's workshop allocation belongs exclusively to Ash school.",
             "Beacon's workshop allocation belongs exclusively to Elm school.",
             "Beacon's workshop allocation belongs exclusively to Yew school."),
            "Workshop hours -> diagnosed faults -> trial breakdowns -> completed tasks and finals selection.",
            "falling", "competitive_allocation",
        ),
        _family(
            13, "clinic_operations",
            "Will Alder clinic's routine appointment waiting time be under 14 days on 30 November?",
            "YES if the clinic's routine-booking audit on 30 November reports a median wait below 14 days; otherwise NO.",
            "Alder books routine visits from the Cedar request queue. Birch is an independent clinic with its own request queue. Appointment capacity is limited, and future patient demand remains uncertain.",
            "Processor K's staffing plan allows 30 request operations per day.",
            "Processor K's audit meeting is scheduled for 7 November.",
            "Processor K's staffing confirmation allows 60 request operations per day; the previous plan allowed 30.",
            "Requests pending in Cedar occupy Alder's future appointment slots. Fewer pending requests free earlier slots for new bookings and shorten their waits. Every operation specified here concerns a real request, and Birch shares neither staff nor appointments with Alder.",
            ("Processor K removes withdrawn requests from the Cedar queue.",
             "Processor K inserts eligible requests into the Cedar queue.",
             "Processor K reviews pending requests within the Birch queue."),
            "Processor operations -> Cedar pending requests -> earlier available slots -> waiting time.",
            "rising", "queue_load",
        ),
        _family(
            14, "conservation",
            "Will at least 800 Rowan saplings survive through November in the trial enclosure?",
            "YES if the end-of-November census counts at least 800 living Rowan saplings in the trial enclosure; otherwise NO.",
            "The Rowan enclosure contains saplings and root-feeding grubs. A screen controls entry of beetle K. Sapling survival also depends on weather and disease, which remain uncertain.",
            "The insectary's October plan supplies 200 beetle K adults to the enclosure.",
            "The insectary's tray inventory is scheduled for 8 October.",
            "The insectary's confirmed October delivery contains 100 beetle K adults; its earlier plan contained 200.",
            "Delivered beetles remain active in the enclosure throughout the month. Root-feeding grubs injure sapling roots, and root injury reduces water uptake and survival. Beetle K has no other interactions with the saplings, grubs, disease, or weather.",
            ("Beetle K eats living sapling roots in the enclosure.",
             "Beetle K eats root-feeding grubs in the enclosure.",
             "Beetle K eats inert fallen debris in the enclosure."),
            "K adults -> roots consumed or root-feeding grubs consumed -> root injury -> sapling survival.",
            "falling", "biological_interaction",
        ),
        _family(
            15, "maritime_logistics",
            "Will at least 25 deep-draught cargo ships reach Merrow's inner dock during November?",
            "YES if Merrow's log records at least 25 deep-draught cargo arrivals at the inner dock during November; otherwise NO.",
            "Deep-draught ships use the Alder channel to enter Merrow. The Birch channel leads to an independent harbour. Merrow has enough potential bookings, but weather and channel access affect arrivals.",
            "Dredger K's work plan specifies 200 cubic metres of sediment per day.",
            "Dredger K's routine crew briefing is held each Monday.",
            "Dredger K's capacity test records 400 cubic metres of sediment per day; its work plan specified 200.",
            "Sediment on Alder's bed reduces its navigable depth. Additional depth opens more tide windows in which deep-draught ships can reach Merrow. The stated work remains inside the tested depth range; Birch's sediment and tides do not affect Alder.",
            ("Dredger K removes sediment from the Alder channel bed.",
             "Dredger K deposits sediment onto the Alder channel bed.",
             "Dredger K relocates sediment within the Birch channel bed."),
            "Dredger sediment rate -> Alder depth -> accessible tide windows -> dock arrivals.",
            "rising", "buffer_flow",
        ),
        _family(
            16, "telecommunications",
            "Will the Fern relay link meet its 99 percent uptime target during November?",
            "YES if the audited Fern link is available for at least 99 percent of November; otherwise NO.",
            "Fern's receiver obtains a direct radio signal and can also receive station K. Fading of the direct signal remains uncertain. K's received signal is weaker than the direct signal across the operating range.",
            "Station K's timetable assigns 120 transmission minutes per day.",
            "Station K's clock inspection is scheduled for 2 November.",
            "Station K's confirmed timetable assigns 60 transmission minutes per day; its previous timetable assigned 120.",
            "A stronger combined signal at Fern's receiver prevents interruptions during fades. K's waveform and phase remain stable during each transmission. An unused-frequency transmission is filtered out and consumes none of Fern's channel capacity.",
            ("Station K transmits opposite-phase copies on Fern's operating frequency.",
             "Station K transmits matching-phase copies on Fern's operating frequency.",
             "Station K transmits matching-phase copies on an unused frequency."),
            "K transmission duration -> destructive/constructive received signal combination -> fade interruptions -> uptime.",
            "falling", "signal_interaction",
        ),
        _family(
            17, "retail",
            "Will Wren Books sell at least 600 copies of its autumn title during November?",
            "YES if Wren Books records at least 600 paid, nonreturned copies of its autumn title during November; otherwise NO.",
            "Wren Books and Lark Books sell rival titles to a fixed local reading-club audience whose members buy one autumn title each. Ibis Books serves a separate region. Readers' final choices remain uncertain.",
            "Harbour Radio's autumn package is scheduled to reach 10,000 distinct listeners.",
            "Harbour Radio's production meeting is booked for 1 October.",
            "Harbour Radio's confirmed autumn package is scheduled to reach 20,000 distinct listeners; the earlier schedule listed 10,000.",
            "The radio package increases awareness of its advertised title, bringing more reading-club members to that title's shop demonstrations. Demonstrations convert some visitors into buyers. The regions do not overlap, and the radio package advertises one title only.",
            ("Harbour Radio's autumn package advertises the Wren Books title.",
             "Harbour Radio's autumn package advertises the Lark Books title.",
             "Harbour Radio's autumn package advertises the Ibis Books title."),
            "Radio reach -> title awareness -> demonstration visits -> purchases from a shared audience.",
            "rising", "competitive_allocation",
        ),
        _family(
            18, "water_treatment",
            "Will the Brook treatment works deliver at least 90 megalitres of certified water in November?",
            "YES if Brook's November certified-water meter totals at least 90 megalitres; otherwise NO.",
            "Brook takes raw water from the Cedar settling basin. Birch belongs to an independent treatment works. Water demand is sufficient, but rainfall and filter maintenance remain uncertain.",
            "Pump K's operating plan moves 30 kilograms of suspended silt per hour.",
            "Pump K's service roster is posted every Wednesday.",
            "Pump K's commissioning test records 15 kilograms of suspended silt per hour; its operating plan specified 30.",
            "Suspended silt in Cedar clogs Brook's filters, forcing cleaning breaks. Fewer cleaning breaks permit additional certified-water output. All planned silt loads are within the treatment works' operating range. Birch's contents cannot enter Brook's system.",
            ("Pump K carries suspended silt into the Cedar basin.",
             "Pump K carries suspended silt from the Cedar basin.",
             "Pump K carries suspended silt within the Birch basin."),
            "Pump silt rate -> Cedar silt load -> filter cleaning breaks -> certified output.",
            "falling", "buffer_flow",
        ),
        _family(
            19, "horticulture",
            "Will the Aster greenhouse produce at least 12,000 marketable blooms during November?",
            "YES if Aster's accepted-flower count reaches 12,000 blooms during November; otherwise NO.",
            "Aster grows a single flower variety whose buds are attacked by mite M. Organism K is introduced on carrier strips. Temperature and other growing conditions remain uncertain.",
            "The strip supplier's November plan contains 40 K-bearing carrier strips.",
            "The strip supplier's packing audit is scheduled for 6 November.",
            "The strip supplier's confirmed November allocation contains 80 K-bearing carrier strips; its earlier plan contained 40.",
            "Additional carrier strips establish more K organisms on greenhouse benches. Damage to unopened buds prevents those buds from becoming marketable blooms. K has no effects other than the feeding behavior specified here and does not alter other greenhouse resources.",
            ("Organism K feeds exclusively upon bud-attacking mite M.",
             "Organism K feeds exclusively upon unopened flower buds.",
             "Organism K feeds exclusively upon discarded carrier material."),
            "Carrier strips -> K population -> mites consumed or buds consumed -> marketable blooms.",
            "rising", "biological_interaction",
        ),
        _family(
            20, "event_production",
            "Will the Finch theatre open its new production by 30 November?",
            "YES if Finch holds its first public performance of the new production by 30 November; otherwise NO.",
            "Finch's rehearsal space must be cleared of crates before full stage rehearsals. The Robin theatre has an independent space and staff. Rehearsal progress and technical faults remain uncertain.",
            "Hauler K's November schedule specifies 20 crate movements per day.",
            "Hauler K's route review is scheduled for 4 November.",
            "Hauler K's confirmed November schedule specifies 10 crate movements per day; its previous schedule specified 20.",
            "Crates stored on Finch's rehearsal floor occupy the stage area and postpone full rehearsals. Additional full rehearsals resolve blocking problems needed before opening. Each stated movement changes floor occupancy, while Robin's floor and operations are independent of Finch.",
            ("Hauler K carries crates onto the Finch rehearsal floor.",
             "Hauler K carries crates off the Finch rehearsal floor.",
             "Hauler K carries crates onto the Robin rehearsal floor."),
            "Hauler movements -> occupied Finch stage space -> full rehearsals -> readiness for opening.",
            "falling", "buffer_flow",
        ),
    ]
    validate_families(rows)
    return rows


def validate_family(family: dict[str, Any]) -> None:
    """Check structural guarantees; this is NOT semantic or independent validation."""
    visible = ("family_id", "domain", "question", "resolution", "background",
               "prior_information", "evidence", "repeat_news")
    for key in visible:
        if not isinstance(family.get(key), str) or not family[key].strip():
            raise ValueError(f"Missing/non-string field {key}")
    if family.get("split") != "development" or family.get("provenance") != PROVENANCE:
        raise ValueError("These materials must remain unvalidated development items")
    if family["repeat_news"] not in family["prior_information"]:
        raise ValueError("Repeated news must already be present verbatim at baseline")
    if family["repeat_news"] == family["evidence"]:
        raise ValueError("Repeated and new evidence must be different items")
    if family["evidence"] in family["prior_information"]:
        raise ValueError("New evidence must not already be present at baseline")
    contexts = family.get("contexts", {})
    if set(contexts) != set(CONTEXTS) or len(set(contexts.values())) != 3:
        raise ValueError("Each family requires three distinct bridge contexts")
    lengths = [len(contexts[key].split()) for key in CONTEXTS]
    if max(lengths) - min(lengths) > 3:
        raise ValueError("Bridge-context lengths differ by more than three words")
    metadata = family.get("private_metadata", {})
    if metadata.get("independently_validated") is not False:
        raise ValueError("Authored candidate status must not imply independent validation")
    if metadata.get("expected_direction") != {"positive": 1, "negative": -1, "broken": 0}:
        raise ValueError("Missing development-author direction labels")
    common = metadata.get("common_context", "")
    clauses = metadata.get("decisive_clauses", {})
    if not common or any(contexts[key] != f"{common} {clauses.get(key, '')}" for key in CONTEXTS):
        raise ValueError("Contexts must differ only in the decisive clause")


def validate_families(families: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    for family in families:
        validate_family(family)
        if family["family_id"] in ids:
            raise ValueError(f"Duplicate family_id: {family['family_id']}")
        ids.add(family["family_id"])


def load_families(path: str | Path | None = None) -> list[dict[str, Any]]:
    source = Path(path) if path is not None else DEFAULT_MATERIALS_PATH
    families = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    validate_families(families)
    return families


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MATERIALS_PATH)
    args = parser.parse_args()
    rows = build_development_families()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(f"Wrote {len(rows)} unvalidated development families to {args.output}")


if __name__ == "__main__":
    main()
