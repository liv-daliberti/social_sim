"""Authored relation schedules for the eleven non-flow parent families.

Every schedule uses all three actors and all three roles.  Role order is the
effect of *increased actor activity* on the original YES outcome: helpful,
harmful, then a channel outside that outcome.  The compiler, rather than these
materials, reverses the first two roles when the evidence reports a decrease.
These fictional development materials have not received human validation.
"""

SPECS = {
    1: {
        "background": (
            "Mira Vale and Oren Pike contest the Westmere mayoral election. Tessa Wynn "
            "contests a council seat in another town. Each voter belongs to one town's "
            "electorate, and each campaign has its own staff and budget. The Westmere "
            "contest remains competitive."
        ),
        "common": (
            "Brightbridge, Clearbrook, and Stonegate each operate a phone bank for one "
            "candidate. More phone-bank hours recruit more canvassers; their door visits "
            "shift votes toward their assigned candidate. Each bank has its own callers "
            "and funding, and additional canvassers stay with their assigned campaign. "
            "A change in one bank's hours leaves the other banks' hours fixed."
        ),
        "background_paraphrase": (
            "Westmere voters will choose between Mira Vale and Oren Pike for mayor. "
            "In a different town, Tessa Wynn is seeking a council seat. Voter rolls, "
            "campaign personnel, and campaign finances are specific to each town and "
            "campaign. Either Westmere candidate could still win."
        ),
        "common_paraphrase": (
            "A candidate receives canvassers from the phone bank listed beside their "
            "name. Brightbridge, Clearbrook, and Stonegate have their own staff and "
            "budgets. Spending more hours making calls brings in more canvassers, whose "
            "visits persuade voters to support that candidate. Those recruits work for "
            "that campaign throughout; revising one bank's hours preserves the other "
            "two schedules."
        ),
        "actors": ("Brightbridge", "Clearbrook", "Stonegate"),
        "roles": ("Mira Vale", "Oren Pike", "Tessa Wynn"),
        "roles_paraphrase": ("Mira Vale", "Oren Pike", "Tessa Wynn"),
        "assignment_format": "{actor}'s canvassers are assigned to {role}.",
        "assignment_paraphrase": "{role} receives the canvassers recruited by {actor}.",
        "names": {
            "Mira Vale": "Nora Reed", "Oren Pike": "Ilan Frost",
            "Tessa Wynn": "Vera Moss", "Westmere": "Dunhaven",
            "Brightbridge": "Copperfield", "Clearbrook": "Silvergrove",
            "Stonegate": "Amberford",
        },
        "repair_notes": [
            "All three campaign recipients appear in every assignment schedule.",
            "Separate voter rolls and fixed other-bank schedules close turnout and resource spillovers.",
        ],
    },
    2: {
        "background": (
            "Eastbank will give one transport-capital allocation to either Canal bus "
            "lanes or Ridge tram upgrades. Grove footpaths is considered by another "
            "district using that district's budget. Each proposal has its own technical "
            "review and construction estimate; the two Eastbank reviews remain open."
        ),
        "common": (
            "Arc Design, Span Design, and Beam Design each review one proposal using "
            "their own engineering staff. Additional hours resolve more construction "
            "uncertainties, lowering that proposal's contingency budget. Eastbank "
            "favors the proposal with the smaller required budget. Findings apply to "
            "the reviewed proposal, and changing one firm's hours leaves the other "
            "firms' hours and the districts' allocations fixed."
        ),
        "background_paraphrase": (
            "The November capital choice in Eastbank is between the Canal bus-lane "
            "plan and the Ridge tram plan, with funding available for one. A different "
            "district pays for and decides on Grove footpaths. Construction costs are "
            "being checked separately for each plan, and neither Eastbank review is "
            "finished."
        ),
        "common_paraphrase": (
            "Lower contingency allowances make a plan more attractive to Eastbank's "
            "council. An engineering review lowers that allowance by settling "
            "construction questions, so more review time improves the assigned plan's "
            "position. Arc Design, Span Design, and Beam Design each supply their own "
            "engineers to one plan. A revised schedule changes only that firm's work; "
            "the other schedules, district funds, and other plans' findings stay as before."
        ),
        "actors": ("Arc Design", "Span Design", "Beam Design"),
        "roles": ("Canal", "Ridge", "Grove"),
        "roles_paraphrase": ("Canal", "Ridge", "Grove"),
        "assignment_format": "{actor}'s engineering contract covers the {role} proposal.",
        "assignment_paraphrase": "The {role} proposal receives its engineering review from {actor}.",
        "names": {
            "Eastbank": "Westridge", "Canal": "Meadow", "Ridge": "Valley",
            "Grove": "Orchard", "Arc Design": "Crown Design",
            "Span Design": "Pier Design", "Beam Design": "Frame Design",
        },
        "repair_notes": [
            "Distinct engineering providers preserve the original proposal-cost mechanism without a shared staff budget.",
            "Falling evidence reverses the helpful and harmful assignments; role order itself remains activity-based.",
        ],
    },
    3: {
        "background": (
            "Northstar Furnishings and Solway Furnishings are the two remaining "
            "bidders for the hotel's desks. Pineway Furnishings supplies an office "
            "buyer in another market. The hotel weighs quoted price and quality. "
            "Each furniture firm has its own customer orders and material contract."
        ),
        "common": (
            "Cedar Components, Hazel Components, and Maple Components each supply "
            "one furniture firm from their own production line. A higher usable-panel "
            "yield lowers the assigned firm's materials bill, and that saving is "
            "passed through to a lower customer quote at the same quality. For a "
            "hotel bidder, the lower quote improves its chance of receiving the desk "
            "contract. A yield change at one supplier leaves the other suppliers' "
            "prices, yields, and output fixed."
        ),
        "background_paraphrase": (
            "The hotel will choose its desk supplier from Northstar Furnishings and "
            "Solway Furnishings, considering both price and quality. Pineway Furnishings "
            "is preparing furniture for an office customer in a different market. "
            "The firms obtain materials under separate contracts and fulfill their "
            "own orders."
        ),
        "common_paraphrase": (
            "When more blanks become usable panels, a components maker can charge "
            "its sole furniture customer less. That customer reduces its quote by the "
            "saving while keeping quality constant; the hotel's two bidders become "
            "more competitive when their quotes fall. Cedar Components, Hazel "
            "Components, and Maple Components operate their own lines. Revising one "
            "line's yield preserves the other makers' output, yield, and prices."
        ),
        "actors": ("Cedar Components", "Hazel Components", "Maple Components"),
        "roles": ("Northstar Furnishings", "Solway Furnishings", "Pineway Furnishings"),
        "roles_paraphrase": ("Northstar Furnishings", "Solway Furnishings", "Pineway Furnishings"),
        "assignment_format": "{actor}'s sole furniture customer is {role}.",
        "assignment_paraphrase": "{role} obtains its panels exclusively from {actor}.",
        "names": {
            "Northstar Furnishings": "Fairhaven Furnishings",
            "Solway Furnishings": "Redcliff Furnishings",
            "Pineway Furnishings": "Bluewater Furnishings",
            "Cedar Components": "Aspen Components",
            "Hazel Components": "Walnut Components",
            "Maple Components": "Poplar Components",
        },
        "repair_notes": [
            "Makes lower cost actually pass through to the quote, closing the original permits-versus-does ambiguity.",
            "The third firm's quote concerns its office customer, avoiding a hotel bid by a nonbidder.",
        ],
    },
    4: {
        "background": (
            "Alton and Bracken compete for the first-submission award for an L-7 "
            "genome assembly. Corven assembles a different species for its own "
            "archive. Each group still needs sequencing and assembly work, and the "
            "award comparison is between Alton's and Bracken's qualifying submissions."
        ),
        "common": (
            "Mosaic Core, Tessera Core, and Pattern Core each deliver sequence reads "
            "to one group using their own instruments. More batches fill more gaps in "
            "the recipient's draft genome, reducing the remaining assembly work and "
            "bringing its completion closer. Reads stay with their assigned project. "
            "Changing one core's batches leaves the other cores' capacity and "
            "deliveries fixed."
        ),
        "background_paraphrase": (
            "The L-7 award goes to the earlier qualifying submission from Alton or "
            "Bracken. Corven is building an assembly of another species to place in "
            "its archive. All three projects have gaps to sequence and work to finish "
            "before their assemblies are complete."
        ),
        "common_paraphrase": (
            "A group can finish its draft sooner when additional sequence batches "
            "close gaps that would otherwise need assembly work. Mosaic Core, Tessera "
            "Core, and Pattern Core run separate instruments and provide these reads "
            "to the project listed for each core. The receiving project keeps those "
            "reads. An adjustment to one core's schedule preserves the amounts and "
            "instrument capacity available from the other two."
        ),
        "actors": ("Mosaic Core", "Tessera Core", "Pattern Core"),
        "roles": ("Alton", "Bracken", "Corven"),
        "roles_paraphrase": ("Alton", "Bracken", "Corven"),
        "assignment_format": "{actor}'s sequencing batches are assigned to {role}.",
        "assignment_paraphrase": "{role} receives the sequence reads produced by {actor}.",
        "names": {
            "Alton": "Darrow", "Bracken": "Elden", "Corven": "Fallow",
            "L-7": "P-6", "Mosaic Core": "Prism Core",
            "Tessera Core": "Facet Core", "Pattern Core": "Lattice Core",
        },
        "repair_notes": [
            "Preserves the race and deadline while specifying the third project's positive purpose.",
            "Dedicated instruments and retained reads rule out capacity substitution and information sharing.",
        ],
    },
    5: {
        "background": (
            "Marlin and Falcon are evenly matched volleyball clubs meeting on "
            "18 November. Heron plays a match in another league. Each club has its "
            "own players and coaches. Fitness and execution during the matches "
            "remain uncertain."
        ),
        "common": (
            "Seabright Court, Bayview Court, and Shoreline Court each provide "
            "specialist serving practice to one club. More practice hours improve "
            "serve placement, creating more difficult returns and more scoring "
            "opportunities for that club. Courts have their own coaches and booked "
            "time; extra sessions supplement the recipient's usual practice. "
            "Changing one court's hours leaves the other clubs' practice and "
            "players fixed."
        ),
        "background_paraphrase": (
            "The 18 November volleyball match pairs Marlin with Falcon, whose "
            "abilities are similar. Heron's fixture belongs to a different league. "
            "Club rosters and coaching staffs are separate, and both fitness and "
            "match-day performance are still uncertain."
        ),
        "common_paraphrase": (
            "A club gains scoring chances by placing serves where the opposing "
            "side finds them harder to return. Additional specialist drills improve "
            "that placement. Seabright Court, Bayview Court, and Shoreline Court "
            "each book their own coaches and facilities for one club's extra drills. "
            "These drills add to ordinary practice; revising a court's booking "
            "preserves the other clubs' sessions and rosters."
        ),
        "actors": ("Seabright Court", "Bayview Court", "Shoreline Court"),
        "roles": ("Marlin", "Falcon", "Heron"),
        "roles_paraphrase": ("Marlin", "Falcon", "Heron"),
        "assignment_format": "{actor}'s specialist sessions are reserved for the {role} club.",
        "assignment_paraphrase": "The {role} club practices its specialist serves at {actor}.",
        "names": {
            "Marlin": "Otter", "Falcon": "Osprey", "Heron": "Kestrel",
            "Seabright Court": "Dawnlight Court", "Bayview Court": "Hillview Court",
            "Shoreline Court": "Ridgeline Court",
        },
        "repair_notes": [
            "All courts and clubs appear in each schedule, with ordinary league membership facts shared across contexts.",
            "Extra practice supplements ordinary practice and uses dedicated coaches and facilities.",
        ],
    },
    8: {
        "background": (
            "Strain R grows in a controlled culture, and its measured fluorescence "
            "is proportional to its living cell count. A reference bead in the vessel "
            "has its own tracer readout. Variable culture inputs leave the final "
            "living-cell count uncertain."
        ),
        "common": (
            "Strain K, strain L, and strain M each grow in a separate feed chamber "
            "and send their secretion into the vessel. More substrate grows more "
            "cells of the fed strain and produces more of its secretion. A nutrient "
            "raises strain R's growth, an inhibitor lowers strain R's growth, and a bead tracer "
            "changes the reference-bead readout. The reported fluorescence measures "
            "strain R alone. Each secretion acts only through its listed role; these effects "
            "retain their direction throughout the stated range. Changing one "
            "strain's feed leaves the other feed chambers and culture inputs fixed."
        ),
        "background_paraphrase": (
            "The culture assay counts light from living strain R cells, with each "
            "living cell contributing the same amount. The vessel also contains a "
            "reference bead monitored on its own channel. Several growing conditions "
            "are variable, so strain R's eventual population is uncertain."
        ),
        "common_paraphrase": (
            "Feeding strain K, strain L, or strain M increases that strain's population "
            "in its own chamber, so more secretion reaches the culture vessel. "
            "The assigned product determines what follows: a nutrient promotes strain R's "
            "growth, an inhibitor suppresses strain R's growth, or a tracer changes the bead's "
            "separate reading. The assay used for resolution records strain R's light only. "
            "Products have exactly their assigned effects with the same directional "
            "response over the planned range. Adjusting one feed preserves the "
            "remaining feeds and the culture's other inputs."
        ),
        "actors": ("Strain K", "Strain L", "Strain M"),
        "roles": ("a growth nutrient for strain R", "a growth inhibitor for strain R", "a tracer for the reference bead"),
        "roles_paraphrase": ("a nutrient that supports strain R's growth", "an inhibitor that suppresses strain R's growth", "a tracer that labels the reference bead"),
        "assignment_format": "{actor}'s secretion serves as {role}.",
        "assignment_paraphrase": "The product released by {actor} is {role}.",
        "names": {
            "Strain R": "Strain T", "strain R": "strain T",
            "Strain K": "Strain V", "strain K": "strain V",
            "Strain L": "Strain W", "strain L": "strain W",
            "Strain M": "Strain X", "strain M": "strain X",
        },
        "repair_notes": [
            "Makes the tracer channel concrete and puts nutrient, inhibitor, and tracer relations in every context.",
            "Separate feeds and a target-only readout remove competition and optical contamination paths.",
            "Stipulates stable directional effects despite the other two secretions being present.",
        ],
    },
    12: {
        "background": (
            "Elm and Ash compete for one district robotics-finals place. Yew enters "
            "another district's trials. Each school has its own team and equipment. "
            "Judges evaluate robot reliability and completion of the assigned tasks."
        ),
        "common": (
            "The Beacon mentors, Summit mentors, and Lantern mentors each teach one "
            "school to diagnose mechanical faults before trials. More workshop hours "
            "resolve more faults, reducing breakdowns and allowing more tasks to be "
            "completed under the judging rules. Each mentor team has its own staff "
            "and equipment. A change in one team's hours leaves the other teams' "
            "workshops and school resources fixed."
        ),
        "background_paraphrase": (
            "The district can send either Elm or Ash to the robotics finals. Yew "
            "seeks a place through a different district. School teams use their own "
            "equipment, and trials reward both a reliable robot and successfully "
            "finished tasks."
        ),
        "common_paraphrase": (
            "Workshop participants learn to find and fix mechanical problems before "
            "their robots are judged. Longer instruction produces fewer trial "
            "breakdowns, which lets a robot finish more scored tasks. The Beacon "
            "mentors, Summit mentors, and Lantern mentors supply their own people "
            "and equipment to one school each. Revising one mentor schedule "
            "preserves the other workshops and the schools' other resources."
        ),
        "actors": ("Beacon", "Summit", "Lantern"),
        "roles": ("Elm", "Ash", "Yew"),
        "roles_paraphrase": ("Elm", "Ash", "Yew"),
        "assignment_format": "The {actor} mentors' workshops are assigned to {role} school.",
        "assignment_paraphrase": "{role} school receives robotics instruction from the {actor} mentors.",
        "names": {
            "Elm": "Linden", "Ash": "Beech", "Yew": "Fir",
            "Beacon": "Horizon", "Summit": "Crest", "Lantern": "Compass",
        },
        "repair_notes": [
            "Preserves the single-place competition while giving every school a named independent mentor team.",
            "The third school's district membership is common to every context.",
        ],
    },
    14: {
        "background": (
            "The Rowan trial enclosure contains saplings, root-feeding grubs, and a "
            "tray of fallen debris. A screen controls the entry of delivered beetles. "
            "Weather and disease also affect sapling survival and remain uncertain."
        ),
        "common": (
            "Beetle K, beetle L, and beetle M remain active in the enclosure during "
            "the month. More adults of a kind consume more of its assigned food. "
            "Root-feeding grubs damage sapling roots; damaged roots take up less "
            "water and reduce survival. Direct consumption also damages roots. "
            "Debris is held in a collection tray whose contents are removed after "
            "the trial. Beetles act only by consuming their assigned food, and "
            "debris processing leaves root conditions fixed. Responses remain "
            "directionally stable across the planned range; a revised delivery "
            "leaves the other beetle populations, weather, and disease fixed."
        ),
        "background_paraphrase": (
            "Saplings and the grubs that eat their roots occupy the Rowan enclosure. "
            "Fallen debris is kept there in a collection tray, and deliveries of "
            "beetles enter through a controlled screen. The eventual survival count "
            "also depends on uncertain disease and weather."
        ),
        "common_paraphrase": (
            "Root injury impairs water uptake and therefore sapling survival. It "
            "can come from root-feeding grubs or from beetles feeding on roots "
            "directly. Beetle K, beetle L, and beetle M stay active for the trial, "
            "and larger populations eat more of the food listed for their kind. "
            "The debris tray is emptied after the trial; eating its contents "
            "preserves conditions around the roots. Each beetle kind affects the "
            "enclosure solely through its listed feeding, with a stable direction "
            "over the scheduled range. Changing one delivery preserves the other "
            "populations and the weather and disease conditions."
        ),
        "actors": ("Beetle K", "Beetle L", "Beetle M"),
        "roles": ("root-feeding grubs", "living sapling roots", "fallen debris in the collection tray"),
        "roles_paraphrase": ("the grubs that attack sapling roots", "the saplings' living roots", "the collection tray's fallen debris"),
        "assignment_format": "{actor} eats {role}.",
        "assignment_paraphrase": "The food consumed by {actor} is {role}.",
        "names": {
            "Rowan": "Alderwood", "Beetle K": "Beetle V", "beetle K": "beetle V",
            "Beetle L": "Beetle W", "beetle L": "beetle W",
            "Beetle M": "Beetle X", "beetle M": "beetle X",
        },
        "repair_notes": [
            "Specifies where debris and its products remain, closing decomposition-to-nutrients spillovers.",
            "Every schedule contains predator, root-feeder, and debris-feeder roles with fixed other populations.",
        ],
    },
    16: {
        "background": (
            "Fern's receiver obtains a direct radio signal on its operating frequency. "
            "A second frequency is recorded by a monitoring receiver. Fading of the "
            "direct signal remains uncertain. Throughout the operating range, the "
            "combined amplitude of all opposite-phase relay copies at Fern stays "
            "below the direct signal's amplitude."
        ),
        "common": (
            "Station K, Station L, and Station M relay copies of the direct waveform. "
            "For copies on Fern's operating frequency, phase is defined at the receiver "
            "relative to the arriving direct signal, and these phases remain stable. Matching-phase "
            "copies strengthen the combined signal and opposite-phase copies weaken "
            "it. A stronger combined signal prevents interruptions during fades. "
            "The monitoring frequency goes through a separate receiver and capacity "
            "allocation. Longer schedules extend the same station's active periods; "
            "the other station schedules, received amplitudes, and waveform remain fixed."
        ),
        "background_paraphrase": (
            "The Fern link listens to its direct signal on one frequency, while a "
            "monitor logs a second frequency. The direct signal may fade. Even "
            "during those fades, the total opposite-phase relay amplitude received "
            "at Fern remains smaller than the direct amplitude."
        ),
        "common_paraphrase": (
            "For relay copies on the direct wave's frequency, compare their phases "
            "when they arrive at Fern to decide whether they add or subtract. Station K, Station L, "
            "and Station M keep these relative phases constant. Matching copies "
            "on the link's frequency increase its combined signal; opposite copies "
            "decrease it. Higher combined strength helps the link stay available "
            "through fades. Copies on the monitor's frequency use that receiver's "
            "own capacity. Additional scheduled minutes continue a station's "
            "existing transmissions, preserving the other timetables, amplitudes, "
            "and the transmitted waveform."
        ),
        "actors": ("Station K", "Station L", "Station M"),
        "roles": (
            "matching-phase copies on Fern's operating frequency",
            "opposite-phase copies on Fern's operating frequency",
            "copies on the monitoring frequency",
        ),
        "roles_paraphrase": (
            "copies aligned with the direct wave on Fern's operating frequency",
            "copies opposed to the direct wave on Fern's operating frequency",
            "copies delivered on the monitor's frequency",
        ),
        "assignment_format": "{actor} transmits {role}.",
        "assignment_paraphrase": "The relay output of {actor} consists of {role}.",
        "names": {
            "Fern": "Moss", "Station K": "Station V",
            "Station L": "Station W", "Station M": "Station X",
        },
        "repair_notes": [
            "Defines phase at the receiver rather than at the transmitter.",
            "Bounds total destructive amplitude, preventing a sign reversal after complete cancellation.",
            "All contexts include both phases and both frequency destinations.",
            "The monitoring receiver has its own capacity; longer schedules extend existing periods.",
        ],
    },
    17: {
        "background": (
            "Wren Books and Lark Books sell rival autumn titles to one local "
            "reading-club audience. Each member buys exactly one of those two titles. "
            "Ibis Books serves a reading club in another region. The clubs have "
            "their own members, and readers' final choices remain uncertain."
        ),
        "common": (
            "Harbour Radio, Meadow Radio, and Beacon Radio each advertise one "
            "bookshop's title to that shop's reading-club region. Greater reach "
            "raises awareness, brings more members to demonstrations, and converts "
            "some visitors into buyers. For the local club, a purchase of one "
            "title replaces a purchase of its rival. Each station has its own "
            "advertising budget; changing its reach leaves the other packages, "
            "club memberships, and one-title purchase rule fixed."
        ),
        "background_paraphrase": (
            "Every member of the local reading club will buy one autumn book, "
            "choosing between the Wren Books title and the Lark Books title. "
            "Members of a club in a different region shop with Ibis Books. "
            "The two clubs have separate memberships, and their readers have "
            "yet to settle their choices."
        ),
        "common_paraphrase": (
            "Hearing a title advertised makes readers more likely to attend its "
            "shop demonstration, where some decide to buy it. Reaching additional "
            "club members therefore increases purchases of the advertised title. "
            "Within the local club, these purchases take the place of the other "
            "title because each member buys one. Harbour Radio, Meadow Radio, "
            "and Beacon Radio fund their own packages for the assigned shops' "
            "regions. Revising one package preserves the others, the membership "
            "lists, and the number of titles each member buys."
        ),
        "actors": ("Harbour Radio", "Meadow Radio", "Beacon Radio"),
        "roles": ("Wren Books", "Lark Books", "Ibis Books"),
        "roles_paraphrase": ("Wren Books", "Lark Books", "Ibis Books"),
        "assignment_format": "{actor}'s autumn package advertises the {role} title.",
        "assignment_paraphrase": "The autumn title from {role} is promoted by {actor}.",
        "names": {
            "Wren Books": "Robin Books", "Lark Books": "Swift Books",
            "Ibis Books": "Crane Books", "Harbour Radio": "Hillside Radio",
            "Meadow Radio": "Valley Radio", "Beacon Radio": "Lantern Radio",
        },
        "repair_notes": [
            "States that local readers buy exactly one of the two target titles, securing the rival-sales direction.",
            "Every station targets its recipient's region; membership and other advertising packages are held fixed.",
        ],
    },
    19: {
        "background": (
            "Aster grows a single flower variety whose unopened buds are attacked "
            "by mite M. Carrier strips introduce organisms to the greenhouse. "
            "Used carrier material is collected in trays. Temperature and other "
            "growing conditions remain uncertain."
        ),
        "common": (
            "Organism K, organism L, and organism N each arrive on their own "
            "carrier strips. More strips establish more organisms of that kind, "
            "which consume more of their assigned food. Damage to unopened buds "
            "prevents marketable blooms; eating bud-attacking mites reduces that "
            "damage. Collected carrier material remains in its trays until removal "
            "after the growing period. Each organism acts only through its assigned "
            "feeding, and tray contents leave growing conditions fixed. Effects "
            "retain their direction throughout the stated range. Revising one "
            "strip allocation leaves the other organisms and greenhouse resources fixed."
        ),
        "background_paraphrase": (
            "The flower crop at Aster consists of one variety. Mite M damages its "
            "buds before they open. Organisms are delivered on strips, and the "
            "spent strip material goes into collection trays. Growing conditions, "
            "including temperature, are still variable."
        ),
        "common_paraphrase": (
            "A damaged bud cannot become a marketable flower, so consuming mites "
            "that attack buds protects the crop. Consuming buds directly damages "
            "it. Organism K, organism L, and organism N enter on separate strips; "
            "a larger strip delivery establishes more feeders and increases "
            "consumption of the food assigned to that kind. Spent carrier stays "
            "in collection trays until the growing period ends, preserving the "
            "plants' conditions. Feeding is each organism's only effect, with "
            "the same directional response over the planned range. Changing "
            "one delivery preserves the other populations and greenhouse resources."
        ),
        "actors": ("Organism K", "Organism L", "Organism N"),
        "roles": ("bud-attacking mite M", "unopened flower buds", "used carrier material in the collection trays"),
        "roles_paraphrase": ("mite M, which attacks unopened buds", "the flowers' unopened buds", "the collection trays' spent carrier material"),
        "assignment_format": "{actor} feeds on {role}.",
        "assignment_paraphrase": "The food consumed by {actor} is {role}.",
        "names": {
            "Aster": "Zinnia", "mite M": "mite Q", "Mite M": "Mite Q",
            "Organism K": "Organism V", "organism K": "organism V",
            "Organism L": "Organism W", "organism L": "organism W",
            "Organism N": "Organism X", "organism N": "organism X",
            "K-bearing": "V-bearing",
        },
        "repair_notes": [
            "Puts mite, bud, and carrier feeding into every schedule using identical relation syntax.",
            "Collection trays close nutrient and resource effects from consumption of discarded carrier material.",
            "Distinct organism and mite labels keep a name-only transformation consistent with K-bearing strips.",
        ],
    },
}
