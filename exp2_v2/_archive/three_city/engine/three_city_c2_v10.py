"""Paper rerun of the natural continuous-profile world with a choice clue.

The numerical data-generating process is intentionally identical to v9. Only
the third prompt arm is changed, so the completed v9 pilot remains untouched.
"""

from engine.three_city_c2_v9 import *  # noqa: F401,F403

PROMPT_ARMS = ("c_only", "abc", "abc_structural_clue")
