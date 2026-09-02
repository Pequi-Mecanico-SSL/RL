# Notes: making the current branch (HEAD, `feat/reward-s1-collision-clustering`) learn again

Worktree: `bisect/wt/fix-relearn`, branch `fix/relearn-and-offenses` (new branch
off HEAD `7668dba`, not merged/pushed anywhere). Goal 1: cross score >= 0.6
like tag `1.0` did. Goal 2 (after goal 1 confirmed): re-enable offense
penalties (double touch, team/opponent defense area) without regressing
learning.

## Goal 1 changes -- bug fixes (not judgment calls)

1. **`src/utils/decorators.py`** -- `decorator_observations`'s `main_robots`/
   `adv_robots` were built from a hardcoded `if "blue" in name` /
   `if "yellow" in name` check, regardless of which color was actually being
   computed (`color_main`). Result: a yellow agent was hosted blue's own
   robots as "itself." Fixed to filter by `color_main`/`color_adv` (the
   variables the loop already computes).

2. **`src/utils/geometry.py`** -- `Geometry2D._invert_coordinates` mutated its
   input dict in place and returned the same object. Since `raw_observations`
   is one shared dict reused across all 6 observation-computing functions
   (and, within `decorator_observations`, across all 6 agents), every call
   compounded the mirror transform on top of whatever the previous call left
   behind -- not a clean "mirror once," but "mirror an uncontrolled number of
   times depending on call order." Fixed to operate on a copy.

3. **`src/observations.py`** -- `oritations_observations` computed the
   normalized-theta feature as `deg2rad(theta)/(2*pi)` directly. For a
   negative raw `theta`, this lands a full period (1.0) away from the same
   physical angle's `[0, 360)` representation, so yellow's (internally
   mirrored-to-[0,360)) theta and blue's raw theta disagreed even when they
   represented the same relative orientation -- `sin`/`cos` were unaffected
   (periodic), only the raw `theta` feature. Fixed to derive it via
   `arctan2(sin, cos)/pi` (tag `1.0`'s own convention for this), which is
   periodic-safe by construction.

   Verified all three with `bisect/probe_obs.py bisect/wt/fix-relearn` --
   purity, mirror-symmetry, and identity all pass (exit 0) after these three
   fixes; all three failed before.

4. **`src/rewards.py`, `r_dist`** -- normalized robot-to-ball distance by
   `norm([field.length, field.goal_width])` instead of
   `norm([field.length, field.width])` -- reuses a `Geometry2D` instance
   originally meant for goal-angle geometry (where `goal_width` is the right
   bound) for a general distance calc, where it isn't. Confirmed via a direct
   diff against tag `1.0`'s own inline `_get_dist_between`, which used
   `field.width`. Fixed to use `field.width`.

## Goal 1 changes -- judgment calls (reverting recent, unproven weight changes)

5. **`src/rewards.py`, `DENSE_REWARDS`** -- `r_speed` weight was `0.3` (down
   from tag `1.0`'s `0.7`) and `r_dist` was `0.5` (up from `1.0`'s `0.1`) --
   i.e. "stay near the ball" was weighted *more* than "move the ball toward
   the goal," inverted from `1.0`'s design. This looked like unproven
   experimentation from this branch's own WIP (`feat/reward-s1-collision-
   clustering`) rather than a deliberate, validated change. Reverted both to
   `1.0`'s proven values (`0.7` / `0.1`).

6. **`src/rewards.py`, `SPARSE_REWARDS["OUTSIDE_REWARD"]`** -- was `0` (no
   penalty for kicking the ball out of bounds), tag `1.0` used `-10`.
   Restored to `-10`.

   `end_on_offense: false` in `config.yaml` was already correct (this is what
   prevents the catastrophic whole-episode-ending-on-a-minor-foul bug found
   during the bisect, `bc53b97`) -- left untouched.

## Goal 2 (not yet applied -- do after goal 1's training run confirms it crosses 0.6)

Set `SPARSE_REWARDS["OPPONENT_DEFENSE_AREA"]`, `["TEAM_DEFENSE_AREA"]`,
`["DOUBLE_TOUCH"]` to modest non-zero penalties (matching `v1.1.0`'s own
`-1` convention, which is empirically known to be compatible with learning --
see `BISECT_REPORT.md`'s ablation section) while keeping `end_on_offense:
false`. `src/simulators/rsoccer.py`'s offense-handling loop already only
*adds* the penalty (`reward_agents[robot_name] += ...`), it does not zero out
the robot's other per-step reward first (unlike `v1.1.0`'s equivalent code),
so this should be even safer than what `v1.1.0` already used successfully.
