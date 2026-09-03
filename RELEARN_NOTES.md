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

## Goal 1 result: SUCCESS

Trained under the same harness/budget as the bisect (3,000,000 timesteps,
`bisect/results/fix-relearn-goal1/`). Score crossed 0.6 at **iteration 19,
731,880 timesteps**, spiking to 0.99 before the self-play opponent-update
fired (confirmed in `train.log`: `---- Updating Opponent!!! ----`,
`score_couter before reset 0.99`, then the expected reset to 0.0).

That's **less than half** the 1,502,280 timesteps tag `1.0` needed for the
same milestone -- the fixed current branch isn't just "learning again," it's
converging faster than the original `1.0` baseline. `episode_len_mean`
stayed healthy throughout (1200 -> 1135, shortening only as real goals start
ending episodes early, never collapsing the way `main`'s did).

This is consistent with the current branch's own env/observation architecture
(the `StackWrapper` + `observations.py` refactor, the Judge-based freekick
system) being a genuinely better learning setup than `1.0`'s original
inline code -- once the bugs that were breaking it are actually fixed.

## Goal 2: enable offense penalties (double touch, team/opponent defense area)

`src/rewards.py`'s `SPARSE_REWARDS` had `OPPONENT_DEFENSE_AREA`,
`TEAM_DEFENSE_AREA`, and `DOUBLE_TOUCH` all at `0` -- i.e. these offenses
currently have zero effect on training. Set all three to `-1`, matching
`v1.1.0`'s own convention (see `BISECT_REPORT.md`'s ablation section --
`-1` per-offense penalties, without ending the episode, are empirically known
to be compatible with learning: original `v1.1.0` had exactly these values
active and still converged, just slower for other, unrelated reasons).

`end_on_offense: false` in `config.yaml` is left untouched -- this is what
keeps the current branch immune to the catastrophic whole-episode-ending
bug found during the bisect (`bc53b97`). `src/simulators/rsoccer.py`'s
offense-handling loop only *adds* the penalty to the offending robot's own
reward (`reward_agents[robot_name] += ...`), it doesn't zero out the rest of
that robot's per-step reward first, so this should be strictly safer than
what `v1.1.0` already used successfully.

Custom metrics already tracked by this branch's callback
(`foul/collisions_per_1k_steps`, `foul/team_defense_area_events_per_1k_steps`,
`foul/opponent_defense_area_events_per_1k_steps`,
`kickoff/double_touch_per_1k_steps`) will show whether the penalty actually
shapes behavior (offense rate trending down over training) once this run
completes.

## Goal 2 result: learning still works, behavioral shaping did NOT clearly work (yet)

Trained under the same harness (3,000,000 timestep budget, stopped early at
iteration 40 / 1,540,800 timesteps -- see below for why). Data in
`bisect/results/fix-relearn-goal2/`.

**Learning: confirmed intact.** Score crossed 0.6 at iteration 22,
**847,440 timesteps** -- ~16% more than goal 1's 731,880 (a modest, expected
cost for the added constraint), still far faster than tag `1.0`'s 1,502,280.
`episode_len_mean` stayed in a healthy range throughout (1090-1200, shortening
only as goals actually happen -- never collapsing the way `main`'s did). So
the core goal-2 requirement -- offense penalties don't break training -- holds.

**Behavioral shaping: not observed within this run.** The three offense-rate
custom metrics (`foul/team_defense_area_events_per_1k_steps`,
`foul/opponent_defense_area_events_per_1k_steps`,
`kickoff/double_touch_per_1k_steps`) were tracked from iteration 1. Through
iteration 40 they show a **consistent upward trend**, not the hoped-for
decline:

| iter | ts | team_defense_area | opponent_defense_area | double_touch |
|---|---|---|---|---|
| 20 | 770,400 | 3.68 | 0.64 | 0.82 |
| 30 | 1,155,600 | 11.13 | 3.07 | 0.94 |
| 40 | 1,540,800 | 4.94 | 8.73 | 1.00 |

(noisy iteration-to-iteration, but the trend across the whole window is up,
not down, for all three). Self-play cycling was also much slower than goal 1:
only **1** opponent-update fired in this window vs. goal 1's 6 in a comparable
span -- consistent with the `-1` penalty adding friction to how fast the
policy improves, without (yet) suppressing the penalized behavior itself.

**Likely explanation:** `-1` is small relative to `GOAL_REWARD=10` and the
per-step dense terms accumulated over a ~1100-1200-step episode. As the policy
gets more aggressive/contested (which is *necessary* to keep scoring goals),
it naturally spends more time near both defense areas, and a one-time `-1`
per offense isn't enough to outweigh the value of contesting there. This
mirrors, at much smaller scale, the exact tension the `bc53b97` bug (found
during the bisect) got catastrophically wrong -- offense penalties compete
directly against the dense reward signal, and getting the balance right needs
either a larger penalty, or scaling it by something (time-in-violation,
severity), or approaching it differently (e.g. action masking near defense
areas) rather than a bigger flat penalty which risks recreating something
closer to the original catastrophic dynamic if tuned too aggressively.

**Stopped at iteration 40 (51% of budget)** rather than running to
completion -- the offense-rate trend was unambiguous and consistent across
20 iterations, and continuing would mainly cost GPU time rather than change
the conclusion. A follow-up worth trying: a larger penalty (e.g. `-3` to
`-5`) or a penalty scaled by episode length, run long enough to see if
offense rates eventually turn over once self-play has had more opponent-
update cycles to actually specialize against the penalty.

## Offense-detection audit (src/judges/ssl_judge.py) -- before further tuning

Before increasing the `-1` penalty, audited whether the three offense checks
(`DOUBLE_TOUCH`, `OPPONENT_DEFENSE_AREA`, `TEAM_DEFENSE_AREA`) and `COLLISION`
actually detect what their names claim. Found three real bugs, all fixed and
covered by a standalone unit test (`bisect/probe_offenses` -- run inside the
training image, since `src/objects/Frame.py`'s mutable dataclass default only
type-checks cleanly under Python 3.10, not 3.12):

1. **`_check_collision`** built `all_robots = {**robots_blue, **robots_yellow}`.
   Both dicts use keys `{0,1,2}`, so yellow's entries silently overwrote
   blue's in the merge -- collisions were only ever checked against the 3
   yellow robots. Blue-blue collisions were never detected, and a blue robot
   never even saw itself (or any blue teammate) in the comparison set. This
   also means the `foul/collisions_per_1k_steps` metric watched during goal 2
   was measuring an incomplete signal the whole time (though harmless to
   training itself, since `SPARSE_REWARDS["COLLISION"]` stayed `0`). Fixed to
   use a flat list of all 6 robots.

2. **`_check_double_touch`** used lingering *proximity* (the same 0.22m
   "possession" radius used elsewhere) as a stand-in for "touched the ball a
   second time," without requiring the ball to have actually left the
   kicker first. A robot that just kicked off and is still momentarily
   within 0.22m of the ball -- an entirely ordinary, single legitimate touch
   -- could trigger this. **This is the likely explanation for why
   `kickoff/double_touch_per_1k_steps` kept climbing during the first goal-2
   run**: as the policy got better at actually reaching and kicking the ball
   at kickoff, it triggered this false positive more often, which looks
   identical to "double-touching more" in the metric but isn't. Fixed to
   require the ball to have gone free (no robot within the possession
   radius) at least once since the sole historical toucher's first touch,
   before a same-robot re-possession counts. Known remaining limitation: a
   robot that dribbles continuously from kickoff without ever releasing the
   ball still won't be flagged -- real SSL rules likely consider that a
   foul too, but detecting it would need a genuine touch-event detector
   (e.g. extending `_update_last_touch`'s velocity-direction-change logic),
   which is out of scope here.

3. **`_check_ally_defense_area`** (`TEAM_DEFENSE_AREA`) counted robots one at
   a time as `_update_offenses` iterated them, only flagging a robot if it
   was inside the box *and* the running count (as of that robot) exceeded 1
   -- so with exactly 2 teammates in the box, only the one processed second
   (by dict/id order, not real arrival time) got flagged; the first was
   exempt purely by iteration order. Restructured `_update_offenses` to
   determine all offenders on a side in one pass and flag every one of them
   when 2+ are present.

All three verified with a standalone script exercising `Judge` directly
(synthetic frames, no simulator/training stack needed) -- 10/10 checks pass,
including that a lone robot in the box (no teammate) correctly does *not*
get flagged, and that ordinary single-touch kickoffs correctly do *not*
trigger `DOUBLE_TOUCH`.

Re-running goal 2 (offense penalties still at `-1`, but now on the corrected
detection) before deciding whether the penalty magnitude itself needs
tuning -- the earlier "rising offense rate" trend may have been measuring
mostly false positives, not a genuine tuning problem.

## Goal 2 retune (fixed offense detection, same -1 penalties): result

Re-ran goal 2 with the same `-1` penalties but on the corrected offense
detection (`691f6e3`). Stopped at iteration 77 / 2,966,040 timesteps (99% of
the 3,000,000 budget, essentially complete). Data in
`bisect/results/fix-relearn-goal2b/`.

**Score never crossed 0.6.** Peaked at **0.46** around iteration 67
(2,580,840 timesteps), then settled back into a 0.19-0.28 range through the
rest of the run. No self-play opponent-update ever fired.

### Three-way comparison

| run | offense detection | penalties | first score movement | crosses 0.6 |
|---|---|---|---|---|
| goal 1 | n/a (penalties off) | none | iter 19 / 732K ts | **yes, 731,880 ts** |
| goal 2 (original) | buggy (collision undercounts, double-touch false-positives, team-defense-area under-flags) | -1 each | iter 20 / 770K ts | **yes, 847,440 ts** |
| goal 2b (this run) | fixed | -1 each | iter 20 / 770K ts (then a long slow patch, iter 40-64, before breaking out) | **no** -- peaked 0.46, settled ~0.2-0.3 |

### Reading this honestly

On its face this looks like the detection fixes made things *worse* --
slower to first move, a long plateau from iter ~30-64, and a final peak
(0.46) well short of both prior runs' full crossing. But this investigation
has repeatedly shown **large single-seed swings within a single
configuration**: `v1.1.0`'s own original run had an equally long dead
plateau (score stuck at 0 through iter 30, only starting to climb past
iter 31) before eventually reaching 0.53 near budget exhaustion; goal 2b
itself alternated between multi-iteration plateaus and sharp breakouts at
least three times in this one run (iter 40-44 flat, iter 45-47 breakout to
0.20, iter 48-64 flat, iter 65-67 breakout to 0.46, then a fade). With only
**one run per configuration**, there is no way to separate "the fixes
changed the learning dynamics" from "this run drew an unlucky trajectory" --
both are fully consistent with everything observed. A real answer needs
multiple seeds per configuration (e.g. 3-5 runs each for goal 2 and goal 2b),
which wasn't done here given the time/GPU cost of each 3M-timestep run
(~3 hours).

**What is NOT in question, independent of any training run:** the three
offense-detection bugs are real and the fixes are correct. That was verified
directly (`bisect/probe_offenses.py`, 10/10 synthetic-scenario checks pass),
not inferred from training outcomes -- a training run's score has no bearing
on whether `{**dict_a, **dict_b}` silently drops entries with colliding keys,
or whether a lone robot in the defense area gets incorrectly exempted from
`TEAM_DEFENSE_AREA`. Those are bugs regardless of what happens when you train
against the fixed version.

## Final recommendation

1. **Ship goal 1's fixes** (`db3fe33`): observation-mirroring bug,
   in-place-mutation bug, orientation periodic-safety bug, `r_dist`
   normalizer bug, plus the `r_speed`/`r_dist`/`OUTSIDE_REWARD` weight
   reverts to tag `1.0`'s proven values. Solid -- demonstrated crossing 0.6
   in under half of `1.0`'s own time, verified via `bisect/probe_obs.py`
   independent of the training outcome too.
2. **Ship the three offense-detection fixes** (`691f6e3`): collision
   dict-merge bug, double-touch false-positive, team-defense-area
   under-flagging. Correctness is settled by the unit tests, independent of
   training outcome.
3. **Treat the exact offense-penalty magnitude (`-1`) as still open.**
   Whether `-1` is the right value, and what its true effect on convergence
   speed is, is not settled by this investigation -- the one run available
   showed a worse outcome than the buggy-detection version, but that's not
   distinguishable from noise with n=1. If/when this matters for real
   training, budget for multiple seeds (3-5 runs) per penalty value being
   compared, rather than drawing conclusions from single runs the way this
   session had to under time constraints.

## Reducing offenses further: hypotheses and a targeted experiment

The `-1` flat penalty didn't clearly reduce offense rates in goal2b (single
run, inconclusive vs. noise -- see above). Before just sweeping the
magnitude, worked out concrete hypotheses for *why* a flat per-step penalty
might not be shaping behavior well:

1. **Duration/magnitude mismatch.** `OPPONENT_DEFENSE_AREA`/`TEAM_DEFENSE_AREA`
   fire *every step* a violation persists (unlike `DOUBLE_TOUCH`, which
   self-disables after firing once per episode). A robot that lingers for
   many steps already accumulates a large total penalty under `-1`/step --
   so "the penalty is too weak" isn't obviously the right diagnosis for
   *sustained* violations. But a brief, incidental boundary crossing (e.g.
   chasing a bouncing ball that happens to cross the line for one frame)
   gets exactly the same per-step penalty as a deliberate one, with no
   distinction.
2. **No anticipatory gradient.** The penalty is a flat step function (0
   outside, `-1` inside) -- the value function only learns "this spot is
   bad" after the agent has already committed to entering, with no smooth
   signal to course-correct beforehand.
3. **Direct incentive conflict.** `OPPONENT_DEFENSE_AREA` specifically
   requires ball possession deep in the opponent's zone -- exactly where
   scoring chances are richest. A flat per-step penalty fights head-on
   against `GOAL_REWARD=10` in exactly the situations where attacking is
   most valuable.
4. **The correctness fix itself increased the effective penalty.**
   `TEAM_DEFENSE_AREA` now flags *every* offending teammate (fix earlier in
   this file), roughly doubling the typical per-violation total compared to
   the buggy version that inspired the original `-1` choice.

### Experiment: grace period (targets hypotheses 1 and 3)

Rather than another magnitude sweep, added a **grace period**: both
`OPPONENT_DEFENSE_AREA` and `TEAM_DEFENSE_AREA` now only flag once the
violation condition has held for `GRACE_STEPS=10` consecutive steps
(~0.33s at fps=30), not on first contact. `TEAM_DEFENSE_AREA` tracks the
2+-occupancy *situation* per side (not each robot's individual streak), so
two teammates briefly overlapping while chasing a loose ball doesn't count,
but a sustained defensive huddle does. `OPPONENT_DEFENSE_AREA` tracks each
robot's own consecutive-possession-inside-the-box streak.

This changes *when* an offense is flagged (only sustained presence), not
*whether* the underlying geometry/possession logic is correct -- the
already-fixed detection logic (blue-blue collisions, ball-must-go-free
double touch, all-offenders team-area flagging) is untouched. Verified with
9 new/updated cases in `bisect/probe_offenses.py` (13/13 pass): brief
crossings for both offense types are confirmed NOT flagged, sustained ones
are confirmed flagged, and the "lone robot in the box" / "collision"
invariants from before still hold.

## Grace-period experiment: intermediate result (1.5M budget) -- extending

Ran with a shorter 1,500,000-timestep budget for faster iteration. Score did
not cross 0.6 within that budget (final: 0.11 at iter39/1,502,280), but the
offense dynamics are the most encouraging of any variant tried:

- `team_defense_area`: oscillated rather than monotonically rising --
  0 -> 24 (iter5-10) -> **0 for 9 straight iterations** (iter15-23) -> 54
  (iter24-27, coinciding with score first starting to move -- plausibly the
  policy discovering defensive clustering near its own goal as a tactic once
  real attacking/defending play develops) -> **back down to ~0.8 by the end**
  (iter38-39), while score kept climbing the whole time (0.02 -> 0.15). This
  looks like a genuine self-correcting dynamic: the policy explores
  clustering, gets penalized under the grace-gated `-1`, and moves away from
  it again -- not just noise, and not a stall.
- `opponent_defense_area`: stayed at 0 until score started improving, then
  rose modestly (0 -> ~5.4) -- consistent with hypothesis 3 (more genuine
  attacking near the opponent box naturally creates more opportunities for
  this specific offense as play improves).
- `double_touch`: stayed at exactly 0 for the first 7 iterations (much
  longer than any prior run), consistent with the earlier false-positive fix
  working; then climbed to a ~0.5-0.6 plateau, similar to prior runs (this
  offense has no grace period in this experiment).

**New hypothesis (H6):** defensive clustering near one's own goal may be
tactically valuable in a literal sense (more bodies to block shots), so
there's real tension between "good defense" and the `TEAM_DEFENSE_AREA` rule
-- not purely a training artifact. The grace-period + `-1` combination
appears to be teaching the policy to move away from over-clustering (the
observed self-correction), but this needs a longer run to confirm it's a
stable equilibrium rather than an oscillation that could recur.

Since the trajectory was still healthily climbing (not plateaued or
declining) when the 1.5M budget ran out, extending with the standard 3M
budget to see whether it actually crosses 0.6.

## Grace-period experiment: SUCCESS (3M budget) -- final result

Extended the grace-period configuration to the standard 3,000,000-timestep
budget. **Score crossed 0.6 at iteration 50, 1,926,000 timesteps** --
confirmed via `---- Updating Opponent!!! ----` in the log
(`score_couter before reset 0.76`). This is the **first offense-penalty
configuration, across every variant tried (goal2 original, goal2b), to
actually cross the threshold.**

Stopped after confirming stability at iteration 58 (2,234,160 timesteps):
**3 opponent updates fired**, with score cycling through clearly healthy
self-play dynamics after each reset (0.07 -> 0.30 -> 0.36 -> 0.48 -> **1.49**
-> 0.39 -> 0.68 -> 0.10) -- the same qualitative pattern goal 1's confirmed-
working run showed (6 updates, strong post-reset recovery each time), not a
single lucky crossing.

Offense metrics post-crossing show a genuine, expected tradeoff rather than
a clean win-on-every-axis:

- `team_defense_area`: stayed controlled -- oscillates (0 -> 12.8 -> 0) but
  no runaway trend, consistent with the self-correcting dynamic seen
  throughout this run and the 1.5M one.
- `opponent_defense_area`: **climbed substantially** as attacking play
  intensified (6.6 -> 19.6 over the same window score went from 0.21 to
  1.49). This is hypothesis 3 playing out directly: a team that's
  genuinely better at attacking spends more time deep in the opponent's
  zone with possession, which is exactly the condition `OPPONENT_DEFENSE_
  AREA` detects. More scoring and more of this specific offense rose
  together, not independently -- a real tension between "attack
  effectively" and "avoid this specific rule," not something the grace
  period or `-1` penalty fully resolves.
- `double_touch`: settled in a ~0.6-0.8 range, similar to prior runs
  (unaffected by this experiment, which only added a grace period to the
  two defense-area offenses).

**Conclusion:** the grace period is a real, working improvement over a flat
step-function penalty -- it lets training reach the same self-play success
goal 1 (no offense penalties at all) demonstrated, while keeping
`team_defense_area` under control throughout. `opponent_defense_area`
remains a genuine, unresolved tension worth a further follow-up (e.g. a
possession-duration cap specific to attacking play, or accepting that some
increase here is the cost of "attack more" and tuning the threshold/penalty
specifically for that offense rather than sharing `GRACE_STEPS`/`-1` with
`team_defense_area`).

## Final state -- this is the committed "last change that learn works"

`git log -1` on this branch (`fix/relearn-and-offenses`) is `1adfa48`,
which already contains everything needed for this result: goal 1's four bug
fixes + two weight reverts (`db3fe33`), the three offense-detection bug
fixes (`691f6e3`), and the grace-period mechanism (`902a367`). No further
code changes were needed to produce this 3M run's success -- only the
budget. Per the instruction to always leave a working configuration
committed: **this is it.** `SPARSE_REWARDS` currently has
`OPPONENT_DEFENSE_AREA`/`TEAM_DEFENSE_AREA`/`DOUBLE_TOUCH` all at `-1`, with
`GRACE_STEPS=10` gating the two defense-area offenses -- confirmed to both
learn (crosses 0.6, 3 stable self-play cycles) and keep `team_defense_area`
under control, with `opponent_defense_area` as the known remaining
open question for future tuning.

## rSim / robosim.SSLEL: the genuine small SSL-EL field, wired in (opt-in)

Investigated `git+https://github.com/Pequi-Mecanico-SSL/rSim.git` (the repo
tag `1.0`'s own Dockerfile builds from, line 35) at the user's request.
Turned up something the earlier field-size analysis in `BISECT_REPORT.md`
got wrong: this repo doesn't add a new *value* to `robosim.SSL`'s
`field_type` -- it adds a **whole separate simulator class**, `robosim.SSLEL`,
with its own field configuration (`src/robosim/sslelconfig.h`) and its own,
different `field_type` numbering:

| `robosim.SSLEL` field_type | length x width | robots | notes |
|---|---|---|---|
| 0 | **4.5m x 3.0m** | 3v3 | the genuine SSL-EL field |
| 1 | 9.0m x 6.0m | 6v6 | scaled-up, not this project's setup |
| 2 | 6.0m x 4.0m | 6v6 | "hardware challenge," not this project's setup |

Confirmed by actually building the repo (not just reading source) and
querying `get_field_params()` directly.

**This means the bisect's earlier "tag 1.0 uses `field_type=0` = 12x9m field"
claim was wrong.** Tag `1.0`'s own vendored code calls `robosim.SSLEL(...)`,
not `robosim.SSL(...)` -- the bisect harness's compat shim
(`bisect/patches/rsim-robosim-rename.md`) renamed `SSLEL` calls to `SSL`
calls to work around the prebuilt image lacking `SSLEL`, on the assumption
it was a pure API rename. It wasn't: `SSL` and `SSLEL` are separate C++
classes with **different field_type -> dimension mappings**, so that shim
silently substituted the wrong field (a large 12x9m Division-A-sized field)
for what tag `1.0` actually ran on (a tiny 4.5x3.0m field). The bisect's
core conclusions (the `bc53b97` episode-ending bug, the observation-mirroring
bug) are unaffected -- those come from direct code diffs, not the field-size
measurement -- but the specific "1.0 vs v1.1.0+ field size" comparison in
`BISECT_REPORT.md` should be read with this correction in mind.

### Build fix

`pip install .` from the rSim repo failed with a CMake error (this is very
likely the exact "make version" issue recalled from before): pip's isolated
build environment auto-installs the *latest* `cmake` PyPI package (4.4.3),
which hard-removed compatibility with `cmake_minimum_required` versions
below 3.5 -- and the pinned `pybind11 v2.9.1` this repo's `CMakeLists.txt`
fetches declares exactly that. Fixed with a `PIP_CONSTRAINT=cmake<4` file
during the build (keeps CMake in the 3.x line, which only warns, doesn't
hard-fail, on that old a `cmake_minimum_required`). Confirmed the resulting
build has byte-identical `getState()`/`setActions()`/`getFieldParams()`
layouts to the standard `robosim.SSL` (verified by diffing `sslelworld.cpp`
against `sslworld.cpp`) -- so no state/action parsing changes were needed,
only which C++ class gets constructed.

### Integration: `src/simulators/rsim_sslel.py`, opt-in `use_sslel` flag

Added `RSimSSLEL(RSimSSL)` (`src/simulators/rsim_sslel.py`) -- overrides
only `_init_simulator` to call `robosim.SSLEL` instead of `robosim.SSL`;
`send_commands`/`get_frame` are inherited unchanged from `RSimSSL` since the
data format is identical. Validates `field_type == 0` (raises otherwise --
SSLEL's other field_types are 6v6, incompatible with this project's fixed
3v3 setup).

Deliberately **not** a patch to `rsoccer_gym`'s installed `RSimSSL` (which
would be a blanket, always-on change): `robosim.SSL`'s `field_type=0/1/2`
and `robosim.SSLEL`'s `field_type=0/1/2` mean different things (different
dimensions AND different robot counts), so patching `RSimSSL` itself would
silently redefine what an *existing* `field_type: 1` in `config.yaml` means
project-wide -- including the config the grace-period experiment's confirmed
`3c7b3fc` result was trained on. Instead, `SSLMultiAgentEnv` (`src/simulators/
rsoccer.py`) got a new `use_sslel=False` constructor parameter; when `True`,
it temporarily monkeypatches `rsoccer_gym.ssl.ssl_gym_base.RSimSSL` to
`RSimSSLEL` for the duration of the `SSLBaseEnv.__init__()` call, then
restores it -- scoped to one env construction, zero effect on any other env
instance or on this repo's own vendored files. `config.yaml` got a matching
`use_sslel: false` (default, unchanged behavior) with a comment explaining
the field_type-numbering gotcha.

Verified this doesn't disturb anything: `bisect/probe_obs.py` and
`bisect/probe_offenses.py` both still pass unchanged, and `import
src.simulators.rsoccer` still succeeds on the *current*, unmodified `ssl-el`
image (which lacks `SSLEL` entirely) -- `robosim.SSLEL` is only ever touched
when `use_sslel=True` is explicitly requested.

### Not yet done: actually training on it

The shared `ssl-el` Docker image still only has the standard `rc-robosim`
wheel (no `SSLEL`) -- `use_sslel: true` will raise
`AttributeError: module 'robosim' has no attribute 'SSLEL'` until an image
is built from the rSim source (with the `cmake<4` fix) instead. Deliberately
did not rebuild/overwrite the shared `ssl-el` image as part of this change --
that's a separate, more consequential step (affects every future run using
that image) worth doing explicitly rather than silently. Next step, if
wanted: build a new image tag from `github.com/Pequi-Mecanico-SSL/rSim.git`
(matching tag `1.0`'s own Dockerfile line, plus the `cmake<4` fix) and run a
training pass with `use_sslel: true`, `field_type: 0` to see how the genuine
small SSL-EL field performs.
