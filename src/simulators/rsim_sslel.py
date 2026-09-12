"""RSimSSLEL: the same RSimSSL wrapper rsoccer_gym provides, but backed by the
team's own robosim.SSLEL simulator class instead of the generic robosim.SSL.

Why a separate class instead of patching rsoccer_gym.Simulators.rsim.RSimSSL
directly: robosim.SSLEL has its OWN field_type numbering, distinct from
robosim.SSL's. field_type=0 on SSLEL is the genuine, small SSL-EL field
(4.5m x 3.0m, 3v3) this project actually plays on; field_type=1/2 on SSLEL
are 6v6 configs (Division B / Hardware Challenge scaled up), not compatible
with this project's fixed 3v3 setup. robosim.SSL's field_type=0/1/2 mean
something else again (12x9m / 9x6m / 6x4m, all nominally 3v3-capable). Same
numbers, three different meanings depending on which simulator class -- so
swapping the binding for every field_type globally (patching RSimSSL itself)
would silently change what an *existing* `field_type: 1` in config.yaml
means. This class is only used when a caller explicitly opts in (see
SSLMultiAgentEnv's use_sslel flag in rsoccer.py), and validates field_type
itself so a caller can't quietly get an incompatible robot-count config.

Verified byte-compatible with RSimSSL's existing state/action parsing before
writing this (see RELEARN_NOTES.md): SSLELWorld::getState()/setActions()/
getFieldParams()/replace() are identical in layout to SSLWorld's -- only the
Config class (and therefore field dimensions) differs. So RSimSSL's own
send_commands/get_frame are reused unchanged; only the simulator constructor
call itself differs.
"""
import robosim

from rsoccer_gym.Simulators.rsim import RSimSSL

# robosim.SSLEL's own field_type values (see rSim's src/robosim/sslelconfig.h,
# SSLELConfig::Field::setFieldType) -- NOT the same meanings as robosim.SSL's.
SSLEL_FIELD_TYPE_3V3 = 0  # 4.5m x 3.0m, 3v3 -- the only one this project's fixed 3v3 setup supports


class RSimSSLEL(RSimSSL):
    def __init__(self, field_type: int, n_robots_blue: int, n_robots_yellow: int, time_step_ms: int):
        if field_type != SSLEL_FIELD_TYPE_3V3:
            raise ValueError(
                f"RSimSSLEL only supports field_type={SSLEL_FIELD_TYPE_3V3} (robosim.SSLEL's "
                f"4.5x3.0m 3v3 config) -- got field_type={field_type}. robosim.SSLEL's other "
                f"field_type values (1, 2) are 6v6 configs, incompatible with this project's "
                f"fixed 3v3 setup. This is a different numbering than robosim.SSL's field_type."
            )
        super().__init__(
            field_type=field_type,
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            time_step_ms=time_step_ms,
        )

    def _init_simulator(self, field_type, n_robots_blue, n_robots_yellow,
                         ball_pos, blue_robots_pos, yellow_robots_pos,
                         time_step_ms):
        return robosim.SSLEL(
            field_type,
            n_robots_blue,
            n_robots_yellow,
            time_step_ms,
            ball_pos,
            blue_robots_pos,
            yellow_robots_pos,
        )
