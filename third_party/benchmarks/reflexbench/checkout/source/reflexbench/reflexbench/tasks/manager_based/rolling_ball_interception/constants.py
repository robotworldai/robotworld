"""Shared geometry constants for ball interception."""

# Shift the entire ramp setup in +X and +Z.
RAMP_X_SHIFT = 0.15
RAMP_Z_SHIFT = 0.0

# Ramp: 0.50 m long, 0.40 m wide, 0.02 m thick, tilted 25 deg around Y.
# High end (+X, away from robot) -> low end (-X, toward robot).
RAMP_TILT_DEG = 25.0
RAMP_LENGTH = 0.50
RAMP_WIDTH = 0.80
RAMP_THICKNESS = 0.02
RAMP_RAIL_THICKNESS = 0.01
RAMP_RAIL_HEIGHT = 0.03
RAMP_CENTER_POS = (0.85 + RAMP_X_SHIFT, 0.0, 0.40 + RAMP_Z_SHIFT)
RAMP_QUAT = (0.9763, 0.0, -0.2164, 0.0)
RAMP_SIZE = (RAMP_LENGTH, RAMP_WIDTH, RAMP_THICKNESS)
RAMP_RAIL_SIZE = (RAMP_LENGTH, RAMP_RAIL_THICKNESS, RAMP_RAIL_HEIGHT)
RAMP_RAIL_LEFT_POS = (
    0.85 + RAMP_X_SHIFT,
    -(0.5 * RAMP_WIDTH + 0.5 * RAMP_RAIL_THICKNESS),
    0.425 + RAMP_Z_SHIFT,
)
RAMP_RAIL_RIGHT_POS = (
    0.85 + RAMP_X_SHIFT,
    0.5 * RAMP_WIDTH + 0.5 * RAMP_RAIL_THICKNESS,
    0.425 + RAMP_Z_SHIFT,
)

# Ball start: 0.1 m above ramp high-end surface.
BALL_START_POS = (1.05 + RAMP_X_SHIFT, 0.0, 0.61 + RAMP_Z_SHIFT)

# White platform below the ramp. The "ramp exit" is defined as the minimum-X
# edge line of this platform, not its center.
RAMP_EXIT_ZONE_CENTER_POS = (0.62 + RAMP_X_SHIFT, 0.0, 0.285 + RAMP_Z_SHIFT)
# Y extent slightly wider than ramp width (same ratio as original 0.84/0.80).
RAMP_EXIT_ZONE_SIZE = (0.10, 1.05 * RAMP_WIDTH, 0.01)

RAMP_EXIT_LINE_X = RAMP_EXIT_ZONE_CENTER_POS[0] - 0.5 * RAMP_EXIT_ZONE_SIZE[0]
RAMP_EXIT_LINE_Z = RAMP_EXIT_ZONE_CENTER_POS[2] + 0.5 * RAMP_EXIT_ZONE_SIZE[2]
RAMP_EXIT_LINE_Y_MIN = -0.5 * RAMP_EXIT_ZONE_SIZE[1]
RAMP_EXIT_LINE_Y_MAX = 0.5 * RAMP_EXIT_ZONE_SIZE[1]

# Representative point on the exit line, with Y=0 as the line midpoint.
RAMP_EXIT_POS = (RAMP_EXIT_LINE_X, 0.0, RAMP_EXIT_LINE_Z)
