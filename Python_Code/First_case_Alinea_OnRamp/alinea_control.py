# alinea_control.py

# ALINEA parameters (tunable)
# Increased rho_crit slightly to allow a bit more mainline density before blocking
rho_crit = 0.05   # Critical density (cars/m) - Adjusted from 0.04
K = 100           # Control gain - Adjusted from 80 (slightly more aggressive)
r_prev = 1.0      # Initial control state


def compute_density(cars, cell_index, cell_length=1.0):
    """Estimate total density in a main road cell (all lanes, cars per meter)."""
    count = sum(
        1 for car in cars
        if car.get("lane") in [0, 1] and cell_index <= car["x"] < cell_index + cell_length
    )
    # The cell_length here is in abstract units (e.g., 1 unit per cell).
    # If a cell represents 100 meters, then:
    return count / (cell_length * 100)  # if 100m cell -> veh/m


def congestion_at_merge(cars, merge_start=2.0, merge_end=3.0, min_gap=0.3):
    """
    Detect congestion in the merge zone based on spacing between mainline cars.
    Returns True if cars are too close to allow safe merge.
    """
    lane_cars = sorted([
        car for car in cars if car.get("lane") in [0, 1] and merge_start <= car["x"] < merge_end
    ], key=lambda c: c["x"])

    for i in range(len(lane_cars) - 1):
        lead = lane_cars[i + 1]
        follower = lane_cars[i]
        if lead["x"] - follower["x"] < min_gap:
            return True

    return False


def alinea(r_prev, rho_measured, congestion=False):
    """
    ALINEA control logic with congestion awareness.

    - Uses standard feedback law
    - Blocks ramp if physical congestion is detected in merge zone
    """
    r_raw = r_prev + K * (rho_crit - rho_measured)
    r_clamped = max(0.0, min(r_raw, 1.0))

    if congestion:
        return 0.0, r_raw  # override to block ramp entry

    return r_clamped, r_raw