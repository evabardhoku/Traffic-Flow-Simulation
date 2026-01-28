import numpy as np

# --- Simulation Parameters ---
# Total frames for a 24-hour simulation, where each frame represents 1 simulated second.
# 24 hours * 60 minutes/hour * 60 seconds/minute = 86400 seconds
TOTAL_FRAMES = 24 * 3600


# --- Traffic Demand Functions ---

def get_mainline_spawn_chance(simulated_seconds):
    """
    Determines the chance of a new car spawning on the mainline based on the
    simulated time of day (24-hour cycle).

    Args:
        simulated_seconds (float): The current simulated time in seconds from the start (0 to 86400).

    Returns:
        float: The probability (0.0 to 1.0) of a mainline car spawning in the current frame.
    """
    # Convert seconds to hours for easier time-based logic
    sim_hour = (simulated_seconds % (24 * 3600)) / 3600.0

    # Base spawn chance (e.g., during off-peak hours)
    spawn_chance = 0.03  # Adjusted base value, slightly lower

    # Morning peak (e.g., 6 AM to 9 AM)
    if 6.0 <= sim_hour < 9.0:
        # Gradually increase to a higher peak
        if sim_hour < 7.5:
            spawn_chance = 0.03 + (0.08 * ((sim_hour - 6.0) / 1.5))  # Rises to max 0.11
        else:
            spawn_chance = 0.11 - (0.04 * ((sim_hour - 7.5) / 1.5))  # Falls slightly

    # Afternoon/Evening peak (e.g., 16 PM to 19 PM)
    elif 16.0 <= sim_hour < 19.0:
        # Gradually increase to a higher peak
        if sim_hour < 17.5:
            spawn_chance = 0.03 + (0.10 * ((sim_hour - 16.0) / 1.5))  # Rises to max 0.13
        else:
            spawn_chance = 0.13 - (0.05 * ((sim_hour - 17.5) / 1.5))  # Falls slightly

    # Late night / Early morning (e.g., 0 AM to 5 AM) - very low traffic
    elif 0.0 <= sim_hour < 5.0:
        spawn_chance = 0.005  # Even lower at night

    # Ensure probability is within [0, 1]
    return max(0.0, min(1.0, spawn_chance))


def get_ramp_spawn_chance(simulated_seconds):
    """
    Determines the chance of a new car spawning on the ramp,
    often mirroring mainline demand patterns but potentially lower.

    Args:
        simulated_seconds (float): The current simulated time in seconds.

    Returns:
        float: The probability (0.0 to 1.0) of a ramp car spawning.
    """
    # For simplicity, let ramp demand be a fraction of mainline demand
    mainline_chance = get_mainline_spawn_chance(simulated_seconds)
    ramp_chance_factor = 0.5  # Ramp traffic is 50% of mainline traffic

    # Add a specific ramp-heavy period or modify based on assumptions
    sim_hour = (simulated_seconds % (24 * 3600)) / 3600.0

    if 7.0 <= sim_hour < 9.0:  # Slight increase during morning rush
        ramp_chance_factor = 0.6
    elif 17.0 <= sim_hour < 19.0:  # Slight increase during evening rush
        ramp_chance_factor = 0.65

    return max(0.0, min(1.0, mainline_chance * ramp_chance_factor))

def get_off_ramp_exit_chance(simulated_seconds):
    """
    Determines the probability of a car deciding to take the off-ramp.
    This will be higher during commute hours when people are heading home.

    Args:
        simulated_seconds (float): The current simulated time in seconds.

    Returns:
        float: The probability (0.0 to 1.0) of a car deciding to exit.
    """
    sim_hour = (simulated_seconds % (24 * 3600)) / 3600.0

    base_exit_chance = 0.02 # Base chance during off-peak

    # Morning peak (e.g., people arriving at work area, some exits)
    if 7.0 <= sim_hour < 9.0:
        base_exit_chance = 0.03 # Slightly higher

    # Afternoon/Evening peak (e.g., people going home, higher exits)
    elif 16.0 <= sim_hour < 19.0:
        if sim_hour < 17.5:
            base_exit_chance = 0.03 + (0.04 * ((sim_hour - 16.0) / 1.5)) # Rises to 0.07
        else:
            base_exit_chance = 0.07 - (0.03 * ((sim_hour - 17.5) / 1.5)) # Falls to 0.04

    # Late night / Early morning - very low exits
    elif 0.0 <= sim_hour < 5.0:
        base_exit_chance = 0.01

    return max(0.0, min(1.0, base_exit_chance))

def format_time(total_seconds):
    """
    Formats a total number of seconds into an HH:MM:SS string.

    Args:
        total_seconds (float): The total number of seconds.

    Returns:
        str: Formatted time string (HH:MM:SS).
    """
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"