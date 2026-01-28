import matplotlib

matplotlib.use('TkAgg')  # 🟢 Use interactive window

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import time

# Assuming these modules are correctly implemented and accessible:
# Ensure these modules are available in your environment or provide their content if they are custom
try:
    from Second_Case_Alinea_On.alinea_control import alinea, compute_density, r_prev, congestion_at_merge
    from traffic_demand import get_mainline_spawn_chance, format_time  # traffic_demand might be shared or a new one
    from traffic_demand import TOTAL_FRAMES as TRAFFIC_DEMAND_TOTAL_FRAMES # Import with alias
    from traffic_demand import get_ramp_spawn_chance
    from Second_Case_Alinea_On.sensors import Sensor  # <-- THIS WAS THE ONE WE JUST FIXED
    from Second_Case_Alinea_On.metanet_analysis import handle_sensor_data, \
        plot_global_analysis  # <-- THIS ALSO NEEDS TO BE UPDATED
except ImportError as e:
    print(
        f"Error importing a module: {e}. Please ensure alinea_control.py, traffic_demand.py, sensors.py, and metanet_analysis.py are in the same directory.")


    # Provide dummy implementations for the sake of runnable code if modules are missing
    # In a real scenario, you'd fix the import paths or provide the files.
    class DummySensor:
        def __init__(self, *args): pass

        def measure(self, *args): pass

        def start(self): pass

        def stop(self): pass

        def get_table(self): return []


    Sensor = DummySensor


    def dummy_alinea(*args, **kwargs):
        return 0.5, 0.5  # Return a default r_prev and raw_r


    alinea = dummy_alinea
    r_prev = 0.5


    def dummy_compute_density(*args, **kwargs):
        return 0.0


    compute_density = dummy_compute_density


    def dummy_congestion_at_merge(*args, **kwargs):
        return False


    congestion_at_merge = dummy_congestion_at_merge


    def dummy_get_mainline_spawn_chance(*args):
        return 0.5


    get_mainline_spawn_chance = dummy_get_mainline_spawn_chance


    def dummy_format_time(seconds):
        return f"{int(seconds / 3600):02d}:{int((seconds % 3600) / 60):02d}:{int(seconds % 60):02d}"


    format_time = dummy_format_time
    TRAFFIC_DEMAND_TOTAL_FRAMES = 3600  # Default total frames if not imported


    def dummy_get_ramp_spawn_chance(*args):
        return 0.3


    get_ramp_spawn_chance = dummy_get_ramp_spawn_chance


    def dummy_handle_sensor_data(*args, **kwargs):
        print("Dummy handle_sensor_data called.")


    handle_sensor_data = dummy_handle_sensor_data


    def dummy_plot_global_analysis(*args, **kwargs):
        print("Dummy plot_global_analysis called.")


    plot_global_analysis = dummy_plot_global_analysis

# --- Simulation Parameters ---
ROAD_LENGTH_KM = 5.0
NUM_CELLS = 12
CELL_LENGTH_METERS = (ROAD_LENGTH_KM * 1000) / NUM_CELLS

SECONDS_PER_FRAME_SIM = 1.0

# Set TOTAL_FRAMES for the animation to a significantly higher value
# This ensures the simulation runs longer, providing more data points and variance for plotting.
# 36000 frames = 10 hours of simulation at 1 second per frame.
TOTAL_FRAMES = 36000 # Increased total frames for longer simulation and more data variance

# --- Visual Display Parameters ---
DISPLAY_CELL_VISUAL_WIDTH = 2.0

# --- Density Scale Parameters ---
MAX_DENSITY_FOR_SCALE = 0.15  # Max density value for color mapping (adjust as needed)
DENSITY_BAR_HEIGHT = 0.1  # Height of the density bar below the road
DENSITY_BAR_Y_OFFSET = -1.0  # Adjusted lower Y-position offset for the density bar

# --- Minimum Speed Thresholds ---
MIN_SPEED_CRITICAL = 0.005  # Absolute minimum speed allowed when truly blocked/congested (very slow crawl)
MIN_SPEED_ALLOW = 0.03  # Minimum speed when ALINEA is "ALLOW" (no overall congestion, can cruise)
MAX_SPEED_MAINLINE_CAR = 0.080  # Max speed for mainline car (from spawn_main_car)
MAX_SPEED_MAINLINE_TRUCK = 0.047  # Max speed for mainline truck (from spawn_main_car)

# --- Merge Specific Parameters (On-Ramp) ---
MERGE_ZONE_START_X = 2.5
MERGE_ZONE_END_X = 2.9  # Where the ramp visually meets/goes under mainline
MERGE_BUFFER_AHEAD = 1.0  # How much space is needed ahead on mainline for a safe merge
MERGE_BUFFER_BEHIND = 0.7  # How much space is needed behind on mainline for a safe merge
MERGE_ACCELERATION_RATE = 0.003  # How quickly ramp cars speed up to match mainline
RAMP_SLOW_DOWN_RATE = 0.001  # How quickly ramp cars slow down if blocked
MAINLINE_MERGE_COOPERATION_RANGE_AHEAD = 1.5  # How far ahead mainline car looks for merging ramp car
MAINLINE_MERGE_COOPERATION_RANGE_BEHIND = 0.5  # How far behind mainline car looks for merging ramp car
MAINLINE_COOPERATION_DECELERATION_RATE = 0.0005  # How much mainline cars slow down to create gap

# --- Off-Ramp Specific Parameters ---
OFF_RAMP_START_X = 8.0  # Cell coordinate where off-ramp starts
OFF_RAMP_END_X = 9.5  # Cell coordinate where off-ramp ends (cars exit simulation)
OFF_RAMP_LENGTH = OFF_RAMP_END_X - OFF_RAMP_START_X
OFF_RAMP_CHANCE = 0.15  # Chance for a car to decide to take the off-ramp if in lane 1
OFF_RAMP_DECISION_ZONE = 0.5  # How many cells before off-ramp start car decides to exit
OFF_RAMP_SPEED_REDUCTION = 0.000000005  # How much cars slow down on off-ramp to simulate curve/exit
OFF_RAMP_LANE_Y = -0.35  # Y-position of the off-ramp lane visually

# --- Sensor Initialization ---
sensors = [
    Sensor(1, SECONDS_PER_FRAME_SIM, CELL_LENGTH_METERS),
    Sensor(2, SECONDS_PER_FRAME_SIM, CELL_LENGTH_METERS),
    Sensor(5, SECONDS_PER_FRAME_SIM, CELL_LENGTH_METERS),
    Sensor(OFF_RAMP_END_X + 0.5, SECONDS_PER_FRAME_SIM, CELL_LENGTH_METERS, is_off_ramp_sensor=True)
    # New sensor after off-ramp
]

recording = False
cars = []
animation_running = True


# --- Car logic (ADJUSTED 'v' VALUES FOR SECONDS_PER_FRAME_SIM = 1.0) ---
def spawn_main_car():
    if np.random.rand() < 0.2:
        return {
            "x": 0.0,
            "v": np.random.uniform(0.026, MAX_SPEED_MAINLINE_TRUCK),
            "lane": 1,
            "color": "brown",
            "type": "truck",
            "exiting": False  # New property for off-ramp decision
        }
    else:
        return {
            "x": 0.0,
            "v": np.random.uniform(0.053, MAX_SPEED_MAINLINE_CAR),
            "lane": np.random.choice([0, 1]),
            "color": "green",
            "type": "car",
            "exiting": False  # New property for off-ramp decision
        }


def spawn_ramp_car():
    if np.random.rand() < 0.15:
        return {
            "x": 1.5,
            "v": np.random.uniform(0.020, 0.033),
            "lane": "ramp",
            "color": "darkred",
            "type": "truck",
            "merged": False,
            "target_lane": 1,  # Ramp cars always aim for lane 1
            "exiting": False
        }
    else:
        return {
            "x": 1.5,
            "v": np.random.uniform(0.026, 0.053),
            "lane": "ramp",
            "color": "orange",
            "type": "car",
            "merged": False,
            "target_lane": 1,
            "exiting": False
        }


def enforce_safe_distance(min_gap=0.2):  # Increased min_gap for better visual spacing
    global r_prev  # We need r_prev to decide minimum speed

    # Main road cars
    for lane in [0, 1]:
        lane_cars = sorted(
            [c for c in cars if c.get("lane") == lane],
            key=lambda c: c["x"]
        )
        for i in range(len(lane_cars) - 1):
            lead = lane_cars[i + 1]
            follower = lane_cars[i]
            gap = lead["x"] - follower["x"]

            # Max speed for this car type
            max_v_for_car = MAX_SPEED_MAINLINE_CAR if follower["type"] == "car" else MAX_SPEED_MAINLINE_TRUCK

            if gap < min_gap:
                # Adjust follower's speed to maintain gap
                follower_target_v = lead["v"] + (gap - min_gap) * 0.1  # Simple proportional control
                follower["v"] = min(follower["v"], follower_target_v, MIN_SPEED_CRITICAL)
            else:
                # If gap is sufficient, car can accelerate.
                min_v_for_lane = MIN_SPEED_ALLOW if r_prev > 0.5 else MIN_SPEED_CRITICAL

                target_v = max(follower["v"], min_v_for_lane)  # Ensure minimum speed
                target_v = min(target_v + 0.002, max_v_for_car)  # Accelerate slightly towards max

                # Still don't go faster than lead car if they're close (prevents fast car closing gap too quickly)
                if gap < min_gap * 2:  # Check if still relatively close
                    target_v = min(target_v, lead["v"] + 0.005)  # Don't get too close to lead's speed

                follower["v"] = target_v

    # Ramp cars (check spacing with other ramp cars only) - This is for ramp-on-ramp collisions
    ramp_cars = sorted(
        [c for c in cars if c.get("lane") == "ramp"],
        key=lambda c: c["x"]
    )
    for i in range(len(ramp_cars) - 1):
        lead = ramp_cars[i + 1]
        follower = ramp_cars[i]
        gap = lead["x"] - follower["x"]
        if gap < min_gap * 0.7:  # Slightly tighter gap for ramp
            follower["v"] = min(follower["v"], lead["v"], MIN_SPEED_CRITICAL)
        else:
            follower["v"] = max(follower["v"], MIN_SPEED_CRITICAL)

    # Off-ramp cars (check spacing with other off-ramp cars)
    off_ramp_cars = sorted(
        [c for c in cars if c.get("lane") == "off_ramp"],
        key=lambda c: c["x"]
    )
    for i in range(len(off_ramp_cars) - 1):
        lead = off_ramp_cars[i + 1]
        follower = off_ramp_cars[i]
        gap = lead["x"] - follower["x"]
        if gap < min_gap * 0.7:
            follower["v"] = min(follower["v"], lead["v"], MIN_SPEED_CRITICAL)
        else:
            follower["v"] = max(follower["v"], MIN_SPEED_CRITICAL)

    # --- Smart overtaking logic (Adjusted distances for new speeds) ---
    for car in cars:
        if "overtake_timer" not in car:
            car["overtake_timer"] = 0

        if car["type"] == "truck" or car["lane"] == "ramp" or car[
            "lane"] == "off_ramp":  # Trucks and ramp/off-ramp cars don't overtake
            continue

        if car.get("lane") == 1:  # RIGHT lane (default)
            # Check if this car wants to exit
            if car.get("exiting") and car["x"] >= OFF_RAMP_START_X - OFF_RAMP_DECISION_ZONE and car[
                "x"] < OFF_RAMP_START_X:
                # Car is in the decision zone for the off-ramp
                # Check if off-ramp lane is clear for switching
                off_ramp_lane_clear = True
                for other in cars:
                    if other == car or other.get("lane") == "off_ramp":
                        continue  # Only check mainline cars in the off_ramp's path for now
                    if (other.get("lane") == 1 and  # Check mainline cars that are where off-ramp cars would be
                            OFF_RAMP_START_X <= other["x"] < OFF_RAMP_START_X + 0.5):  # A small zone
                        off_ramp_lane_clear = False
                        break

                if off_ramp_lane_clear:
                    car["lane"] = "off_ramp"
                    car["color"] = "purple"  # Visual indicator for off-ramp cars
                    car["v"] = min(car["v"], MIN_SPEED_ALLOW)  # Slow down slightly for ramp

            # Overtaking logic for mainline cars
            blocked = False
            for other in cars:
                if other == car or other.get("lane") != 1:
                    continue
                # Check for blocking car: if lead is slower and too close
                if 0 < other["x"] - car["x"] < 0.7 and other["v"] < car[
                    "v"] * 0.9:  # Made condition more sensitive (0.9 instead of 0.8)
                    blocked = True
                    break

            if blocked:
                lane_clear = True
                for o in cars:
                    if o == car or o.get("lane") != 0:
                        continue
                    # Check if target lane is clear (e.g., within 1.0 cells around for safe merge)
                    if abs(car["x"] - o["x"]) < 1.2:  # Increased buffer for lane change
                        lane_clear = False
                        break

                if lane_clear:
                    car["lane"] = 0  # switch to left
                    car["overtake_timer"] = 25  # Frames to stay in left lane (slightly increased)


        elif car.get("lane") == 0:  # LEFT lane
            car["overtake_timer"] = max(0, car["overtake_timer"] - 1)

            if car["overtake_timer"] == 0:
                safe_to_return = True
                for o in cars:
                    if o == car or o.get("lane") != 1:
                        continue
                    if abs(car["x"] - o["x"]) < 1.2:  # Use same buffer for returning
                        safe_to_return = False
                        break

                if safe_to_return:
                    car["lane"] = 1
                    car["v"] = max(car["v"], MIN_SPEED_ALLOW)  # Ensure it doesn't slow down too much on return


def update_car_positions():
    enforce_safe_distance()  # Apply safe distance within lanes first

    # Mainline car cooperation for merging ramp cars
    for mainline_car in cars:
        if mainline_car["lane"] == 1 and mainline_car["type"] != "truck":  # Only cars in merge lane, not trucks
            # Check if there's a ramp car trying to merge ahead
            ramp_car_trying_to_merge = None
            for r_car in cars:
                if (r_car["lane"] == "ramp" and not r_car.get("merged") and
                        MERGE_ZONE_START_X <= r_car["x"] < MERGE_ZONE_END_X + 0.5):  # Ramp car in or near merge zone

                    # Is the ramp car trying to merge in *my* vicinity?
                    if (mainline_car["x"] < r_car["x"] + MAINLINE_MERGE_COOPERATION_RANGE_AHEAD and
                            mainline_car["x"] > r_car["x"] - MAINLINE_MERGE_COOPERATION_RANGE_BEHIND):
                        ramp_car_trying_to_merge = r_car
                        break  # Found a relevant ramp car

            if ramp_car_trying_to_merge:
                # Check if there's a car behind this mainline_car that would get too close if it slows
                car_behind_is_too_close = False
                for other_mainline in cars:
                    if (other_mainline["lane"] == 1 and other_mainline["x"] < mainline_car["x"] and
                            mainline_car["x"] - other_mainline["x"] < MERGE_BUFFER_BEHIND * 1.5):  # A bit more buffer
                        car_behind_is_too_close = True
                        break

                # If no car is too close behind, try to cooperate by gently decelerating
                if not car_behind_is_too_close:
                    mainline_car["v"] = max(mainline_car["v"] - MAINLINE_COOPERATION_DECELERATION_RATE, MIN_SPEED_ALLOW)
                else:
                    # If car behind is too close, prioritize safe following distance over cooperation
                    pass  # Let enforce_safe_distance handle it

    for car in cars:
        # Handle ramp car merging logic
        if car["lane"] == "ramp":
            if not car.get("merged") and car["x"] >= MERGE_ZONE_START_X:
                # Only attempt merge if in merge zone and not yet merged

                mainline_cars_lane1 = [
                    c for c in cars
                    if c["lane"] == 1
                ]

                gap_found = True

                # Check if a mainline car is too close ahead or behind
                for other in mainline_cars_lane1:
                    if (car["x"] - MERGE_BUFFER_BEHIND < other["x"] < car["x"] + MERGE_BUFFER_AHEAD):
                        gap_found = False
                        break

                if gap_found:
                    # Merge successful!
                    car["lane"] = 1  # Change to mainline lane
                    car["color"] = "green"  # Indicate successful merge
                    car["merged"] = True
                    # Accelerate to match mainline flow or desired speed
                    car["v"] = max(car["v"], MIN_SPEED_ALLOW, np.random.uniform(0.04, 0.07))  # Give it a speed boost
                else:
                    # No gap found, car must slow down on ramp
                    car["v"] = max(car["v"] - RAMP_SLOW_DOWN_RATE, MIN_SPEED_CRITICAL)  # Gradually slow down
                    car["color"] = "red"  # Indicate blocked merge (visual feedback)

            # Failsafe: if car passes the merge_end_x and hasn't merged, it's stuck.
            # Force it to merge, but it might still overlap if logic isn't perfect.
            # This is a last resort to prevent cars from disappearing or getting stuck infinitely.
            if car["x"] >= MERGE_ZONE_END_X + 0.1 and not car.get("merged"):  # A bit beyond the visual merge end
                car["lane"] = 1
                car["color"] = "green"
                car["v"] = max(car["v"], MIN_SPEED_ALLOW)  # Give it a boost to move
                car["merged"] = True
                # print(f"DEBUG: Forced merge for car at x={car['x']:.2f}") # Comment out for less console spam

        # Handle off-ramp car logic
        elif car["lane"] == 1 and not car.get("exiting"):  # Only mainline cars in lane 1 that haven't decided to exit
            if car["x"] >= OFF_RAMP_START_X - OFF_RAMP_DECISION_ZONE:  # Car is in the decision zone
                if np.random.rand() < OFF_RAMP_CHANCE:  # Random chance to decide to exit
                    car[
                        "exiting"] = True  # Mark car as deciding to exit (lane change will happen in enforce_safe_distance)

        elif car["lane"] == "off_ramp":
            car["v"] = max(car["v"] - OFF_RAMP_SPEED_REDUCTION, MIN_SPEED_CRITICAL)  # Slow down on off-ramp
            # Cars exiting via off-ramp are removed at OFF_RAMP_END_X

        # Apply movement
        car["x"] += car["v"]

        # Failsafe for cars that somehow got stuck on ramp beyond visual road
        if car["lane"] == "ramp" and car[
            "x"] >= NUM_CELLS - 1:  # If a ramp car reaches end of road, force merge and move off
            car["lane"] = 1
            car["color"] = "green"
            car["v"] = 0.05  # Give it a boost to get off screen
            car["merged"] = True


def ramp_car_position_y(x):
    # Adjusted curve for a smoother visual transition
    ramp_start_x = 1.5
    ramp_end_x = 2.8  # Adjusted endpoint for merge
    y_ramp_start = -0.35
    y_ramp_end = 0.05  # Y-position of lane 1

    if x < ramp_start_x:
        return y_ramp_start
    elif x > ramp_end_x:
        return y_ramp_end
    else:
        t = (x - ramp_start_x) / (ramp_end_x - ramp_start_x)
        # Using a smoothstep function (3t^2 - 2t^3) for interpolation
        y_interpolated = y_ramp_start + (y_ramp_end - y_ramp_start) * (3 * t ** 2 - 2 * t ** 3)
        return y_interpolated


def off_ramp_car_position_y(x):
    # Curve for off-ramp cars
    off_ramp_start_x = OFF_RAMP_START_X
    off_ramp_end_x = OFF_RAMP_END_X
    y_mainline = 0.05  # Y-position of lane 1
    y_off_ramp_end = OFF_RAMP_LANE_Y  # Final Y-position on the off-ramp

    if x < off_ramp_start_x:
        return y_mainline
    elif x > off_ramp_end_x:
        return y_off_ramp_end
    else:
        t = (x - off_ramp_start_x) / (off_ramp_end_x - off_ramp_start_x)
        # Using a smoothstep function (3t^2 - 2t^3) for interpolation
        y_interpolated = y_mainline + (y_off_ramp_end - y_mainline) * (3 * t ** 2 - 2 * t ** 3)
        return y_interpolated


def draw_cars(ax, car_artists):
    for artist in car_artists:
        artist.remove()
    car_artists.clear()

    for car in cars:
        if car.get("lane") == 0:
            y = 0.4  # left lane
        elif car.get("lane") == 1:
            y = 0.05  # right lane
        elif car["lane"] == "ramp":
            y = ramp_car_position_y(car["x"])
        elif car["lane"] == "off_ramp":
            y = off_ramp_car_position_y(car["x"])
        else:
            y = 0.4  # Default, should not happen

        display_x = car["x"] * DISPLAY_CELL_VISUAL_WIDTH

        radius = 0.1 if car["type"] == "car" else 0.15
        circle = Circle((display_x, y), radius, color=car["color"], zorder=5)  # Cars on top
        ax.add_patch(circle)
        car_artists.append(circle)


def remove_offscreen_cars():
    """
    Remove cars that have completely exited the road segment.

    Cars should be removed once they have fully passed through the last cell
    to prevent them from getting stuck at the boundary and causing traffic jams.
    """
    global cars

    # Define the exit threshold - cars are removed when they pass this point
    # Adding a small buffer (0.2) beyond NUM_CELLS to ensure cars have fully
    # exited the last cell before removal
    exit_threshold = NUM_CELLS + 0.2

    # Keep only cars that haven't reached the exit threshold
    initial_count = len(cars)
    cars = [car for car in cars if car["x"] < exit_threshold]

    # Debug logging (comment out in production)
    removed_count = initial_count - len(cars)
    if removed_count > 0:
        print(f"DEBUG: Removed {removed_count} cars that exited the road (x >= {exit_threshold})")

    # Additional safety check: if any car is somehow stuck at the very end
    # (between NUM_CELLS and exit_threshold), force them to move forward
    for car in cars:
        if car["x"] >= NUM_CELLS and car["v"] < 0.01:  # Nearly stopped at the boundary
            print(f"DEBUG: Forcing stuck car at x={car['x']:.2f} to move forward")
            car["v"] = max(car["v"], 0.05)  # Give it enough speed to exit


def draw_static_road(ax):
    road_y = 0
    lane_height = 0.3

    # Draw road segments (background, zorder=2)
    for i in range(NUM_CELLS):
        rect1 = Rectangle((i * DISPLAY_CELL_VISUAL_WIDTH, road_y), DISPLAY_CELL_VISUAL_WIDTH, lane_height,
                          edgecolor='black', facecolor='lightgray', lw=1, zorder=2)
        rect2 = Rectangle((i * DISPLAY_CELL_VISUAL_WIDTH, road_y + 0.3), DISPLAY_CELL_VISUAL_WIDTH, lane_height,
                          edgecolor='black', facecolor='lightgray', lw=1, zorder=2)
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        cell_text = ax.text(i * DISPLAY_CELL_VISUAL_WIDTH + DISPLAY_CELL_VISUAL_WIDTH * 0.5, road_y + 0.65,
                            f'Cell {i}', ha='center', va='bottom', fontsize=8, color='gray', zorder=2)
        cell_text._static_label = True  # Mark as static

    # Draw the on-ramp with a more integrated look using Polygon
    on_ramp_start_cell = 1.3  # Cell coordinate where ramp starts
    on_ramp_end_cell = 2.9  # Cell coordinate where ramp ends at mainline (visually under)

    on_ramp_y_start = -0.35  # Y-coordinate at ramp start
    on_ramp_y_merge_under = -0.05  # Y-position where ramp visually goes "under" the road

    visual_on_ramp_start_x = on_ramp_start_cell * DISPLAY_CELL_VISUAL_WIDTH
    visual_on_ramp_end_x = on_ramp_end_cell * DISPLAY_CELL_VISUAL_WIDTH

    on_ramp_poly_points = [
        (visual_on_ramp_start_x, on_ramp_y_start),
        (visual_on_ramp_start_x, on_ramp_y_start + 0.25),
        (visual_on_ramp_end_x, on_ramp_y_merge_under + 0.2),
        (visual_on_ramp_end_x, on_ramp_y_merge_under),
    ]
    on_ramp_polygon = Polygon(on_ramp_poly_points, closed=True, color='sienna', alpha=0.7, edgecolor='none',
                              zorder=1)  # Ramp behind road
    ax.add_patch(on_ramp_polygon)

    on_ramp_text_obj = ax.text(
        (on_ramp_start_cell + (on_ramp_end_cell - on_ramp_start_cell) / 2) * DISPLAY_CELL_VISUAL_WIDTH,
        on_ramp_y_start - 0.1, 'On-Ramp', ha='center', va='bottom', fontsize=9, color='white', weight='bold', zorder=1)
    on_ramp_text_obj._static_label = True

    # --- Draw the Off-Ramp ---
    off_ramp_start_visual_x = OFF_RAMP_START_X * DISPLAY_CELL_VISUAL_WIDTH
    off_ramp_end_visual_x = OFF_RAMP_END_X * DISPLAY_CELL_VISUAL_WIDTH

    # Off-ramp starts from lane 1 (y=0.05) and curves downwards
    off_ramp_poly_points = [
        (off_ramp_start_visual_x, 0.05),  # Top-left (from mainline lane 1)
        (off_ramp_start_visual_x, 0.05 + 0.2),  # Bottom-left (start of a wider ramp)
        (off_ramp_end_visual_x, OFF_RAMP_LANE_Y + 0.2),  # Top-right (end of ramp)
        (off_ramp_end_visual_x, OFF_RAMP_LANE_Y),  # Bottom-right (end of ramp)
    ]
    off_ramp_polygon = Polygon(off_ramp_poly_points, closed=True, color='purple', alpha=0.7, edgecolor='none', zorder=1)
    ax.add_patch(off_ramp_polygon)

    off_ramp_text_obj = ax.text((OFF_RAMP_START_X + OFF_RAMP_END_X) / 2 * DISPLAY_CELL_VISUAL_WIDTH,
                                OFF_RAMP_LANE_Y - 0.1, 'Off-Ramp', ha='center', va='bottom', fontsize=9, color='white',
                                weight='bold', zorder=1)
    off_ramp_text_obj._static_label = True

    # Draw sensors (placed over road) - Adjusted higher
    main_road_center_y = road_y + lane_height  # Center between lanes, or slightly above lane 1
    # Existing sensors
    for i in [1, 2, 5]:
        sensor_x_visual = (i + 0.5) * DISPLAY_CELL_VISUAL_WIDTH
        sensor_y = main_road_center_y + 0.5  # Increased offset to move sensors significantly higher
        sensor = Circle((sensor_x_visual, sensor_y), 0.05, color='blue', alpha=0.8, zorder=4)  # Higher zorder
        ax.add_patch(sensor)
        sensor_text = ax.text(sensor_x_visual, sensor_y + 0.1, f'S{i}', ha='center', fontsize=9, color='blue',
                              weight='bold', zorder=4)
        sensor_text._static_label = True

    # New sensor for off-ramp
    off_ramp_sensor_x_visual = (OFF_RAMP_END_X + 0.5) * DISPLAY_CELL_VISUAL_WIDTH  # Half a cell after the off-ramp ends
    off_ramp_sensor_y = main_road_center_y + 0.5
    off_ramp_sensor = Circle((off_ramp_sensor_x_visual, off_ramp_sensor_y), 0.05, color='darkgreen', alpha=0.8,
                             zorder=4)  # Different color
    ax.add_patch(off_ramp_sensor)
    off_ramp_sensor_text = ax.text(off_ramp_sensor_x_visual, off_ramp_sensor_y + 0.1, 'S_OFF', ha='center', fontsize=9,
                                   color='darkgreen', weight='bold', zorder=4)
    off_ramp_sensor_text._static_label = True

    ax.set_xlim(-0.5 * DISPLAY_CELL_VISUAL_WIDTH, (NUM_CELLS + 1) * DISPLAY_CELL_VISUAL_WIDTH)
    ax.set_ylim(DENSITY_BAR_Y_OFFSET - DENSITY_BAR_HEIGHT - 0.15, 2.0)  # Adjusted Y-limit further for higher time
    ax.set_aspect('equal')
    ax.axis('off')


# --- Density Calculation for all cells ---
def compute_all_cell_densities(cars):
    densities = [0.0] * NUM_CELLS
    for i in range(NUM_CELLS):
        cell_start_x = i
        cell_end_x = i + 1
        cars_in_cell_lane0 = [car for car in cars if cell_start_x <= car["x"] < cell_end_x and car["lane"] == 0]
        cars_in_cell_lane1 = [car for car in cars if cell_start_x <= car["x"] < cell_end_x and car["lane"] == 1]

        densities[i] = (len(cars_in_cell_lane0) + len(cars_in_cell_lane1)) / 2.0  # Density per lane segment
    return densities


# --- Draw Density Scale ---
def draw_density_scale(ax, densities):
    # Remove only the density bars and their texts before redrawing
    for artist in ax.get_children():
        if isinstance(artist, Rectangle) and hasattr(artist, '_density_bar'):
            artist.remove()
        if isinstance(artist, plt.Text) and hasattr(artist, '_density_text'):
            artist.remove()

    cmap = plt.cm.get_cmap('RdYlGn_r')

    for i, density in enumerate(densities):
        norm_density = min(density / MAX_DENSITY_FOR_SCALE, 1.0)
        color = cmap(norm_density)

        rect = Rectangle(
            (i * DISPLAY_CELL_VISUAL_WIDTH, DENSITY_BAR_Y_OFFSET),
            DISPLAY_CELL_VISUAL_WIDTH,
            DENSITY_BAR_HEIGHT,
            facecolor=color,
            edgecolor='black',
            lw=0.5,
            alpha=0.6,
            zorder=1
        )
        rect._density_bar = True
        ax.add_patch(rect)

        txt = ax.text(i * DISPLAY_CELL_VISUAL_WIDTH + DISPLAY_CELL_VISUAL_WIDTH * 0.5,
                      DENSITY_BAR_Y_OFFSET + DENSITY_BAR_HEIGHT / 2,
                      f'{density:.2f}', ha='center', va='center', fontsize=7, color='black', weight='bold', zorder=2)
        txt._density_text = True

    # Density label
    density_label_x = (NUM_CELLS / 2) * DISPLAY_CELL_VISUAL_WIDTH
    density_label_y = DENSITY_BAR_Y_OFFSET - 0.1  # Position below the bars
    # Ensure this label is not removed by the general text clearing logic in update
    if not hasattr(ax, '_density_label_obj'):
        ax._density_label_obj = ax.text(density_label_x, density_label_y,
                                        'Density (Cars/Cell Segment)',
                                        ha='center', va='top', fontsize=8, color='dimgray')
    else:
        ax._density_label_obj.set_position((density_label_x, density_label_y))
        ax._density_label_obj.set_text('Density (Cars/Cell Segment)')


# --- Animation callbacks ---
last_frame_time = time.perf_counter()
fps_rolling_avg = 0.0
fps_alpha = 0.1


def init():
    return []


def update(frame):
    global r_prev, last_frame_time, fps_rolling_avg

    current_time_perf = time.perf_counter()
    elapsed_real_time_for_frame = current_time_perf - last_frame_time
    last_frame_time = current_time_perf

    if elapsed_real_time_for_frame > 0:
        instant_fps = 1.0 / elapsed_real_time_for_frame
        fps_rolling_avg = fps_rolling_avg * (1 - fps_alpha) + instant_fps * fps_alpha

    current_sim_seconds = frame * SECONDS_PER_FRAME_SIM

    # Clear only dynamic texts (not static labels or density texts)
    for txt in ax.texts:
        if not hasattr(txt, '_static_label') and not hasattr(txt, '_density_text') and txt != getattr(ax,
                                                                                                      '_density_label_obj',
                                                                                                      None):
            txt.remove()

    # --- Display Info when Paused ---
    if not animation_running:
        sim_time_paused = format_time(current_sim_seconds)
        rho_paused = compute_density(cars, cell_index=2)

        ax.text(NUM_CELLS * DISPLAY_CELL_VISUAL_WIDTH / 2, 1.8, f"PAUSED",  # Further higher position
                fontsize=20, weight='bold', ha='center', color='gray')
        ax.text(NUM_CELLS * DISPLAY_CELL_VISUAL_WIDTH / 2, 1.7, f"Frame {frame}",  # Adjusted lower
                fontsize=12, ha='center', color='dimgray')
        ax.text(0, 1.7,  # Adjusted lower
                f"Time: {sim_time_paused} | Merge Cell Density (Cell 2): {rho_paused:.3f}",
                fontsize=10, color='gray', ha='left')
        ax.text(NUM_CELLS * DISPLAY_CELL_VISUAL_WIDTH, 1.7, f"FPS: {fps_rolling_avg:.1f}",  # Adjusted lower
                fontsize=10, color='red', ha='right')
        # Ensure density scale is drawn even when paused
        all_densities = compute_all_cell_densities(cars)
        draw_density_scale(ax, all_densities)
        return car_artists

    # === 1. MAINLINE TRAFFIC DEMAND ===
    spawn_chance = get_mainline_spawn_chance(current_sim_seconds)
    if np.random.rand() < spawn_chance:
        if not any(car.get("lane") in [0, 1] and car["x"] < 0.2 for car in cars):
            cars.append(spawn_main_car())

    # === 2. ALINEA Control and Ramp Car Spawning ===
    rho_merge_cell = compute_density(cars, cell_index=2)  # Density at merge cell
    is_congested = congestion_at_merge(cars)

    if frame % 600 == 0:
        r_prev, raw_r = alinea(r_prev, rho_merge_cell, congestion=is_congested)

    ramp_chance = get_ramp_spawn_chance(current_sim_seconds)
    if np.random.rand() < ramp_chance and r_prev > 0.5:
        # Check if there's space on the ramp entrance
        if not any(
                car.get("lane") == "ramp" and MERGE_ZONE_START_X - 1.0 <= car["x"] < MERGE_ZONE_START_X - 0.5 for car in
                cars):
            cars.append(spawn_ramp_car())

    # === 3. Update Cars and Measure Sensors ===
    update_car_positions()
    remove_offscreen_cars()
    draw_cars(ax, car_artists)

    if recording:
        for sensor in sensors:
            sensor.measure(cars, frame)

    # === 4. Display UI Labels ===
    sim_time_display = format_time(current_sim_seconds)

    # Main time (top center, higher)
    ax.text(NUM_CELLS * DISPLAY_CELL_VISUAL_WIDTH / 2, 1.8, f"{sim_time_display}",  # Further higher position
            fontsize=20, weight='bold', ha='center', color='black')

    # Frame number (just below time)
    ax.text(NUM_CELLS * DISPLAY_CELL_VISUAL_WIDTH / 2, 1.7, f"Frame {frame}",  # Adjusted lower
            fontsize=12, ha='center', color='dimgray')

    # ALINEA/Density info (top left)
    ax.text(0, 1.7,  # Adjusted lower
            f"ALINEA: {'ALLOW' if r_prev > 0.5 else 'BLOCK'} | Merge Cell Density (Cell 2): {rho_merge_cell:.3f}",
            fontsize=10, color='darkblue', ha='left')

    # FPS display (top right)
    ax.text(NUM_CELLS * DISPLAY_CELL_VISUAL_WIDTH, 1.7, f"FPS: {fps_rolling_avg:.1f}",  # Adjusted lower
            fontsize=10, color='red', ha='right')

    # === 5. Draw Density Scale ===
    all_densities = compute_all_cell_densities(cars)
    draw_density_scale(ax, all_densities)

    return car_artists


# --- Button Callbacks (Remain unchanged) ---
def start(event):
    global animation_running
    animation_running = True


def stop(event):
    global animation_running
    animation_running = False


# --- Setup Figure and Buttons ---
fig, ax = plt.subplots(figsize=(14, 6))  # Increased figure height for more space
plt.subplots_adjust(bottom=0.25, top=0.92, left=0.05, right=0.95)  # Adjust margins

draw_static_road(ax)

# Mark initial static text elements after draw_static_road
for text_obj in ax.texts:
    if 'Cell' in text_obj.get_text() or 'On-Ramp' in text_obj.get_text() or 'S' in text_obj.get_text() or 'Off-Ramp' in text_obj.get_text():
        text_obj._static_label = True

car_artists = []

# Create buttons
ax_start = plt.axes([0.7, 0.05, 0.1, 0.075])
ax_stop = plt.axes([0.81, 0.05, 0.1, 0.075])

btn_start = Button(ax_start, 'Start')
btn_stop = Button(ax_stop, 'Stop')

btn_start.on_clicked(start)
btn_stop.on_clicked(stop)

ax_start_rec = plt.axes([0.1, 0.05, 0.15, 0.075])
ax_stop_rec = plt.axes([0.27, 0.05, 0.15, 0.075])

btn_rec_start = Button(ax_start_rec, 'Start Recording')
btn_rec_stop = Button(ax_stop_rec, 'Stop Recording')


def start_recording(event):
    global recording
    recording = True
    for s in sensors:
        s.start()


def stop_recording(event):
    print("DEBUG: stop_recording function called!")
    global recording
    recording = False
    for s in sensors:
        s.stop()

    # The modules for analysis are assumed to be present for this to work
    try:
        # Corrected import path for metanet_analysis
        from Second_Case_Alinea_On.metanet_analysis import handle_sensor_data, plot_global_analysis
        for sensor in sensors:
            # Corrected attribute from cell_index to location_cell_x
            print(f"DEBUG: Processing sensor at cell_index={sensor.cell_index} data...")

            try:
                handle_sensor_data(
                    sensor.cell_index,
                    sensor.get_table(),
                    seconds_per_frame=SECONDS_PER_FRAME_SIM,
                    cell_length_meters=CELL_LENGTH_METERS,
                    road_length_km=ROAD_LENGTH_KM
                )
            except ValueError as ve:
                print(f"ERROR: Plotting for sensor at x={sensor.cell_index} failed due to data issues: {ve}")
                print("This usually means the data collected was insufficient or malformed for the 3D plot.")
            except Exception as e:
                print(f"AN UNEXPECTED ERROR OCCURRED during sensor data processing for sensor at x={sensor.cell_index}: {e}")


        print("DEBUG: Calling plot_global_analysis...")
        plot_global_analysis()
        print("\nRecording stopped and data processing attempt finished.")
    except ImportError:
        print("Analysis modules (metanet_analysis) not found. Skipping data processing and plotting.")


btn_rec_start.on_clicked(start_recording)
btn_rec_stop.on_clicked(stop_recording)

# Start animation
ani = FuncAnimation(
    fig, update,
    frames=TOTAL_FRAMES,
    init_func=init,
    interval=50,
    blit=False
)

plt.show()
