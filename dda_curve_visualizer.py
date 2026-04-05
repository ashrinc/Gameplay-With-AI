#!/usr/bin/env python3
"""
DDA Curve Visualizer
Records DDA decisions from running game and plots difficulty curve over time.

Run the game normally, and this script will display a live-updating plot showing:
- Enemy Count
- Enemy Speed
- Spawn Delay (higher = slower spawning)
- Player Performance Score
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import time
import subprocess
import sys

# Initialize plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DDA - Difficulty Adjustment Curve', fontsize=16)

# Store last 60 seconds of data
max_points = 120  # 60 seconds at ~2s per DDA update
times = deque(maxlen=max_points)
enemy_counts = deque(maxlen=max_points)
enemy_speeds = deque(maxlen=max_points)
spawn_delays = deque(maxlen=max_points)
player_skills = deque(maxlen=max_points)

def parse_dda_output(line):
    """Extract DDA parameters from console output line."""
    try:
        # Format: [AI/... Wave:X ...] ... → enemies=6 spd=100 hp=55 fire=1.80 ast=6 spawn=1.00s
        if "→" not in line or "enemies=" not in line:
            return None

        parts = line.split("→")[-1]  # Get the right side of →

        data = {}

        # Extract enemies=X
        if "enemies=" in parts:
            data['enemies'] = int(parts.split("enemies=")[1].split()[0])

        # Extract spd=X
        if "spd=" in parts:
            data['speed'] = float(parts.split("spd=")[1].split()[0])

        # Extract spawn=X
        if "spawn=" in parts:
            spawn_str = parts.split("spawn=")[1].split("s")[0]
            data['spawn_delay'] = float(spawn_str)

        # Extract player skill (DYING, STRUGGLING, OK, GOOD, DOMINATING)
        if "DYING" in line:
            data['skill'] = 0.0
        elif "STRUGGLING" in line:
            data['skill'] = 0.25
        elif "OK" in line:
            data['skill'] = 0.5
        elif "GOOD" in line:
            data['skill'] = 0.75
        elif "DOMINATING" in line:
            data['skill'] = 1.0
        else:
            data['skill'] = 0.5

        return data if len(data) >= 2 else None
    except:
        return None

def update_plot(frame):
    """Update the plot with new data."""
    for ax in axes.flat:
        ax.clear()

    if len(times) > 0:
        # Plot 1: Enemy Count
        ax = axes[0, 0]
        ax.plot(list(times), list(enemy_counts), 'r-', linewidth=2, marker='o')
        ax.set_ylabel('Enemy Count', fontsize=11)
        ax.set_ylim(0, 14)
        ax.grid(True, alpha=0.3)
        ax.set_title('Enemy Count Over Time')

        # Plot 2: Enemy Speed
        ax = axes[0, 1]
        ax.plot(list(times), list(enemy_speeds), 'b-', linewidth=2, marker='s')
        ax.set_ylabel('Speed (px/s)', fontsize=11)
        ax.set_ylim(50, 200)
        ax.grid(True, alpha=0.3)
        ax.set_title('Enemy Speed Over Time')

        # Plot 3: Spawn Delay (CRITICAL - higher = slower spawning)
        ax = axes[1, 0]
        ax.plot(list(times), list(spawn_delays), 'g-', linewidth=2, marker='^')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Spawn Interval (s)', fontsize=11, color='green')
        ax.set_ylim(0, 2)
        ax.grid(True, alpha=0.3)
        ax.set_title('⚠️ Spawn Delay (KEY METRIC)')
        ax.tick_params(axis='y', labelcolor='green')

        # Plot 4: Player Skill Assessment
        ax = axes[1, 1]
        ax.plot(list(times), list(player_skills), 'purple', linewidth=2, marker='D')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Player Skill', fontsize=11)
        ax.set_ylim(0, 1)
        ax.fill_between(range(len(player_skills)), list(player_skills), alpha=0.3, color='purple')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
        ax.grid(True, alpha=0.3)
        ax.set_title('Player Performance Score')
        ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()

# Start the game subprocess and read its output
print("🎮 Starting game with output capture...")
print("📊 Plotting DDA curve in real-time...")

try:
    process = subprocess.Popen(
        ["python3", "game_3d.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    start_time = time.time()

    # Read output in a background manner
    def read_game_output():
        for line in process.stdout:
            elapsed = time.time() - start_time
            data = parse_dda_output(line)

            if data:
                times.append(elapsed)
                enemy_counts.append(data.get('enemies', 6))
                enemy_speeds.append(data.get('speed', 100))
                spawn_delays.append(data.get('spawn_delay', 1.0))
                player_skills.append(data.get('skill', 0.5))

                print(f"[{elapsed:.1f}s] Enemies={data.get('enemies', '?')} "
                      f"Speed={data.get('speed', '?')} "
                      f"SpawnDelay={data.get('spawn_delay', '?'):.2f}s")

    # Create animation
    ani = animation.FuncAnimation(fig, update_plot, interval=500, blit=False)

    # Run the plot
    plt.show()

except KeyboardInterrupt:
    print("\n✓ Stopping visualization...")
    process.terminate()
except Exception as e:
    print(f"✗ Error: {e}")
