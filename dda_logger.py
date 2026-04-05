#!/usr/bin/env python3
"""
DDA Logger - Lightweight game monitoring
Tails game output and logs DDA decisions to CSV and creates plot after session.

Usage: python dda_logger.py
"""

import subprocess
import csv
import time
import sys
import re
from pathlib import Path

# Create log file
log_file = Path("dda_session.csv")
log_file.unlink(missing_ok=True)

print("🎮 Starting game with DDA logging...\n")

with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'time_s', 'action', 'enemy_count', 'enemy_speed', 'enemy_hp',
        'fire_rate', 'asteroid_count', 'spawn_delay', 'player_skill', 'wave'
    ])

    try:
        process = subprocess.Popen(
            ["python3", "game_3d.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        start_time = time.time()

        # Action map
        action_names = {
            '0': 'HOLD/AI',
            '1': 'HARDER+',
            '2': 'EASIER-',
            '3': 'HARDER++',
            '4': 'EASIER--'
        }

        skill_map = {
            'DYING': 0.0,
            'STRUGGLING': 0.25,
            'OK': 0.5,
            'GOOD': 0.75,
            'DOMINATING': 1.0
        }

        for line in process.stdout:
            print(line, end='')
            elapsed = time.time() - start_time

            # Parse: [AI/... (SKILL) Wave:X] ACTION → enemies=X spd=X hp=X fire=X ast=X spawn=Xs
            if "→" in line and "enemies=" in line:
                try:
                    # Extract action name
                    action = 'HOLD/AI'
                    for act_name in action_names.values():
                        if act_name in line:
                            action = act_name
                            break

                    # Extract metrics
                    match_enemies = re.search(r'enemies=(\d+)', line)
                    match_speed = re.search(r'spd=([\d.]+)', line)
                    match_hp = re.search(r'hp=([\d.]+)', line)
                    match_fire = re.search(r'fire=([\d.]+)', line)
                    match_ast = re.search(r'ast=(\d+)', line)
                    match_spawn = re.search(r'spawn=([\d.]+)', line)
                    match_wave = re.search(r'Wave:(\d+)', line)

                    # Extract skill
                    skill = 0.5
                    for skill_name, score in skill_map.items():
                        if skill_name in line:
                            skill = score
                            break

                    # Write to CSV
                    writer.writerow([
                        f"{elapsed:.1f}",
                        action,
                        match_enemies.group(1) if match_enemies else "?",
                        match_speed.group(1) if match_speed else "?",
                        match_hp.group(1) if match_hp else "?",
                        match_fire.group(1) if match_fire else "?",
                        match_ast.group(1) if match_ast else "?",
                        match_spawn.group(1) if match_spawn else "?",
                        f"{skill:.2f}",
                        match_wave.group(1) if match_wave else "?",
                    ])
                    f.flush()

                except Exception as e:
                    pass

    except KeyboardInterrupt:
        print("\n✓ Logging stopped")
        process.terminate()

print(f"\n✅ Data saved to {log_file}")
print(f"📊 To plot: python dda_plot.py")
