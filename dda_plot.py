#!/usr/bin/env python3
"""
DDA Curve Plotter - Visualize difficulty adjustments after game session
Reads dda_session.csv and creates 4-panel analysis plot.

Usage: python dda_plot.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

log_file = Path("dda_session.csv")

if not log_file.exists():
    print(f"❌ {log_file} not found!")
    print("Run: python dda_logger.py (while playing)")
    sys.exit(1)

# Read log
df = pd.read_csv(log_file)

if df.empty:
    print("❌ No data in CSV file!")
    sys.exit(1)

# Convert columns to numeric
for col in ['time_s', 'enemy_count', 'enemy_speed', 'enemy_hp', 'fire_rate', 'asteroid_count', 'spawn_delay', 'player_skill']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('📊 Space Dogfight - DDA Analysis Curve', fontsize=18, fontweight='bold')

# Color actions
action_colors = {
    'HOLD/AI': 'gray',
    'HARDER+': 'red',
    'HARDER++': 'darkred',
    'EASIER-': 'lightblue',
    'EASIER--': 'darkblue'
}

# ─────────────────────────────────────────────────────────
# PLOT 1: Enemy Count (should fluctuate 2-12)
# ─────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(df['time_s'], df['enemy_count'], 'ro-', linewidth=2, markersize=6, label='Current')
ax.axhline(y=6, color='green', linestyle='--', alpha=0.5, label='Default (6)')
ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Min (2)')
ax.axhline(y=12, color='orange', linestyle='--', alpha=0.5, label='Max (12)')
ax.fill_between(df['time_s'], 2, 12, alpha=0.1, color='gray')
ax.set_ylabel('Enemy Count', fontsize=12, fontweight='bold')
ax.set_title('👾 Enemy Count Over Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)
ax.set_ylim(0, 14)

# ─────────────────────────────────────────────────────────
# PLOT 2: Enemy Speed (should fluctuate 60-180)
# ─────────────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(df['time_s'], df['enemy_speed'], 'bs-', linewidth=2, markersize=6, label='Current')
ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Default (100)')
ax.axhline(y=60, color='red', linestyle='--', alpha=0.5, label='Min (60)')
ax.axhline(y=180, color='orange', linestyle='--', alpha=0.5, label='Max (180)')
ax.fill_between(df['time_s'], 60, 180, alpha=0.1, color='gray')
ax.set_ylabel('Speed (px/sec)', fontsize=12, fontweight='bold')
ax.set_title('⚡ Enemy Speed Over Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)
ax.set_ylim(50, 200)

# ─────────────────────────────────────────────────────────
# PLOT 3: Spawn Delay (KEY METRIC - why no enemies!)
# ─────────────────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(df['time_s'], df['spawn_delay'], 'g^-', linewidth=3, markersize=8, label='Spawn Delay', color='darkgreen')
ax.axhline(y=1.0, color='blue', linestyle='--', alpha=0.5, label='Default (1.0s)')
ax.axhline(y=0.25, color='orange', linestyle='--', alpha=0.3, label='Min (0.25s)')
ax.axhline(y=1.6, color='red', linestyle='--', alpha=0.3, label='Max (1.6s - TOO SLOW!)')
ax.fill_between(df['time_s'], 0.25, 1.6, alpha=0.1, color='yellow')
ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
ax.set_ylabel('Seconds Between Spawns', fontsize=12, fontweight='bold', color='darkgreen')
ax.set_title('⚠️ SPAWN DELAY (Why No Enemies?)', fontsize=13, fontweight='bold', color='darkgreen')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)
ax.set_ylim(0, 2)
ax.tick_params(axis='y', labelcolor='darkgreen')

# ─────────────────────────────────────────────────────────
# PLOT 4: Player Skill (0=dying, 1=dominating)
# ─────────────────────────────────────────────────────────
ax = axes[1, 1]
ax.plot(df['time_s'], df['player_skill'], 'mo-', linewidth=2, markersize=6)
ax.fill_between(df['time_s'], df['player_skill'], alpha=0.3, color='purple')
ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Balanced (0.5)')
ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.3, label='Struggle Zone (<0.3)')
ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.3, label='Dominating Zone (>0.7)')
ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
ax.set_ylabel('Player Skill Score', fontsize=12, fontweight='bold')
ax.set_title('🎯 Player Performance Assessment', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('dda_curve.png', dpi=150, bbox_inches='tight')
print("✅ Plot saved to: dda_curve.png")

# Print statistics
print("\n" + "="*60)
print("📊 STATISTICS")
print("="*60)
print(f"\nTotal Time: {df['time_s'].max():.1f} seconds")
print(f"Total Actions: {len(df)}")
print(f"Average Action Rate: {len(df) / (df['time_s'].max() or 1):.1f} decisions/second")

print(f"\n👾 ENEMY COUNT:")
print(f"   Range: {df['enemy_count'].min():.0f} - {df['enemy_count'].max():.0f}")
print(f"   Mean: {df['enemy_count'].mean():.1f}")

print(f"\n⚡ ENEMY SPEED:")
print(f"   Range: {df['enemy_speed'].min():.0f} - {df['enemy_speed'].max():.0f}")
print(f"   Mean: {df['enemy_speed'].mean():.1f}")

print(f"\n⏱️  SPAWN DELAY (seconds between enemy spawns):")
print(f"   Range: {df['spawn_delay'].min():.2f}s - {df['spawn_delay'].max():.2f}s")
print(f"   Mean: {df['spawn_delay'].mean():.2f}s")
print(f"   ⚠️  If mean > 1.2s, enemies spawn too slowly!")

print(f"\n🎯 PLAYER SKILL:")
print(f"   Range: {df['player_skill'].min():.2f} - {df['player_skill'].max():.2f}")
print(f"   Mean: {df['player_skill'].mean():.2f}")

action_dist = df['action'].value_counts()
print(f"\n🎮 ACTION DISTRIBUTION:")
for action, count in action_dist.items():
    pct = 100 * count / len(df)
    print(f"   {action:12s}: {count:3d} times ({pct:5.1f}%)")

print("\n" + "="*60)
print("💡 ANALYSIS TIPS:")
print("="*60)
if df['spawn_delay'].mean() > 1.2:
    print("⚠️  HIGH SPAWN DELAY - Model thinks you're dominating!")
    print("    The game is making enemies spawn slower to keep challenge.")
    print("    Try: Make mistakes, get hit, die intentionally to reset.")

if df['player_skill'].mean() > 0.7:
    print("✓ Model thinks you're DOMINATING!")
    print("  It will keep increasing difficulty.")

if df['enemy_count'].mean() < 3:
    print("⚠️  Very few enemies - check if model is stuck in EASIER mode.")

plt.show()
