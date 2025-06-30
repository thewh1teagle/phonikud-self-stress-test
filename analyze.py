#!/usr/bin/env python3
"""
WER Analysis Script 📊
Analyzes Word Error Rate (WER) results from different TTS models/systems
Lower WER values indicate better performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up matplotlib style  
plt.style.use('default')

def load_data():
    """Load and clean the results data 📂"""
    print("🔍 Loading results data...")
    
    # Read CSV file
    df = pd.read_csv('results.csv')
    
    # Handle missing values in phonikud_enhanced column
    df['phonikud_enhanced'] = df['phonikud_enhanced'].fillna(0)
    
    print(f"✅ Loaded {len(df)} records")
    return df

def print_summary_stats(df):
    """Print detailed WER summary statistics with emojis 📈"""
    print("\n" + "="*60)
    print("📊 WER ANALYSIS SUMMARY")
    print("="*60)
    
    systems = ['roboshaul_nakdimon', 'phonikud', 'phonikud_enhanced']
    system_emojis = ['🤖', '🎯', '⚡']
    
    for system, emoji in zip(systems, system_emojis):
        print(f"\n{emoji} {system.upper().replace('_', ' ')}")
        print("-" * 40)
        
        values = df[system].dropna()
        mean_wer = values.mean()
        std_wer = values.std()
        min_wer = values.min()
        max_wer = values.max()
        median_wer = values.median()
        
        print(f"📊 Mean WER: {mean_wer:.2f}")
        print(f"📏 Std Dev: {std_wer:.2f}")
        print(f"🎯 Median WER: {median_wer:.2f}")
        print(f"✅ Best WER: {min_wer} (lower is better)")
        print(f"❌ Worst WER: {max_wer}")
        print(f"🔢 Total Samples: {len(values)}")
        
        # WER performance tiers
        perfect_pct = (values == 0).mean() * 100
        excellent_pct = (values <= 0.5).mean() * 100
        good_pct = (values <= 1.0).mean() * 100
        acceptable_pct = (values <= 2.0).mean() * 100
        
        print(f"🏆 Perfect (WER=0): {perfect_pct:.1f}%")
        print(f"⭐ Excellent (≤0.5): {excellent_pct:.1f}%")
        print(f"👍 Good (≤1.0): {good_pct:.1f}%")
        print(f"📝 Acceptable (≤2.0): {acceptable_pct:.1f}%")
        
        # WER distribution
        wer_counts = values.value_counts().sort_index()
        print(f"📋 WER Distribution:")
        for wer, count in wer_counts.items():
            percentage = (count / len(values)) * 100
            bar = "█" * int(percentage // 5)
            quality = "🏆" if wer == 0 else "⭐" if wer <= 0.5 else "👍" if wer <= 1.0 else "⚠️"
            print(f"   WER {wer}: {count:3d} ({percentage:5.1f}%) {bar} {quality}")

def create_visualizations(df):
    """Create WER-appropriate visualizations 📈"""
    print("\n🎨 Creating WER visualizations...")
    
    systems = ['roboshaul_nakdimon', 'phonikud', 'phonikud_enhanced']
    colors = ['#2E8B57', '#4169E1', '#DC143C']  # Green, Blue, Red
    
    # 1. Bar chart of average WER (lower is better)
    plt.figure(figsize=(10, 6))
    means = [df[system].dropna().mean() for system in systems]
    system_names = []
    for s in systems:
        name = s.replace('_', ' ').title()
        if 'phonikud' in s.lower():
            name += ' (ours)'
        system_names.append(name)
    
    # Color bars based on performance (green=best, red=worst)
    sorted_indices = np.argsort(means)
    bar_colors = ['green' if i == sorted_indices[0] else 'orange' if i == sorted_indices[1] else 'red' 
                  for i in range(len(means))]
    
    bars = plt.bar(system_names, means, color=bar_colors, alpha=0.7, edgecolor='black')
    plt.ylabel('← Word Error Rate (WER)', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars (WER and Accuracy)
    # Adjust y-axis limits to give more room for labels
    max_wer = max(means)
    plt.ylim(0, max_wer * 1.3)  # Add 30% more space at the top
    
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        accuracy = max(0, (1 - mean_val) * 100)  # Convert WER to accuracy percentage
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'WER: {mean_val:.2f}\nAcc: {accuracy:.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('wer_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 Saved WER comparison as 'wer_comparison.png'")
    plt.show()
    
    # 2. Box plot for WER distribution (better for error rate data)
    plt.figure(figsize=(10, 6))
    
    wer_data = [df[system].dropna() for system in systems]
    box_plot = plt.boxplot(wer_data, labels=system_names, patch_artist=True)
    
    # Color boxes based on median performance
    medians = [data.median() for data in wer_data]
    sorted_med_indices = np.argsort(medians)
    box_colors = ['lightgreen' if i == sorted_med_indices[0] else 'lightblue' if i == sorted_med_indices[1] else 'lightcoral' 
                  for i in range(len(medians))]
    
    for patch, color in zip(box_plot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('📦 WER Distribution by System (Lower is Better)', fontsize=14, fontweight='bold')
    plt.ylabel('Word Error Rate (WER)')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('wer_distribution.png', dpi=300, bbox_inches='tight')
    print("💾 Saved WER distribution as 'wer_distribution.png'")
    plt.show()
    
    # 3. Success rate plot (percentage of samples with WER <= threshold)
    plt.figure(figsize=(10, 6))
    
    thresholds = [0, 0.5, 1.0, 1.5, 2.0]
    
    for system, color in zip(systems, colors):
        success_rates = []
        scores = df[system].dropna()
        
        for threshold in thresholds:
            success_rate = (scores <= threshold).mean() * 100
            success_rates.append(success_rate)
        
        plt.plot(thresholds, success_rates, 'o-', color=color, linewidth=2, 
                label=system.replace('_', ' ').title(), markersize=6)
    
    plt.title('🎯 Success Rate vs WER Threshold', fontsize=14, fontweight='bold')
    plt.xlabel('WER Threshold')
    plt.ylabel('Success Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    
    # Add percentage labels
    for i, threshold in enumerate(thresholds):
        plt.axvline(x=threshold, color='gray', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wer_success_rate.png', dpi=300, bbox_inches='tight')
    print("💾 Saved success rate plot as 'wer_success_rate.png'")
    plt.show()
    
    print("✅ All WER plots created successfully!")

def print_insights(df):
    """Print key insights and recommendations for WER data 💡"""
    print("\n" + "="*60)
    print("💡 WER ANALYSIS INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    systems = ['roboshaul_nakdimon', 'phonikud', 'phonikud_enhanced']
    means = [df[system].dropna().mean() for system in systems]
    
    # Find best and worst performing systems (lower WER is better)
    best_idx = np.argmin(means)
    worst_idx = np.argmax(means)
    
    print(f"\n🏆 BEST PERFORMER: {systems[best_idx].replace('_', ' ').title()}")
    print(f"   Average WER: {means[best_idx]:.2f}")
    
    print(f"\n📈 NEEDS IMPROVEMENT: {systems[worst_idx].replace('_', ' ').title()}")
    print(f"   Average WER: {means[worst_idx]:.2f}")
    
    # Calculate improvement potential
    improvement = means[worst_idx] - means[best_idx]
    improvement_pct = (improvement / means[worst_idx]) * 100 if means[worst_idx] > 0 else 0
    print(f"\n🎯 IMPROVEMENT POTENTIAL: {improvement:.2f} WER points ({improvement_pct:.1f}% reduction)")
    
    # Consistency analysis
    stds = [df[system].dropna().std() for system in systems]
    most_consistent_idx = np.argmin(stds)
    
    print(f"\n⚖️  MOST CONSISTENT: {systems[most_consistent_idx].replace('_', ' ').title()}")
    print(f"   Standard Deviation: {stds[most_consistent_idx]:.2f}")
    
    # WER threshold analysis
    print(f"\n📊 WER PERFORMANCE BREAKDOWN:")
    thresholds = [0, 0.5, 1.0, 2.0]
    threshold_labels = ["Perfect (0)", "Excellent (≤0.5)", "Good (≤1.0)", "Acceptable (≤2.0)"]
    
    for i, system in enumerate(systems):
        scores = df[system].dropna()
        print(f"\n   🔹 {system.replace('_', ' ').title()}:")
        
        for threshold, label in zip(thresholds, threshold_labels):
            pct = (scores <= threshold).mean() * 100
            print(f"      {label}: {pct:.1f}%")
    
    # Overall system ranking
    print(f"\n🏅 SYSTEM RANKING (Best to Worst):")
    sorted_systems = [(systems[i], means[i]) for i in np.argsort(means)]
    for rank, (system, mean_wer) in enumerate(sorted_systems, 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        print(f"   {emoji} #{rank}: {system.replace('_', ' ').title()} (WER: {mean_wer:.2f})")

def main():
    """Main WER analysis function 🚀"""
    print("🎯 Starting WER (Word Error Rate) Analysis")
    print("=" * 50)
    
    try:
        # Load data
        df = load_data()
        
        # Print summary statistics
        print_summary_stats(df)
        
        # Create visualizations
        create_visualizations(df)
        
        # Print insights
        print_insights(df)
        
        print("\n" + "="*60)
        print("✅ Analysis Complete! 🎉")
        print("📊 Check the generated PNG files for visualizations")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
