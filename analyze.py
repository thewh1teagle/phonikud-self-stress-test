#!/usr/bin/env python3
"""
WER Analysis Script 📊
Analyzes Word Error Rate (WER) results from different TTS models/systems
Lower WER values indicate better performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import re
from pathlib import Path
from math import pi

# Set up matplotlib style  
plt.style.use('default')

def count_hebrew_words(text):
    """Count Hebrew words in text using Unicode range \u0590-\u05f4"""
    if pd.isna(text) or text == '':
        return 0
    
    # Hebrew character range: \u0590-\u05f4
    hebrew_pattern = r'[\u0590-\u05f4]+'
    hebrew_words = re.findall(hebrew_pattern, text)
    return len(hebrew_words)

def load_and_merge_data():
    """Load and merge sentence data with WER results 📂"""
    print("🔍 Loading and merging data...")
    
    # Read sentence data
    sentences_df = pd.read_csv('llm1.csv')
    print(f"✅ Loaded {len(sentences_df)} sentences from llm1.csv")
    
    # Read WER results with error handling for malformed lines
    try:
        results_df = pd.read_csv('results.csv', on_bad_lines='skip')
        print(f"✅ Loaded {len(results_df)} results from results.csv")
    except Exception as e:
        print(f"⚠️  Warning reading results.csv: {e}")
        # Try reading with more permissive settings
        results_df = pd.read_csv('results.csv', sep=',', on_bad_lines='skip', engine='python')
        print(f"✅ Loaded {len(results_df)} results from results.csv (with error handling)")
    
    # Clean results data
    print(f"🧹 Cleaning results data...")
    
    # Remove duplicate rows (keep first occurrence)
    results_df = results_df.drop_duplicates(subset=['id'], keep='first')
    print(f"   Removed duplicates, {len(results_df)} unique results remain")
    
    # Handle missing values in phonikud_enhanced column
    results_df['phonikud_enhanced'] = results_df['phonikud_enhanced'].fillna(0)
    
    # Create matching ID format: llm1_1, llm1_2, etc.
    sentences_df['result_id'] = 'llm1_' + sentences_df['id'].astype(str)
    
    # Merge dataframes
    merged_df = sentences_df.merge(results_df, left_on='result_id', right_on='id', how='inner', suffixes=('', '_result'))
    print(f"✅ Merged {len(merged_df)} records successfully")
    
    # Count Hebrew words in each sentence
    merged_df['words_count'] = merged_df['phrase'].apply(count_hebrew_words)
    
    # Calculate actual WER (Word Error Rate) = incorrect_words_count / total_words_count
    systems = ['roboshaul_nakdimon', 'phonikud', 'phonikud_enhanced']
    for system in systems:
        # Create new column with actual WER values
        merged_df[f'{system}_wer'] = merged_df.apply(
            lambda row: row[system] / row['words_count'] if row['words_count'] > 0 else 0, 
            axis=1
        )
    
    return merged_df

def create_json_report(df):
    """Create detailed JSON report with per-sentence analysis 📋"""
    print("\n📝 Creating JSON report...")
    
    systems = ['roboshaul_nakdimon', 'phonikud', 'phonikud_enhanced']
    
    # Calculate average WER for each system (using calculated WER values)
    avg_wer = {}
    for system in systems:
        wer_column = f'{system}_wer'
        avg_wer[system] = float(df[wer_column].dropna().mean())
    
    # Create detailed report structure
    report = {
        "summary": {
            "total_sentences": len(df),
            "average_wer": avg_wer,
            "systems_analyzed": systems
        },
        "sentences": []
    }
    
    # Add individual sentence details
    for _, row in df.iterrows():
        sentence_data = {
            "id": row['result_id'],  # Use result_id (llm1_1, llm1_2, etc.) instead of numeric id
            "sentence": row['phrase'],
            "words_count": int(row['words_count']),
            "wer": {}
        }
        
        # Add incorrect word count and calculated WER for each system
        for system in systems:
            incorrect_count = row[system]
            wer_value = row[f'{system}_wer']
            
            if pd.notna(incorrect_count) and pd.notna(wer_value):
                sentence_data["wer"][system] = {
                    "incorrect_words": int(incorrect_count),
                    "wer": float(wer_value)
                }
            else:
                sentence_data["wer"][system] = {
                    "incorrect_words": None,
                    "wer": None
                }
        
        report["sentences"].append(sentence_data)
    
    # Save to JSON file
    with open('report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("💾 Saved detailed report as 'report.json'")
    
    # Print summary
    print(f"\n📊 REPORT SUMMARY:")
    print(f"Total sentences analyzed: {len(df)}")
    print(f"Average Hebrew words per sentence: {df['words_count'].mean():.1f}")
    print(f"\nAverage WER by system:")
    for system, wer in avg_wer.items():
        print(f"  {system}: {wer:.3f}")
    
    return report

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

        
        # Use calculated WER values instead of raw incorrect word counts
        wer_column = f'{system}_wer'
        values = df[wer_column].dropna()
        mean_wer = values.mean()
        std_wer = values.std()
        min_wer = values.min()
        max_wer = values.max()
        median_wer = values.median()
        
        print(f"📊 Mean WER: {mean_wer:.3f}")
        print(f"📏 Std Dev: {std_wer:.3f}")
        print(f"🎯 Median WER: {median_wer:.3f}")
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
    """Create WER visualization 📈"""
    print("\n🎨 Creating WER visualization...")
    
    systems = ['roboshaul_nakdimon', 'phonikud', 'phonikud_enhanced']
    
    # Bar chart of average WER (lower is better)
    plt.figure(figsize=(10, 6))
    means = [df[f'{system}_wer'].dropna().mean() for system in systems]
    system_names = []
    for s in systems:
        name = s.replace('_', ' ').title()
        if 'phonikud' in s.lower():
            name += ' (ours)'
        system_names.append(name)
    
    # Sort systems with roboshaul at the end
    # First, get phonikud systems sorted by performance
    phonikud_data = [(mean, name) for mean, name in zip(means, system_names) if 'roboshaul' not in name.lower()]
    roboshaul_data = [(mean, name) for mean, name in zip(means, system_names) if 'roboshaul' in name.lower()]
    
    # Sort phonikud systems by performance (best to worst)
    phonikud_sorted = sorted(phonikud_data, key=lambda x: x[0])
    
    # Combine: phonikud systems first, then roboshaul
    sorted_data = phonikud_sorted + roboshaul_data
    sorted_means, sorted_names = zip(*sorted_data)
    
    # Color bars based on WER performance
    # Find the best, middle, and worst WER values
    all_wers = list(sorted_means)
    best_wer = min(all_wers)
    worst_wer = max(all_wers)
    
    bar_colors = []
    for wer_val in sorted_means:
        if wer_val == best_wer:
            bar_colors.append('green')  # Best WER
        elif wer_val == worst_wer:
            bar_colors.append('yellow')  # Worst WER (roboshaul)
        else:
            bar_colors.append('red')  # Middle WER
    
    bars = plt.bar(sorted_names, sorted_means, color=bar_colors, alpha=0.7, edgecolor='black')
    plt.ylabel('← Word Error Rate (WER)', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    # Adjust y-axis limits to give more room for labels
    max_wer = max(sorted_means)
    plt.ylim(0, max_wer * 2.0)  # Double the space at the top for labels
    
    for bar, mean_val in zip(bars, sorted_means):
        height = bar.get_height()
        # Use a relative offset based on the scale instead of fixed 0.05
        offset = max_wer * 0.05  # 5% of the maximum WER value
        plt.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{mean_val:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('wer_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 Saved WER comparison as 'wer_comparison.png'")
    plt.show()
    
    print("✅ WER plot created successfully!")

def create_outstanding_visualizations(df):
    """Create improved excellence visualization that makes the BEST performer visually outstanding! 🏆"""
    print("\n✨ Creating outstanding WER excellence visualization...")
    
    systems = ['roboshaul_nakdimon', 'phonikud', 'phonikud_enhanced']
    
    # Get the data
    means = [df[f'{system}_wer'].dropna().mean() for system in systems]
    system_names = []
    for s in systems:
        name = s.replace('_', ' ').title()
        if 'phonikud' in s.lower():
            name += ' ⭐'  # Mark your systems
        system_names.append(name)
    
    # IMPROVED INVERTED WER BAR CHART - Shows "how much BETTER" each system is
    plt.figure(figsize=(14, 8))
    
    # Calculate "excellence score" = max_wer - current_wer (higher is better)
    max_wer = max(means)
    excellence_scores = [max_wer - wer for wer in means]
    
    # Create horizontal bar chart with improved styling
    bars = plt.barh(system_names, excellence_scores, height=0.6,
                    color=['#FFD700' if 'phonikud' in name.lower() else '#B0B0B0' for name in system_names],
                    edgecolor='#333333', linewidth=2, alpha=0.9)
    
    # Make the best one really stand out with gradient effect
    best_idx = np.argmax(excellence_scores)
    bars[best_idx].set_color('#FFD700')  # Gold for winner
    bars[best_idx].set_alpha(1.0)
    bars[best_idx].set_linewidth(3)
    bars[best_idx].set_edgecolor('#B8860B')  # Dark gold edge
    
    # Style the non-winners more subtly
    for i, bar in enumerate(bars):
        if i != best_idx:
            bar.set_color('#D3D3D3')  # Light gray
            bar.set_alpha(0.7)
    
    # Clean axis styling
    plt.xlabel('Excellence Score (Higher = Better Performance)', fontsize=16, fontweight='bold', color='#333333')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    
    # Improved value labels with better positioning
    max_score = max(excellence_scores)
    for i, (bar, score, wer) in enumerate(zip(bars, excellence_scores, means)):
        width = bar.get_width()
        y_pos = bar.get_y() + bar.get_height()/2
        
        # Position labels smartly
        if width > max_score * 0.6:  # If bar is long, put label inside
            label_x = width - max_score * 0.05
            ha = 'right'
            color = '#333333' if i == best_idx else '#666666'
        else:  # If bar is short, put label outside
            label_x = width + max_score * 0.02
            ha = 'left'
            color = '#333333'
        
        plt.text(label_x, y_pos,
                f'WER: {wer:.3f}\nExcellence: {score:.3f}', 
                ha=ha, va='center', fontweight='bold', fontsize=12, color=color)
        
        # Add winner badge for the best
        if i == best_idx:
            plt.text(width + max_score * 0.15, y_pos, 
                    '🏆', ha='left', va='center', fontsize=20)
    
    # Subtle grid
    plt.grid(True, alpha=0.2, axis='x', linestyle='--')
    
    # Better spacing and layout
    plt.gca().invert_yaxis()  # Best performer at top
    plt.margins(x=0.15)  # Add margin for labels
    plt.tight_layout()
    
    # Save with higher quality
    plt.savefig('wer_excellence_chart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("💾 Saved improved excellence chart as 'wer_excellence_chart.png'")
    plt.show()
    
    print("🎉 Outstanding visualization created successfully!")

def print_insights(df):
    """Print key insights and recommendations for WER data 💡"""
    print("\n" + "="*60)
    print("💡 WER ANALYSIS INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    systems = ['roboshaul_nakdimon', 'phonikud', 'phonikud_enhanced']
    means = [df[f'{system}_wer'].dropna().mean() for system in systems]
    
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
    stds = [df[f'{system}_wer'].dropna().std() for system in systems]
    most_consistent_idx = np.argmin(stds)
    
    print(f"\n⚖️  MOST CONSISTENT: {systems[most_consistent_idx].replace('_', ' ').title()}")
    print(f"   Standard Deviation: {stds[most_consistent_idx]:.2f}")
    
    # WER threshold analysis
    print(f"\n📊 WER PERFORMANCE BREAKDOWN:")
    thresholds = [0, 0.5, 1.0, 2.0]
    threshold_labels = ["Perfect (0)", "Excellent (≤0.5)", "Good (≤1.0)", "Acceptable (≤2.0)"]
    
    for i, system in enumerate(systems):
        scores = df[f'{system}_wer'].dropna()
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
        df = load_and_merge_data()
        
        # Print summary statistics
        print_summary_stats(df)
        
        # Create visualizations
        create_visualizations(df)
        
        # Print insights
        print_insights(df)
        
        # Create JSON report
        create_json_report(df)
        
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
