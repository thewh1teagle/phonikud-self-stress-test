#!/usr/bin/env python3
"""
WER Analysis Script 📊
Analyzes Word Error Rate (WER) results from different TTS models/systems
Lower WER values indicate better performance
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path

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
    
    # Get system names dynamically from CSV columns (exclude 'id')
    systems = [col for col in results_df.columns if col != 'id']
    print(f"📋 Found systems: {systems}")
    
    # Clean results data
    print(f"🧹 Cleaning results data...")
    
    # Remove duplicate rows (keep first occurrence)
    results_df = results_df.drop_duplicates(subset=['id'], keep='first')
    print(f"   Removed duplicates, {len(results_df)} unique results remain")
    
    # Handle missing values in system columns
    for system in systems:
        results_df[system] = results_df[system].fillna(0)
    
    # Create matching ID format: llm1_1, llm1_2, etc.
    sentences_df['result_id'] = 'llm1_' + sentences_df['id'].astype(str)
    
    # Merge dataframes
    merged_df = sentences_df.merge(results_df, left_on='result_id', right_on='id', how='inner', suffixes=('', '_result'))
    print(f"✅ Merged {len(merged_df)} records successfully")
    
    # Count Hebrew words in each sentence
    merged_df['words_count'] = merged_df['phrase'].apply(count_hebrew_words)
    
    # Calculate actual WER (Word Error Rate) = incorrect_words_count / total_words_count
    for system in systems:
        # Create new column with actual WER values
        merged_df[f'{system}_wer'] = merged_df.apply(
            lambda row: row[system] / row['words_count'] if row['words_count'] > 0 else 0, 
            axis=1
        )
    
    return merged_df, systems

def create_json_report(df, systems):
    """Create detailed JSON report with per-sentence analysis 📋"""
    print("\n📝 Creating JSON report...")
    
    # Calculate average WER and total errors for each system
    avg_wer = {}
    total_errors = {}
    for system in systems:
        wer_column = f'{system}_wer'
        avg_wer[system] = float(df[wer_column].dropna().mean())
        total_errors[system] = int(df[system].sum())  # Sum of all incorrect words
    
    # Create detailed report structure
    report = {
        "summary": {
            "total_sentences": len(df),
            "total_errors": total_errors,
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
    print(f"\nTotal errors by system:")
    for system, errors in total_errors.items():
        print(f"  {system}: {errors} errors")
    print(f"\nAverage WER by system:")
    for system, wer in avg_wer.items():
        print(f"  {system}: {wer:.3f}")
    
    return report

def print_summary_stats(df, systems):
    """Print detailed WER summary statistics with emojis 📈"""
    print("\n" + "="*60)
    print("📊 WER ANALYSIS SUMMARY")
    print("="*60)
    
    system_emojis = ['🎯', '⚡', '🤖', '🔊', '📡', '🎵', '🔮']  # More emojis for flexibility
    
    for i, system in enumerate(systems):
        emoji = system_emojis[i % len(system_emojis)]  # Cycle through emojis
        print(f"\n{emoji} {system.upper().replace('_', ' ').replace('-', ' ')}")
        print("-" * 40)
        
        # Use calculated WER values instead of raw incorrect word counts
        wer_column = f'{system}_wer'
        values = df[wer_column].dropna()
        mean_wer = values.mean()
        std_wer = values.std()
        min_wer = values.min()
        max_wer = values.max()
        median_wer = values.median()
        total_errors = df[system].sum()
        
        print(f"📊 Mean WER: {mean_wer:.3f}")
        print(f"📏 Std Dev: {std_wer:.3f}")
        print(f"🎯 Median WER: {median_wer:.3f}")
        print(f"✅ Best WER: {min_wer} (lower is better)")
        print(f"❌ Worst WER: {max_wer}")
        print(f"🔢 Total Samples: {len(values)}")
        print(f"🚨 Total Errors: {total_errors}")
        
        # WER performance tiers
        perfect_pct = (values == 0).mean() * 100
        excellent_pct = (values <= 0.5).mean() * 100
        good_pct = (values <= 1.0).mean() * 100
        acceptable_pct = (values <= 2.0).mean() * 100
        
        print(f"🏆 Perfect (WER=0): {perfect_pct:.1f}%")
        print(f"⭐ Excellent (≤0.5): {excellent_pct:.1f}%")
        print(f"👍 Good (≤1.0): {good_pct:.1f}%")
        print(f"📝 Acceptable (≤2.0): {acceptable_pct:.1f}%")

def print_insights(df, systems):
    """Print key insights and recommendations for WER data 💡"""
    print("\n" + "="*60)
    print("💡 WER ANALYSIS INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    means = [df[f'{system}_wer'].dropna().mean() for system in systems]
    
    # Find best and worst performing systems (lower WER is better)
    best_idx = np.argmin(means)
    worst_idx = np.argmax(means)
    
    print(f"\n🏆 BEST PERFORMER: {systems[best_idx].replace('_', ' ').replace('-', ' ').title()}")
    print(f"   Average WER: {means[best_idx]:.3f}")
    
    print(f"\n📈 NEEDS IMPROVEMENT: {systems[worst_idx].replace('_', ' ').replace('-', ' ').title()}")
    print(f"   Average WER: {means[worst_idx]:.3f}")
    
    # Calculate improvement potential
    improvement = means[worst_idx] - means[best_idx]
    improvement_pct = (improvement / means[worst_idx]) * 100 if means[worst_idx] > 0 else 0
    print(f"\n🎯 IMPROVEMENT POTENTIAL: {improvement:.3f} WER points ({improvement_pct:.1f}% reduction)")
    
    # Overall system ranking
    print(f"\n🏅 SYSTEM RANKING (Best to Worst):")
    sorted_systems = [(systems[i], means[i]) for i in np.argsort(means)]
    for rank, (system, mean_wer) in enumerate(sorted_systems, 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📍"
        print(f"   {emoji} #{rank}: {system.replace('_', ' ').replace('-', ' ').title()} (WER: {mean_wer:.3f})")

def main():
    """Main WER analysis function 🚀"""
    print("🎯 Starting WER (Word Error Rate) Analysis")
    print("=" * 50)
    
    try:
        # Load data
        df, systems = load_and_merge_data()
        
        # Print summary statistics
        print_summary_stats(df, systems)
        
        # Print insights
        print_insights(df, systems)
        
        # Create JSON report
        create_json_report(df, systems)
        
        print("\n" + "="*60)
        print("✅ Analysis Complete! 🎉")
        print("📊 Check the generated JSON report")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
