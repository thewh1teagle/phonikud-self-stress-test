import pandas as pd
import re
import json

target_csv = 'data/result_kobi.csv'

def count_hebrew_words(text):
    """Count Hebrew words in text using Unicode range \u0590-\u05f4"""
    if pd.isna(text) or text == '':
        return 0
    
    # Hebrew character range: \u0590-\u05f4
    hebrew_pattern = r'[\u0590-\u05f4]+'
    hebrew_words = re.findall(hebrew_pattern, text)
    return len(hebrew_words)


def main():
    # Read text_llm1.csv and create phrase dictionary
    text_df = pd.read_csv('data/text_llm1.csv')
    phrase_dict = dict(zip(text_df['id'], text_df['phrase']))
    
    # Read result file with error handling
    try:
        result_df = pd.read_csv(target_csv, on_bad_lines='skip')
    except:
        # If that fails, try with different parameters
        result_df = pd.read_csv(target_csv, sep=',', on_bad_lines='skip', engine='python')
    
    # Get model names (all columns except the first one which is id)
    model_names = result_df.columns[1:].tolist()
    
    # Calculate total errors and WER for each model
    total_errors = {}
    average_wer = {}
    total_sentences = len(result_df)
    
    for model_name in model_names:
        model_total_errors = result_df[model_name].sum()
        total_words = sum(count_hebrew_words(phrase_dict[row['id']]) 
                         for _, row in result_df.iterrows() 
                         if row['id'] in phrase_dict)
        wer = model_total_errors / total_words if total_words > 0 else 0
        
        total_errors[model_name] = int(model_total_errors)
        average_wer[model_name] = wer
    
    # Create sentences data
    sentences = []
    for _, row in result_df.iterrows():
        sentence_id = row['id']
        if sentence_id in phrase_dict:
            sentence_text = phrase_dict[sentence_id]
            word_count = count_hebrew_words(sentence_text)
            
            sentence_data = {
                'id': sentence_id,
                'sentence': sentence_text,
                'word_count': word_count,
                'models': {}
            }
            
            for model_name in model_names:
                errors = int(row[model_name])
                wer = errors / word_count if word_count > 0 else 0
                sentence_data['models'][model_name] = {
                    'errors': errors,
                    'wer': round(wer, 4)
                }
            
            sentences.append(sentence_data)
    
    # Create final report
    report = {
        'summary': {
            'total_sentences': total_sentences,
            'total_errors': total_errors,
            'average_wer': average_wer
        },
        'sentences': sentences
    }
    
    # Write report.json
    with open('report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"Report generated with {len(sentences)} sentences and {len(model_names)} models")
    print(f"Models: {', '.join(model_names)}")


if __name__ == "__main__":
    main()