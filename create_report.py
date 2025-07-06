import pandas as pd
import re
import json

target_csv = 'data/result_yakov.csv'

def count_hebrew_words(text):
    return len(re.findall(r'[\u0590-\u05f4]+', str(text))) if pd.notna(text) else 0

def main():
    text_df = pd.read_csv('data/text_llm1.csv')
    phrase_dict = dict(zip(text_df['id'], text_df['phrase']))
    
    result_df = pd.read_csv(target_csv, on_bad_lines='skip')
    model_names = result_df.columns[1:].tolist()
    
    sentences = []
    for _, row in result_df.iterrows():
        if row['id'] in phrase_dict:
            sentence_text = phrase_dict[row['id']]
            word_count = count_hebrew_words(sentence_text)
            sentences.append({
                'id': row['id'],
                'sentence': sentence_text,
                'word_count': word_count,
                'models': {model: {'errors': int(row[model]), 
                                 'wer': round(int(row[model]) / word_count if word_count > 0 else 0, 4)} 
                          for model in model_names}
            })
    
    total_errors = {model: sum(s['models'][model]['errors'] for s in sentences) for model in model_names}
    total_words = sum(s['word_count'] for s in sentences)
    average_wer = {model: total_errors[model] / total_words for model in model_names}
    
    report = {
        'summary': {
            'total_sentences': len(sentences),
            'total_words': total_words,
            'total_errors': total_errors,
            'average_wer': average_wer
        },
        'sentences': sentences
    }
    
    with open('report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"Report generated with {len(sentences)} sentences and {len(model_names)} models")
    print('Created report.json')

if __name__ == "__main__":
    main()