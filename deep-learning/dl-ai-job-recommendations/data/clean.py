import pandas as pd
import numpy as np
import re

def clean_df(input_path, output_path='job_postings_cleaned.csv', max_length=200):
    df = pd.read_csv(input_path)

    # Compute annual salary

    def compute_salary(row, hours_per_year=2080):
        pay_period = str(row.get('pay_period', '')).upper()
        if not np.isnan(row.get('med_salary', np.nan)):
            salary_val = row['med_salary']
        elif not np.isnan(row.get('min_salary', np.nan)) and not np.isnan(row.get('max_salary', np.nan)):
            salary_val = (row['min_salary'] + row['max_salary']) / 2.0
        else:
            return np.nan
        if pay_period == 'HOURLY':
            salary_val *= hours_per_year
        return salary_val

    df['annual_salary'] = df.apply(compute_salary, axis=1)
    df = df.dropna(subset=['annual_salary']).reset_index(drop=True)

    # Select features and initial cleaning

    # Drop high-missing, ID/URL, and salary component columns
    drop_cols = [
        'skills_desc', 'closed_time', 'med_salary',
        'max_salary', 'min_salary',
        'job_id', 'company_id', 'job_posting_url', 'application_url', 'posting_domain'
    ]
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Convert timestamp columns from ms to datetime
    for col in ['listed_time', 'original_listed_time', 'expiry']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit='ms')

    # Extract state from location
    df['state'] = df['location'].str.split(',').str[-1].str.strip().fillna('UNKNOWN')

    # Normalize categorical columns
    for col in df.select_dtypes('object').columns:
        df[col] = df[col].fillna('UNKNOWN').str.strip()

    # Fill numeric NaNs
    for col in df.select_dtypes(['float64','int64']).columns:
        df[col] = df[col].fillna(0)

    # Drop columns and clean text fields

    # Drop unwanted columns
    df.drop(columns=['original_listed_time', 'expiry', 'currency', 'compensation_type', 'applies', 'listed_time'],
            errors='ignore', inplace=True)

    # Clean state codes to two-letter uppercase or UNKNOWN
    def clean_state(s):
        if isinstance(s, str) and len(s.strip()) == 2 and s.strip().isalpha():
            return s.strip().upper()
        return 'UNKNOWN'

    df['state'] = df['state'].apply(clean_state)

    # Clean descriptions: strip HTML, remove URLs, lowercase, remove punctuation, collapse whitespace
    def clean_description(text):
        text = str(text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'http\S+', ' ', text)
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    df['description'] = df['description'].apply(clean_description)

    # Remove short descriptions and truncate to max_length tokens

    def tokenize(text):
        return text.split()

    # Filter out descriptions <20 tokens
    df = df[df['description'].apply(lambda x: len(tokenize(x)) >= 20)].reset_index(drop=True)

    # Truncate to max_length tokens
    def clip_to_max_length(text):
        tokens = tokenize(text)
        return ' '.join(tokens[:max_length]) if len(tokens) > max_length else text

    df['description'] = df['description'].apply(clip_to_max_length)
    df.to_csv(output_path, index=False)

    return df