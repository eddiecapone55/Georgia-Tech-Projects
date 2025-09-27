import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, DataCollatorWithPadding

class JobTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Drop rows with missing or too-short descriptions
        self.df = self.df[self.df['description'].str.split().str.len() >= 20]
        # Truncate to max_length tokens
        def clip(text):
            toks = text.split()
            return ' '.join(toks[:self.max_length])
        self.df['description'] = self.df['description'].apply(clip)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text  = f"TITLE: {row['title']}. "
        text += f"PAY PERIOD: {row['pay_period']}. "
        text += f"STATE: {row['state']}. "
        text += f"VIEWS: {row['views']}. "
        text += f"WORK TYPE: {row['formatted_work_type']}. "
        text += f"REMOTE ALLOWED: {row['remote_allowed']}. "
        text += f"APPLICATION TYPE: {row['application_type']}. "
        text += f"EXPERIENCE LEVEL: {row['formatted_experience_level']}. "
        text += f"SPONSORED: {row['sponsored']}. "
        text += f"DESCRIPTION: {row['description']}."

        # tokenize the combined text
        enc = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids':      enc.input_ids.squeeze(0),
            'attention_mask': enc.attention_mask.squeeze(0),
            'labels':         torch.tensor(row['salary_quantile'], dtype=torch.long)
        }
class JobFusionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, cat_cols):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cat_cols = cat_cols
        # build category maps
        self.cat_maps = {
            col: {v:i for i,v in enumerate(sorted(self.df[col].unique()))}
            for col in cat_cols
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = f"{row['title']} {self.tokenizer.sep_token} {row['description']}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids      = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        struct = torch.tensor([
            self.cat_maps[col][row[col]] for col in self.cat_cols
        ], dtype=torch.long)

        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'struct':         struct,
            'label':          torch.tensor(row['salary_quantile'], dtype=torch.long)
        }


def load_data(path, cfg):
    df = pd.read_csv(path)
    # build quantile labels
    df['salary_quantile'] = pd.qcut(
        df['annual_salary'],
        q=cfg['data']['salary_classes'],
        labels=list(range(cfg['data']['salary_classes']))
    ).astype(int)

    columns_to_drop = ['annual_salary', 
                       'normalized_salary',
                       'fips', 
                       'company_name',
                       'work_type',
                       'zip_code']

    df = df.drop(columns = columns_to_drop)
    
    train_val, test = train_test_split(
        df,
        test_size=cfg['data']['test_size'],
        stratify=df['salary_quantile'],
        random_state=cfg['train']['seed']
    )
    train, val = train_test_split(
        train_val,
        test_size=cfg['data']['val_size'],
        stratify=train_val['salary_quantile'],
        random_state=cfg['train']['seed']
    )
    return train, val, test
