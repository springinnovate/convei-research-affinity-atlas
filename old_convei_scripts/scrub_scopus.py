import numpy as np
import re
import glob

import pandas


def main():
    # matches = re.findall(
    #     pattern,
    #     'B. Rouet-Leduc; richp@gmail Disaster Prevention Research Institute, Kyoto University, Japan; email: rouetleduc.bertrand.5s@kyoto-u.ac.jp',
    #     re.IGNORECASE)
    # print(matches)

    df = pandas.DataFrame()
    for csv_path in glob.glob('data/scopus_2024_05_28/*.csv'):
        print(csv_path)
        df = pandas.concat(
            (df, pandas.read_csv(csv_path, na_values=['', ' '], low_memory=False)),
            ignore_index=True)
    df['Correspondence Address'] = df['Correspondence Address'].fillna('')
    original_count = len(df)
    print(f'original count: {original_count}')
    df = df.drop_duplicates(subset='DOI', keep='first')
    duplciates_dropped_count = len(df)
    print(f'DOI duplicates dropped: {duplciates_dropped_count}')
    email_pattern = r'[\w\.-]+@[\w\.-]+\.(?:edu|org|gov)'
    df = df[df['Correspondence Address'].str.contains(
        email_pattern, regex=True)]
    missing_email_dropped = len(df)
    print(f'Empty correspondance no/email dropped: {missing_email_dropped} {100*missing_email_dropped/duplciates_dropped_count:.2f}%')
    df['Emails'] = df['Correspondence Address'].str.findall(email_pattern)
    all_emails = [email for sublist in df['Emails'] for email in sublist]
    valid_email_count = len(all_emails)
    print(
        f'Unique emails: {valid_email_count} '
        f'{100*valid_email_count/missing_email_dropped:.2f}%')
    print(df['Emails'])
    # filter ema


if __name__ == '__main__':
    main()
