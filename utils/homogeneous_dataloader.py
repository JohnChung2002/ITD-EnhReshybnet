import pandas as pd
import numpy as np
import itertools
import pickle
from datetime import datetime
from tqdm import tqdm

def load_homogeneous_cert_data(data_path="./data", method="undersampling_Reshybnet", version="r4.2", year=None, month=None, multiple_months=None):
    with open(f"{data_path}/users.csv", "r") as f:
        users = pd.read_csv(f)

    with open(f"{data_path}/day{version}.csv", "r") as f:
        day = pd.read_csv(f)
    day['key_date'] = day['starttime'].apply(lambda ts: str(datetime.fromtimestamp(ts).date()))

    with open(f"{data_path}/email.csv", "r") as f:
        email = pd.read_csv(f)

    email.drop(['pc','from', 'size', 'attachments', 'content'], axis=1, inplace=True)
    #split by semicolon and remove emails that do not end in @dtaa.com
    email['to'] = email['to'].str.split(';').apply(lambda x: [] if not isinstance(x, list) else [i for i in x if '@dtaa.com' in i])
    email['cc'] = email['cc'].str.split(';').apply(lambda x: [] if not isinstance(x, list) else [i for i in x if '@dtaa.com' in i])
    email['bcc'] = email['bcc'].str.split(';').apply(lambda x: [] if not isinstance(x, list) else [i for i in x if '@dtaa.com' in i])
    #combine to cc and bcc columns
    email['to'] = email['to'] + email['cc'] + email['bcc']
    email.drop(['cc', 'bcc'], axis=1, inplace=True)
    # drop rows where to is empty list
    email = email[email['to'].map(len) > 0]
    email['user_id'] = email['user'].apply(lambda x: users[users['user_id'] == x].index[0])
    email['key_date'] = pd.to_datetime(email['date']).dt.date.astype(str)

    with open(f"{data_path}/logon.csv", "r") as f:
        logon = pd.read_csv(f)
    
    logon['key_date'] = pd.to_datetime(logon['date']).dt.date.astype(str)
    pcs = logon.groupby(['user', 'key_date'])['pc'].apply(lambda x: list(set(x)))
    pcs = pcs.reset_index()
    pcs['user_id'] = pcs['user'].map(lambda x: users[users['user_id'] == x].index[0])

    if method == 'undersampling_Reshybnet':
        with open(f"{data_path}/user_date.pkl", "rb") as f:
            user_date = pickle.load(f)
        user_date['key_date'] = pd.to_datetime(user_date['date']).dt.date
        user_date['user_id'] = user_date['user_index'].apply(lambda x: users[users['user_id'] == x].index[0])
        # Create a key column in both DataFrames combining user and date.
        # For day, use the 'user' column; for user_date, we assume the user identifier is 'user_id'.
        day['merge_key'] = day['user'].astype(str) + '_' + day['key_date'].astype(str)
        day['merge_key'] = day['merge_key'].apply(lambda x: x.strip())
        user_date['merge_key'] = user_date['user_id'].astype(str) + '_' + user_date['key_date'].astype(str)
        user_date['merge_key'] = user_date['merge_key'].apply(lambda x: x.strip())

        # Now create the mask: it will be True for rows in 'day' whose key exists in 'user_date'
        mask = day['merge_key'].isin(user_date['merge_key'])

        # Optionally, filter the DataFrame:
        filtered_day = day[mask]
        filtered = filtered_day.drop(columns=['merge_key']).reset_index(drop=True)
    elif method == "undersampling":
        # Separate the data into two groups based on 'insider' value
        insider = day[day['insider'] != 0]
        normal = day[day['insider'] == 0]

        # Randomly sample 954 entries from each group
        sampled_insider_1 = insider.sample(n=954, random_state=42)
        sampled_insider_0 = normal.sample(n=954, random_state=42)

        filtered = pd.concat([sampled_insider_1, sampled_insider_0], ignore_index=True).reset_index(drop=True)
    elif method == "one_month" and multiple_months is None:
        year_month = f"{year}-{month:02d}"
        filtered = day[day['key_date'].str.startswith(year_month)].reset_index(drop=True)
        email = email[email['key_date'].str.startswith(year_month)]
        pcs = pcs[pcs['key_date'].str.startswith(year_month)]
    elif method == "one_month" and multiple_months is not None:
        # multiple months is how many months to go forward from the given month
        year_month = f"{year}-{month:02d}"
        print(f"Processing month: {month}")
        filtered = day[day['key_date'].str.startswith(year_month)].reset_index(drop=True)
        email = email[email['key_date'].str.startswith(year_month)]
        pcs = pcs[pcs['key_date'].str.startswith(year_month)]
        for i in range(1, multiple_months + 1):
            print(f"Processing month: {month + i}")
            year_month = f"{year}-{month + i:02d}"
            filtered = pd.concat([filtered, day[day['key_date'].str.startswith(year_month)]], ignore_index=True)
            email = pd.concat([email, email[email['key_date'].str.startswith(year_month)]], ignore_index=True)
            pcs = pd.concat([pcs, pcs[pcs['key_date'].str.startswith(year_month)]], ignore_index=True)        
    filtered['insider'] = filtered['insider'].apply(lambda x: 1 if x != 0 else 0)
    return filtered, users, email, pcs

def none_homogeneous_relationship(df):
    # Create a list of empty lists for src and dst
    connections = [
        [],  # src
        []   # dst
    ]

    # Iterate over the DataFrame rows
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Append the same index to both src and dst
        connections[0].append(index)
        connections[1].append(index)

    return connections

# Function to create supervisor-to-user and intra-user connections
def user_hierarchical_relationship(df, users):
    # Precompute a mapping from each user (df['user'] value) to a list of df indices (user-days)
    # df['user'] is assumed to store the users' index from the users DataFrame.
    user_day_map = df.groupby('user').groups  # dict: user index -> Index of df rows

    # Create a mapping from user_id (string) to its index (i.e. row label in users)
    user_to_index = {row['user_id']: idx for idx, row in users.iterrows()}

    # Group subordinate users by supervisor. 
    # This creates a dictionary: supervisor -> list of subordinate user_ids.
    sup_to_userids = users.groupby('sup')['user_id'].apply(list).to_dict()

    connections = [[], []]

    # Process hierarchical edges for each supervisor.
    for supervisor, subordinate_ids in tqdm(sup_to_userids.items(), desc="Processing supervisors"):
        # Skip if supervisor is NaN or not in our user mapping.
        if pd.isna(supervisor) or supervisor not in user_to_index:
            continue

        supervisor_index = user_to_index[supervisor]
        supervisor_days = user_day_map.get(supervisor_index, [])

        if isinstance(supervisor_days, pd.DataFrame):
            if supervisor_days.empty:
                continue

        if isinstance(supervisor_days, list):
            if not supervisor_days:
                continue

        # Convert subordinate user_ids to their corresponding indices (filtering out missing ones)
        subordinate_indexes = [user_to_index[uid] for uid in subordinate_ids if uid in user_to_index]

        # 1. Add edges from supervisor days to each subordinate's days.
        for subordinate in subordinate_indexes:
            subordinate_days = user_day_map.get(subordinate, [])
            for src, dst in itertools.product(supervisor_days, subordinate_days):
                connections[0].append(src)
                connections[1].append(dst)
                connections[0].append(dst)
                connections[1].append(src)

        # 2. Add edges among subordinate users (all pairs of subordinate user-days)
        for u1, u2 in itertools.combinations(subordinate_indexes, 2):
            days_u1 = user_day_map.get(u1, [])
            days_u2 = user_day_map.get(u2, [])
            for src, dst in itertools.product(days_u1, days_u2):
                connections[0].append(src)
                connections[1].append(dst)
                connections[0].append(dst)
                connections[1].append(src)

    # Process self edges (within each user): connect all user-days from the same user.
    for user_id, user_index in tqdm(user_to_index.items(), desc="Processing self connections"):
        user_days = user_day_map.get(user_index, [])
        for src, dst in itertools.combinations(user_days, 2):
            connections[0].append(src)
            connections[1].append(dst)
            connections[0].append(dst)
            connections[1].append(src)

    return connections

def merge_all_dates_for_each_user(df):
    # Custom function to merge lists in the 'to' column
    def merge_to_lists(series):
        merged = []
        for item in series:
            # Make sure each item is a list
            if not isinstance(item, list):
                item = [item]
            merged.extend(item)
        # Remove duplicates while preserving order
        return list(dict.fromkeys(merged))
    
    # Group by user (or you could use 'user_id' if that’s the unique identifier)
    merged_df = df.groupby("user").agg({
        "id": "first",          # You can choose how to handle these fields
        "date": lambda x: list(x),      # Gather all dates (if needed)
        "to": merge_to_lists,   # Merge the lists from all rows
        "user_id": "first",      # Assuming user_id is unique per user
        "key_date": lambda x: list(x),  # Gather all key_date values
    }).reset_index()
    
    return merged_df

def email_communication_relationship(df, emails, users):
    # merge all user-day entries to only have one entry per user for the entire timeframe
    email = merge_all_dates_for_each_user(emails)
    email.drop(['date', 'key_date'], axis=1, inplace=True)

    # Precompute a mapping from user email to user index from the users DataFrame.
    user_map = dict(zip(users['user_id'], users.index))
    user_email_map = dict(zip(users['email'], users.index))
    user_node_map = df.groupby('user').apply(lambda group: group.index.tolist()).to_dict()
    
    connections = [
        [], # src
        []  # dst
    ]

    for _, row in tqdm(email.iterrows(), total=email.shape[0]):
        user = user_map[row['user']]
        if user not in user_node_map:
            continue
        for to in row['to']:
            if to not in user_email_map:
                continue
            src_list = user_node_map[user]
            if user_email_map[to] not in user_node_map:
                continue
            dst_list = user_node_map[user_email_map[to]]
            for src, dst in list(itertools.product(src_list, dst_list)):
                connections[0].append(src)
                connections[1].append(dst)
                connections[0].append(dst)
                connections[1].append(src)

    return connections

def email_day_communication_relationship(df, emails, users):
    # Precompute a mapping from merge_key to index in the day DataFrame.
    day_map = dict(zip(df['merge_key'], df.index))
    
    # Precompute a mapping from user email to user index from the users DataFrame.
    user_map = dict(zip(users['email'], users.index))

    connections = [
        [], # src
        []  # dst
    ]

    for _, row in tqdm(emails.iterrows(), desc="Processing email data", total=emails.shape[0]):
        key = row['merge_key']  # Sender's merge_key, e.g. "209_2010-01-02"
        date = row['key_date']  # Date of the email
        
        # Get sender's node from day_map.
        if key not in day_map:
            continue
        sender_node = day_map[key]
        
        # Process each recipient email in the 'to' field.
        for user_email in row['to']:
            # Look up the recipient's user index.
            if user_email not in user_map:
                continue
            recipient_user_index = user_map[user_email]
            
            # Construct the recipient's merge_key (format: "{user_index}_{date}").
            recipient_key = f"{recipient_user_index}_{date}"
            
            # If the recipient's merge_key exists, add an edge.
            if recipient_key in day_map:
                recipient_node = day_map[recipient_key]
                connections[0].append(sender_node)
                connections[1].append(recipient_node)

    return connections

def merge_pc_entries(df):
    merged_df = df.groupby('pc').agg({
        'key_date': lambda x: list(x.unique()),  # unique dates for each PC
        'user_id': lambda x: list(set(sum(x, [])))  # flatten lists and get unique user_ids
    }).reset_index()
    return merged_df

def device_sharing_relationship(df, pcs):
    pc = pcs.explode('pc').groupby(['key_date', 'pc'])['user_id'].apply(list).reset_index()
    pc = merge_pc_entries(pc)
    pc.drop(['key_date'], axis=1, inplace=True)

    user_node_map = df.groupby('user').apply(lambda group: group.index.tolist()).to_dict()

    connections = [
        [], #src
        [] #dst
    ]

    for _, row in pc.iterrows():
        shared_users = row['user_id']
        for user1, user2 in itertools.combinations(shared_users, 2):
            if user1 not in user_node_map or user2 not in user_node_map:
                continue
            src_list = user_node_map[user1]
            dst_list = user_node_map[user2]
            for src, dst in list(itertools.product(src_list, dst_list)):
                connections[0].append(src)
                connections[1].append(dst)
                connections[0].append(dst)
                connections[1].append(src)

    return connections

def device_day_sharing_relationship(df, pcs):
    user_day_index = df.reset_index().set_index(['user', 'key_date'])['index'].to_dict()
    shared_pc_groups = pcs.explode('pc').groupby(['key_date', 'pc'])['user_id'].apply(list).reset_index()

    connections = [[], []]

    for _, row in shared_pc_groups.iterrows():
        shared_users = row['user_id']
        key_date = row['key_date']

        # Create all unique pairs of users who used the same PC
        for user1, user2 in itertools.combinations(shared_users, 2):
            # Get the index positions from feature_user_day
            if (user1, key_date) in user_day_index and (user2, key_date) in user_day_index:
                src = user_day_index[(user1, key_date)]
                dst = user_day_index[(user2, key_date)]

                # Append to connections list
                connections[0].append(src)
                connections[1].append(dst)
    
    return connections