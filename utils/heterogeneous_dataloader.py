from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd
import pickle

def load_heterogeneous_cert_data(data_path="./data", method='undersampling_Reshybnet', version='r4.2', year=None, month=None):
    with open(f"{data_path}/users.csv", "r") as f:
        users = pd.read_csv(f)

    # read only first line of csv to get column names without loading the entire file
    with open(f"{data_path}/2010-01-filled.csv", "r") as f:
        columns = f.readline().strip().split(',')

    activity_df = pd.DataFrame(columns=columns)

    year_month = {
        '2010': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        '2011': ['01', '02', '03', '04', '05'],
    }

    if method == 'undersampling_Reshybnet':
        with open(f"{data_path}/user_date.pkl", "rb") as f:
            user_date = pickle.load(f)
        user_date['user_id'] = user_date['user_index'].apply(lambda x: users[users['user_id'] == x].index[0])
        user_date['key_date'] = pd.to_datetime(user_date['date']).dt.date.astype(str)

        for year in year_month:
            for month in year_month[year]:
                df = user_date[user_date['key_date'].str.startswith(f'{year}-{month}')]
                if df.empty:
                    continue

                df['merge_key'] = df['user_id'].astype(str) + '_' + df['key_date'].astype(str)
        
                temp_activity_df = pd.read_csv(f"{data_path}/{year}-{month}-filled.csv")
                temp_activity_df['key_date'] = temp_activity_df['timestamp'].apply(lambda ts: str(datetime.fromtimestamp(ts).date()))
                temp_activity_df['merge_key'] = temp_activity_df['user'].astype(str) + '_' + temp_activity_df['key_date'].astype(str)

                mask = temp_activity_df['merge_key'].isin(df['merge_key'])
                filtered_df = temp_activity_df[mask]
                activity_df = pd.concat([activity_df, filtered_df])
                
    elif method == 'undersampling':
        with open(f"{data_path}/day{version}.csv", "r") as f:
            day = pd.read_csv(f)
        day['key_date'] = day['starttime'].apply(lambda ts: str(datetime.fromtimestamp(ts).date()))

        # Separate the data into two groups based on 'insider' value
        insider = day[day['insider'] != 0]
        normal = day[day['insider'] == 0]

        # Randomly sample 954 entries from each group
        sampled_insider_1 = insider.sample(n=954, random_state=42)
        sampled_insider_0 = normal.sample(n=954, random_state=42)

        filtered = pd.concat([sampled_insider_1, sampled_insider_0], ignore_index=True).reset_index(drop=True)

        for year in year_month:
            for month in year_month[year]:
                df = filtered[filtered['key_date'].str.startswith(f'{year}-{month}')]
                if df.empty:
                    continue

                df['merge_key'] = df['user'].astype(str) + '_' + df['key_date'].astype(str)
        
                temp_activity_df = pd.read_csv(f"{data_path}/{year}-{month}-filled.csv")
                temp_activity_df['key_date'] = temp_activity_df['timestamp'].apply(lambda ts: str(datetime.fromtimestamp(ts).date()))
                temp_activity_df['merge_key'] = temp_activity_df['user'].astype(str) + '_' + temp_activity_df['key_date'].astype(str)

                mask = temp_activity_df['merge_key'].isin(df['merge_key'])
                filtered_df = temp_activity_df[mask]
                activity_df = pd.concat([activity_df, filtered_df])
        
    elif method == 'one_month':
        year_month = f"{year}-{month:02d}"
        with open(f"{data_path}/{year_month}-filled.csv", "r") as f:
            activity_df = pd.read_csv(f)
        activity_df['key_date'] = activity_df['timestamp'].apply(lambda ts: str(datetime.fromtimestamp(ts).date()))

    activity_df.reset_index(drop=True, inplace=True)
    activity_df['insider'] = activity_df['insider'].apply(lambda x: 1 if x != 0 else 0)

    f_col = {
        'logon': ['timestamp', 'user', 'day', 'week', 'pc', 'isworkhour', 'isafterhour', 'isweekend', 'isweekendafterhour', 'n_allact', 'allact_n-pc0', 'allact_n-pc1', 'allact_n-pc2', 'allact_n-pc3', 'n_logon', 'logon_n-pc0', 'logon_n-pc1', 'logon_n-pc2', 'logon_n-pc3'],
        'device': ['timestamp', 'user', 'day', 'week', 'pc', 'isworkhour', 'isafterhour', 'isweekend', 'isweekendafterhour', 'n_allact', 'allact_n-pc0', 'allact_n-pc1', 'allact_n-pc2', 'allact_n-pc3', 'n_usb', 'usb_mean_usb_dur', 'usb_n-pc0', 'usb_n-pc1', 'usb_n-pc2', 'usb_n-pc3'],
        'file': ['timestamp', 'user', 'day', 'week', 'pc', 'isworkhour', 'isafterhour', 'isweekend', 'isweekendafterhour', 'n_allact', 'allact_n-pc0', 'allact_n-pc1', 'allact_n-pc2', 'allact_n-pc3', 'n_file', 'file_mean_file_len', 'file_mean_file_depth', 'file_mean_file_nwords', 'file_n-disk0', 'file_n-disk1', 'file_n-pc0', 'file_n-pc1', 'file_n-pc2', 'file_n-pc3', 'file_n_otherf', 'file_otherf_mean_file_len', 'file_otherf_mean_file_depth', 'file_otherf_mean_file_nwords', 'file_otherf_n-disk0', 'file_otherf_n-disk1', 'file_otherf_n-pc0', 'file_otherf_n-pc1', 'file_otherf_n-pc2', 'file_otherf_n-pc3', 'file_n_compf', 'file_compf_mean_file_len', 'file_compf_mean_file_depth', 'file_compf_mean_file_nwords', 'file_compf_n-disk0', 'file_compf_n-disk1', 'file_compf_n-pc0', 'file_compf_n-pc1', 'file_compf_n-pc2', 'file_compf_n-pc3', 'file_n_phof', 'file_phof_mean_file_len', 'file_phof_mean_file_depth', 'file_phof_mean_file_nwords', 'file_phof_n-disk0', 'file_phof_n-disk1', 'file_phof_n-pc0', 'file_phof_n-pc1', 'file_phof_n-pc2', 'file_phof_n-pc3', 'file_n_docf', 'file_docf_mean_file_len', 'file_docf_mean_file_depth', 'file_docf_mean_file_nwords', 'file_docf_n-disk0', 'file_docf_n-disk1', 'file_docf_n-pc0', 'file_docf_n-pc1', 'file_docf_n-pc2', 'file_docf_n-pc3', 'file_n_txtf', 'file_txtf_mean_file_len', 'file_txtf_mean_file_depth', 'file_txtf_mean_file_nwords', 'file_txtf_n-disk0', 'file_txtf_n-disk1', 'file_txtf_n-pc0', 'file_txtf_n-pc1', 'file_txtf_n-pc2', 'file_txtf_n-pc3', 'file_n_exef', 'file_exef_mean_file_len', 'file_exef_mean_file_depth', 'file_exef_mean_file_nwords', 'file_exef_n-disk0', 'file_exef_n-disk1', 'file_exef_n-pc0', 'file_exef_n-pc1', 'file_exef_n-pc2', 'file_exef_n-pc3'],
        'email': ['timestamp', 'user', 'day', 'week', 'pc', 'isworkhour', 'isafterhour', 'isweekend', 'isweekendafterhour', 'n_allact', 'allact_n-pc0', 'allact_n-pc1', 'allact_n-pc2', 'allact_n-pc3', 'n_email', 'email_mean_n_des', 'email_mean_n_atts', 'email_mean_n_exdes', 'email_mean_n_bccdes', 'email_mean_email_size', 'email_mean_email_text_slen', 'email_mean_email_text_nwords', 'email_n-Xemail1', 'email_n-exbccmail1', 'email_n-pc0', 'email_n-pc1', 'email_n-pc2', 'email_n-pc3'],
        'http': ['timestamp', 'user', 'day', 'week', 'pc', 'isworkhour', 'isafterhour', 'isweekend', 'isweekendafterhour', 'n_allact', 'allact_n-pc0', 'allact_n-pc1', 'allact_n-pc2', 'allact_n-pc3', 'n_http', 'http_mean_url_len', 'http_mean_url_depth', 'http_mean_http_c_len', 'http_mean_http_c_nwords', 'http_n_otherf', 'http_otherf_mean_url_len', 'http_otherf_mean_url_depth', 'http_otherf_mean_http_c_len', 'http_otherf_mean_http_c_nwords', 'http_n_socnetf', 'http_socnetf_mean_url_len', 'http_socnetf_mean_url_depth', 'http_socnetf_mean_http_c_len', 'http_socnetf_mean_http_c_nwords', 'http_n_cloudf', 'http_cloudf_mean_url_len', 'http_cloudf_mean_url_depth', 'http_cloudf_mean_http_c_len', 'http_cloudf_mean_http_c_nwords', 'http_n_jobf', 'http_jobf_mean_url_len', 'http_jobf_mean_url_depth', 'http_jobf_mean_http_c_len', 'http_jobf_mean_http_c_nwords', 'http_n_leakf', 'http_leakf_mean_url_len', 'http_leakf_mean_url_depth', 'http_leakf_mean_http_c_len', 'http_leakf_mean_http_c_nwords', 'http_n_hackf', 'http_hackf_mean_url_len', 'http_hackf_mean_url_depth', 'http_hackf_mean_http_c_len', 'http_hackf_mean_http_c_nwords'],
    }

    activities = {
        'logon': activity_df[activity_df['node_type'] == 'logon'][f_col['logon']].reset_index(drop=True),
        'device': activity_df[activity_df['node_type'] == 'device'][f_col['device']].reset_index(drop=True),
        'file': activity_df[activity_df['node_type'] == 'file'][f_col['file']].reset_index(drop=True),
        'email': activity_df[activity_df['node_type'] == 'email'][f_col['email']].reset_index(drop=True),
        'http': activity_df[activity_df['node_type'] == 'http'][f_col['http']].reset_index(drop=True),
    }

    labels = {
        'logon': activity_df[activity_df['node_type'] == 'logon']['insider'].to_numpy(),
        'device': activity_df[activity_df['node_type'] == 'device']['insider'].to_numpy(),
        'file': activity_df[activity_df['node_type'] == 'file']['insider'].to_numpy(),
        'email': activity_df[activity_df['node_type'] == 'email']['insider'].to_numpy(),
        'http': activity_df[activity_df['node_type'] == 'http']['insider'].to_numpy(),
    }

    # Loop over each activity type and its corresponding reset DataFrame
    for node_type, df_subset in activities.items():
        mask = activity_df['node_type'] == node_type
        # Assign new node indices (reset index values) for these rows
        activity_df.loc[mask, 'node'] = range(len(df_subset))
        # convert node row to integer
    activity_df['node'] = activity_df['node'].astype(int)

    return activity_df, activities, labels, users 

def get_edges_dict():
    return {
        "logon_associate_logon" : [[], []],
        "logon_associate_device" : [[], []],
        "logon_associate_http" : [[], []],
        "logon_associate_file" : [[], []],
        "logon_associate_email": [[], []],
        "device_associate_logon" : [[], []],
        "device_associate_device" : [[], []],
        "device_associate_http" : [[], []],
        "device_associate_file" : [[], []],
        "device_associate_email": [[], []],
        "http_associate_logon" : [[], []],
        "http_associate_device" : [[], []],
        "http_associate_http" : [[], []],
        "http_associate_file" : [[], []],
        "http_associate_email": [[], []],
        "file_associate_logon" : [[], []],
        "file_associate_device" : [[], []],
        "file_associate_http" : [[], []],
        "file_associate_file" : [[], []],
        "file_associate_email": [[], []],
        "email_associate_logon": [[], []],
        "email_associate_device": [[], []],
        "email_associate_http": [[], []],
        "email_associate_file": [[], []],
        "email_associate_email": [[], []]        
    }

def none_heterogeneous_edges(activities):
    all_logon_index = activities['logon'].index.to_list()
    all_device_index = activities['device'].index.to_list()
    all_http_index = activities['http'].index.to_list()
    all_file_index = activities['file'].index.to_list()
    all_email_index = activities['email'].index.to_list()
    return {
        "logon_associate_logon" : [all_logon_index, all_logon_index],
        "logon_associate_device" : [[], []],
        "logon_associate_http" : [[], []],
        "logon_associate_file" : [[], []],
        "logon_associate_email" : [[], []],
        "device_associate_logon" : [[], []],
        "device_associate_device" : [all_device_index, all_device_index],
        "device_associate_http" : [[], []],
        "device_associate_file" : [[], []],
        "device_associate_email" : [[], []],
        "http_associate_logon" : [[], []],
        "http_associate_device" : [[], []],
        "http_associate_http" : [all_http_index, all_http_index],
        "http_associate_file" : [[], []],
        "http_associate_email" : [[], []],
        "file_associate_logon" : [[], []],
        "file_associate_device" : [[], []],
        "file_associate_http" : [[], []],
        "file_associate_file" : [all_file_index, all_file_index],
        "file_associate_email" : [[], []],
        "email_associate_logon" : [[], []],
        "email_associate_device" : [[], []],
        "email_associate_http" : [[], []],
        "email_associate_file" : [[], []],
        "email_associate_email" : [all_email_index, all_email_index]
    }

# Log2Vec Edge Processing Functions
def edge_associate(data, j, edges):
    if len(data) < 2:
        return
    data1 = data.iloc[j-1]
    data2 = data.iloc[j]
    edges[f"{data1['node_type']}_associate_{data2['node_type']}"][0].append(data1['node'])
    edges[f"{data1['node_type']}_associate_{data2['node_type']}"][1].append(data2['node'])
    edges[f"{data2['node_type']}_associate_{data1['node_type']}"][0].append(data2['node'])
    edges[f"{data2['node_type']}_associate_{data1['node_type']}"][1].append(data1['node'])

def next_edge_associate(today_data, next_day_data, edges):
    if len(today_data) >= 1 and len(next_day_data) >= 1:
        data1 = today_data.iloc[0]
        data2 = next_day_data.iloc[0]
        edges[f"{data1['node_type']}_associate_{data2['node_type']}"][0].append(data1['node'])
        edges[f"{data1['node_type']}_associate_{data2['node_type']}"][1].append(data2['node'])
    if len(today_data) >= 2 and len(next_day_data) >= 2:
        data1 = today_data.iloc[-1]
        data2 = next_day_data.iloc[-1]
        edges[f"{data1['node_type']}_associate_{data2['node_type']}"][0].append(data1['node'])
        edges[f"{data1['node_type']}_associate_{data2['node_type']}"][1].append(data2['node'])

def process_rule_1(today_data, edges):
    for j in range(1, len(today_data)):
        edge_associate(today_data, j, edges)

def process_rule_2_3(today_data, edges):
    pc_encodings = today_data["pc"].unique()
    for pc in pc_encodings:
        process_rule_2(today_data, pc, edges)
        process_rule_3(today_data, pc, edges)

def process_rule_2(today_data, pc, edges):
    same_pc_data = today_data[today_data["pc"] == pc]
    for j in range(1, len(same_pc_data)):
        edge_associate(same_pc_data, j, edges)

def process_rule_3(today_data, pc, edges):
    same_pc_data = today_data[today_data["pc"] == pc]
    for node_type in ['logon', 'device', 'http', 'file']:
        same_day_user_data = same_pc_data[same_pc_data["node_type"] == node_type]
        for j in range(1, len(same_day_user_data)):
            edge_associate(same_day_user_data, j, edges)

def process_rule_4(today_data, next_day_data, edges):
    next_edge_associate(today_data, next_day_data, edges)

def process_rule_5_6(today_data, next_day_data, edges):
    merge_data = today_data._append(next_day_data)
    pc_encodings = merge_data["pc"].unique()
    for pc in pc_encodings:
        process_rule_5(today_data, next_day_data, pc, edges)
        process_rule_6(today_data, next_day_data, pc, edges)

def process_rule_5(today_data, next_day_data, pc, edges):
    today_same_pc_data = today_data[today_data["pc"] == pc]
    next_day_same_pc_data = next_day_data[next_day_data["pc"] == pc]
    next_edge_associate(today_same_pc_data, next_day_same_pc_data, edges)

def process_rule_6(today_data, next_day_data, pc, edges):
    today_same_pc_data = today_data[today_data["pc"] == pc]
    next_day_same_pc_data = next_day_data[next_day_data["pc"] == pc]

    for node_type in ['logon', 'device', 'http', 'file']:
        today_node_data = today_same_pc_data[today_same_pc_data["node_type"] == node_type]
        next_day_node_data = next_day_same_pc_data[next_day_same_pc_data["node_type"] == node_type]
        next_edge_associate(today_node_data, next_day_node_data, edges)

def process_sequential_day(date, chronological_df, delta):
    edges = get_edges_dict()

    today_data = chronological_df[chronological_df["key_date"] == date.strftime('%Y-%m-%d')].sort_values(by=['timestamp'])

    process_rule_1(today_data, edges)

    if (date + timedelta(days=1)) <= (date + timedelta(days=delta.days)):
        next_day_data = chronological_df[chronological_df["key_date"] == (date + timedelta(days=1)).strftime('%Y-%m-%d')].sort_values(by=['timestamp'])
        next_edge_associate(today_data, next_day_data, edges)

    return edges


def process_log2vec_day(date, chronological_df, delta):
    # Initialize a local edges dictionary for this day's processing
    edges = get_edges_dict()

    today_data = chronological_df[chronological_df["key_date"] == date.strftime('%Y-%m-%d')].sort_values(by=['timestamp'])

    # Rule 1 - Chronological Activity
    process_rule_1(today_data, edges)

    # Rule 2, 3 - Same PC Activity, Same Operation, etc.
    process_rule_2_3(today_data, edges)

    # Rule 4, 5, 6 - Next day associations
    if (date + timedelta(days=1)) <= (date + timedelta(days=delta.days)):
        next_day_data = chronological_df[chronological_df["key_date"] == (date + timedelta(days=1)).strftime('%Y-%m-%d')].sort_values(by=['timestamp'])
        process_rule_4(today_data, next_day_data, edges)
        process_rule_5_6(today_data, next_day_data, edges)

    return edges

def merge_edges(edges_list):
    merged_edges = get_edges_dict()
    for edges in edges_list:
        for key in merged_edges.keys():
            merged_edges[key][0].extend(edges[key][0])
            merged_edges[key][1].extend(edges[key][1])

    return merged_edges

def process_sequential_edges(chronological_df, start_date, end_date):
    results = []
    
    delta = end_date - start_date

    for i in tqdm(range(delta.days), desc="Processing Days for Rules"):
        date = start_date + timedelta(days=i)
        out = process_sequential_day(
            date,
            chronological_df,
            delta
        )
        results.append(out)

    # Merge edges from all days
    edges = merge_edges(results)
    return edges

def process_log2vec_edges(chronological_df, start_date, end_date):
    results = []
    
    delta = end_date - start_date

    for i in tqdm(range(delta.days), desc="Processing Days for Rules"):
        date = start_date + timedelta(days=i)
        out = process_log2vec_day(
            date,
            chronological_df,
            delta
        )
        results.append(out)

    # Merge edges from all days
    edges = merge_edges(results)
    return edges