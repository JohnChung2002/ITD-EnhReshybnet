import pandas as pd
from datetime import datetime
from tqdm import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

def search_node_for_row(row):
    # Get the user id from the users dataframe.
    user = users[users.index == int(row['user'])]['user_id'].values[0]
    timestamp = row['key_timestamp']
    for log_file in log_files:
        # Construct and print the grep command.
        cmd = f'grep "{timestamp}" {args.data_path}/{log_file}.csv | grep "{user}"'
        print(f"Searching: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout.strip():
            return log_file
    return None

parser = argparse.ArgumentParser(description='CERT Activity Labeling')
parser.add_argument('--version', choices=['r4.2', 'r5.2', 'r6.2'], default='r4.2', help='Dataset version to use')
parser.add_argument('--data_path', type=str, default='/fred/oz382/dataset/CERT/r4.2', help='Path to the dataset')
parser.add_argument('--extracted_data_path', type=str, default='/fred/oz382/dataset/CERT/r4.2/ExtractedData-comb', help='Path to the dataset')

args = parser.parse_args()

with open(f"{args.extracted_data_path}/users.csv", "r") as f:
    users = pd.read_csv(f)

log_files = ['logon', 'device', 'email', 'file', 'http']

activity_df = pd.DataFrame(columns=['timestamp', 'user', 'day', 'week', 'pc', 'isworkhour', 'isafterhour', 'isweekend', 'isweekendafterhour', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin', 'O', 'C', 'E', 'A', 'N', 'n_allact', 'allact_n-pc0', 'allact_n-pc1', 'allact_n-pc2', 'allact_n-pc3', 'n_logon', 'logon_n-pc0', 'logon_n-pc1', 'logon_n-pc2', 'logon_n-pc3', 'n_usb', 'usb_mean_usb_dur', 'usb_n-pc0', 'usb_n-pc1', 'usb_n-pc2', 'usb_n-pc3', 'n_file', 'file_mean_file_len', 'file_mean_file_depth', 'file_mean_file_nwords', 'file_n-disk0', 'file_n-disk1', 'file_n-pc0', 'file_n-pc1', 'file_n-pc2', 'file_n-pc3', 'file_n_otherf', 'file_otherf_mean_file_len', 'file_otherf_mean_file_depth', 'file_otherf_mean_file_nwords', 'file_otherf_n-disk0', 'file_otherf_n-disk1', 'file_otherf_n-pc0', 'file_otherf_n-pc1', 'file_otherf_n-pc2', 'file_otherf_n-pc3', 'file_n_compf', 'file_compf_mean_file_len', 'file_compf_mean_file_depth', 'file_compf_mean_file_nwords', 'file_compf_n-disk0', 'file_compf_n-disk1', 'file_compf_n-pc0', 'file_compf_n-pc1', 'file_compf_n-pc2', 'file_compf_n-pc3', 'file_n_phof', 'file_phof_mean_file_len', 'file_phof_mean_file_depth', 'file_phof_mean_file_nwords', 'file_phof_n-disk0', 'file_phof_n-disk1', 'file_phof_n-pc0', 'file_phof_n-pc1', 'file_phof_n-pc2', 'file_phof_n-pc3', 'file_n_docf', 'file_docf_mean_file_len', 'file_docf_mean_file_depth', 'file_docf_mean_file_nwords', 'file_docf_n-disk0', 'file_docf_n-disk1', 'file_docf_n-pc0', 'file_docf_n-pc1', 'file_docf_n-pc2', 'file_docf_n-pc3', 'file_n_txtf', 'file_txtf_mean_file_len', 'file_txtf_mean_file_depth', 'file_txtf_mean_file_nwords', 'file_txtf_n-disk0', 'file_txtf_n-disk1', 'file_txtf_n-pc0', 'file_txtf_n-pc1', 'file_txtf_n-pc2', 'file_txtf_n-pc3', 'file_n_exef', 'file_exef_mean_file_len', 'file_exef_mean_file_depth', 'file_exef_mean_file_nwords', 'file_exef_n-disk0', 'file_exef_n-disk1', 'file_exef_n-pc0', 'file_exef_n-pc1', 'file_exef_n-pc2', 'file_exef_n-pc3', 'n_email', 'email_mean_n_des', 'email_mean_n_atts', 'email_mean_n_exdes', 'email_mean_n_bccdes', 'email_mean_email_size', 'email_mean_email_text_slen', 'email_mean_email_text_nwords', 'email_n-Xemail1', 'email_n-exbccmail1', 'email_n-pc0', 'email_n-pc1', 'email_n-pc2', 'email_n-pc3', 'n_http', 'http_mean_url_len', 'http_mean_url_depth', 'http_mean_http_c_len', 'http_mean_http_c_nwords', 'http_n_otherf', 'http_otherf_mean_url_len', 'http_otherf_mean_url_depth', 'http_otherf_mean_http_c_len', 'http_otherf_mean_http_c_nwords', 'http_n_socnetf', 'http_socnetf_mean_url_len', 'http_socnetf_mean_url_depth', 'http_socnetf_mean_http_c_len', 'http_socnetf_mean_http_c_nwords', 'http_n_cloudf', 'http_cloudf_mean_url_len', 'http_cloudf_mean_url_depth', 'http_cloudf_mean_http_c_len', 'http_cloudf_mean_http_c_nwords', 'http_n_jobf', 'http_jobf_mean_url_len', 'http_jobf_mean_url_depth', 'http_jobf_mean_http_c_len', 'http_jobf_mean_http_c_nwords', 'http_n_leakf', 'http_leakf_mean_url_len', 'http_leakf_mean_url_depth', 'http_leakf_mean_http_c_len', 'http_leakf_mean_http_c_nwords', 'http_n_hackf', 'http_hackf_mean_url_len', 'http_hackf_mean_url_depth', 'http_hackf_mean_http_c_len', 'http_hackf_mean_http_c_nwords', 'insider'])

if args.version == 'r4.2':
    year_month = {
        '2010': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        '2011': ['01', '02', '03', '04', '05'],
    }
else:
    year_month = {
        '2010': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        '2011': ['01', '02', '03', '04', '05', '06'],
    }

for year in year_month.keys():
    for month in year_month[year]:
        df = pd.read_csv(f"{args.extracted_data_path}/split_by_month/{year}-{month}.csv")
        #log_df['key_timestamp'] = log_df['date'].apply(lambda ts: str(datetime.fromtimestamp(ts).strftime('%m/%d/%Y %H:%M:%S')))
        df['key_timestamp'] = df['timestamp'].apply(lambda ts: pd.to_datetime(ts, unit='s').strftime('%m/%d/%Y %H:%M:%S'))
        df['node_type'] = df.apply(lambda row: 
            'logon' if row['n_logon'] == 1 else 
            'device' if row['n_usb'] == 1 else 
            'file' if row['n_file'] == 1 else 
            'http' if row['n_http'] == 1 else 
            'email' if row['n_email'] == 1 else 
            None, axis=1
        )
        # for log_file in tqdm(log_files):
        #     with open(f"/fred/oz382/dataset/CERT/r4.2/{log_file}.csv", "r") as f:
        #         log_df = pd.read_csv(f)
        #         # convert timestamp to datetime '%m/%d/%Y %H:%M:%S'
        activity_end = df[df['node_type'].isnull()]
        print("Start searching for missing node_type")
        results = {}
        # Use ThreadPoolExecutor to search for missing node types in parallel.
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_index = {
                executor.submit(search_node_for_row, row): index 
                for index, row in activity_end.iterrows()
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    node_type = future.result()
                except Exception as exc:
                    print(f"Index {idx} generated an exception: {exc}")
                    node_type = None
                results[idx] = node_type
        
        # Update the node_type column with the results.
        for idx, node_type in results.items():
            df.loc[idx, 'node_type'] = node_type

        df.to_csv(f"{args.extracted_data_path}/split_by_month/{year}-{month}-filled.csv", index=False)
