import pandas as pd
from pathlib import Path
import shutil


# meta data reading for XPS data
def read_spectrum_meta_data(path, tools_to_read = ['H1', 'J4', 'K1', 'K4']):    
     #meta data for data
    df_md = pd.DataFrame(columns=['tool', 'wafer_id', 'datetime', 'path', 'lot_id'])
    
    for tool in tools_to_read:
        # create the path for the tool
        path_tool_data = path / tool
        # if no such tool directory exists, continue to the next tool
        if not path_tool_data.exists():
            continue
        paths = [Path(path) for path in path_tool_data.rglob('*.spe')]
        # if no data found, continue to the next tool
        if len(paths) == 0:
            print(f'No data found for the tool: {tool}')
            continue
        # start reading the meta data and create columns
        # assume structure like:
        # RawData_USERNAME_2026-05-29_09384401\NTA000004.0H\NTA000004.23_20260517-152621\G94.NTA000004.23.01.20260517.152441.spe
        df_tmp = pd.DataFrame({'path': paths})
        df_tmp['wafer_id'] = df_tmp['path'].apply(lambda x: x.parents[0].name.split('_')[0])
        df_tmp['lot_id'] = df_tmp['path'].apply(lambda x: x.parents[1].name)
        df_tmp['datetime'] = df_tmp['path'].apply(lambda x: x.parents[0].name.split('_')[1])
        df_tmp['datetime'] = pd.to_datetime(df_tmp['datetime'], format='%Y%m%d-%H%M%S')
        df_tmp['tool'] = tool
        # append the read meta data
        df_md = pd.concat([df_md, df_tmp[df_md.columns]], ignore_index=True)

    # fix data type
    df_md = df_md.convert_dtypes()
    return df_md



def generate_standard_file_name(data_row, append=None):
    tool = str(data_row['tool'])
    date = str(data_row['datetime'].date())
    time = str(data_row['datetime'].time()).replace(':','-')
    wafer_id = str(data_row['wafer_id'])
    lst = [tool, date, time, wafer_id]
    file_name = '_'.join(lst)
    if append: file_name += str(append)
    return file_name

def copy_and_rename_spe(df, path_spe, dry_run=False):    
    df_tmp = df.copy()
    path_spe.mkdir(parents=True, exist_ok=True) 
    df_tmp['path_spe'] = df_tmp.apply(lambda x: path_spe/generate_standard_file_name(x, '.spe'), axis=1)
    # data copy
    for index, row in df_tmp.iterrows():
        if dry_run: break
        shutil.copy2(row['path'], row['path_spe'])    
    return df_tmp

def copy_and_rename_vms(df, path_vms, dry_run=False):
    df_tmp = df.copy()
    # copy vms files from the spe folder        
    path_vms.mkdir(parents=True, exist_ok=True) 
    df_tmp['path_vms'] = df_tmp.apply(lambda x: path_vms/generate_standard_file_name(x, '.vms'), axis=1)
    
    # note that the file name may become low-cased
    # it will try without lowering the case first and then lowering case when the previous fails
    df_tmp['path_spe_tranformed']       = df_tmp.apply(lambda x: x['path_spe'].parent / str(str(x['path_spe'].stem)+'.vms'), axis=1) 
    df_tmp['path_spe_tranformed_lower'] = df_tmp.apply(lambda x: x['path_spe'].parent / str(str(x['path_spe'].stem).lower()+'.vms'), axis=1)
    
    # copy vms files
    missing_count = 0
    lower_case_used = False
    upper_case_used = False
    for index, row in df_tmp.iterrows():
        if dry_run: break
        if row['path_spe_tranformed'].exists:
            if not upper_case_used: upper_case_used = True
            shutil.copy2(row['path_spe_tranformed'], row['path_vms'])
        elif row['path_spe_tranformed_lower'].exists:
            if not lower_case_used: lower_case_used = True
            shutil.copy2(row['path_spe_tranformed_lower'], row['path_vms'])
        else:
            missing_count += 1
    
    # show results and save meta data to a csv file
    print(f'Missing {missing_count} vms file(s)')
    print(f'Case used: Lower={lower_case_used}, Upper={upper_case_used}')
   
    return df_tmp