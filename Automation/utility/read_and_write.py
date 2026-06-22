import pandas as pd
from pathlib import Path
import spe_xps_reader
import shutil
import copy
from scipy.interpolate import CubicSpline
import numpy as np

# meta data reading for XPS data
def read_spectrum_meta_data(path: Path, tools_to_read = None)-> pd.DataFrame:    
    # TODO: check path uniqueness
    # tool mapping dictionary
    tool_mapping = {'G94' : 'H1',
                    'G40' : 'J4',
                    'G43' : 'K1',
                    'V3103' : 'K4',                    
                    }
    # meta data for data
    df_md = pd.DataFrame(columns=['tool', 'wafer_id', 'datetime', 'path', 'lot_id'])
    
    if tools_to_read is not None:
        # when data was seperated by tool directories
        for tool in tools_to_read:
            # create the path for the tool
            path_tool_data = path / tool
            # if no such tool directory exists, continue to the next tool
            if not path_tool_data.exists():
                continue
            paths = list(path_tool_data.rglob('*.spe'))
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
            # TODO: no die info here
            # append the read meta data
            df_md = pd.concat([df_md, df_tmp[df_md.columns]], ignore_index=True)
    else:
        # when data was not seprated by tool directories
        paths = list(path.rglob('*.spe'))
        if len(paths) == 0:
            raise Exception(f'No data found for the directory: {path}')
        # hard-coded eraser for other non-relevent spe file in the path
        # TODO: a more specific path for spe file search
        paths_tmp = paths.copy()
        to_remove = set()
        for path in paths:
            if 'RawData' not in str(path):
                to_remove.add(path)
        paths = list(set(paths) - to_remove)
        # start reading
        df_tmp = pd.DataFrame({'path': paths})
        df_tmp['wafer_id'] = df_tmp['path'].apply(lambda x: x.parents[0].name.split('_')[0])
        df_tmp['lot_id'] = df_tmp['path'].apply(lambda x: x.parents[1].name)
        df_tmp['datetime'] = df_tmp['path'].apply(lambda x: x.parents[0].name.split('_')[1])
        df_tmp['datetime'] = pd.to_datetime(df_tmp['datetime'], format='%Y%m%d-%H%M%S')
        df_tmp['tool'] = df_tmp['path'].apply(lambda x: tool_mapping[x.stem.split('.')[0]])
        df_tmp['parent'] = df_tmp['path'].apply(lambda x: x.parents[0].name)
        df_tmp['stem'] = df_tmp['path'].apply(lambda x: x.stem)
        df_tmp = df_tmp.sort_values(by=['path']).reset_index(drop=True)
        df_tmp['die'] = df_tmp.groupby(by=['parent', 'stem']).cumcount()
        df_md = df_tmp.drop(['parent', 'stem'], axis=1)

    # fix data type
    df_md = df_md.convert_dtypes()
    return df_md



def T7_reader(path_to_T7_data):
    # TODO: check the actual file format
    # TODO: remove so only 2 columns (lot id or wafer id and T7_code) remain
    df = pd.read_csv(path_to_T7_data)
    return df



def meta_data_create_T7(df_meta, path_to_T7_data):
    df_T7 = T7_reader(path_to_T7_data)
    df_T7['lot'] = df_T7['lot_id'].apply(lambda x: str(x).split('.')[0])
    df_tmp = df_meta.copy()
    df_tmp['lot'] = df_tmp['wafer_id'].apply(lambda x: str(x).split('.')[0])

    df_tmp = pd.merge(df_tmp, df_T7, on='lot', suffixes=('', '_y'))
    df_tmp = df_tmp[list(df_meta.columns)+['T7_code']]
    if any(df_tmp['T7_code'].isna()):
        print(f'Missing T7 code for the following {sum(df_tmp['T7_code'].isna())} wafer(s):')
        print(df_tmp[df_tmp['T7_code'].isna()]['wafer_id'])
        raise Exception('Missing T7 code for certain wafer(s)')
    return df_tmp



def create_paired_source_target_DataFrame(df_meta:pd.DataFrame, time_tolerance_in_hours=12, filter_out_non_paired_data=False):
    df_tmp = df_meta.copy()
    df_tmp = df_tmp.sort_values('datetime') # merge_asof request sorted comparison key

    # change tolerance time to time-delta object
    time_tolerance_in_hours = pd.Timedelta(f'{time_tolerance_in_hours}h')

    lst = []
    tools = df_tmp['tool'].unique()
    for tool_source in tools:
        for tool_target in tools:
            if tool_source == tool_target:
                continue
            # specify source and target tool
            df_source = df_tmp.query('tool == @tool_source').copy()
            df_target = df_tmp.query('tool == @tool_target').copy()
            # duplicate datetime so it is not removed after merge_asof (since it is the merge reference)
            df_target['datetime_target'] = df_target['datetime']
            # apply merge_asof and calcualte time difference
            df_source = pd.merge_asof(df_source, df_target, on='datetime', by=['die', 'T7_code'], direction='nearest', tolerance=time_tolerance_in_hours, suffixes=('','_target'))
            df_source['datetime_difference'] = df_source['datetime_target'] - df_source['datetime']
            df_source['datetime_difference'] = df_source['datetime_difference'].abs()
            lst.append(df_source)
    df_pair = pd.concat(lst)
    df_pair = df_pair[list(df_meta.columns)+['tool_target', 'wafer_id_target', 'path_target', 'datetime_difference', 'datetime_target', ]]
    if filter_out_non_paired_data:
        df_pair.dropna(inplace=True, ignore_index=True)
    return df_pair



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
    for row in df_tmp.itertuples(index=False):
        if dry_run: break
        shutil.copy2(row.path, row.path_spe)    
    return df_tmp



def copy_and_rename_vms(df, path_vms, dry_run=False):
    df_tmp = df.copy()
    # copy vms files from the spe folder        
    path_vms.mkdir(parents=True, exist_ok=True) 
    df_tmp['path_vms'] = df_tmp.apply(lambda x: path_vms/generate_standard_file_name(x, '.vms'), axis=1)
    
    # note that the file name may become low-cased
    # it will try without lowering the case first and then lowering case when the previous fails
    # df_tmp['path_spe_tranformed']       = df_tmp.apply(lambda x: x['path_spe'].parent / str(str(x['path_spe'].stem)+'.vms'), axis=1) 
    # df_tmp['path_spe_tranformed_lower'] = df_tmp.apply(lambda x: x['path_spe'].parent / str(str(x['path_spe'].stem).lower()+'.vms'), axis=1)
    df_tmp['path_spe_tranformed'] = df_tmp['path_spe'].apply(lambda x: x.with_suffix('.vms'))
    df_tmp['path_spe_tranformed_lower'] = df_tmp['path_spe'].apply(lambda x: x.parent / f"{x.stem.lower()}.vms")
    
    # copy vms files
    missing_count = 0
    lower_case_used = False
    upper_case_used = False
    for row in df_tmp.itertuples(index=False):
        if dry_run: break
        if row.path_spe_tranformed.exists():
            if not upper_case_used: upper_case_used = True
            shutil.copy2(row.path_spe_tranformed, row.path_vms)
        elif row.path_spe_tranformed_lower.exists():
            if not lower_case_used: lower_case_used = True
            shutil.copy2(row.path_spe_tranformed, row.path_vms)
        else:
            missing_count += 1
    
    # show results and save meta data to a csv file
    print(f'Missing {missing_count} vms file(s)')
    print(f'Case used: Lower={lower_case_used}, Upper={upper_case_used}')
   
    return df_tmp



def spectrum_specification_for_interpolation(spectra, scaling_factor=1, verbosity=1):
    interpolation_specification = dict()
    property_dict = {'region' : [], 
                     'BE_min' : [], 
                     'BE_max' : [], 
                     'BE_step': [], 
                     'I_min'  : [], 
                     'I_max'  : [], 
                     'number_of_points' : [],
                     'path'   : [],
                     }
    # read properties for each spectrum
    for path, spectrum in spectra.items():
        for region, dct in spectrum.items():
            property_dict['region'].append(region)
            property_dict['BE_min'].append(dct['data']['binding_energy'].min())
            property_dict['BE_max'].append(dct['data']['binding_energy'].max())
            property_dict['BE_step'].append((dct['data']['binding_energy'].max()-dct['data']['binding_energy'].min())/(len(dct['data'])-1))
            property_dict['I_min'].append(dct['data']['intensity'].min())
            property_dict['I_max'].append(dct['data']['intensity'].max())
            property_dict['number_of_points'].append(len(dct['data']))
            property_dict['path'].append(path)
    df = pd.DataFrame(property_dict)

    # determine specification for interpolation
    BE_step = df['BE_step'].abs().min()
    for region in df['region'].unique():
        df_tmp = df.query('region == @region')
        interpolation_specification[region] = (df_tmp['BE_min'].max(), df_tmp['BE_max'].min(), BE_step/scaling_factor)

    return interpolation_specification


def interpolation_spline(spectra, interpolation_specification):
    # interpolation_specification = {region:(BE_min, BE_max, BE_step)}
    spectra_interpolated = copy.deepcopy(spectra)
    for path, spectrum in spectra_interpolated.items():
        for region, dct in spectrum.items():
            # get data, sort (needed by spline), and create spline
            df = dct['data']
            df = df.sort_values(by=['binding_energy'], ascending=True, ignore_index=True)
            spline = CubicSpline(df['binding_energy'], df['intensity'], extrapolate=True)
            # setup BE grids
            BE_start = interpolation_specification[region][0]
            BE_end = interpolation_specification[region][1]
            BE_step = interpolation_specification[region][2]
            # grid points that does not cross BE_end
            num_segments = int(np.floor((BE_end - BE_start) / BE_step))
            truncated_end = BE_start + (num_segments * BE_step)
            grid = np.linspace(BE_start, truncated_end, num=num_segments + 1)
            # perform interpolation
            df_tmp = pd.DataFrame({'binding_energy' : grid, 'intensity' : spline(grid)})
            # update relavent data
            dct['data'] = df_tmp
            # dct['energy_range'] = (grid[0], grid[-1])
            # dct['energy_start'] = grid[0]      
            # dct['energy_step'] = grid[1] - grid[0]     

    return spectra_interpolated



def SPE_file_reader(df_meta, use_custom=False):
    dct = dict()
    if use_custom:
        pass
    else:
        for path in df_meta['path']:
            dct[path] = SPE_file_reader_single(path)
    return dct



def SPE_file_reader_single(path_to_file):    
    # Use spe reader from: https://github.com/gkerherve/spe_reader  (pip install spe-xps-reader)
    # Another possible reader: https://pypi.org/project/xps-export/ (pip install xps-export)
    parsed = spe_xps_reader.extract_all_regions(path_to_file)
    spectrum = dict()
    # fill spectrum data
    for region in parsed['regions_data']:
        # TODO: maybe we can get rid off this extra ['data'] dictionary layer
        spectrum[region['name']] = dict()
        # spectrum[region['name']]['binding_energy'] = region['be_values']
        # spectrum[region['name']]['intensity'] = region['corrected_intensities']
        spectrum[region['name']]['data'] = pd.DataFrame({'binding_energy' : region['be_values'], 'intensity' : region['corrected_intensities']})
        # spectrum[region['name']]['energy_range'] = (region['be_values'][0], region['be_values'][-1])
        # spectrum[region['name']]['energy_start'] = region['be_values'][0] # TODO: this assume at least 1 data point       
        # spectrum[region['name']]['energy_step'] = region['be_values'][1] - region['be_values'][0] # TODO: this assume at least 2 data points       
    return spectrum



def SPE_file_reader_single_customized(path_to_file):
    spectrum = dict()
    # dummy function for future development
    return spectrum

