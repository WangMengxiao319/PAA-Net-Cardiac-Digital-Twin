'''Sensitive analysis of QRS-T for different scar simulation'''
from utils_folder.metrics import DTW
from utils_folder.file_process import *
import pandas as pd
import matplotlib .pyplot as plt
import numpy as np
import seaborn as sns
import neurokit2 as nk
from scipy.signal import find_peaks, peak_prominences

def main_global_SA(data_dir):
    '''Global sensitive analysis'''
    # 1. To the baseline
    # global_SA_baseline(data_dir)
    # 2. For each MI types
    global_SA_MI_type(data_dir)
    
def global_SA_baseline(data_dir = "results/generate_dataset_DHlab_data_Scar_CV_APD",visualize = True):
    '''dissimilarity of each MI scenario to the baseline in each lead'''
    
    lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    baseline_type = "healthy"
    MI_location_list = ['A1', 'A2', 'A3', 'A4', 'B1_small', 'B1_large', 'B2', 'B3']
    MI_transmural_extent_list = ['subendo', 'transmural']
    MI_types = []
    for loc in MI_location_list:
        for extent in MI_transmural_extent_list:
            MI_types.append(f"{loc}_{extent}")

    output_dir = 'results/sensitive_analysis/global'
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(f'{output_dir}/global_dtw_baseline_vs_MI.csv'):
        cases = list_subfolders(data_dir)
        
        global_DTW = []
        for index,case in enumerate(cases):
            # if index == 0:
            # Load baseline ECG
            baseline_file = f"{data_dir}/{case}/{baseline_type}/predicted_ecg.csv"
            baseline_ecg = pd.read_csv(baseline_file,header=None).to_numpy()
            print('index:',index)
            # print("shape of baseline_ecg:", baseline_ecg.shape)  # (8, signal_length)
            for MI_type in MI_types:
                # Load MI ECG
                MI_file = f"{data_dir}/{case}/{MI_type}/predicted_ecg.csv"
                MI_ecg = pd.read_csv(MI_file,header=None).to_numpy()
                # print(f"shape of {MI_type}:", MI_ecg.shape)  # (8, signal_length)
                # ***Compute DTW for each lead
                dtw_values = DTW(baseline_ecg, MI_ecg)   # (8,)
                print(f"Case: {case}, MI_type: {MI_type}, DTW: {dtw_values}")
                # Store to pandas
                global_DTW.append({
                    "case": case,
                    "MI_type": MI_type,
                    **{f"lead_{lead_names[i]}": dtw_values[i] for i in range(len(lead_names))}
                })
        pd_global_DTW = pd.DataFrame(global_DTW)

        pd_global_DTW.to_csv(f'{output_dir}/global_dtw_baseline_vs_MI.csv', index=False)
    else:
        pd_global_DTW = pd.read_csv(f'{output_dir}/global_dtw_baseline_vs_MI.csv')
    
    # Calculate mean and max across leads between each MI type and baseline, and average across cases
    # pd_global_DTW
    lead_columns = ['lead_I', 'lead_II', 'lead_V1', 'lead_V2', 'lead_V3', 'lead_V4', 'lead_V5', 'lead_V6']
    pd_global_DTW['lead_avg'] = pd_global_DTW[lead_columns].mean(axis=1)
    pd_global_DTW['lead_max'] = pd_global_DTW[lead_columns].max(axis=1)
    
    # mean
    pd_global_DTW_mean = pd_global_DTW.groupby('MI_type').mean(numeric_only=True)
    pd_global_DTW_mean = pd_global_DTW_mean.reindex(index = MI_types)

    # add mean of each lead
    lead_mean = pd_global_DTW_mean.mean(axis=1)
    print(lead_mean)
    # add one column at the end
    pd_global_DTW_mean['Average'] = lead_mean

    pd_global_DTW_mean_index = pd_global_DTW_mean.index.tolist()
    # rename
    pd_global_DTW_mean.index = [ECG_rename(name) for name in pd_global_DTW_mean_index]
    
    pd_global_DTW_mean.to_csv(f'{output_dir}/global_dtw_baseline_vs_MI_mean.csv', index=True)
    

    if visualize:
        plt.figure(figsize=(12,8))
        sns.heatmap(pd_global_DTW_mean, annot=True, fmt=".1f", square=True, cmap="Blues") # cmap=sns.cubehelix_palette(as_cmap=True)
        plt.title("Global DTW Dissimilarity to Baseline ECG", fontsize=16)
        plt.savefig(f'{output_dir}/fig_global_dtw_baseline_vs_MI_mean.pdf')
                
        
def dtw_ecg(ecg1, ecg2):
    '''Compute DTW dissimilarity between two ECG signals'''
    # ecg1, ecg2: (n_lead, signal_length)
    dissimularity_score_lead = DTW(ecg1, ecg2)  # (n_lead,)
    dissimularity_score_avg = np.mean(dissimularity_score_lead)
    
    return dissimularity_score_avg, dissimularity_score_lead

def global_SA_MI_type(data_dir = "results/generate_dataset_DHlab_data_Scar_CV_APD",visual_QRS_mutual_dissimilarity = True):
    '''Maximum and average dissimilarity (DTWmax and DTWavg) between each MI scenario of all leads'''
    lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    baseline_type = "healthy"
    MI_location_list = ['A1', 'A2', 'A3', 'A4', 'B1_small', 'B1_large', 'B2', 'B3']
    MI_transmural_extent_list = ['subendo', 'transmural']
    MI_types = []
    for loc in MI_location_list:
        for extent in MI_transmural_extent_list:
            MI_types.append(f"{loc}_{extent}")
            
    output_dir = 'results/sensitive_analysis/global'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(f'{output_dir}/global_mutual_DTW_max.csv'):
        cases = list_subfolders(data_dir)
        for index,case in enumerate(cases):
            print("Processing index:", index, "case:", case)
            # if index < 2:
            # Load baseline ECG
            baseline_file = f"{data_dir}/{case}/{baseline_type}/predicted_ecg.csv"
            baseline_ecg = pd.read_csv(baseline_file,header=None).to_numpy()
            print("shape of baseline_ecg:", baseline_ecg.shape)  # (8, signal_length)
            ECG_list = list()
            ECG_name_list = list()
            for MI_type in MI_types:
                # Load MI ECG
                MI_file = f"{data_dir}/{case}/{MI_type}/predicted_ecg.csv"
                MI_ecg = pd.read_csv(MI_file,header=None).to_numpy()
                ECG_name = ECG_rename(MI_type)
                ECG_list.append(MI_ecg)
                ECG_name_list.append(ECG_name)
            
            pd_mutual_dissimilarity_avg_each, pd_mutual_dissimilarity_max_each = calculate_global_dissimilarity_matched(ECG_list, ECG_name_list, case) 
            if index == 0:
                pd_mutual_dissimilarity_avg = pd_mutual_dissimilarity_avg_each
                pd_mutual_dissimilarity_max = pd_mutual_dissimilarity_max_each
            else:
                pd_mutual_dissimilarity_avg = pd.concat([pd_mutual_dissimilarity_avg_each, pd_mutual_dissimilarity_avg])
                pd_mutual_dissimilarity_max = pd.concat([pd_mutual_dissimilarity_max_each, pd_mutual_dissimilarity_max])

    
        pd_mutual_dissimilarity_avg.index.name = 'Scenario name'
        pd_mutual_dissimilarity_avg = pd_mutual_dissimilarity_avg.groupby(level='Scenario name').mean(numeric_only=True)
        pd_mutual_dissimilarity_avg = pd_mutual_dissimilarity_avg.reindex(index = list(reversed(ECG_name_list)))
        pd_mutual_dissimilarity_avg.to_csv(f'{output_dir}/global_mutual_DTW_avg.csv', index=True)  

        pd_mutual_dissimilarity_max.index.name = 'Scenario name'
        pd_mutual_dissimilarity_max = pd_mutual_dissimilarity_max.groupby(level='Scenario name').mean(numeric_only=True)
        pd_mutual_dissimilarity_max = pd_mutual_dissimilarity_max.reindex(index = list(reversed(ECG_name_list)))
        pd_mutual_dissimilarity_max.to_csv(f'{output_dir}/global_mutual_DTW_max.csv', index=True)       
    else:
        pd_mutual_dissimilarity_avg = pd.read_csv(f'{output_dir}/global_mutual_DTW_avg.csv', index_col=0)
        pd_mutual_dissimilarity_max = pd.read_csv(f'{output_dir}/global_mutual_DTW_max.csv', index_col=0)
        
        # Rename the first column and first row.
        pd_mutual_dissimilarity_avg.index = (
            pd_mutual_dissimilarity_avg.index.map(ECG_rename)
        )
        pd_mutual_dissimilarity_max.index = (
            pd_mutual_dissimilarity_max.index.map(ECG_rename)
        )

        # Rename the columns.
        pd_mutual_dissimilarity_avg.columns = (
            pd_mutual_dissimilarity_avg.columns.map(ECG_rename)
        )
        pd_mutual_dissimilarity_max.columns = (
            pd_mutual_dissimilarity_max.columns.map(ECG_rename)
        )
        
    if visual_QRS_mutual_dissimilarity:
        
        anti_diag_mask = np.flip(np.eye(pd_mutual_dissimilarity_avg.shape[0], dtype=bool), axis=1)
        #####
        ###
        #
        mask_UP = np.flip(np.triu(np.ones_like(pd_mutual_dissimilarity_avg, dtype=bool)),0)   
        mask_UP[anti_diag_mask] = False # Force the anti-diagonal to remain visible.
            #
          ###
        #####
        mask_LOW = np.flip(np.tril(np.ones_like(pd_mutual_dissimilarity_avg, dtype=bool)),0)
        mask_LOW[anti_diag_mask] = False # Force the anti-diagonal to remain visible.
        # Create annotation array
        plt.figure(figsize=(12,10))
        sns.heatmap(pd_mutual_dissimilarity_avg, annot=True, mask=mask_UP, fmt=".1f", square=True, cmap="Blues") # cmap=sns.cubehelix_palette(as_cmap=True)
        plt.savefig(f'{output_dir}/fig_mutual_DTW_avg.pdf')

        plt.figure(figsize=(12,10))
        # Getting the lower Triangle of the co-relation matrix
        sns.heatmap(pd_mutual_dissimilarity_max, annot=True, mask=mask_LOW, fmt=".1f", square=True, cmap="GnBu") # crest
        plt.savefig(f'{output_dir}/fig_mutual_DTW_max.pdf')
        
        # merge into one figure
        plt.figure(figsize=(7, 5))
        ax = plt.gca()

        # Upper triangle (average).
        sns.heatmap(
            pd_mutual_dissimilarity_avg,
            mask=mask_UP,
            annot=False,
            fmt=".1f",
            square=True,
            cmap="Blues",
            cbar=True,
            cbar_kws={"label": "DTW avg","shrink":0.8,"pad":0},
            annot_kws={"size": 12},
            ax=ax
        )
        # Lower triangle (maximum).
        sns.heatmap(
            pd_mutual_dissimilarity_max,
            mask=mask_LOW,
            annot=False,
            fmt=".1f",
            square=True,
            cmap="GnBu",
            cbar=True,
            cbar_kws={"label": "DTW max","shrink":0.8,"pad":0.01},
            annot_kws={"size": 12},
            ax=ax,
            
        )
        cbar_avg = ax.collections[0].colorbar  # Colorbar for the first heatmap.
        cbar_avg.ax.tick_params(labelsize=7)   # Tick-label size.
        cbar_avg.set_label("DTW avg", fontsize=8)  # Label font size.
        
        cbar_avg2 = ax.collections[1].colorbar  # Colorbar for the second heatmap.
        cbar_avg2.ax.tick_params(labelsize=7)   # Tick-label size.
        cbar_avg2.set_label("DTW max", fontsize=8)  # Label font size.

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right',fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(),fontsize=8)
        
        # Hide the y-axis label.
        ax.set_ylabel("")
        # plt.title("Mutual QRS DTW Dissimilarity (Upper: Avg, Lower: Max)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.96, 1])
        plt.savefig(f'{output_dir}/fig_mutual_DTW_avg_max_fused.pdf', bbox_inches='tight')
        plt.close()
        
def calculate_global_dissimilarity_matched(ecgs, ecg_name, mesh_name):  
    output_dir = 'results/sensitive_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    n_signal = len(ecgs)
 
    global_dissimilarity_avg_listlist = {}
    global_dissimilarity_max_listlist = {}
    
    for i_signal in range(n_signal):
        global_dissimilarity_avg_list = {}
        global_dissimilarity_max_list = {}
        for j_signal in range(n_signal):
            # ecg_signal = np.expand_dims(ecgs[i_signal], axis=0)
            dissimularity_score_avg, dissimularity_score_lead = dtw_ecg(ecgs[i_signal], ecgs[j_signal])
            global_dissimilarity_avg_list[ecg_name[j_signal]] = dissimularity_score_avg
            global_dissimilarity_max_list[ecg_name[j_signal]] = max(dissimularity_score_lead)
        global_dissimilarity_avg_listlist[ecg_name[i_signal]] = global_dissimilarity_avg_list
        global_dissimilarity_max_listlist[ecg_name[i_signal]] = global_dissimilarity_max_list
    
    visual_peak = False
    if visual_peak:
        plt.figure(figsize=(20,16))
        # Getting the Upper Triangle of the co-relation matrix
        pd_mutual_dissimilarity_avg = pd.DataFrame.from_dict(global_dissimilarity_avg_listlist)
        matrix = np.flip(np.triu(pd_mutual_dissimilarity_avg), 0)
        sns.heatmap(pd_mutual_dissimilarity_avg, annot=True, mask=matrix, fmt=".1f", cmap=sns.cubehelix_palette(as_cmap=True)) # , vmin=0, vmax=25
        plt.savefig(f'{output_dir}/QRS_mutual_dissimilarity_avg.pdf')

        plt.figure(figsize=(20,16))
        # Getting the lower Triangle of the co-relation matrix
        pd_mutual_dissimilarity_max = pd.DataFrame.from_dict(global_dissimilarity_max_listlist)
        matrix = np.flip(np.tril(pd_mutual_dissimilarity_max), 0)
        sns.heatmap(pd_mutual_dissimilarity_max, annot=True, mask=matrix, fmt=".1f", cmap="crest")
        plt.savefig(f'{output_dir}/QRS_mutual_dissimilarity_max.pdf')
    
    pd_dissimilarity_avg = pd.DataFrame.from_dict(global_dissimilarity_avg_listlist)
    pd_dissimilarity_avg.insert(0, 'mesh_name', [mesh_name]*n_signal)

    pd_dissimilarity_max = pd.DataFrame.from_dict(global_dissimilarity_max_listlist)
    pd_dissimilarity_max.insert(0, 'mesh_name', [mesh_name]*n_signal)

    return pd_dissimilarity_avg, pd_dissimilarity_max


def main_local_SA(data_dir):
    '''Local sensitive analysis, especially for different QRST phenotypes'''   
    # 1. ***Split fiducial points: QRS onset, R peak, QRS offset, T peak, T end
    # 2. Obtain phenotypes based on fiducial points
    # calculate_qrst_phenotype(plot_figure=False, data_dir=data_dir)
    # 3. Sensitive analysis for each phenotype
    visualize_local()
    
def find_fiducial_points_Julia(ecg_lead, lat,lead_name ='I', plot_figure=True):
    '''Find fiducial points in a given ECG lead signal'''
    dV = abs(np.gradient(ecg_lead))
    ddV = abs(np.gradient(dV))
    dV[0:2] = 0.0 # remove gradient artefacts
    ddV[0:2] = 0.0
    # Find Q start
    dVTOL_end_of_Twave = 0.002 # 0.0002 # mV/ms # TODO This trick will only work with simulated signals
    if lead_name != 'V6':
        dvTOL_start_of_Twave = 0.002 # 0.0002 # mV/ms
    else:
        dvTOL_start_of_Twave = 0.001
    # Find qrs end
    max_lat = np.max(lat)
    qrs_end_idx = int(max_lat)  # QRS offset
    
    # Find L point
    fs = 1000
    ST_period = 0.06
    L_idx = int(qrs_end_idx + fs*ST_period)

    QRS_segment = ecg_lead[:qrs_end_idx]
    ST_segment = ecg_lead[qrs_end_idx:]
    t_amplitude = abs(ST_segment).max()
    t_peak_idx = np.where(abs(ST_segment) == t_amplitude)[0][0] + qrs_end_idx  # T peak
    # Find T-wave end
    i = len(ecg_lead) - 1
    for i in range(len(ecg_lead) - 1, t_peak_idx, -1):
        if (dV[i] > dVTOL_end_of_Twave):
            break
    t_end_idx = i  # T end
    # Find T-wave start
    j = qrs_end_idx
    # for j in range(t_peak_idx-int(0.5*(t_end_idx-t_peak_idx)), qrs_end_idx,-1):
    for j in range(qrs_end_idx+10,t_peak_idx,1):
        if (dV[j] > dvTOL_start_of_Twave):
            break
    t_start_idx = j  # T start
    
        
    # plot
    if plot_figure:
        plt.figure(figsize=(6,4))
        plt.plot(ecg_lead, label=f'ECG')
        # plt.axvline(x=max_lat, color='r', linestyle='--', label='QRS End')
        plt.plot(qrs_end_idx, ecg_lead[qrs_end_idx], 'ro', label='QRS End point')
        plt.plot(t_peak_idx, ecg_lead[t_peak_idx], 'go', label='T Peak point')
        plt.plot(t_end_idx, ecg_lead[t_end_idx], 'bo', label='T End point')
        plt.plot(t_start_idx, ecg_lead[t_start_idx], 'mo', label='T Start point')
        plt.title('Baseline ECG with QRS End Indicated')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.legend(ncol=2)
        plt.show()
        
    points = {'J':qrs_end_idx,'L':L_idx,'T1':t_start_idx,'T':t_peak_idx,'T2':t_end_idx }
    return points,QRS_segment, ST_segment, t_amplitude
        
def nk_ecg_analysis(signal,fs):
    '''NeuroKit2 ECG Analysis'''
    # Plot the ECG signal
    # plt.plot(signal)
    # plt.show()
    sig_length = signal.shape[-1]

    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs,show=False)
    # print(rpeaks)

    # Delineate the ECG signal
    _, waves_peak = nk.ecg_delineate(signal,
                                     rpeaks,
                                     sampling_rate=fs,
                                     method="dwt",  # Can be one of ["dwt", "peak","cwt"]
                                     show=False,
                                     show_type='all')
    # Rename the fields.
    # ECG_R_Onsets-->ECG_Q_Onsets
    waves_peak['ECG_Q_Onsets'] = waves_peak.pop('ECG_R_Onsets')
    # ECG_R_Offsets-->ECG_S_Offsets
    waves_peak['ECG_S_Offsets'] = waves_peak.pop('ECG_R_Offsets')

    # for i in waves_peak.keys():
    #     print(i)
    #     print(waves_peak[i])
    #     print(len(waves_peak[i]))

    # Add the R peak.
    waves_peak['ECG_R_Peaks'] = rpeaks['ECG_R_Peaks']
    # Convert lists in the dictionary to arrays.
    for key, value in waves_peak.items():
        value = np.array(value)
        value = value[~np.isnan(value)]
        value = value.astype(int)
        # Omit values outside the signal range, including negative values.
        value = value[value < sig_length]
        value = value[value > 0]
        waves_peak[key] = value

    # check the result
    # Align the other points by heartbeat according to the R-peak positions.
        # p_onsets, p_peaks, p_offsets, Q_onsets, and Q_peaks precede the R peak.
        # S_peaks, S_offsets, T_onsets, T_peaks, and T_offsets follow the R peak.
    front_points = ['ECG_P_Onsets', 'ECG_P_Peaks', 'ECG_P_Offsets', 'ECG_Q_Onsets', 'ECG_Q_Peaks']
    end_points = ['ECG_S_Peaks', 'ECG_S_Offsets', 'ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets']
    waves_peak_new = {key:np.full(len(waves_peak['ECG_R_Peaks']),-1) for key in waves_peak.keys()}
    incomplete_beats = []
    for i in range(len(waves_peak['ECG_R_Peaks'])):
        waves_peak_new['ECG_R_Peaks'][i] = waves_peak['ECG_R_Peaks'][i]
        for key, value in waves_peak.items():
            # Build a new dictionary keyed like waves_peak and align its values to the R peaks.
            if key in front_points:
                # Points between the previous and current R peaks.
                if i == 0:
                    value_front = 0
                else:
                    value_front = waves_peak['ECG_R_Peaks'][i-1]
                value_end = waves_peak['ECG_R_Peaks'][i]
            elif key in end_points:
                # Points between the current and next R peaks.
                value_front = waves_peak['ECG_R_Peaks'][i]
                if i == len(waves_peak['ECG_R_Peaks'])-1:
                    value_end = sig_length
                else:
                    value_end = waves_peak['ECG_R_Peaks'][i+1]

            if key != 'ECG_R_Peaks':
                # Find values between value_front and value_end.
                value = value[value >= value_front]
                value = value[value < value_end]
                # print(len(waves_peak[key]))
                # print('key',key, 'value',value)
                if len(value) > 0:
                    waves_peak_new[key][i] = value[0]
                else:
                    if i not in incomplete_beats:
                        incomplete_beats.append(i)
    # print(incomplete_beats)
    # # Remove incomplete-beat data.
    # for key, value in waves_peak_new.items():
    #     waves_peak_new[key] = np.delete(value,incomplete_beats)


    # plot
    # plt.figure(figsize=(10, 5))
    # plt.plot(signal)
    # # plt.scatter(rpeaks['ECG_R_Peaks'], signal[rpeaks['ECG_R_Peaks']], color='red')
    # for key, value in waves_peak.items():
    #     if key == 'ECG_R_Peaks':
    #         plt.scatter(value, signal[value], color='red',label=key)
    #     # Convert lists to arrays and omit NaN values.
    #     else:
    #         plt.scatter(value, signal[value],label=key)
    # plt.legend()
    # plt.show()

    # print(waves_peak_new)
    # Heart-rate variability.
    # ecg_hrv = nk.ecg_hrv(signal, rpeaks, sampling_rate=fs)
    return waves_peak_new

# def find_fiducial_points_nk(ecg_lead, lat, lead_name ='I', plot_figure=True):
#     '''Find fiducial points using neurokit2
#     Currently unusable: P waves are missing and NeuroKit appears unable to find enough points. Only R and S peaks work.'''
    
#     ecg_data_tiled = np.tile(ecg_lead, 10) 
#     signal_length = len(ecg_lead)
#     fs = 1000
#     ecg_points = nk_ecg_analysis(ecg_data_tiled, fs)
#     # S_Offsets_all = ecg_points.get('ECG_S_Offsets', np.array([]))
#     # S_Offsets_candidate = S_Offsets_all[
#     #     (S_Offsets_all >= signal_length) & (S_Offsets_all < signal_length * 2)
#     # ]
#     # qrs_end_idx = S_Offsets_candidate[0]
    
#     T_start_all = ecg_points.get('ECG_T_Onsets', np.array([]))
#     T_start_candidate = T_start_all[
#         (T_start_all >= signal_length) & (T_start_all < signal_length * 2)
#     ]
#     t_start_idx = T_start_candidate[0]
    
#     T_end_all = ecg_points.get('ECG_T_Offsets', np.array([]))
#     T_end_candidate = T_end_all[
#         (T_end_all >= signal_length) & (T_end_all < signal_length * 2)
#     ]
#     t_end_idx = T_end_candidate[0]
    
#     T_peak_all = ecg_points.get('ECG_T_Peaks', np.array([]))
#     T_peak_candidate = T_peak_all[
#         (T_peak_all >= signal_length) & (T_peak_all < signal_length * 2)
#     ]
#     t_peak_idx = T_peak_candidate[0]
    
            
#     # plot
#     if plot_figure:
#         plt.figure(figsize=(6,4))
#         plt.plot(ecg_lead, label=f'ECG')
#         # plt.axvline(x=max_lat, color='r', linestyle='--', label='QRS End')
#         plt.plot(qrs_end_idx, ecg_lead[qrs_end_idx], 'ro', label='QRS End point')
#         plt.plot(t_peak_idx, ecg_lead[t_peak_idx], 'go', label='T Peak point')
#         plt.plot(t_end_idx, ecg_lead[t_end_idx], 'bo', label='T End point')
#         plt.plot(t_start_idx, ecg_lead[t_start_idx], 'mo', label='T Start point')
#         plt.title('Baseline ECG with QRS End Indicated')
#         plt.xlabel('Time (ms)')
#         plt.ylabel('Amplitude')
#         plt.legend(ncol=2)
#         plt.show()
#     return qrs_end_idx, t_start_idx, t_end_idx



def calculate_pathological_Q(ecg_signal):

    eps = 1e-4
    
    ecg_signal_each_lead = ecg_signal
    peaks, _ = find_peaks(ecg_signal_each_lead, height=0)
    if len(peak_prominences(ecg_signal_each_lead, peaks)[0]) > 0:
        peak_R = peaks[np.argmax(peak_prominences(ecg_signal_each_lead, peaks)[0])] 
    else:
        peak_R = np.where(abs(ecg_signal_each_lead) < (np.min(abs(ecg_signal_each_lead)) + eps))[0]
    peaks, _ = find_peaks(-ecg_signal_each_lead, height=0)
    if len(peak_prominences(-ecg_signal_each_lead, peaks)[0]) > 0:
        peak_Q = peaks[np.argmax(peak_prominences(-ecg_signal_each_lead, peaks)[0])]            
    else:
        peak_Q = np.where(abs(ecg_signal_each_lead) < (np.min(abs(ecg_signal_each_lead)) + eps))[0]
    
    if not isinstance(peak_R, np.int64):
        if peak_R.shape[0] > 1:
            peak_R = peak_R[1]
        else:
            peak_R = peak_R[0]

    if not isinstance(peak_Q, np.int64):
        if peak_Q.shape[0] > 1:
            peak_Q = peak_Q[1]
        else:
            peak_Q = peak_Q[0]

    R_amplitude = ecg_signal_each_lead[peak_R] 
    Q_amplitude = - ecg_signal_each_lead[peak_Q] 
    Q_R_amplitude_ratio = abs(Q_amplitude/R_amplitude)

    # print(np.min(abs(ecg_signal_each_lead)))           
    signal_zero = (np.where(abs(ecg_signal_each_lead) < (np.min(abs(ecg_signal_each_lead)) + eps))[0]).tolist()
    Q_nearest_zero = min(signal_zero, key=lambda x: abs(x-peak_Q))
    Q_duration = abs(Q_nearest_zero - peak_Q)/1000

    visual_check = False
    if visual_check:
        plt.plot(ecg_signal_each_lead)
        plt.plot(peak_Q, ecg_signal_each_lead[peak_Q], "x")
        plt.plot(peak_R, ecg_signal_each_lead[peak_R], "o")
        # plt.plot(Q_nearest_zero, ecg_signal_each_lead[Q_nearest_zero], "o")
        plt.plot(np.zeros_like(ecg_signal_each_lead), "--", color="gray")
        plt.show()
    
    if Q_R_amplitude_ratio > 0.25:
        Pathological_Q = 1
    elif Q_duration > 0.03:
        Pathological_Q = 1
    else:
        Pathological_Q = 0    

    Q_duration = Q_duration * 1000
    return Pathological_Q, Q_R_amplitude_ratio, Q_duration

def calculate_QRS_fractionation(ecg_signal):
    peaks_R, _ = find_peaks(ecg_signal, height=0.01) 
    peaks_Q, _ = find_peaks(-ecg_signal, height=0.01)
    QRS_fractionation = peaks_R.shape[0] + peaks_Q.shape[0]
    if QRS_fractionation > 1:
        QRS_fractionation = QRS_fractionation - 3 # except Q, R, S peak
    else:
        QRS_fractionation = 0
    QRS_fractionation
            
    return QRS_fractionation

    
def calculate_qrst_phenotype(plot_figure=True, data_dir="results/generate_dataset_DHlab_data_Scar_CV_APD"):
    '''Calculate QRST phenotypes based on fiducial points'''
    
    normal_type = "healthy"
    MI_location_list = ['A1', 'A2', 'A3', 'A4', 'B1_small', 'B1_large', 'B2', 'B3']
    MI_transmural_extent_list = ['subendo', 'transmural']
    
    MI_types = [f"{loc}_{extent}" for loc in MI_location_list for extent in MI_transmural_extent_list]
    all_types = [normal_type] + MI_types
    
    output_dir = 'results/sensitive_analysis/local'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/qrst_phenotype.csv'
    
    cases = list_subfolders(data_dir)
    
    # If the file exists, remove it first to avoid appending duplicate old data.
    if os.path.exists(output_file):
        os.remove(output_file)
    
    for index, case in enumerate(cases):
        # if index < 2:  # Process only the first two cases.
        for baseline_type in all_types:
            print("Processing case:", case, "type:", baseline_type)
            
            lat_file = f"{data_dir}/{case}/{baseline_type}/lat_simulation.csv"
            baseline_file = f"{data_dir}/{case}/{baseline_type}/predicted_ecg.csv"
            
            lat = pd.read_csv(lat_file, header=None).to_numpy().squeeze()
            baseline_ecg = pd.read_csv(baseline_file, header=None).to_numpy()
            
            lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            
            phenotype_rows = []
            
            for i, ecg_lead in enumerate(baseline_ecg):
                # ------Find fiducial points-----------
                points, QRS_segment, ST_segment, t_amplitude = find_fiducial_points_Julia(ecg_lead, lat, lead_names[i], plot_figure)
                
                # ------Calculate phenotypes-----------
                qrs_end_idx = points['J']
                t_peak_idx = points['T']
                t_end_idx = points['T2']
                L_idx = points['L']
                
                # QRS phenotypes
                QRS_duration = qrs_end_idx
                Pathological_Q, Q_R_amplitude_ratio, Q_duration = calculate_pathological_Q(QRS_segment)
                QRS_fractionation = calculate_QRS_fractionation(ecg_lead)
                
                # ST-T phenotypes
                t_sign = np.sign(ST_segment[t_peak_idx - qrs_end_idx])
                t_peak = t_sign * t_amplitude
                t_min = np.amin(ST_segment)
                t_max = abs(np.amax(ST_segment))
                t_polarity = (t_max + t_min) / (max(abs(t_max), abs(t_min)))
                qt_dur = t_end_idx
                t_pe = t_end_idx - t_peak_idx
                ST_amplitude = ecg_lead[L_idx]
                
                phenotype_rows.append({
                    "id": index,
                    "case": case,
                    "baseline_type": baseline_type,
                    "lead": lead_names[i],
                    "QRS duration": QRS_duration,
                    "Pathological Q": Pathological_Q,
                    "QR ratio": Q_R_amplitude_ratio,
                    "Q duration": Q_duration,
                    "QRS fractionation": QRS_fractionation,
                    "T peak": t_peak,
                    "T polarity": t_polarity,
                    "ST amplitude": ST_amplitude,
                    "QT duration": qt_dur,
                    "T duration": t_pe
                })
            
            # Save the CSV after each baseline_type is processed.
            df_phenotype = pd.DataFrame(phenotype_rows)
            if not os.path.exists(output_file):
                df_phenotype.to_csv(output_file, index=False)
            else:
                df_phenotype.to_csv(output_file, index=False, mode='a', header=False)
def visualize_local():
    '''Visualize the lobal sensitive analysis for each phenotype'''
    input_file = 'results/sensitive_analysis/local/qrst_phenotype.csv'
    output_dir = 'results/sensitive_analysis/local'
    df = pd.read_csv(input_file)
    df['baseline_type'] = df['baseline_type'].apply(ECG_rename)
    
    normal_type = "healthy"
    MI_location_list = ['A1', 'A2', 'A3', 'A4', 'B1_small', 'B1_large', 'B2', 'B3']
    MI_transmural_extent_list = ['subendo', 'transmural']
    
    MI_types = [f"{loc}_{extent}" for loc in MI_location_list for extent in MI_transmural_extent_list]
    all_types = [normal_type] + MI_types
    
    all_types = [ECG_rename(name) for name in all_types]
    # Select numeric columns
    numeric_cols = ['QRS duration', 'Pathological Q', 'QR ratio', 
                    'Q duration', 'QRS fractionation', 'T peak', 'T polarity', 
                    'ST amplitude', 'QT duration', 'T duration']

    # Calculate healthy stats
    healthy_df = df[df['baseline_type'] == 'healthy']
    healthy_mean = healthy_df[numeric_cols].mean()
    healthy_std = healthy_df[numeric_cols].std()

    # Calculate stats for all types
    grouped_means = df.groupby('baseline_type')[numeric_cols].mean()
    # Calculate Z-scores (Standardized Difference) relative to healthy
    # (Mean_Type - Mean_Healthy) / Std_Healthy
    z_scores = (grouped_means - healthy_mean) / healthy_std

    # Drop 'healthy' row from z_scores for visualization (since it will be 0)
    z_scores_plot = z_scores.drop('healthy')

    # Plot Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(z_scores_plot, annot=True, cmap='RdBu_r', center=0, fmt=".2f")
    plt.title('Standardized Difference of Phenotypes vs Healthy (Z-score)')
    plt.ylabel('Baseline Type')
    plt.xlabel('Phenotype Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_difference.png')

    # Plot Boxplots for key features
    # We select a few important ones to avoid overcrowding
    key_features = numeric_cols

    fig, axes = plt.subplots(len(key_features), 1, figsize=(14, 6 * len(key_features)))

    for i, feature in enumerate(key_features):
        # Sort order by median of the feature to make it readable
        order = all_types
        
        sns.boxplot(data=df, x='baseline_type', y=feature, ax=axes[i], order=order, palette="Set3")
        axes[i].set_title(f'Distribution of {feature} by Type')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        axes[i].axhline(healthy_mean[feature], color='r', linestyle='--', label='Healthy Mean')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/boxplots_key_features.png')
    
    
    # Clustermap
    # We use the same z_scores_plot data
    g = sns.clustermap(
        z_scores_plot, 
        cmap='Blues',       
        annot=True, 
        fmt=".2f", 
        figsize=(8, 8),
        dendrogram_ratio=(0.15, 0.15), # Adjust the side dendrogram ratio to leave more room for the heatmap.
        cbar_pos=(0.02, 0.8, 0.03, 0.15), # Reposition the colorbar to avoid overlap.
        cbar = False,
    )
    g.ax_cbar.set_visible(False)
    g.ax_heatmap.set_xlabel('Phenotype features')
    g.ax_heatmap.set_ylabel('scenario types')
    # 3. Rotate axis labels to prevent overlap.
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')

    # 4. Save as PDF.
    g.savefig(f'{output_dir}/clustermap_difference.pdf', bbox_inches='tight')
    
    

if __name__ == "__main__":
    data_dir = r"output\generate_dataset_DHlab_data_Scar_CV_APD_chronic_new_v1"
    # data_dir = r"/path/to/dataset/generate_dataset_DHlab_data_Scar_CV_APD"
    # main_global_SA(data_dir=data_dir)
    main_local_SA(data_dir=data_dir)
    # global_SA_baseline(data_dir=data_dir)
    # global_SA_MI_type(data_dir=data_dir)
