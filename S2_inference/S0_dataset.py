import os
import random
import numpy as np
import torch
import glob
import torch.utils.data as data
import sys

sys.path.append('.')
sys.path.append('..')
from utils.utils import visualize_PC_with_label
import re
import pandas as pd
from collections import defaultdict
from scipy.signal import decimate
from utils.dataset_utils import *

def split_dataset(data_dir = r"/path/to/dataset/DHlab_valid"):


    # 1. Read all sample filenames.
    df = pd.read_csv(os.path.join(data_dir, 'valid_list.csv'))
    filenames = df['filename'].tolist()
    
    filenames = [os.path.splitext(x)[0] for x in filenames]

    # 2. Parse patient_id from each filename.
    patient_to_samples = defaultdict(list)

    for s in filenames:
        patient_id = s.split('_')[0]   # IMMC001
        patient_to_samples[patient_id].append(s)

    patients = list(patient_to_samples.keys())

    # 3. Shuffle at the patient level.
    np.random.seed(42)
    np.random.shuffle(patients)

    # 4. Determine split sizes.
    n = len(patients)
    train_p = patients[:int(0.7 * n)]
    val_p   = patients[int(0.7 * n):int(0.85 * n)]
    test_p  = patients[int(0.85 * n):]

    # 5. Write list files.
    def write_list(filename, patient_ids):
        with open(filename, "w") as f:
            for pid in patient_ids:
                for sample in patient_to_samples[pid]:
                    f.write(sample + "\n")
    dataset_dir = 'dataset'
    os.makedirs( dataset_dir, exist_ok=True)
    write_list(f"{dataset_dir}/train.list", train_p)
    write_list(f"{dataset_dir}/val.list", val_p)
    write_list(f"{dataset_dir}/test.list", test_p)
    
def get_max_ecg_length(dataset_choose= 'v1'):


    path = 'dataset/'
    filenames = [f for f in os.listdir(path)
        if os.path.isdir(os.path.join(path, f))]

    max_length = 0
    for filename in filenames:
        print(filename)
        datapath = path + filename + f'/{dataset_choose}/'

        signal_files = glob.glob(datapath + 'ECG_simulated_*' + '.csv')
        num_signal = len(signal_files)
        for id in range(num_signal):
            ECG_value = np.loadtxt(signal_files[id], delimiter=',')
            if ECG_value.shape[1] > max_length:
                max_length = ECG_value.shape[1]
    print('Max ECG length:', max_length)

class LoadDataset(data.Dataset):
    '''Use one ECG and one MI type as a case-level/instance-level sample.'''
    def __init__(self, path='dataset/', num_input=2048, split='train', ecg_segment = 'QRST'):
        self.path = path
        self.num_input = num_input
        self.use_cobiveco = True
        self.data_augment = False
        self.signal_length = 512  # Max ECG length: 1086 （one heartbeat)
        self.ecg_segment = ecg_segment

        with open(path + '{}.list'.format(split), 'r') as f:
            filenames = [line.strip() for line in f]

        self.metadata = list()
        self.filenames = filenames
        for filename in filenames:
            # print(filename)
            datapath = path + filename + '/'
            simulated_path = datapath + 'v1/'

            unit = 0.1
            nodesXYZ, label_index = getCobiveco_vtu(datapath + 'mesh_cobiveco_AHA17.vtu')
            
            PC_XYZ_labeled = np.concatenate((unit*nodesXYZ, label_index), axis=1)           
            electrode_node = np.loadtxt(datapath + filename + '_electrode_xyz.csv', delimiter=',')
            Coord_base_apex = np.loadtxt(datapath + filename + '_coarse_nodefield_ab_cut.csv', delimiter=',')
            Coord_apex, Coord_base = Coord_base_apex[1], Coord_base_apex[0] 
            electrode_index = 4*np.ones(electrode_node.shape[0], dtype=np.int32)
            electrode_XYZ_labeled = np.concatenate((unit*electrode_node, electrode_index[..., np.newaxis]), axis=1)
            
            signal_files = glob.glob(simulated_path + 'ECG_simulated_*' + '.csv')
            num_signal = len(signal_files)
            # print(num_signal)
            for id in range(num_signal):
                signal_file_name = os.path.basename(signal_files[id])        
                MI_type = signal_file_name.split('ECG_simulated_')[1].replace('.csv', '')
                
                lat_file = simulated_path +  f'LAT_simulated_{MI_type}.csv'
                lat = pd.read_csv(lat_file, header=None).to_numpy().squeeze()
                max_lat = np.max(lat)
                qrs_end_idx = int(max_lat)  # QRS offset

                MI_index = np.zeros(nodesXYZ.shape[0], dtype=np.int32)
                ECG_value = np.loadtxt(signal_files[id], delimiter=',') # (8, N)

                QRS_segment = ECG_value[:,:qrs_end_idx]
                ST_segment = ECG_value[:,qrs_end_idx:]
                
                if self.ecg_segment == 'QRS':
                    ECG_value = QRS_segment
                elif self.ecg_segment == 'ST':
                    ECG_value = ST_segment
                else:
                    ECG_value = ECG_value
                # resample from 1000 Hz to 500 Hz
                # Downsample each lead independently.
                ECG_value_500 = decimate(
                    ECG_value,
                    q=2,              # Downsampling factor: 1000 -> 500
                    axis=1,           # Time dimension
                    ftype='fir',      # FIR anti-aliasing filter (recommended)
                    zero_phase=True   # Zero phase to avoid phase distortion
                )
                # print('ECG_value shape:', ECG_value.shape)
                cur_len = ECG_value_500.shape[1]
                if cur_len < self.signal_length:
                    ECG_value_u = np.pad(ECG_value_500, ((0, 0), (0, self.signal_length-ECG_value_500.shape[1])), 'constant')   
                else:
                    ECG_value_u = ECG_value_500[:, :self.signal_length]

                
                if MI_type == 'B1_large_transmural_slow' or MI_type == 'normal' or MI_type == 'A2_30_40_transmural':
                    continue

                if re.compile(r'5_transmural|0_transmural', re.IGNORECASE).search(MI_type): # remove apical MI size test case
                    continue

                if re.compile(r'AHA', re.IGNORECASE).search(MI_type): # remove randomly generated MI
                    continue

                # if not re.compile(r'5_transmural|0_transmural', re.IGNORECASE).search(MI_type) and not (MI_type == 'A2_transmural'): # remove apical MI size test case
                #     continue

                # if not re.compile(r'AHA', re.IGNORECASE).search(MI_type): # test only random MI!
                #     continue
                #             
                # if MI_type.find('subendo') != -1:
                #     continue
 
                # if MI_type != 'B3_transmural' and MI_type != 'A3_transmural' and MI_type != 'A2_transmural':
                #     continue  

                # print(MI_type)             

                if MI_type != 'healthy':
                    # Scar_filename = f'{simulated_path}{filename}_coarse_lvscarnodes_{MI_type}.csv'
                    # BZ_filename = f'{simulated_path}{filename}_coarse_lvborderzonenodes_{MI_type}.csv'
                    Scar_filename = f'{simulated_path}{filename}_coarse_new_v1_lvscarnodes_{MI_type}.csv'
                    BZ_filename = f'{simulated_path}{filename}_coarse_new_v1_lvborderzonenodes_{MI_type}.csv'

                    if MI_type == 'B1_large_transmural_slow':
                        Scar_filename = Scar_filename.replace('_slow', '')
                        BZ_filename = BZ_filename.replace('_slow', '')

                    Scar_node = np.unique((np.loadtxt(Scar_filename, delimiter=',')-1).astype(int))
                    BZ_node = np.unique((np.loadtxt(BZ_filename, delimiter=',')-1).astype(int))
                    MI_index[Scar_node] = 1
                    MI_index[BZ_node] = 2
                ECG_array = np.array(ECG_value_u)
                MI_array = np.array(MI_index)
                MI_type_id = np.array(id)
                # print(MI_type_id)
                
                partial_PC_labeled_array, idx_remained = resample_pcd(PC_XYZ_labeled, self.num_input)
                partial_MI_lab_array = MI_array[idx_remained]
                partial_PC_labeled_array_coarse, idx_remained = resample_pcd(PC_XYZ_labeled, self.num_input//4)             
                # visualize_PC_with_label(partial_PC_labeled_array[:, 0:3], partial_MI_array)
                partial_PC_electrode_labeled_array = partial_PC_labeled_array # np.concatenate((partial_PC_labeled_array, electrode_XYZ_labeled), axis=0)
                partial_PC_electrode_XYZ = partial_PC_electrode_labeled_array[:, 0:3]
                partial_PC_electrode_lab = partial_PC_electrode_labeled_array[:, 3:]
                # partial_MI_lab_array = partial_MI_lab_array + np.where(partial_PC_electrode_labeled_array[0:self.num_input, -1]==1.0, 3, 0)
                # visualize_PC_with_label(partial_PC_labeled_array[:, 0:3], partial_MI_lab_array)

                partial_PC_electrode_XYZ_normalized = normalize_data(partial_PC_electrode_XYZ, Coord_apex)
                if self.data_augment:
                    scaling = random.uniform(0.8, 1.2)
                    partial_PC_electrode_XYZ_normalized = scaling*translate_point(jitter_point(rotate_point(partial_PC_electrode_XYZ_normalized, np.random.random()*np.pi)))
                partial_PC_electrode_XYZ_normalized_labeled = np.concatenate((partial_PC_electrode_XYZ_normalized, partial_PC_electrode_lab), axis=1)

                partial_PC_electrode_XYZ_normalized_coarse = normalize_data(partial_PC_labeled_array_coarse[:, 0:3], Coord_apex)
                partial_PC_electrode_XYZ_normalized_labeled_coarse = np.concatenate((partial_PC_electrode_XYZ_normalized_coarse, partial_PC_labeled_array_coarse[:, 3:]), axis=1)

                self.metadata.append((partial_PC_electrode_XYZ_normalized_labeled, partial_MI_lab_array, ECG_array, partial_PC_electrode_XYZ_normalized_labeled_coarse, MI_type, filename))

    def __getitem__(self, index):
        partial_PC_electrode_XYZ_normalized_labeled, partial_MI_array, ECG_array, partial_PC_electrode_XYZ, MI_type, filename = self.metadata[index]

        partial_input = torch.from_numpy(partial_PC_electrode_XYZ_normalized_labeled).float()
        gt_MI = torch.from_numpy(partial_MI_array).long()
        ECG_input = torch.from_numpy(ECG_array).float()
        partial_input_coarse = torch.from_numpy(partial_PC_electrode_XYZ).float()

        return partial_input, ECG_input, gt_MI, partial_input_coarse, MI_type,filename

    def __len__(self):
        return len(self.metadata)


class NUHDataset_validation(data.Dataset):
    '''
    '''
    def __init__(self, CONFIG = None, path='dataset/', split='test',
                 num_input=1024*4):
        if CONFIG is None:
            CONFIG = {
            "ecg": "/path/to/dataset/ECG_series",
            "ecg_paired": "/path/to/dataset/ECG_series_paired_heartbeats",
            "mri": "/path/to/dataset/mesh_conversion_InSilicoHeartGen",
            "mri_csv": "/path/to/dataset/DHlab_valid",
            "metadata_path": "/path/to/dataset/metadata.csv",
            "paired_csv_path": "/path/to/dataset/paired_ecg_mri.csv"
            }
        self.CONFIG = CONFIG
        
        self.paired_csv_path = CONFIG["paired_csv_path"]
        
        if not os.path.exists(self.paired_csv_path):
            paired_ecg_mri_NUH(CONFIG)
            
        
        with open(path + '{}.list'.format(split), 'r') as f:
            filenames = [line.strip() for line in f]
        
        self.df_paired = pd.read_csv(self.paired_csv_path, encoding="gbk")
        # self.df_paired = self.df_paired[self.df_paired['valid'] == 1]
        
        # # choose only the samples in the split list
        self.df_paired = self.df_paired[self.df_paired['Mesh_Folder'].isin(filenames)]
                
        self.num_input = num_input
        self.use_cobiveco = True
        self.data_augment = False
        self.signal_length = 512  # Max ECG length: 668 （one heartbeat)

    def __getitem__(self, index):
        try:
            row = self.df_paired.iloc[index]
            mesh_name = row['Mesh_Folder']
            ecg_folder_name = row['ECG_Folder'].split('.pdf')[0]
            
            Mesh_path = os.path.join(self.CONFIG["mri"], mesh_name, "ensi", "mesh_cobiveco_AHA17_scar_transmural.vtu")
            ECG_path = os.path.join(self.CONFIG["ecg_paired"], f"{ecg_folder_name}.npy")
            
            # ------------ Read MRI data (.vtu) -------------
            # load point cloud nodes
            nodesXYZ, label_index = getCobiveco_vtu(Mesh_path)
            # load scar
            MI_array = getScarLabel_vtu(Mesh_path)  # 0/1 label
            # process pointcload
            unit = 0.1 # convert from cm to m
            PC_XYZ_labeled = np.concatenate((unit*nodesXYZ, label_index), axis=1)
            Coord_base_apex_path = os.path.join(self.CONFIG["mri_csv"], mesh_name, f"{mesh_name}_coarse_nodefield_ab_cut.csv")         
            Coord_base_apex = np.loadtxt(Coord_base_apex_path, delimiter=',')
            Coord_apex, Coord_base = Coord_base_apex[1], Coord_base_apex[0] 
            partial_PC_labeled_array, idx_remained = resample_pcd(PC_XYZ_labeled, self.num_input)
            partial_MI_lab_array = MI_array[idx_remained]
            partial_PC_labeled_array_coarse, idx_remained = resample_pcd(PC_XYZ_labeled, self.num_input//4)             
            # visualize_PC_with_label(partial_PC_labeled_array[:, 0:3], partial_MI_array)
            partial_PC_electrode_labeled_array = partial_PC_labeled_array # np.concatenate((partial_PC_labeled_array, electrode_XYZ_labeled), axis=0)
            partial_PC_electrode_XYZ = partial_PC_electrode_labeled_array[:, 0:3]
            partial_PC_electrode_lab = partial_PC_electrode_labeled_array[:, 3:]
            # partial_MI_lab_array = partial_MI_lab_array + np.where(partial_PC_electrode_labeled_array[0:self.num_input, -1]==1.0, 3, 0)
            # visualize_PC_with_label(partial_PC_labeled_array[:, 0:3], partial_MI_lab_array)

            partial_PC_electrode_XYZ_normalized = normalize_data(partial_PC_electrode_XYZ, Coord_apex)
            if self.data_augment:
                scaling = random.uniform(0.8, 1.2)
                partial_PC_electrode_XYZ_normalized = scaling*translate_point(jitter_point(rotate_point(partial_PC_electrode_XYZ_normalized, np.random.random()*np.pi)))
            partial_PC_electrode_XYZ_normalized_labeled = np.concatenate((partial_PC_electrode_XYZ_normalized, partial_PC_electrode_lab), axis=1)

            partial_PC_electrode_XYZ_normalized_coarse = normalize_data(partial_PC_labeled_array_coarse[:, 0:3], Coord_apex)
            partial_PC_electrode_XYZ_normalized_labeled_coarse = np.concatenate((partial_PC_electrode_XYZ_normalized_coarse, partial_PC_labeled_array_coarse[:, 3:]), axis=1)


            # --------------Read ECG data (.npy) ------------
            ecg = load_ecg_npy_8leaads(ECG_path)
            print('ecg.shape',ecg.shape)
            # resample from 1000 Hz to 500 Hz
            # Downsample each lead independently.
            # ECG_value_500 = ecg
            ECG_value_500 = decimate(
                ecg,
                q=2,              # Downsampling factor: 1000 -> 500
                axis=1,           # Time dimension
                ftype='fir',      # FIR anti-aliasing filter (recommended)
                zero_phase=True   # Zero phase to avoid phase distortion
            )
            # print('ECG_value shape:', ECG_value.shape)
            cur_len = ECG_value_500.shape[1]
            if cur_len < self.signal_length:
                ECG_value_u = np.pad(ECG_value_500, ((0, 0), (0, self.signal_length-ECG_value_500.shape[1])), 'constant')   
            else:
                ECG_value_u = ECG_value_500[:, :self.signal_length]
            ECG_array = np.array(ECG_value_u)
            
            # --------------convert to tensor--------------
            partial_input = torch.from_numpy(partial_PC_electrode_XYZ_normalized_labeled).float()
            gt_MI = torch.from_numpy(partial_MI_lab_array).long()
            ECG_input = torch.from_numpy(ECG_array).float()
            partial_input_coarse = torch.from_numpy(partial_PC_electrode_XYZ_normalized_labeled_coarse).float()
            
            return partial_input, ECG_input, gt_MI, partial_input_coarse, mesh_name, ecg_folder_name
        except FileNotFoundError:
            print(f"跳过缺失文件: {Mesh_path}")
            # Strategy A: recursively return the next sample.
            new_idx = (index + 1) % len(self)
            return self.__getitem__(new_idx)
    def __len__(self):
        return len(self.df_paired)  
    
    def get_pairing_dict(self):
        return self.df_paired



    
if __name__ == '__main__':
    # split_dataset()
    # get_max_ecg_length(dataset_choose='v1')
    
    
    ROOT = 'dataset/'

    # train_dataset = LoadDataset(path=ROOT,  split='train')
    # val_dataset = LoadDataset(path=ROOT, split='val')
    # test_dataset = LoadDataset(path=ROOT, split='test')
    # print("\033[33mTraining dataset\033[0m has {} pair of partial and ground truth point clouds".format(len(train_dataset)))
    # print("\033[33mValidation dataset\033[0m has {} pair of partial and ground truth point clouds".format(len(val_dataset)))
    # print("\033[33mTesting dataset\033[0m has {} pair of partial and ground truth point clouds".format(len(test_dataset)))

    # # visualization
    # partial_input, ECG_input, gt_MI, partial_input_coarse, MI_type = train_dataset[random.randint(0, len(train_dataset))-1]
    # print("partial input shape:", partial_input.shape) #torch.Size([2048, 10])
    # print("ECG input shape:", ECG_input.shape) # ([8, 512])
    # print("ground truth MI shape:", gt_MI.shape) # ([2048])
    # print("partial input coarse point cloud has {} points".format(len(partial_input_coarse))) # 512
    # print("MI type:", MI_type)
    
    
    # test_dataset = LoadDataset(path=ROOT, split='test')
    # file_names = test_dataset.filenames
    # print(type(file_names))
    # print(file_names[0])
    
    
    # extract_dataset()
    test_dataset = NUHDataset_validation()
    partial_input, ECG_input, gt_MI, partial_input_coarse, mesh_name, ecg_folder_name = test_dataset[0]
    print(ECG_input.shape)
    
    
    
