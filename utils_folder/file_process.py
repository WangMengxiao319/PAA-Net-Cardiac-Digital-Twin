import os
import shutil
import glob

def list_subfolders(path):
    """obtain the list of subfolders in a given path"""
    return [f for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f))]
    
def ECG_rename(ECG_name):
    ECG_name = ECG_name.replace('A1', 'Septal').replace('A2', 'Apical').replace('A3', 'Ext anterior').replace('A4', 'Lim anterior')
    ECG_name = ECG_name.replace('B1', 'Lateral').replace('B2', 'Inferior').replace('B3', 'Inferolateral')
    # ECG_name = ECG_name.replace('normal', 'Baseline')
    ECG_name = ECG_name.replace('Lateral_small_transmu','Lateral Transmural(small)')
    ECG_name = ECG_name.replace('Lateral_large_transmu','Lateral Transmural(large)')
    ECG_name = ECG_name.replace('transmu', 'Transmural')
    ECG_name = ECG_name.replace('_',' ')

    return ECG_name

def extract_healthy_samples():
    data_dir_a = 'output/generate_dataset_DHlab_data_Scar_CV_APD_chronic'
    data_dir_b = 'output/generate_dataset_DHlab_data_Scar_CV_APD_chronic_new_v1'
    
    # Copy healthy folders from directory A to directory B without renaming their parent folders.
    subfolders_a = list_subfolders(data_dir_a)
    for folder in subfolders_a:
        healthy_folder_a = os.path.join(data_dir_a, folder, 'healthy')
        healthy_folder_b = os.path.join(data_dir_b, folder, 'healthy')
        if os.path.exists(healthy_folder_a):
            os.makedirs(healthy_folder_b, exist_ok=True)
            for file_name in os.listdir(healthy_folder_a):
                source_file = os.path.join(healthy_folder_a, file_name)
                target_file = os.path.join(healthy_folder_b, file_name)
                if os.path.isfile(source_file):
                    shutil.copy(source_file, target_file)
                    print(f"Copy {source_file} to {target_file}")
        
def combine_simulated_data_forAI():
    '''Combine ECG, mesh, and CSV files from multiple folders into one dataset directory.'''
    # Copy each sample's ECG and mesh files into a new folder.
    mesh_vtu_path=r'/path/to/dataset/mesh_conversion_InSilicoHeartGen'   # Traverse all sample subfolders containing mesh_cobiveco_AHA17.vtu and labels_final.vtk.
    mesh_csv_path = r'/path/to/dataset/DHlab_valid'  # CSV files containing lvborderzonenodes, lvscarnodes, or electrode_xyz.
    ecg_path = r'/path/to/simulation/output/generate_dataset_DHlab_data_Scar_CV_APD_chronic'  # Copy predicted_ecg, rename it after its parent folder, and append _ecg.
    
    dataset_path =r'/path/to/inference/dataset'
    
    os.makedirs(dataset_path, exist_ok=True)

    # Traverse all cases using the ECG folder as the reference.
    case_list = [
        d for d in os.listdir(ecg_path)
        if os.path.isdir(os.path.join(ecg_path, d))
    ]

    print(f'Found {len(case_list)} cases.')

    for case in case_list:
        print(f'Processing {case} ...')

        case_dst = os.path.join(dataset_path, case)
        os.makedirs(case_dst, exist_ok=True)

        # ========= 1. copy ECG (use parent folder name) =========
        ecg_case_dir = os.path.join(ecg_path, case)

        if os.path.exists(ecg_case_dir):
            for subdir in os.listdir(ecg_case_dir):
                subdir_path = os.path.join(ecg_case_dir, subdir)

                if not os.path.isdir(subdir_path):
                    continue

                ecg_file = os.path.join(subdir_path, 'predicted_ecg.csv')
                if os.path.exists(ecg_file):
                    dst_name = f'ECG_simulated_{subdir}.csv'
                    dst_path = os.path.join(case_dst, dst_name)

                    shutil.copy(ecg_file, dst_path)
                    
                lat_file = os.path.join(subdir_path, 'lat_simulation.csv')
                if os.path.exists(lat_file):
                    dst_name = f'LAT_simulated_{subdir}.csv'
                    dst_path = os.path.join(case_dst, dst_name)

                    shutil.copy(lat_file, dst_path)


        # ========= 2. labels_final.vtk (case root directory) =========
        mesh_case_dir = os.path.join(mesh_vtu_path, case)
        if os.path.exists(mesh_case_dir):
            label_path = os.path.join(mesh_case_dir, 'labels_final.vtk')
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(case_dst, 'labels_final.vtk'))

        # ========= 3. mesh_cobiveco_AHA17.vtu (ensi subdirectory) =========
        ensi_dir = os.path.join(mesh_case_dir, 'ensi')
        if os.path.exists(ensi_dir):
            mesh_vtu = os.path.join(ensi_dir, 'mesh_cobiveco_AHA17.vtu')
            if os.path.exists(mesh_vtu):
                shutil.copy(
                    mesh_vtu,
                    os.path.join(case_dst, 'mesh_cobiveco_AHA17.vtu')
                )

        # ========= 4. CSV files =========
        csv_case_dir = os.path.join(mesh_csv_path, case)
        if os.path.exists(csv_case_dir):
            for fname in os.listdir(csv_case_dir):
                if fname.endswith('.csv'):
                    shutil.copy(
                        os.path.join(csv_case_dir, fname),
                        os.path.join(case_dst, fname)
                    )

    print('✅ Dataset integration finished.')

def combine_simulated_data_forAI_v0():
    '''Move scar- and simulation-related content into each case subfolder.'''
    dataset_path = r'/path/to/inference/dataset'

    # Define the wildcard patterns to match.
    patterns = [
        '*simulated*', 
        '*lvborderzonenodes*', 
        '*lvscarnodes*'
    ]

    for case in os.listdir(dataset_path):
        case_dir = os.path.join(dataset_path, case)
        
        # Ensure this is a directory rather than a loose file under dataset_path.
        if not os.path.isdir(case_dir):
            continue
            
        # Define the target v0 subfolder.
        dest_dir = os.path.join(case_dir, 'v0')
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Iterate over the wildcard patterns.
        for pattern in patterns:
            # Build the complete search path, for example .../case1/*scar*.csv.
            # For files ending in CSV, use f'{pattern}.csv'.
            search_pattern = os.path.join(case_dir, f"{pattern}.csv")
            
            # glob.glob returns all matching file paths.
            for file_path in glob.glob(search_pattern):
                file_name = os.path.basename(file_path)
                target_path = os.path.join(dest_dir, file_name)
                
                print(f"Moving {file_name} to {dest_dir}")
                shutil.move(file_path, target_path)

def combine_simulated_data_forAI_v1():
    '''
    Move scar- and simulation-related content into each case subfolder.
    (generate_dataset_DHlab_data_Scar_CV_APD_chronic_new_v1)
    '''
    dataset_path = r'/path/to/inference/dataset'
    mesh_csv_path = r'/path/to/dataset/DHlab_valid_scar_new_v1'  # CSV files containing lvborderzonenodes, lvscarnodes, or electrode_xyz.
    ecg_path = r'/path/to/simulation/output/generate_dataset_DHlab_data_Scar_CV_APD_chronic_new_v1'  # Copy predicted_ecg, rename it after its parent folder, and append _ecg.
    
    # Define the wildcard patterns to match.
    patterns = [
        '*lvborderzonenodes*', 
        '*lvscarnodes*'
    ]

    for case in os.listdir(dataset_path):
        case_dir = os.path.join(dataset_path, case)
        mesh_csv_dir = os.path.join(mesh_csv_path, case)
        ecg_case_dir = os.path.join(ecg_path, case)
        # Ensure this is a directory rather than a loose file under dataset_path.
        if not os.path.isdir(case_dir):
            continue
            
        # Define the target v1 subfolder.
        dest_dir = os.path.join(case_dir, 'v1')
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Iterate over the wildcard patterns.
        for pattern in patterns:
            # Build the complete search path, for example .../case1/*scar*.csv.
            # For files ending in CSV, use f'{pattern}.csv'.
            search_pattern = os.path.join(mesh_csv_dir, f"{pattern}.csv")
            # glob.glob returns all matching file paths.
            for file_path in glob.glob(search_pattern):
                file_name = os.path.basename(file_path)
                target_path = os.path.join(dest_dir, file_name)
                
                print(f"coping {file_name} to {dest_dir}")
                shutil.copy(file_path, target_path)
                
        # Simulated ECG and LAT files.
        for subdir in os.listdir(ecg_case_dir):
            subdir_path = os.path.join(ecg_case_dir, subdir)

            if not os.path.isdir(subdir_path):
                continue

            ecg_file = os.path.join(subdir_path, 'predicted_ecg.csv')
            if os.path.exists(ecg_file):
                dst_name = f'ECG_simulated_{subdir}.csv'
                dst_path = os.path.join(dest_dir, dst_name)
                shutil.copy(ecg_file, dst_path)
                
            lat_file = os.path.join(subdir_path, 'lat_simulation.csv')
            if os.path.exists(lat_file):
                dst_name = f'LAT_simulated_{subdir}.csv'
                dst_path = os.path.join(dest_dir, dst_name)
                print(f"coping {subdir} to {dest_dir}")
                shutil.copy(lat_file, dst_path)
        # copy_healthy_from_v0_to_v1
        v0_dir = os.path.join(case_dir, 'v0')
        pattern_healthy = '*healthy*'
        search_pattern_healthy = os.path.join(v0_dir, f"{pattern_healthy}.csv")
        for file_path in glob.glob(search_pattern_healthy):
            file_name = os.path.basename(file_path)
            target_path = os.path.join(dest_dir, file_name)
            
            print(f"coping {file_name} to {dest_dir}")
            shutil.copy(file_path, target_path)
                
        
    
def vu_dataset():
    '''
    mesh_conversion_InSilicoHeartGen: Coarse.vtu, fibre.vtk; 
    DHlab_valid: .csv;
    generate_dataset_DHlab_data_Scar_CV_APD_chronic: healthy/lat_simulation.csv
    '''
    # Copy each sample's ECG and mesh files into a new folder.
    mesh_vtu_path=r'/path/to/dataset/mesh_conversion_InSilicoHeartGen'   # Traverse all sample subfolders containing mesh_cobiveco_AHA17.vtu and labels_final.vtk.
    mesh_csv_path = r'/path/to/dataset/DHlab_valid'  # CSV files containing lvborderzonenodes, lvscarnodes, or electrode_xyz.
    lat_path = r'/path/to/simulation/output/generate_dataset_DHlab_data_Scar_CV_APD_chronic'  # Copy predicted_ecg, rename it after its parent folder, and append _ecg.
    
    dataset_path =r'/path/to/dataset/cases_with_LAT'
    
    os.makedirs(dataset_path, exist_ok=True)

    # Traverse all cases using the LAT folder as the reference.
    case_list = [
        d for d in os.listdir(lat_path)
        if os.path.isdir(os.path.join(lat_path, d))
    ]

    print(f'Found {len(case_list)} cases.')

    for case in case_list:
        print(f'Processing {case} ...')

        case_dst = os.path.join(dataset_path, case)
        os.makedirs(case_dst, exist_ok=True)

        # ========= 1. copy ECG (use parent folder name) =========
        ecg_case_dir = os.path.join(lat_path, case)

        if os.path.exists(ecg_case_dir):
            subdir_path = os.path.join(ecg_case_dir, 'healthy')

            if not os.path.isdir(subdir_path):
                continue

            ecg_file = os.path.join(subdir_path, 'lat_simulation.csv')
            if os.path.exists(ecg_file):
                dst_name = f'lat_simulation.csv'
                dst_path = os.path.join(case_dst, dst_name)

                shutil.copy(ecg_file, dst_path)


        # ========= Case root directory =========
        mesh_case_dir = os.path.join(mesh_vtu_path, case)
        if os.path.exists(mesh_case_dir):
            label_path = os.path.join(mesh_case_dir, 'Coarse.vtu')
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(case_dst, 'Coarse.vtu'))

        # ========= ensi subdirectory =========
        ensi_dir = os.path.join(mesh_case_dir, 'ensi')
        if os.path.exists(ensi_dir):
            mesh_vtk = os.path.join(ensi_dir, 'fibre.vtk')
            if os.path.exists(mesh_vtk):
                shutil.copy(
                    mesh_vtk,
                    os.path.join(case_dst, 'fibre.vtk')
                )
            mesh_vtu = os.path.join(ensi_dir, 'mesh_cobiveco_AHA17.vtu')
            if os.path.exists(mesh_vtu):
                shutil.copy(
                    mesh_vtu,
                    os.path.join(case_dst, 'mesh_cobiveco_AHA17.vtu')
                )
                
                
            

        # ========= 4. CSV files =========
        csv_case_dir = os.path.join(mesh_csv_path, case)
        if os.path.exists(csv_case_dir):
            for fname in os.listdir(csv_case_dir):
                if fname.endswith('.csv') and 'lvborderzonenodes' not in fname and 'lvscarnodes' not in fname:
                    shutil.copy(
                        os.path.join(csv_case_dir, fname),
                        os.path.join(case_dst, fname)
                    )

    print('✅ Dataset integration finished.')
    
def calculate_unique_patient():
    '''
    Count the unique patients in the dataset.
    '''
    dataset_path =r'/path/to/inference/dataset'
    
    patient_set = set()
    
    for case in os.listdir(dataset_path):
        case_dir = os.path.join(dataset_path, case)
        if os.path.isdir(case_dir):
            # Assume cases follow the naming convention "PatientID_SomeOtherInfo".
            patient_id = case.split('_')[0]  # Extract PatientID.
            patient_set.add(patient_id)
    
    print(f'Unique patients count: {len(patient_set)}') # 130 cases from 99 patients
        
if __name__== "__main__":
    # extract_healthy_samples()
    # combine_simulated_data_forAI()
    # vu_dataset()
    # combine_simulated_data_forAI_v0()
    # combine_simulated_data_forAI_v1()
    calculate_unique_patient()
