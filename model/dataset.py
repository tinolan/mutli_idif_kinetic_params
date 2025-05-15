import os
import torch
import numpy as np
import glob
import SimpleITK as sitk
from monai.data import CacheDataset
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils_kinetic import PET_2TC_KM
from utils.utils_torch import torch_interp_1d
from utils.set_root_paths import root_data_path
from utils.set_root_paths import root_idif_path


class DynPETDataset(CacheDataset):

    def __init__(self, config, dataset_type): 
        # Enforce determinism
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        # Read global config
        self.config = config
        self.dataset_type = dataset_type
        self.patch_size = self.config["patch_size"]

        # Create config for each dataset type from global config
        self.train_config = {"patient_list": self.config["patient_list"]["train"], "slices_per_patient": int(self.config["slices_per_patient_train"])}
        #self.train_config = {"patient_list": ["28"], "slices_per_patient": 1}

        self.val_config = {"patient_list": self.config["patient_list"]["validation"], "slices_per_patient": int(self.config["slices_per_patient_val"])}
        #self.val_config = {"patient_list": ["28"], "slices_per_patient": 1}

        if self.config["slices_per_patient_test"] == "None":
            self.test_config = {"patient_list": self.config["patient_list"]["test"], "slices_per_patient": 1}     # Take all the slices from the test patients
        else: 
            self.test_config = {"patient_list": self.config["patient_list"]["test"], "slices_per_patient": self.config["slices_per_patient_test"]}

        # Select the correct config
        self.aorta_idif = dict()
        self.portal_idif = dict()
        self.data = list()
        if dataset_type == "train":
            self.build_dataset(self.train_config)
        elif dataset_type == "validation":
            self.build_dataset(self.val_config)
        elif dataset_type == "test":
            self.build_dataset(self.test_config)
        else: 
            print("ERROR: dataset type not supported!")
            return

    def __getitem__(self, idx):                  
        return self.data[idx]

    def __len__(self):
        return int(self.length)
    
    def build_dataset(self, current_config):
        self.current_dataset_size = current_config["slices_per_patient"] * len(current_config["patient_list"])     
        print("Creating dataset", self.dataset_type, ":", current_config)
            
        self.patient_list = current_config["patient_list"]
        for p in self.patient_list:
            self.load_txt_data(p)

        load_data = self.load_data()
        if load_data is None:   
            self.read_dynpet()
            print("Dataset", self.dataset_type, "was saved in", self.save_data_folder)
        else:                   
            self.data = load_data
        
        self.length = len(self.data) 
        print("Dataset", self.dataset_type, "has", self.current_dataset_size, "slices!\n")
        return

    def load_txt_data(self, patient):
        tac_txt_path = os.path.join(root_data_path, "DynamicPET/TAC", f"DynamicFDG_{patient}_TAC.txt")
        aorta_idif_txt_path = os.path.join(root_data_path, "DynamicPET/IDIF", f"DynamicFDG_{patient}_IDIF.txt")
        #portal_idif_txt_path = os.path.join(root_idif_path, "portal_vein_split_IDIF_ownSeg", f"IDIF_Patient_{patient}.txt")
        portal_idif_txt_path = os.path.join(root_idif_path, "ureter_IDIF_ownSeg", f"IDIF_Patient_{patient}.txt")


        # Read acquisition time
        data = pd.read_csv(tac_txt_path, sep="\t")
        data['start[seconds]'] = data['start[seconds]'].apply(lambda x: x / 60)
        time_stamp = data['start[seconds]'].values
        self.time_stamp = torch.Tensor(np.around(time_stamp, 2))

        # Define interpolated time axis for convolution
        step = 0.1
        self.t = torch.Tensor(np.arange(self.time_stamp[0], self.time_stamp[-1], step))

        # Read and interpolate IDIF
        data = pd.read_csv(aorta_idif_txt_path, sep="\t").rolling(1).mean()
        idif_sample_time = torch.Tensor(data["sample-time[minutes]"])
        aorta_idif = torch.Tensor(data["plasma[kBq/cc]"])
        data = pd.read_csv(portal_idif_txt_path, sep="\t").rolling(1).mean()
        portal_idif = torch.Tensor(data["plasma[kBq/cc]"])
        self.aorta_idif[patient] = torch_interp_1d(self.t, idif_sample_time, aorta_idif)
        self.portal_idif[patient] = torch_interp_1d(self.t, idif_sample_time, portal_idif)

        return 

    def read_dynpet(self): 
        data_list = list()
        for patient in self.patient_list:
            patient_folder = glob.glob(os.path.join(root_data_path, "DynamicPET", f"*DynamicFDG_{patient}"))[0]

            # Load lung label
            label_map_path = glob.glob(patient_folder + "/NIFTY/Resampled/labels.nii.gz")[0]
            lung_label = sitk.GetArrayFromImage(sitk.ReadImage(label_map_path)).astype(bool)
            label_map_ = sitk.GetArrayFromImage(sitk.ReadImage(label_map_path))
            lung_mask = np.isin(label_map_, [6, 7])

            if self.current_dataset_size == 1:                
                if self.dataset_type == "train":
                    self.slices = [205]
                else:  # Validation und Test
                    self.slices = [205]
                print(f"\tPatient: {patient}; using custom slice: {self.slices}")
            else:
                self.slices_per_patients = int(self.current_dataset_size / len(self.patient_list))
                
                if self.current_dataset_size == 1: 
                    self.slices = [205]
                    print(f"\tPatient: {patient}; N_slices={len(self.slices)}/1 ; slices:", self.slices)
                else: 
                    bb_path = patient_folder + "/NIFTY/Resampled/bb.nii.gz"
                    bb_ = sitk.GetArrayFromImage(sitk.ReadImage(bb_path))
                    indexes = np.nonzero(bb_)
                    top = indexes[0][-1]
                    bottom = indexes[0][0]
                    step = max(np.floor((top - bottom) / self.slices_per_patients), 1)
                    hom_pick = torch.arange(bottom, top, step)

                    pick = hom_pick
                    if top - bottom < self.slices_per_patients:            
                        self.slices = hom_pick[0:self.slices_per_patients]
                    else:
                        c = int(len(pick)/2)
                        s = int(self.slices_per_patients/2)
                        self.slices = pick[c-s:c+s+1]

                    print(f"\tPatient: {patient}; N_slices={len(self.slices)}/{top - bottom}; slices:", self.slices)

            size = self.patch_size
            pet_list = glob.glob(patient_folder + "/NIFTY/Resampled/PET_*.nii.gz")
            current_data = torch.zeros((len(self.slices), len(pet_list), size, size))          

            for i, p in enumerate(pet_list):
                current_pet = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.float32))
                slice_size = current_pet[0, :, :].shape
                slice_center = torch.tensor(slice_size)[0] / 2
                for j, slice_idx in enumerate(self.slices): 
                    slice_idx = int(slice_idx)
                    current_slice = current_pet[slice_idx,
                                                int(slice_center)-int(size/2):int(slice_center)+int(size/2), 
                                                int(slice_center)-int(size/2):int(slice_center)+int(size/2)]

                    # Apply lung mask to current slice
                    lung_slice = lung_mask[slice_idx,
                                            int(slice_center)-int(size/2):int(slice_center)+int(size/2),
                                            int(slice_center)-int(size/2):int(slice_center)+int(size/2)]
                    current_slice[~lung_slice] = 0  # set non-lung values to zero

                    current_data[j, i, :, :] = current_slice / 1000  # von Bq/ml zu kBq/ml

            for j in range(len(self.slices)):
                data_list.append([patient, self.slices[j], current_data[j, :, :]])
                
        data_list = data_list[0:self.current_dataset_size]
        self.data = data_list
        torch.save(data_list, self.save_data_folder + "/data" + str(self.patient_list) + ".pt")
        
        return data_list


    
    def load_data(self):
        folder_name = f"{self.dataset_type}_N{self.current_dataset_size}_P{self.patch_size}"
        self.save_data_folder = os.path.join("***/data/dataset_kidney_cf", folder_name)

        if not os.path.exists(self.save_data_folder):
            os.makedirs(self.save_data_folder)

        file_name = f"data{self.patient_list}.pt"
        if os.path.exists(os.path.join(self.save_data_folder, file_name)):
            return torch.load(os.path.join(self.save_data_folder, file_name))
        else:
            print(f"WARNING: {file_name} does not exist!")
            return None
