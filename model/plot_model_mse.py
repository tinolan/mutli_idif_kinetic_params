import os
import glob
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from utils.set_root_paths import root_data_path
from natsort import natsorted

checkpoint_path = "***/data/checkpoint/normal/output/rose-water-1_last/nifty"
checkpoint_path = "***/data/checkpoint/portal_input_k3/output/zesty-firefly-17_last/nifty"
checkpoint_path = "***/data/checkpoint/pulmonary_input_alpha/output/lunar-cloud-1_best/nifty"
checkpoint_path = "***/data/checkpoint/kidney/output/spring-pyramid-1_best/nifty"
checkpoint_path = "***/data/checkpoint/bladder/output/devout-disco-2_best/nifty"

name = "rose-water-1_last"
name = "zesty-firefly-17_last"
name = "lunar-cloud-1_best"
name = "spring-pyramid-1_best"
name = "devout-disco-2_best"

model_path = "***/data/plots/normal_model"
model_path = "***/data/plots/portal_model"
model_path = "***/data/plots/pulm_model"
model_path = "***/data/plots/kidney_model"
model_path = "***/data/plots/bladder_model"

root_idif_path = "***/data/IDIF"

# ===== Globale Parameter und Funktionen =====

# Lade Zeitstempel und Zeitvektor (angenommen, diese gelten für alle Patienten)
time_stamp = torch.load(os.path.join(root_data_path, "DynamicPET/time_stamp.pt"))
t = torch.load(os.path.join(root_data_path, "DynamicPET/t.pt"))

def torch_conv(in1, in2):
    in1 = in1.unsqueeze(0).unsqueeze(0)
    in2 = in2.unsqueeze(0).unsqueeze(0)
    in1_flip = torch.flip(in1, (0, 2))
    out = torch.conv1d(in1_flip, in2, padding=in1.shape[2])
    out = out[0, 0, in1.shape[2]+1:]
    out = torch.flipud(out)
    return out 

def torch_interp_1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])
    indexes = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indexes = torch.clamp(indexes, 0, len(m) - 1)
    return m[indexes] * x + b[indexes]

def reduce_to_600(values):
    step = len(values) // 600
    reduced_values = [max(values[i:i+step]) for i in range(0, len(values), step)]
    return torch.tensor(reduced_values[:600], dtype=torch.float32)

def PET_normal(t: torch.Tensor, k1, k2, k3, Vb, aorta_idif_interp: torch.Tensor) -> torch.Tensor:
    k1, k2, k3, Vb = map(lambda x: torch.tensor(x, dtype=torch.float32), [k1, k2, k3, Vb])
    step = 0.1
    a = aorta_idif_interp  # IDIF-Kurve als 1D-Tensor
    e = (k2 + k3) * t
    b = k1 / (k2 + k3) * (k3 + k2 * torch.exp(-e))
    c = torch_conv(a, b) * step
    PET = (1 - Vb) * c + Vb * a
    return PET

def PET_organ(t: torch.Tensor, k1=None, k2=None, k3=None, Vb=None, alpha=None, beta=None, aorta_idif_interp=None, organ_idif_interp=None, organ=None) -> torch.Tensor:
    if k1 is not None:
        k1 = torch.tensor(k1, dtype=torch.float32)
    if k2 is not None:
        k2 = torch.tensor(k2, dtype=torch.float32)
    if k3 is not None:
        k3 = torch.tensor(k3, dtype=torch.float32)
    if Vb is not None:
        Vb = torch.tensor(Vb, dtype=torch.float32)

    step = 0.1
    if organ == "liver":
        a = alpha * aorta_idif_interp + beta * organ_idif_interp
        e = (k2 + k3) * t
        b = k1 / (k2 + k3) * (k3 + k2 * torch.exp(-e))
        c = torch_conv(a, b) * step
        PET = (1 - Vb) * c + Vb * a
    elif organ == "lung":
        a = alpha * aorta_idif_interp + beta * organ_idif_interp
        e = (k2 + k3) * t
        b = k1 / (k2 + k3) * (k3 + k2 * torch.exp(-e))
        c = torch_conv(a, b) * step
        PET = (1 - Vb) * c + Vb * a
    elif organ == "kidney" or organ == "bladder1":
        alpha = torch.tensor(alpha, dtype=torch.float32)

        a = aorta_idif_interp
        ureter = organ_idif_interp
        e = (k2 + k3) * t
        b = k1 / (k2 + k3) * (k3 + k2 * torch.exp(-e))
        c = torch_conv(a, b) * step
        c_u = torch_conv(ureter, b) * step
        PET = (1 - Vb) * (c + (alpha * c_u)) + Vb * a

    elif organ == "bladder":
        """ a = alpha * aorta_idif_interp + beta * organ_idif_interp
        cumsum_a = torch.cumsum(a, dim=0) * step
        PET = k1 * cumsum_a """

        a = alpha * aorta_idif_interp + beta * organ_idif_interp
        e = (k2 + k3) * t
        b = k1 / (k2 + k3) * (k3 + k2 * torch.exp(-e))
        c = torch_conv(a, b) * step
        PET = (1 - Vb) * c + Vb * a
    
    return PET

multi = True

organs = ["liver", "lung", "kidney", "bladder"]
organs = ["bladder"]
patient_list = ["28", "08", "20", "24", "25", "26", "03"]
patient_list = natsorted(patient_list)

all_mse_values = {organ: {} for organ in organs}
tac_save_dir = os.path.join(model_path, "TAC_values")
os.makedirs(tac_save_dir, exist_ok=True)

for patient in patient_list:
    print(f"\n--- Verarbeite Patient {patient} ---")

    # ==== Check ob alles für diesen Patienten schon existiert ====
    all_done = True
    for organ in organs:
        mse_path = os.path.join(model_path, f"MSE_{patient}_{organ}.txt")
        tac_real_path = os.path.join(tac_save_dir, f"TAC_real_{patient}_{organ}.pt")
        tac_pred_path = os.path.join(tac_save_dir, f"TAC_pred_{patient}_{organ}.pt")
        if not (os.path.exists(mse_path) and os.path.exists(tac_real_path) and os.path.exists(tac_pred_path)):
            all_done = False
            break

    if all_done:
        print(f"→ Alle Organe für Patient {patient} sind bereits berechnet. Überspringe komplett.")
        continue

    if multi == False:
        # ==== Lade kinetische Parameter ====
        kinetic_files = {
            "k1": os.path.join(checkpoint_path, (f"{patient}_0_" + name +".nii.gz")),
            "k2": os.path.join(checkpoint_path, (f"{patient}_1_" + name + ".nii.gz")),
            "k3": os.path.join(checkpoint_path, (f"{patient}_2_" + name +".nii.gz")),
            "Vb": os.path.join(checkpoint_path, (f"{patient}_3_" + name + ".nii.gz"))
        }
    elif multi == True:
        if organ == "liver":
            kinetic_files = {
                "k1": os.path.join(checkpoint_path, (f"{patient}_0_" + name +".nii.gz")),
                "k2": os.path.join(checkpoint_path, (f"{patient}_1_" + name + ".nii.gz")),
                "k3": os.path.join(checkpoint_path, (f"{patient}_2_" + name +".nii.gz")),
                "Vb": os.path.join(checkpoint_path, (f"{patient}_3_" + name + ".nii.gz")),
                "alpha": os.path.join(checkpoint_path, (f"{patient}_4_" + name + ".nii.gz")),
                "beta": os.path.join(checkpoint_path, (f"{patient}_5_" + name + ".nii.gz"))
            }
        elif organ == "lung":
            kinetic_files = {
                "k1": os.path.join(checkpoint_path, (f"{patient}_0_" + name +".nii.gz")),
                "k2": os.path.join(checkpoint_path, (f"{patient}_1_" + name + ".nii.gz")),
                "k3": os.path.join(checkpoint_path, (f"{patient}_2_" + name +".nii.gz")),
                "Vb": os.path.join(checkpoint_path, (f"{patient}_3_" + name + ".nii.gz")),
                "alpha": os.path.join(checkpoint_path, (f"{patient}_4_" + name + ".nii.gz")),
                "beta": os.path.join(checkpoint_path, (f"{patient}_5_" + name + ".nii.gz"))
            }
        elif organ =="bladder":
            kinetic_files = {
                "k1": os.path.join(checkpoint_path, (f"{patient}_0_" + name +".nii.gz")),
                "k2": os.path.join(checkpoint_path, (f"{patient}_1_" + name + ".nii.gz")),
                "k3": os.path.join(checkpoint_path, (f"{patient}_2_" + name +".nii.gz")),
                "Vb": os.path.join(checkpoint_path, (f"{patient}_3_" + name + ".nii.gz")),
                "alpha": os.path.join(checkpoint_path, (f"{patient}_4_" + name + ".nii.gz")),
                "beta": os.path.join(checkpoint_path, (f"{patient}_5_" + name + ".nii.gz"))
            }
        elif organ == "kidney":
            kinetic_files = {
                "k1": os.path.join(checkpoint_path, (f"{patient}_0_" + name +".nii.gz")),
                "k2": os.path.join(checkpoint_path, (f"{patient}_1_" + name + ".nii.gz")),
                "k3": os.path.join(checkpoint_path, (f"{patient}_2_" + name +".nii.gz")),
                "Vb": os.path.join(checkpoint_path, (f"{patient}_3_" + name + ".nii.gz")),
                "alpha": os.path.join(checkpoint_path, (f"{patient}_4_" + name + ".nii.gz")),
            }
    kinetic_params = {}
    for key, path in kinetic_files.items():
        image = sitk.ReadImage(path)
        array = sitk.GetArrayFromImage(image)
        kinetic_params[key] = array

    # ==== Lade PET Daten (dynamisch) und Slice-Auswahl ====
    PET_list = glob.glob(os.path.join(root_data_path, f"DynamicPET/*DynamicFDG_{patient}/NIFTY/Resampled/PET_*.nii.gz"))
    pet_images = [sitk.GetArrayFromImage(sitk.ReadImage(pet_path)) for pet_path in PET_list]

    slices_per_patients = 500
    patient_folder = glob.glob(os.path.join(root_data_path, "DynamicPET", f"*DynamicFDG_{patient}"))[0]
    bb_path = patient_folder + "/NIFTY/Resampled/bb.nii.gz"
    bb_ = sitk.GetArrayFromImage(sitk.ReadImage(bb_path))
    indexes = np.nonzero(bb_)
    top = indexes[0][-1]
    bottom = indexes[0][0]
    step = max(np.floor((top - bottom) / slices_per_patients), 1)
    hom_pick = torch.arange(bottom, top, step)
    pick = hom_pick
    if top - bottom < slices_per_patients:            
        slices = hom_pick[0:slices_per_patients]
    else:
        c = int(len(pick)/2)
        s = int(slices_per_patients/2)
        slices = pick[c-s:c+s+1]

    size = 112
    current_data = torch.zeros((len(pet_images), len(slices), size, size))          

    for i, p in enumerate(PET_list):
        current_pet = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.float32))
        slice_size = current_pet[0, :, :].shape
        slice_center = torch.tensor(slice_size)[0] / 2
        for j, slice_idx in enumerate(slices): 
            slice_idx = int(slice_idx)
            current_slice = current_pet[slice_idx,
                                        int(slice_center)-int(size/2):int(slice_center)+int(size/2), 
                                        int(slice_center)-int(size/2):int(slice_center)+int(size/2)]
            current_data[i, j, :, :] = current_slice / 1000  # Bq/ml → kBq/ml

    # ==== Lade Labelmaps ====
    label_map = None
    bladder_map = None
    label_map_path = glob.glob(os.path.join(root_data_path, f"DynamicPET/*DynamicFDG_{patient}/NIFTY/Resampled/labels.nii.gz"))
    if label_map_path:
        label_map = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(label_map_path[0])).astype(np.float32))

    bladder_path = glob.glob(os.path.join("***/data", f"segmentationsAndResample/*DynamicFDG_{patient}/urinary_bladder.nii.gz"))
    if bladder_path:
        bladder_map = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(bladder_path[0])).astype(np.float32))

    for organ in organs:
        print(f"\nOrgan: {organ}")
        
        mse_path = os.path.join(model_path, f"MSE_{patient}_{organ}.txt")
        tac_real_path = os.path.join(tac_save_dir, f"TAC_real_{patient}_{organ}.pt")
        tac_pred_path = os.path.join(tac_save_dir, f"TAC_pred_{patient}_{organ}.pt")

        if os.path.exists(mse_path) and os.path.exists(tac_real_path) and os.path.exists(tac_pred_path):
            print(f"→ Ergebnisse für {patient} - {organ} existieren bereits. Überspringe.")
            continue

        label_data = torch.zeros((len(slices), size, size))          
        slice_center = int(label_map[0, :, :].shape[0] / 2)

        if organ != "bladder":
            for j, slice_idx in enumerate(slices): 
                slice_idx = int(slice_idx)
                current_slice = label_map[slice_idx,
                                        slice_center-int(size/2):slice_center+int(size/2),
                                        slice_center-int(size/2):slice_center+int(size/2)]
                label_data[j, :, :] = current_slice
            label_data = label_data.numpy()
            if organ == "liver":
                organ_mask = (label_data == 3)
            elif organ == "lung":
                organ_mask = np.isin(label_data, [4, 5])
            elif organ == "kidney":
                organ_mask = np.isin(label_data, [6, 7])
        else:
            for j, slice_idx in enumerate(slices): 
                slice_idx = int(slice_idx)
                current_slice = bladder_map[slice_idx,
                                            slice_center-int(size/2):slice_center+int(size/2),
                                            slice_center-int(size/2):slice_center+int(size/2)]
                label_data[j, :, :] = current_slice
            label_data = label_data.numpy()
            organ_mask = (label_data == 1)

        organ_coords = np.argwhere(organ_mask)
        tac_matrix = []
        for coord in organ_coords:
            x, y, z = coord
            tac_values = [current_data[i, x, y, z].item() for i in range(current_data.shape[0])]
            tac_tensor = torch.tensor(tac_values, dtype=torch.float32)
            tac_interp = reduce_to_600(torch_interp_1d(t, time_stamp, tac_tensor))
            tac_matrix.append(tac_interp.numpy())
        tac_matrix = np.array(tac_matrix)
        tac_mean = np.mean(tac_matrix, axis=0)

        mean_params = {}
        for key, array in kinetic_params.items():
            organ_values = array[organ_mask]
            mean_val = np.mean(organ_values)
            mean_params[key] = mean_val

        if multi == True:
            if organ == "liver":
                organ_idif_txt_path = os.path.join(root_idif_path, "portal_vein_split_IDIF_ownSeg", f"IDIF_Patient_{patient}.txt")
            elif organ == "lung":
                organ_idif_txt_path = os.path.join(root_idif_path, "pulmonary_artery_IDIF_ownSeg", f"IDIF_Patient_{patient}.txt")
            elif organ == "bladder" or organ == "kidney":
                organ_idif_txt_path = os.path.join(root_idif_path, "ureter_IDIF_ownSeg", f"IDIF_Patient_{patient}.txt")
            organ_idif = pd.read_csv(organ_idif_txt_path, sep="\t")
            organ_idif = torch.tensor(organ_idif["plasma[kBq/cc]"].values, dtype=torch.float32)
            organ_idif_interp = reduce_to_600(torch_interp_1d(t, time_stamp, organ_idif))

        aorta_idif_txt_path = os.path.join(root_data_path, "DynamicPET/IDIF", f"DynamicFDG_{patient}_IDIF.txt")
        data_idif = pd.read_csv(aorta_idif_txt_path, sep="\t")
        aorta_idif = torch.tensor(data_idif["plasma[kBq/cc]"].values, dtype=torch.float32)
        aorta_idif_interp = reduce_to_600(torch_interp_1d(t, time_stamp, aorta_idif))

        if multi == True:
            if organ == "liver" or organ == "lung":
                predicted_TAC = PET_organ(t,
                                    mean_params["k1"],
                                    mean_params["k2"],
                                    mean_params["k3"],
                                    mean_params["Vb"],
                                    mean_params["alpha"],
                                    mean_params["beta"],
                                    aorta_idif_interp,
                                    organ_idif_interp,
                                    organ)
            elif organ == "kidney":
                predicted_TAC = PET_organ(t,
                                    k1=mean_params["k1"],
                                    k2=mean_params["k2"],
                                    k3=mean_params["k3"],
                                    Vb=mean_params["Vb"],
                                    alpha=mean_params["alpha"],
                                    aorta_idif_interp=aorta_idif_interp,
                                    organ_idif_interp=organ_idif_interp,
                                    organ=organ)
            elif organ == "bladder":
                predicted_TAC = PET_organ(t,
                                    k1=mean_params["k1"],
                                    k2=mean_params["k2"],
                                    k3=mean_params["k3"],
                                    Vb=mean_params["Vb"],
                                    alpha=mean_params["alpha"],
                                    beta=mean_params["beta"],
                                    aorta_idif_interp=aorta_idif_interp,
                                    organ_idif_interp=organ_idif_interp,
                                    organ=organ)

        else:
            predicted_TAC = PET_normal(t,
                                   mean_params["k1"],
                                   mean_params["k2"],
                                   mean_params["k3"],
                                   mean_params["Vb"],
                                   aorta_idif_interp)

        mse = np.mean((tac_mean - predicted_TAC.detach().numpy()) ** 2)
        all_mse_values[organ][patient] = mse
        print(f"MSE für {organ}: {mse}")

        torch.save(torch.tensor(tac_mean, dtype=torch.float32), tac_real_path)
        torch.save(predicted_TAC.detach(), tac_pred_path)

        with open(mse_path, "w") as f:
            f.write(str(mse))

        plt.figure(figsize=(10, 6))
        plt.plot(t.numpy(), tac_mean, label="Measured Mean TAC", linewidth=2)
        plt.plot(t.numpy(), predicted_TAC.detach().numpy(), label="Predicted TAC", linestyle="--", linewidth=2)
        plt.xlabel("Time")
        plt.ylabel("TAC")
        plt.title(f"TAC Comparison for Patient {patient} ({organ})")
        plt.legend()
        plt.grid(True)
        save_path_TAC = os.path.join(model_path, f"TAC_Comparison_{patient}_{organ}.png")
        plt.savefig(save_path_TAC, dpi=300)
        plt.close()

# ==== MSE-Barplots speichern ====
for organ, mse_values in all_mse_values.items():
    plt.figure(figsize=(8, 6))
    patients_sorted = sorted(mse_values.keys())
    mse_list = [mse_values[p] for p in patients_sorted]
    plt.bar(patients_sorted, mse_list, color="skyblue")
    plt.xlabel("Patient")
    plt.ylabel("MSE")
    plt.title(f"MSE between Measured and Predicted TAC ({organ})")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    save_path_MSE = os.path.join(model_path, f"MSE_Comparison_{organ}.png")
    plt.savefig(save_path_MSE, dpi=300)
    plt.close()

print("Alle Plots und Ergebnisse wurden gespeichert.")
