import os
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from scipy.stats import pearsonr
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset import DynPETDataset
from network.unet_blocks import UNet
from network.unet_blocks_ST import UNet_ST
from utils.utils_kinetic import PET_2TC_KM_batch
from utils.utils_main import make_save_folder_struct, reconstruct_prediction, apply_final_activation
from utils.utils_logging import log_slice, log_curves, mask_data
from utils.utils_torch import torch_interp_Nd, torch_interp_1d, torch_conv
from utils.set_root_paths import root_path, root_checkpoints_path, root_data_path
import utils.similaritymeasures_torch as similaritymeasures_torch
from scipy.optimize import curve_fit


torch.cuda.empty_cache()

if not torch.cuda.is_available():   
  current_gpu = None    
  machine = "cpu"
  print("*** ERROR: no GPU available ***")
else:
  machine = "cuda:0"
  current_gpu = [0]

class SpaceTempUNet(pl.LightningModule):
  
  def __init__(self, config):

    # Enforce determinism
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)
    
    super(SpaceTempUNet, self).__init__()

    # Read configuration file and add new info if needed
    self.config = config
    self.checkpoint_path = None
    self.config["output_size"] = 6 # Updated to include alpha and beta
    self.config["mask_loss"] = True
    self.config["multi_clamp_params"] = {
      "k1": (0.01, 10),
      "k2": (0.01, 10),
      "k3": (0.001, 1),
      "Vb": (0, 0.2),
      "alpha": (0, 1),
      "beta": (0, 1)
    }
    self.config["patch_size"] = 112
    self.config["use_pearson_metric"] = False

    print(self.config)

    if self.config["use_spatio_temporal_unet"]:
      self.model = UNet_ST(in_channels=1, out_channels=self.config["output_size"], config=self.config)
    else:
      self.model = UNet(in_channels=1, out_channels=self.config["output_size"], config=self.config)

    frame_duration = [10, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10, 10, 10, 10, 30, 30, 30, 30, 30, 30, 30, 30, 60, 60, 60, 60, 120, 120, 120, 120, 120, 300, 300, 300, 300, 300, 300, 300, 300, 300]
    self.loss_frame_duration = frame_duration
    self.frame_duration = np.array(frame_duration) / 60  # from s to min
    self.frame_duration_batch = torch.from_numpy(np.array(self.frame_duration)).unsqueeze(-1).repeat(1, self.config["patch_size"]*self.config["patch_size"]).to(machine)

  def setup(self, stage): 
    self.stage = stage
    self.patch_size = self.config["patch_size"]
    
    if stage == "fit":
      self.train_dataset = DynPETDataset(self.config, "train")
      self.aorta_idif_train_set = self.train_dataset.aorta_idif
      self.portal_idif_train_set = self.train_dataset.portal_idif

      self.val_dataset = DynPETDataset(self.config, "validation")
      self.aorta_idif_val_set = self.val_dataset.aorta_idif
      self.portal_idif_val_set = self.val_dataset.portal_idif

      self.t = self.train_dataset.t.to(machine)
      self.time_stamp = self.train_dataset.time_stamp.to(machine)
      self.t_batch = self.t.repeat(self.patch_size*self.patch_size, 1, 1)
      self.time_stamp_batch = self.time_stamp.repeat(self.patch_size*self.patch_size, 1, 1)

    if stage == "test":
      self.test_dataset = DynPETDataset(self.config, "test")
      self.aorta_idif_test_set = self.test_dataset.aorta_idif
      self.portal_idif_test_set = self.test_dataset.portal_idif
      self.t = self.test_dataset.t.to(machine)
      self.time_stamp = self.test_dataset.time_stamp.to(machine)
      self.t_batch = self.t.repeat(self.patch_size*self.patch_size, 1, 1)
      self.time_stamp_batch = self.time_stamp.repeat(self.patch_size*self.patch_size, 1, 1)

    
  def forward(self, x):
    logits_params = self.model(x)
    return logits_params
  
  def loss_function(self, pred_TAC, real_TAC, k1, k2, k3, Vb, alpha=None, beta=None):        
    # calculate loss 
    #loss = similaritymeasures_torch.mse(pred_TAC.to(machine).double(), real_TAC.to(machine).double())

    loss_frame_duration = torch.from_numpy(np.array(self.loss_frame_duration)).unsqueeze(-1).repeat(1, self.config["patch_size"]*self.config["patch_size"]).to(machine)
    weights =  loss_frame_duration / sum(self.loss_frame_duration)

    weighted_mse = torch.mean(weights * (pred_TAC.to(machine).double() - real_TAC.to(machine).double())**2)
    loss = weighted_mse

    loss = torch.square(pred_TAC - real_TAC)
    safe_real_TAC = torch.nan_to_num(real_TAC, nan=float('nan'))

    max_vals, _ = torch.max(safe_real_TAC, dim=0, keepdim=True)

    loss = loss /(max_vals + 1e-8)

    loss = torch.nanmean(loss)

    #loss = similaritymeasures_torch.mse(pred_TAC.to(machine).double(), real_TAC.to(machine).double())

    # Wenn abs als Final Activation genutzt wird, füge Regularisierungen hinzu
    if self.config["final_activation"] == "abs":

      k1_penalty = 1000*torch.mean(torch.relu(k1 - 10) + torch.relu(0.01 - k1))  # Bestrafe k1 > 2
      k2_penalty = 1000*torch.mean(torch.relu(k2 - 10) + torch.relu(0.4 - k2))  # Bestrafe k2 > 3
      k3_penalty = 1000*torch.mean(torch.relu(k3 - 0.05) + torch.relu(0.0001 - k3))  # Bestrafe k3 > 1
      Vb_penalty = 1000*torch.mean(torch.relu(Vb - 0.2) + torch.relu(0.0001-Vb)) # Bestrafe Vb > 1

      # Gesamtstrafe für k-Werte
      param_penalty = (k1_penalty + k2_penalty + k3_penalty + Vb_penalty)      

      # Strafe für alpha außerhalb von [0, 1]
      alpha_penalty = 1000*torch.mean(torch.relu(alpha - 0.4))
      beta_penalty = 1000*torch.mean(torch.relu(beta - 1) + torch.relu(0.5-beta))


      # Skaliere die Strafen und füge sie zum Verlust hinzu
      #loss += alpha_penalty + param_penalty + beta_penalty
      
    return loss
  
  def train_dataloader(self):
      train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=12)
      return train_loader

  def val_dataloader(self):
      val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.config["batch_size"], shuffle=False, num_workers=12)
      return val_loader
  
  def test_dataloader(self):
      # If batch_size!=1 test_set and test_epoch_end may not work as expected
      test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=12)
      return test_loader

  def accumulate_func(self, batch, logits, return_value=None):
    p = batch[0]              # Patient-ID(s)
    # Anstelle der originalen TACs aus dem Batch
    # holen wir jetzt die Curve-Fit TACs als Referenz:
    curve_fit_TAC_batch, _ = self.apply_curve_fit_on_batch(batch)

    # Wir ersetzen diesen Schritt nun:
    TAC_batch = curve_fit_TAC_batch  # Verwende die Curve-Fit TACs als "Ground Truth"
    TAC_batch = TAC_batch.squeeze(1)
    b, t, h, w = TAC_batch.shape  # Annahme: TAC_batch hat Form [b, 1, T, w, h]
    # Ursprünglicher Code nahm die gemessenen TACs an:
    TAC_batch_old = batch[2]      # [b, T, w, h]
    TAC_batch_old = torch.reshape(TAC_batch_old, [b, 62, h*w])    
    # Falls nötig, entferne den Kanal, um z. B. [b, T, w, h] zu erhalten:
    
    # Erstelle einen Platzhalter für die vom Modell vorhergesagten TACs
    TAC_pred_batch = torch.zeros_like(TAC_batch_old)
    
    # Berechne die kinetischen Parameter (mit finaler Aktivierung) und forme sie passend um
    kinetic_params = apply_final_activation(logits, self.config)  # [b, output_size, 1, w, h]
    kinetic_params = torch.reshape(kinetic_params, [b, self.config["output_size"], h * w, 1])
    kinetic_params = kinetic_params.repeat(1, 1, 1, len(self.t))  # [b, output_size, w*h, len(t)]
    logits = torch.reshape(logits, [b, self.config["output_size"], h * w, 1])
    
    loss = 0
    metric_mse = 0
    metric_mae = 0
    metric_abc = 0
    cosine_sim = 0
    r2 = 0
    chi2 = 0
    pearson = 0
    pearson_p = 0
    counts_pearson = 0
    
    # Sicherstellen, dass h*w der erwarteten Patchgröße entspricht
    if h * w != self.config["patch_size"] * self.config["patch_size"]:
        print("ERROR: falsche Patchgröße!")
        return

    time_stamp_batch = self.time_stamp_batch[:, 0, :].permute((1, 0))
    
    # Iteriere über die Batch-Samples
    for i in range(b):
        # Verwende die curve-fit TAC als Ziel
        current_curve_fit_TAC = TAC_batch[i, :, :]  # Shape: [T, w*h]
        
        # Berechne den vom Modell vorhergesagten TAC anhand der kinetischen Parameter
        current_kinetic_params = kinetic_params[i, :, :, :]
        current_TAC_pred_long, _ = self.make_curve(current_kinetic_params, p[i])
        
        # Passe die Zeitdimension an: von 600 Zeitpunkten zurück auf T (z. B. 62)
        current_TAC_pred_long = current_TAC_pred_long[0, :, :].permute((1, 0))
        current_TAC_pred = torch_interp_Nd(self.time_stamp_batch[:, 0, :],
                                           self.t_batch[:, 0, :],
                                           current_TAC_pred_long)
        current_TAC_pred = current_TAC_pred.permute((1, 0))

        current_curve_fit_TAC = current_curve_fit_TAC.reshape(600, 12544).permute((1, 0)).to(self.device)

        current_curve_fit_TAC = torch_interp_Nd(self.time_stamp_batch[:, 0, :], self.t_batch[:, 0, :], current_curve_fit_TAC).permute((1, 0))
        
        # Interpoliere kinetische Parameter – wie k1, k2, k3, Vb, alpha – falls notwendig
        k1 = torch_interp_Nd(self.time_stamp_batch[:, 0, :], self.t_batch[:, 0, :], kinetic_params[i, 0, :, :]).permute((1, 0))
        k2 = torch_interp_Nd(self.time_stamp_batch[:, 0, :], self.t_batch[:, 0, :], kinetic_params[i, 1, :, :]).permute((1, 0))
        k3 = torch_interp_Nd(self.time_stamp_batch[:, 0, :], self.t_batch[:, 0, :], kinetic_params[i, 2, :, :]).permute((1, 0))
        Vb = torch_interp_Nd(self.time_stamp_batch[:, 0, :], self.t_batch[:, 0, :], kinetic_params[i, 3, :, :]).permute((1, 0))
        alpha = torch_interp_Nd(self.time_stamp_batch[:, 0, :], self.t_batch[:, 0, :], kinetic_params[i, 4, :, :]).permute((1, 0))
        beta = torch_interp_Nd(self.time_stamp_batch[:, 0, :], self.t_batch[:, 0, :], kinetic_params[i, 5, :, :]).permute((1, 0))
        
        # Optional: Anwenden einer Maske basierend auf AUC 
        if self.config["mask_loss"]:
            AUC = torch.trapezoid(current_curve_fit_TAC, time_stamp_batch, dim=0)
            maskk = AUC > 10
            mask = maskk.repeat(62, 1)
            current_curve_fit_TAC = current_curve_fit_TAC * mask
            current_TAC_pred = current_TAC_pred * mask
            k1 = k1 * mask
            k2 = k2 * mask
            k3 = k3 * mask
            Vb = Vb * mask
            alpha = alpha * mask
            beta = beta * mask

        # Speichere den vorhergesagten TAC für das aktuelle Sample
        TAC_pred_batch[i, :, :] = current_TAC_pred
        
        # Berechne den Verlust
        if return_value is None or return_value == "Loss":
            loss += self.loss_function(current_TAC_pred, current_curve_fit_TAC, k1, k2, k3, Vb, alpha=alpha, beta=beta)
        
        if return_value is None or return_value == "Metric":  
          if self.config["mask_loss"]:
              square = torch.square(current_TAC_pred.to(machine) - current_curve_fit_TAC.to(machine))
              absolute = torch.abs(current_TAC_pred.to(machine) - current_curve_fit_TAC.to(machine))
              if len(mask[mask>0]) > 0:
                  div = len(mask[mask>0])
              else:
                  div = 1
              metric_mse += torch.sum(square).item() / div
              metric_mae += torch.sum(absolute).item() / div

              cosine_sim_slice = torch.nn.functional.cosine_similarity(current_TAC_pred.to(machine), current_curve_fit_TAC.to(machine), 0)
              if len(maskk[maskk>0]) > 0:
                  div = len(maskk[maskk>0])
              else:
                  div = 1
              cosine_sim += torch.sum(cosine_sim_slice).item() / div

              weights = (self.frame_duration_batch * torch.exp(-0.00631 * time_stamp_batch)) / (current_curve_fit_TAC.to(machine))
              weights = torch.nan_to_num(weights, posinf=1)
              chi2_slice = torch.sum(torch.multiply(square, weights), axis=0)
              chi2 += torch.sum(chi2_slice).item() / div
          else:
              metric_mse += similaritymeasures_torch.mse(current_TAC_pred.to(machine), current_curve_fit_TAC.to(machine)).item()
              metric_mae += similaritymeasures_torch.mae(current_TAC_pred.to(machine), current_curve_fit_TAC.to(machine)).item()
              cosine_sim += torch.mean(torch.nn.functional.cosine_similarity(current_TAC_pred.to(machine), current_curve_fit_TAC.to(machine), 0)).item()

              square = torch.square(current_TAC_pred.to(machine) - current_curve_fit_TAC.to(machine))
              weights = (self.frame_duration_batch * torch.exp(-0.00631 * time_stamp_batch)) / (current_curve_fit_TAC.to(machine))
              weights = torch.nan_to_num(weights, posinf=1)
              chi2_slice = torch.sum(torch.multiply(square, weights), axis=0)
              chi2 += torch.mean(chi2_slice).item()
          
          # The following metrics are the same independently from self.config["mask_loss"]
          r2 += similaritymeasures_torch.r2(current_TAC_pred.to(machine), current_curve_fit_TAC.to(machine)).item()

          if self.config["use_pearson_metric"]:
              for j in range(h*w):
                  current_pred_TAC = current_TAC_pred[:, j]
                  current_real_TAC = current_curve_fit_TAC[:, j]
                  if self.config["mask_loss"]: 
                      if torch.sum(current_pred_TAC) == 0 or torch.sum(current_real_TAC) == 0:
                          continue
                      else:
                          current_pearson, current_p = pearsonr(current_pred_TAC.cpu().detach().numpy(), current_real_TAC.cpu().detach().numpy())
                          pearson += current_pearson
                          pearson_p += current_p
                          counts_pearson += 1

    loss = loss/b           # Use b instead of self.config["batch_size"] to accomodate for batches of different size (like the last one)
    #loss_dict =  {"loss": loss}

    metric_mse = metric_mse/b
    metric_mae = metric_mae/b
    metric_abc = metric_abc/b
    cosine_sim = cosine_sim/b
    r2 = r2/b
    if self.config["use_pearson_metric"]:
      pearson_p = pearson_p/counts_pearson
      pearson = pearson/counts_pearson
    
    metric_dict = {"mse": metric_mse, "mae": metric_mae, "cosine_sim": cosine_sim, "r2": r2}
    if self.config["use_pearson_metric"]:
      metric_dict["pearson_corr"] = pearson
      metric_dict["pearson_p"] = pearson_p
    
    TAC_pred_batch = torch.reshape(TAC_pred_batch, [b, 1, 62, h, w])

    
    # Lade die Ground-Truth kinetischen Parameter (als Liste von Tensoren)
    curve_fit_tac, gt_params_list = self.apply_curve_fit_on_batch(batch)

    predicted_params = apply_final_activation(logits, self.config)  # Erwartete Form: [B, 6, w*h, 1]
   
    gt_params = []
    for i in range(b):
        
        gt = gt_params_list[i][:6, :, :].clone().detach().to(self.device)
        gt_params.append(gt)
    gt_params = torch.stack(gt_params, dim=0)  # [B, 6, w, h]
    gt_params = gt_params.reshape(b, 6, 12544) # [B, 6, w*h]

    predicted_params = predicted_params.squeeze(-1) # [B, 6, w*h]

    # Berechne den MSE-Loss zwischen den vorhergesagten Parametern und den Ground Truth
    params_mse = torch.square(predicted_params.squeeze(0) - gt_params.squeeze(0))
    params_mse = torch.nanmean(params_mse)

    loss += 10*params_mse

    loss_dict = {"loss": loss}
    #metric_dict = {"param_MSE": params_mse}

    if return_value is None:
      return loss_dict, metric_dict, TAC_pred_batch
    elif return_value == "Loss":
      return loss_dict
    elif return_value == "Metric":
      return metric_dict, TAC_pred_batch


  def accumulate_loss(self, batch, logits):
    return self.accumulate_func(batch, logits, return_value="Loss")
  
  def accumulate_metric(self, batch, logits):
    return self.accumulate_func(batch, logits, return_value="Metric")
  
  def accumulate_loss_and_metric(self, batch, logits):
    return self.accumulate_func(batch, logits, return_value=None)

  def apply_curve_fit_on_batch(self, batch):
    fitted_TAC_list = []
    params_list = []
    # Hole den Zeitvektor als numpy-Array (z. B. Länge T)
    t_file = torch.load(root_data_path + "/DynamicPET/t.pt")
    time_stamp = torch.load(root_data_path + "/DynamicPET/time_stamp.pt")

    batch_size = len(batch[0])
    _, T, w, h = batch[2].shape  # TAC_mes_batch hat Form [b, 1, T, w, h]
    
    # Verzeichnis, in dem die gefitteten Ergebnisse gespeichert werden
    save_dir = self.config.get("curve_fit_save_path", "./curve_fit_results")
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(batch_size):
        # Extrahiere Patienten-ID und Slice-Index
        patient = batch[0][i]
        slice_index = batch[1][i].item() if torch.is_tensor(batch[1][i]) else batch[1][i]
        # TAC des Samples: Form [T, w, h]
        tac = batch[2][i].squeeze(0)
        
        # Initialisiere leere Arrays für den gefitteten TAC und die Parameter (pro Voxel)
        fitted_TAC_sample = np.zeros((600, w, h), dtype=np.float32)
        param_maps = {
            "k1": np.zeros((w, h), dtype=np.float32),
            "k2": np.zeros((w, h), dtype=np.float32),
            "k3": np.zeros((w, h), dtype=np.float32),
            "Vb": np.zeros((w, h), dtype=np.float32),
            "alpha": np.zeros((w, h), dtype=np.float32),
            "beta": np.zeros((w, h), dtype=np.float32),
            "fit_success": np.zeros((w, h), dtype=np.uint8)  # 1: fit erfolgreich, 0: fehlgeschlagen
        }
        
        # Dateiname zur Speicherung (z. B. pro Patient und Slice)
        filename = os.path.join(save_dir, f"{patient}_slice{slice_index}_curve_fit_lung_full.npz")
        if os.path.exists(filename):
            # Lade bereits berechnete Ergebnisse
            data = np.load(filename)
            fitted_TAC_sample = data["fitted_TAC"]
            for key in param_maps.keys():
                param_maps[key] = data[key]
        else:
            # Definiere das Modell, das für das Curve Fit genutzt wird
            def model_func(t, k1, k2, k3, Vb, alpha, beta):
                step = 0.1
                # Hole die IDIFs als numpy-Arrays (angenommen, diese sind im Trainset gespeichert)
                if self.stage == "fit" and patient in self.aorta_idif_train_set.keys():
                    aorta = self.aorta_idif_train_set[patient]
                    portal = self.portal_idif_train_set[patient]
                elif self.stage == "fit" and patient in self.aorta_idif_val_set.keys():
                    aorta = self.aorta_idif_val_set[patient]
                    portal = self.portal_idif_val_set[patient]
                elif self.stage == "test" and patient in self.aorta_idif_test_set.keys():
                    aorta = self.aorta_idif_test_set[patient]
                    portal = self.portal_idif_test_set[patient]
                else: 
                    print("ERROR: IDIF of patient " + str(patient) + " not found!")
                    return
                a = alpha * aorta + beta * portal
                e = (k2 + k3) * t
                b = k1 / (k2 + k3) * (k3 + k2 * np.exp(-e))
                c = torch_conv(a, b) * step
                PET = (1 - Vb) * c + Vb * a
                return PET
            
            # Parameter-Initialisierung und Bounds
            p0 = [0.1, 0.1, 0.001, 0.001, 0.0, 0.0]
            bounds_lower = [0.01, 0.01, 0.001, 0.001, 0.0, 0.0]
            bounds_upper = [10, 10, 1, 0.2, 1, 1]
            
            # Iteriere über alle räumlichen Positionen
            for x in range(w):
                for y in range(h):
                    tac_voxel = tac[:, x, y]# Zeitserie dieses Voxels, Länge T
                    # Prüfe, ob der AUC über dem Schwellenwert liegt (z. B. > 10)
                    auc = torch.trapz(tac_voxel.cuda(), time_stamp.cuda())
                    tac_voxel = torch_interp_1d(t_file, time_stamp.cpu(), tac_voxel.cpu())

                    if auc <= 10:
                        # Bei zu geringem Signal: Original-TAC verwenden, Parameter bleiben 0
                        tac_low = torch.zeros_like(tac_voxel)
                        fitted_TAC_sample[:, x, y] = tac_low
                        continue
                    try:
                        popt, _ = curve_fit(
                            model_func, t_file, tac_voxel,
                            p0=p0, bounds=(bounds_lower, bounds_upper),
                            diff_step=0.001
                        )
                        fitted_TAC_voxel = model_func(t_file.cpu(), *popt)
                        # Speichere die gefitteten Parameter
                        param_maps["k1"][x, y] = popt[0]
                        param_maps["k2"][x, y] = popt[1]
                        param_maps["k3"][x, y] = popt[2]
                        param_maps["Vb"][x, y] = popt[3]
                        param_maps["alpha"][x, y] = popt[4]
                        param_maps["beta"][x, y] = popt[5]
                        param_maps["fit_success"][x, y] = 1  # Fit war erfolgreich
                    except Exception as e:
                        print(f"Curve fit failed für Voxel ({x}, {y}) bei Patient {patient} Slice {slice_index}: {e}")
                        fitted_TAC_voxel = tac_voxel  # Fallback: Original-TAC
                        param_maps["fit_success"][x, y] = 0  # Fit fehlgeschlagen
                    fitted_TAC_sample[:, x, y] = fitted_TAC_voxel
            # Speichere die Ergebnisse als komprimierte npz-Datei
            np.savez_compressed(filename,
                                fitted_TAC=fitted_TAC_sample,
                                k1=param_maps["k1"],
                                k2=param_maps["k2"],
                                k3=param_maps["k3"],
                                Vb=param_maps["Vb"],
                                alpha=param_maps["alpha"],
                                beta=param_maps["beta"],
                                fit_success=param_maps["fit_success"])
        
        # Füge das gefittete Ergebnis für dieses Sample der Liste hinzu (als Tensor mit passender Dimension)
        fitted_TAC_list.append(torch.tensor(fitted_TAC_sample, dtype=torch.float32).unsqueeze(0))  # Form: (1, T, w, h)
        # Staple die Parameter in einem Tensor (Form: [6, w, h])
        param_map_tensor = torch.stack([
            torch.tensor(param_maps["k1"], dtype=torch.float32),
            torch.tensor(param_maps["k2"], dtype=torch.float32),
            torch.tensor(param_maps["k3"], dtype=torch.float32),
            torch.tensor(param_maps["Vb"], dtype=torch.float32),
            torch.tensor(param_maps["alpha"], dtype=torch.float32),
            torch.tensor(param_maps["beta"], dtype=torch.float32),
            torch.tensor(param_maps["fit_success"], dtype=torch.float32)
        ], dim=0)
        params_list.append(param_map_tensor)
    
    # Erstelle den Batch: Form [b, 1, T, w, h]
    fitted_TAC_batch = torch.stack(fitted_TAC_list, dim=0)
    return fitted_TAC_batch, params_list


  
  def make_curve(self, kinetic_params, patient):
    # Prepare the IDIF --> can't be moved outside the for loop because the patient can change inside the batch
    if self.stage == "fit" and patient in self.aorta_idif_train_set.keys():
        current_aorta_idif = self.aorta_idif_train_set[patient]
        current_portal_idif = self.portal_idif_train_set[patient]
    elif self.stage == "fit" and patient in self.aorta_idif_val_set.keys():
        current_aorta_idif = self.aorta_idif_val_set[patient]
        current_portal_idif = self.portal_idif_val_set[patient]
    elif self.stage == "test" and patient in self.aorta_idif_test_set.keys():
        current_aorta_idif = self.aorta_idif_test_set[patient]
        current_portal_idif = self.portal_idif_test_set[patient]
    else: 
        print("ERROR: IDIF of patient " + str(patient) + " not found!")
        return

    aorta_idif_batch = current_aorta_idif.repeat(1, self.patch_size * self.patch_size, 1)
    portal_idif_batch = current_portal_idif.repeat(1, self.patch_size * self.patch_size, 1)


    # Prepare the kinetic parameters
    kinetic_params = kinetic_params.permute((1, 0, 2))
    k1 = kinetic_params[:, 0, :].unsqueeze(1)
    k2 = kinetic_params[:, 1, :].unsqueeze(1)        
    k3 = kinetic_params[:, 2, :].unsqueeze(1)
    Vb = kinetic_params[:, 3, :].unsqueeze(1)
    alpha = kinetic_params[:, 4, :].unsqueeze(0)  # Alpha for Aorta (1, h*w, 600)
    beta = kinetic_params[:, 5, :].unsqueeze(0)  # Beta for Portal (1, h*w, 600)


    # Compute the TAC using updated model
    current_pred_curve = PET_2TC_KM_batch(
      aorta_idif_batch.to(self.device),
      portal_idif_batch.to(self.device),
      self.t_batch.to(self.device),
      k1,
      k2,
      k3,
      Vb,
      alpha,
      beta
      
    )
    
    return current_pred_curve, None


  def training_step(self, batch, batch_idx):
    # Wende den voxelweisen Curve Fit an – damit wird für jeden Voxel der beste TAC errechnet
    TAC_mes_batch = torch.unsqueeze(batch[2], 1)  # adding channel dimension --> [b, 1, 62, w, h]

    fitted_TAC_batch, curve_fit_params = self.apply_curve_fit_on_batch(batch)
    b, x, t, h, w = fitted_TAC_batch.shape
    out_t = 62
    output = torch.zeros((b, x, out_t, h, w), device=self.device, dtype=fitted_TAC_batch.dtype)

    for batch_idx in range(b):
      for x_idx in range(x):
        # reshape [t, h, w] zu [t, h*w]
        current_TAC = fitted_TAC_batch[batch_idx, x_idx].reshape(t, h * w).permute(1, 0).to(self.device)  # [h*w, t]
        # Interpolation durchführen: [h*w, 600] -> [h*w, 62]
        interpolated_TAC = torch_interp_Nd(
            self.time_stamp_batch[:, 0, :],  
            self.t_batch[:, 0, :],           
            current_TAC                             
        )  # Ergebnis: [h*w, 62]

        # Zurückformen zu [62, h, w]
        interpolated_TAC = interpolated_TAC.permute(1, 0).reshape(out_t, h, w)

        # Speichern
        output[batch_idx, x_idx] = interpolated_TAC
    #test = fitted_TAC_batch.reshape(b, x, t, h*w).permute((1, 0)).to(self.device)
    #test = torch_interp_Nd(self.time_stamp_batch[:, 0, :], self.t_batch[:, 0, :], test).permute((1, 0))
    test = output.reshape(b, 1, 62, 112, 112)

    # Verwende den gefitteten TAC als Input für das Netz (hier ggf. noch Padding hinzufügen)
    x = torch.nn.functional.pad(test, (0, 0, 0, 0, 1, 1))  # padding --> [b, 1, 64, w, h]

    logits_params = self.forward(x)
    
    loss_dict, metric_dict, TAC_pred_batch = self.accumulate_loss_and_metric(batch=batch, logits=logits_params)
    
    self.log('train_loss', loss_dict["loss"].item(), on_step=False, on_epoch=True, batch_size=self.config["batch_size"])
    metric_dict_prefixed = {f"train_{k}": v for k, v in metric_dict.items()}
    self.log_dict(metric_dict_prefixed, on_step=False, on_epoch=True, batch_size=self.config["batch_size"])

    
    if batch_idx % 5 == 0:
      if self.config["mask_loss"]:
          test, TAC_mes_batch, TAC_pred_batch, logits_params = mask_data(test, TAC_mes_batch, TAC_pred_batch, logits_params, self.time_stamp, patch_size=self.patch_size)
      
      fig = log_curves(test[:, :, 0:62, :, :].cpu().detach().numpy(), TAC_pred_batch.cpu().detach().numpy(), self.time_stamp.to("cpu"), self.time_stamp.to("cpu"), self.current_epoch)
      wandb.log({"TAC (training batch: "+str(batch_idx)+")": wandb.Image(fig)})
      plt.close()

      kinetic_params = apply_final_activation(logits_params, self.config)
      fig = log_slice(self.config, test, kinetic_params)
      wandb.log({"Slice (training batch " + str(batch_idx) + ")": wandb.Image(fig)})
      plt.close()
    
    return {"loss": loss_dict["loss"]}

  def validation_step(self, batch, batch_idx):
    # batch = [patient, slice, TAC]
    TAC_mes_batch = torch.unsqueeze(batch[2], 1) # adding channel dimension --> [b, 1, 62, w, h]

    fitted_TAC_batch, curve_fit_params = self.apply_curve_fit_on_batch(batch)
    b, x, t, h, w = fitted_TAC_batch.shape
    out_t = 62
    output = torch.zeros((b, x, out_t, h, w), device=self.device, dtype=fitted_TAC_batch.dtype)

    for batch_idx in range(b):
      for x_idx in range(x):
        # reshape [t, h, w] zu [t, h*w]
        current_TAC = fitted_TAC_batch[batch_idx, x_idx].reshape(t, h * w).permute(1, 0).to(self.device)  # [h*w, t]
        # Interpolation durchführen: [h*w, 600] -> [h*w, 62]
        interpolated_TAC = torch_interp_Nd(
            self.time_stamp_batch[:, 0, :],  
            self.t_batch[:, 0, :],           
            current_TAC                             
        )  # Ergebnis: [h*w, 62]

        # Zurückformen zu [62, h, w]
        interpolated_TAC = interpolated_TAC.permute(1, 0).reshape(out_t, h, w)

        # Speichern
        output[batch_idx, x_idx] = interpolated_TAC
    #test = fitted_TAC_batch.reshape(b, x, t, h*w).permute((1, 0)).to(self.device)
    #test = torch_interp_Nd(self.time_stamp_batch[:, 0, :], self.t_batch[:, 0, :], test).permute((1, 0))
    test = output.reshape(b, 1, 62, 112, 112)

    # Verwende den gefitteten TAC als Input für das Netz (hier ggf. noch Padding hinzufügen)
    x = torch.nn.functional.pad(test, (0, 0, 0, 0, 1, 1))  # padding --> [b, 1, 64, w, h]

    logits_params = self.forward(x)        # [b, 6, 1, w, h]

    loss_dict, metric_dict, TAC_pred_batch = self.accumulate_loss_and_metric(batch=batch, logits=logits_params)

    self.log('val_loss', loss_dict["loss"].item(), on_step=False, on_epoch=True, batch_size=self.config["batch_size"])
    metric_dict_prefixed = {f"val_{k}": v for k, v in metric_dict.items()}
    self.log_dict(metric_dict_prefixed, on_step=False, on_epoch=True, batch_size=self.config["batch_size"])

    # Prepare data to log
    if batch_idx % 5 == 0:
      if self.config["mask_loss"]:
        test, TAC_mes_batch, TAC_pred_batch, logits_params = mask_data(test, TAC_mes_batch, TAC_pred_batch, logits_params, self.time_stamp, patch_size=self.patch_size)

      # Log TAC                
      fig = log_curves(test[:, :, 0:62, :, :].cpu().detach().numpy(), TAC_pred_batch.cpu().detach().numpy(), self.time_stamp.to("cpu"), self.time_stamp.to("cpu"), self.current_epoch)
      fig_curve = {"TAC (validation batch: "+str(batch_idx)+")": wandb.Image(fig)}
      plt.close()

      # Log slices
      kinetic_params = apply_final_activation(logits_params, self.config)
      fig = log_slice(self.config, test, kinetic_params)
      fig_slice = {"Slice (validation batch: "+str(batch_idx)+")": wandb.Image(fig)}
      plt.close()

      return {"fig_slice": fig_slice, "fig_curve": fig_curve}
    else:
      return {'val_loss': loss_dict["loss"].item(), "fig_slice": None, "fig_curve": None}


  def validation_epoch_end(self, outputs):
    for o in outputs:
      if not o["fig_slice"] is None:  wandb.log(o["fig_slice"])
      if not o["fig_curve"] is None:  wandb.log(o["fig_curve"])
    return
  
  def test_step(self, batch, batch_idx):
    # batch = [patient, slice, TAC]
    patients_in_batch = batch[0]
    slices_in_batch = batch[1]
    TAC_mes_batch = torch.unsqueeze(batch[2], 1) # adding channel dimension --> [b, 1, 62, w, h]
    fitted_TAC_batch, curve_fit_params = self.apply_curve_fit_on_batch(batch)

    b, x, t, h, w = fitted_TAC_batch.shape
    out_t = 62
    output = torch.zeros((b, x, out_t, h, w), device=self.device, dtype=fitted_TAC_batch.dtype)

    for batch_idx in range(b):
      for x_idx in range(x):
        # reshape [t, h, w] zu [t, h*w]
        current_TAC = fitted_TAC_batch[batch_idx, x_idx].reshape(t, h * w).permute(1, 0).to(self.device)  # [h*w, t]
        # Interpolation durchführen: [h*w, 600] -> [h*w, 62]
        interpolated_TAC = torch_interp_Nd(
            self.time_stamp_batch[:, 0, :],  
            self.t_batch[:, 0, :],           
            current_TAC                             
        )  # Ergebnis: [h*w, 62]

        # Zurückformen zu [62, h, w]
        interpolated_TAC = interpolated_TAC.permute(1, 0).reshape(out_t, h, w)

        # Speichern
        output[batch_idx, x_idx] = interpolated_TAC
    #test = fitted_TAC_batch.reshape(b, x, t, h*w).permute((1, 0)).to(self.device)
    #test = torch_interp_Nd(self.time_stamp_batch[:, 0, :], self.t_batch[:, 0, :], test).permute((1, 0))
    test = output.reshape(b, 1, 62, 112, 112)

    # Verwende den gefitteten TAC als Input für das Netz (hier ggf. noch Padding hinzufügen)
    x = torch.nn.functional.pad(test, (0, 0, 0, 0, 1, 1))  # padding --> [b, 1, 64, w, h]

    logits_params = self.forward(x)        # [b, 6, 1, w, h]

    metric_dict, TAC_pred_batch = self.accumulate_metric(batch=batch, logits=logits_params)
    kinetic_params = apply_final_activation(logits_params, self.config)

    if self.config["mask_loss"]:
      test, TAC_mes_batch, TAC_pred_batch, kinetic_params = mask_data(test, TAC_mes_batch, TAC_pred_batch, kinetic_params, self.time_stamp, patch_size=self.patch_size)

    # Save predictions
    current_run_name = wandb.run.name
    checkpoint_path = self.checkpoint_path  # Zugriff auf den Checkpoint-Pfad
  
    self.img_path, self.pd_path, self.pt_path, self.nifty_path = make_save_folder_struct(current_run_name, root_checkpoints_path, checkpoint_path)
    to_save = [patients_in_batch, slices_in_batch, kinetic_params]
    s = int(slices_in_batch.item())
    torch.save(to_save, os.path.join(self.pt_path, "P_"+str(patients_in_batch[0])+"_S_"+str(s)+"_B_"+str(batch_idx)+".pt"))

    # Prepare data to log
    if batch_idx % 50 == 0:
      if not len(slices_in_batch) == 1: s = slices_in_batch

      # Log TAC             
      log_curves(test[:, :, 0:62, :, :].cpu().detach().numpy(), TAC_pred_batch.cpu().detach().numpy(), self.time_stamp.to("cpu"), self.time_stamp.to("cpu"), self.current_epoch)
      plt.savefig(os.path.join(self.img_path, "TAC_P_"+str(patients_in_batch[0])+"_S_"+str(s)+"_B_"+str(batch_idx)+".png"))
      plt.close()

      # Log slices
      log_slice(self.config, test, kinetic_params)
      plt.savefig(os.path.join(self.img_path, "slices_P_"+str(patients_in_batch[0])+"_S_"+str(s)+"_B_"+str(batch_idx)+".png"))
      plt.close()

    return {"patients_in_batch": patients_in_batch, "slices_in_batch": slices_in_batch, "metric_dict": metric_dict}

  def test_epoch_end(self, outputs):
    run_name = os.path.split(os.path.split(self.pd_path)[0])[1]
    summary = dict()

    for o in outputs:
      metric_dict = o["metric_dict"]
      patients_in_batch = o["patients_in_batch"]
      slices_in_batch = o["slices_in_batch"]
      for i in range(len(patients_in_batch)):
        p = patients_in_batch[i]
        if not p in summary.keys(): summary[p] = dict()
        for j in range(len(slices_in_batch)):
          s = int(slices_in_batch[j].item())
          if patients_in_batch[j] == p:
            summary[p][s] = dict()
            summary[p][s]["MSE"] = metric_dict["mse"]
            summary[p][s]["MAE"] = metric_dict["mae"]
            summary[p][s]["CosineSim"] = metric_dict["cosine_sim"]

    for p in summary.keys():
      current_df = pd.DataFrame.from_dict(summary[p])
      # This file contains the metrics per slice. It allows to identify slices with bad peformance. 
      # It is also used during evaluation phase to compute the metrics on the whole dataset
      current_df.to_excel(os.path.join(self.pd_path, p+"_metric_per_slice_"+run_name+".xlsx"))

    # Reconstruct the 3D kinetic parameters volumes
    reconstruct_prediction(self.pt_path, self.nifty_path)
    return 

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75, 100], gamma=1, verbose=True)
    return [optimizer], [scheduler]
    
def train_unet(resume_path=None, enable_testing=False):
  wandb.finish()

  wandb.login(key="6b4147eb19882f693f5cbb529444079d6e432eff")
  
  wandb.init(project="DynPET_Portal_Train", config=os.path.join(root_path, "config.yaml"))

  # Set up Weights&Biases Logger
  wandb_logger = WandbLogger()
  
  unet = SpaceTempUNet(wandb.config)

  # Set up the checkpoints
  checkpoint_path = os.path.join(root_checkpoints_path, "checkpoints", wandb.run.name)
  print("Checkpoints will be saved in: ", checkpoint_path)
  checkpoint_callback = ModelCheckpoint(  monitor="val_loss",
                                          dirpath=checkpoint_path,
                                          save_top_k=1,
                                          mode="min",
                                          every_n_epochs=5,
                                          save_last=True,
                                      )
  early_stop_callback = EarlyStopping(  monitor="val_loss", 
                                        min_delta=0, 
                                        patience=25,
                                        verbose=True, 
                                        mode="min",
                                        check_finite=True)

  trainer = pl.Trainer(gpus=current_gpu,
                        max_epochs=unet.config["epochs"],
                        enable_checkpointing=True,
                        num_sanity_val_steps=1,
                        log_every_n_steps=unet.config["log_freq"],
                        check_val_every_n_epoch=unet.config["val_freq"],
                        callbacks=[checkpoint_callback, early_stop_callback],
                        logger=wandb_logger,
                        resume_from_checkpoint=resume_path
                    )
  
  trainer.fit(unet)

  if enable_testing:
    trainer.test(ckpt_path="best")
    trainer.test(ckpt_path="last")
  
  # Close Weights&Biases Logger 
  wandb.finish()

def test_unet(checkpoint_path=None):

  wandb.login(key="6b4147eb19882f693f5cbb529444079d6e432eff")
  
  if wandb.run is not None:
    wandb.finish()

  wandb.init(project="DynPET_Portal_Test", config=os.path.join(root_path, "config.yaml"))

  # Set up Weights&Biases Logger
  wandb_logger = WandbLogger()
  if checkpoint_path is None:
    checkpoint_path = os.path.join(root_checkpoints_path, "checkpoints", wandb.config["saved_checkpoint"])
  print("Testing on", checkpoint_path)
  
  unet = SpaceTempUNet(wandb.config)
  unet.checkpoint_path = checkpoint_path  # Speichere den Checkpoint-Pfad im Modell


  trainer = pl.Trainer(gpus=current_gpu,
                        max_epochs=unet.config["epochs"],
                        enable_checkpointing=True,
                        num_sanity_val_steps=1,
                        log_every_n_steps=unet.config["log_freq"],
                        check_val_every_n_epoch=unet.config["val_freq"],
                        logger=wandb_logger,
                      )
  
  trainer.test(unet, ckpt_path=checkpoint_path)
  
  # Close Weights&Biases Logger 
  wandb.finish()

if __name__ == '__main__':
  
  ### TRAIN ###
  train_unet(resume_path=None, enable_testing=False)

  ### TEST ###
  #test_unet()
  