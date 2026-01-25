 
# -*- coding: utf-8 -*-
import os
import copy
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vision_toolkit.segmentation.segmentation_algorithms.I_HOV import pre_process_IHOV
from vision_toolkit.segmentation.segmentation_algorithms.I_FC import pre_process_IFC
from vision_toolkit.segmentation.segmentation_algorithms.I_CNN import (
    pre_process_ICNN,
    CNN1D,
    train_cnn1d_windows,  
)

from vision_toolkit.utils.segmentation_utils import (
    interval_merging,
    filter_binary_intervals_by_duration,
    filter_ternary_intervals_by_duration,
)

from vision_toolkit.segmentation.basic_processing import oculomotor_series as ocs

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None


class LearningSegmentation:
     
    def __init__(self, sampling_frequency, segmentation_method, task, **kwargs):
        """
        

        Parameters
        ----------
        sampling_frequency : TYPE
            DESCRIPTION.
        segmentation_method : TYPE
            DESCRIPTION.
        task : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if segmentation_method == "I_HOV":
            kwargs.setdefault("distance_type", "euclidean")
        else:
            kwargs.setdefault("distance_type", "angular")

        self.config = {
            "sampling_frequency": sampling_frequency,
            "segmentation_method": segmentation_method,
            "task": task,
            "distance_projection": kwargs.get("distance_projection"),
            "distance_type": kwargs.get("distance_type"),
            "size_plan_x": kwargs.get("size_plan_x"),
            "size_plan_y": kwargs.get("size_plan_y"),
            "smoothing": kwargs.get("smoothing", "savgol"),
            "verbose": kwargs.get("verbose", True),

            # duration constraints (seconds)
            "min_fix_duration": kwargs.get("min_fix_duration", 7e-2),
            "max_fix_duration": kwargs.get("max_fix_duration", 2.0),

            # ternary only
            "min_pursuit_duration": kwargs.get("min_pursuit_duration", 1e-1),
            "max_pursuit_duration": kwargs.get("max_pursuit_duration", 2.0),
        }

        # smoothing params
        if self.config["smoothing"] in ("moving_average", "speed_moving_average"):
            self.config["moving_average_window"] = kwargs.get("moving_average_window", 5)
        elif self.config["smoothing"] == "savgol":
            self.config["savgol_window_length"] = kwargs.get("savgol_window_length", 31)
            self.config["savgol_polyorder"] = kwargs.get("savgol_polyorder", 3)

        # method-specific config
        if segmentation_method == "I_HOV":
            self.config.update(
                {
                    "IHOV_duration_threshold": kwargs.get("IHOV_duration_threshold", 0.2),
                    "IHOV_averaging_threshold": kwargs.get("IHOV_averaging_threshold", 0.2),
                    "IHOV_angular_bin_nbr": kwargs.get("IHOV_angular_bin_nbr", 36),
                }
            )
        elif segmentation_method == "I_FC":
            self.config.update(
                {
                    "IFC_bcea_prob": kwargs.get("IFC_bcea_prob", 0.68),
                    "IFC_i2mc": kwargs.get("IFC_i2mc", True),
                    "IFC_i2mc_window_duration": kwargs.get("IFC_i2mc_window_duration", 0.2),
                    "IFC_i2mc_moving_threshold": kwargs.get("IFC_i2mc_moving_threshold", 0.02),
                    # If your I_FC still standardizes internally, keep this False and rely on sklearn Pipeline
                    "IFC_internal_standardize": kwargs.get("IFC_internal_standardize", False),
                }
            )
        elif segmentation_method == "I_CNN":
            self.config.update(
                {
                    "ICNN_temporal_window_size": int(kwargs.get("ICNN_temporal_window_size", 251)),
                    "ICNN_batch_size": int(kwargs.get("ICNN_batch_size", 1024)),
                    "ICNN_learning_rate": float(kwargs.get("ICNN_learning_rate", 1e-3)),
                    "ICNN_num_epochs": int(kwargs.get("ICNN_num_epochs", 25)),
                    "ICNN_cudnn_benchmark": bool(kwargs.get("ICNN_cudnn_benchmark", True)),
                }
            )
            # enforce odd window
            if self.config["ICNN_temporal_window_size"] % 2 == 0:
                self.config["ICNN_temporal_window_size"] += 1
        else:
            raise ValueError(f"Unsupported segmentation_method: {segmentation_method}")

        # Add parameters to each segmentation method for clasifiers ??
        self._clf_factory = {
                    "rf": RandomForestClassifier(max_depth=10, max_features="sqrt"),
                    "svm": SVC(),
                    "knn": KNeighborsClassifier(n_neighbors=3),
                }

   
    @staticmethod
    def _load_df(input_):
        if isinstance(input_, pd.DataFrame):
            return input_
        return pd.read_csv(input_)


    def _preprocess_single_df(self, df):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        data_set : TYPE
            DESCRIPTION.
        config_i : TYPE
            DESCRIPTION.

        """
        cfg = copy.deepcopy(self.config)
        basic = ocs.OcculomotorSeries.generate(df, cfg)
        data_set = basic.get_data_set()
        config_i = basic.get_config()
        config_i["nb_samples"] = len(df)
        
        return data_set, config_i


    @staticmethod
    def _filter_dataset_by_mask(data_set, mask):
        """
        

        Parameters
        ----------
        data_set : TYPE
            DESCRIPTION.
        mask : TYPE
            DESCRIPTION.

        Returns
        -------
        ds_f : TYPE
            DESCRIPTION.

        """
        n = None
        # infer n from x_array if exists
        if "x_array" in data_set:
            n = len(data_set["x_array"])
        ds_f = {}
        for k, v in data_set.items():
            try:
                if n is not None and hasattr(v, "__len__") and len(v) == n:
                    ds_f[k] = np.asarray(v)[mask]
                else:
                    ds_f[k] = v
            except Exception:
                ds_f[k] = v
        return ds_f


    @staticmethod
    def _cnn_classes(task):
        
        if task == "binary":
            return 2
        if task == "ternary":
            return 3
        raise ValueError("task must be 'binary' or 'ternary'.")

   
    @classmethod
    def fit(
        cls,
        input_dfs,
        event_dfs,
        sampling_frequency,
        segmentation_method,
        task,
        classifier="rf",
        **kwargs,
    ):
        """
        

        Parameters
        ----------
        input_dfs : TYPE
            DESCRIPTION.
        event_dfs : TYPE
            DESCRIPTION.
        sampling_frequency : TYPE
            DESCRIPTION.
        segmentation_method : TYPE
            DESCRIPTION.
        task : TYPE
            DESCRIPTION.
        classifier : TYPE, optional
            DESCRIPTION. The default is "rf".
        **kwargs : TYPE
            DESCRIPTION.
         : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        path : TYPE
            DESCRIPTION.

        """
        mt = cls(
            sampling_frequency=sampling_frequency,
            segmentation_method=segmentation_method,
            task=task,
            **kwargs,
        )

        inputs = input_dfs if isinstance(input_dfs, (list, tuple)) else [input_dfs]
        labels = event_dfs if isinstance(event_dfs, (list, tuple)) else [event_dfs]
        if len(inputs) != len(labels):
            raise ValueError("Option A: input_dfs and event_dfs must have same length.")

        
        if segmentation_method in ("I_HOV", "I_FC"):
            if classifier not in mt._clf_factory:
                raise ValueError("classifier must be one of: 'rf', 'svm', 'knn'.")

            Xs, ys = [], []
            for inp, lab in zip(inputs, labels):
                df = mt._load_df(inp)
                df_lab = mt._load_df(lab)

                y = df_lab["event_label"].to_numpy().astype(int)
                if len(y) != len(df):
                    raise ValueError("Label length mismatch with input length.")

                keep = y > 0
                y = y[keep].astype(int)

                data_set, cfg_i = mt._preprocess_single_df(df)
                data_set = mt._filter_dataset_by_mask(data_set, keep)
                cfg_i["nb_samples"] = int(np.sum(keep))

                if segmentation_method == "I_HOV":
                    X = pre_process_IHOV(data_set, cfg_i)
                else:
                    X = pre_process_IFC(data_set, cfg_i)

                Xs.append(np.asarray(X))
                ys.append(np.asarray(y))

            if len(Xs) == 0:
                raise ValueError("No training samples after filtering labels > 0.")

            X_all = np.vstack(Xs)
            y_all = np.concatenate(ys)

            pipe = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("clf", mt._clf_factory[classifier]),
                ]
            )
            expected = {1,2} if task=="binary" else {1,2,3}
            got = set(np.unique(y_all))
            if not got.issubset(expected):
                raise ValueError(f"Unexpected labels {sorted(got)} for task={task}. Expected subset of {sorted(expected)}.")

            pipe.fit(X_all, y_all)

            path = (
                        f"vision_toolkit/segmentation/segmentation_algorithms/trained_models/"
                        f"{segmentation_method}/i_{classifier}_{task}.joblib"
                    )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(pipe, path)
            return path

        # ---- CNN method ----
        if segmentation_method == "I_CNN":
            if torch is None:
                raise RuntimeError("PyTorch is not available in this environment.")

            classes = mt._cnn_classes(task)

            Ws, ys0 = [], []
            for inp, lab in zip(inputs, labels):
                df = mt._load_df(inp)
                df_lab = mt._load_df(lab)

                y = df_lab["event_label"].to_numpy().astype(int)
                if len(y) != len(df):
                    raise ValueError("Label length mismatch with input length.")

                keep = y > 0
                y = y[keep].astype(int)
                if y.size == 0:
                    continue

                data_set, cfg_i = mt._preprocess_single_df(df)
                data_set = mt._filter_dataset_by_mask(data_set, keep)
                cfg_i["nb_samples"] = int(np.sum(keep))

                # windows per trial (avoids boundary artefacts)
                W = pre_process_ICNN(data_set, cfg_i)  # (n_kept, 2, win)
                Ws.append(W.astype(np.float32))

                # optional sanity-check on label set
                expected = {1, 2} if task == "binary" else {1, 2, 3}
                got = set(np.unique(y))
                if not got.issubset(expected):
                    raise ValueError(
                        f"Unexpected labels {sorted(got)} for task={task}. Expected subset of {sorted(expected)}."
                    )
                
                # labels for NLLLoss must be 0..classes-1
                y0 = (y.astype(np.int64) - 1)    
                # y0 in 0..K-1
                if y0.min() < 0 or y0.max() >= classes:
                    raise ValueError(
                        f"Labels out of range for task={task} classes={classes}: "
                        f"min={y0.min()} max={y0.max()}"
                    )
                ys0.append(y0)

            if len(Ws) == 0:
                raise ValueError("No training samples/windows for CNN after filtering labels > 0.")

            Xw = np.concatenate(Ws, axis=0)
            y0 = np.concatenate(ys0, axis=0)

            # Train on pre-windowed data
            path = train_cnn1d_windows(Xw, y0, mt.config)
            return path

        raise ValueError(f"Unsupported segmentation_method for fit(): {segmentation_method}")

 
    @classmethod
    def predict_raw(
        cls,
        input_,
        sampling_frequency,
        segmentation_method,
        task,
        classifier="rf",
        **kwargs,
    ):
        """
        

        Parameters
        ----------
        input_ : TYPE
            DESCRIPTION.
        sampling_frequency : TYPE
            DESCRIPTION.
        segmentation_method : TYPE
            DESCRIPTION.
        task : TYPE
            DESCRIPTION.
        classifier : TYPE, optional
            DESCRIPTION. The default is "rf".
        **kwargs : TYPE
            DESCRIPTION.
         : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        mt = cls(
            sampling_frequency=sampling_frequency,
            segmentation_method=segmentation_method,
            task=task,
            **kwargs,
        )
        df = mt._load_df(input_)
        data_set, cfg_i = mt._preprocess_single_df(df)

       
        if segmentation_method in ("I_HOV", "I_FC"):
            if classifier not in mt._clf_factory:
                raise ValueError("classifier must be one of: 'rf', 'svm', 'knn'.")
            model_path = (
                        f"vision_toolkit/segmentation/segmentation_algorithms/trained_models/"
                        f"{segmentation_method}/i_{classifier}_{task}.joblib"
                    )
            pipe = joblib.load(model_path)

            if segmentation_method == "I_HOV":
                X = pre_process_IHOV(data_set, cfg_i)
            else:
                X = pre_process_IFC(data_set, cfg_i)

            return np.asarray(pipe.predict(np.asarray(X))).astype(int)

       
        if segmentation_method == "I_CNN":
            if torch is None:
                raise RuntimeError("PyTorch is not available in this environment.")

            classes = mt._cnn_classes(task)
            path = (
                        "vision_toolkit/segmentation/segmentation_algorithms/trained_models/"
                        f"I_CNN/i_cnn_{task}.pt"
                    )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = CNN1D(
                numChannels=2,
                classes=classes,
                input_length=int(mt.config["ICNN_temporal_window_size"]),
            )
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()

            w_data = pre_process_ICNN(data_set, cfg_i)  # (n,2,W)
            X = torch.tensor(w_data, dtype=torch.float32)

            loader = DataLoader(
                TensorDataset(X),
                batch_size=int(mt.config["ICNN_batch_size"]),
                shuffle=False,
            )

            raw_idx = []
            with torch.no_grad():
                for (xb,) in loader:
                    xb = xb.to(device)
                    out = model(xb)
                    raw_idx.extend(out.argmax(dim=1).cpu().numpy())

         
            preds = (np.asarray(raw_idx) + 1).astype(int)
            return preds

        raise ValueError(f"Unsupported segmentation_method: {segmentation_method}")

   
    @classmethod
    def predict(
        cls,
        input_,
        sampling_frequency,
        segmentation_method,
        task,
        classifier="rf",
        **kwargs,
    ):
        """
        Single input only.
        Returns masks+intervals, then applies your post-processing:
          - binary  -> filter_binary_intervals_by_duration
          - ternary -> filter_ternary_intervals_by_duration
        """
        mt = cls(
            sampling_frequency=sampling_frequency,
            segmentation_method=segmentation_method,
            task=task,
            **kwargs,
        )

        preds = cls.predict_raw(
            input_=input_,
            sampling_frequency=sampling_frequency,
            segmentation_method=segmentation_method,
            task=task,
            classifier=classifier,
            **kwargs,
        )

        if task == "binary":
            is_fix = preds == 1
            is_sac = preds == 2

            raw = {
                "is_fixation": is_fix,
                "fixation_intervals": interval_merging(np.where(is_fix)[0]),
                "is_saccade": is_sac,
                "saccade_intervals": interval_merging(np.where(is_sac)[0]),
            }

            return filter_binary_intervals_by_duration(
                raw,
                sampling_frequency=mt.config["sampling_frequency"],
                min_fix_duration=mt.config["min_fix_duration"],
                max_fix_duration=mt.config["max_fix_duration"],
            )

        if task == "ternary":
            is_fix = preds == 1
            is_sac = preds == 2
            is_pur = preds == 3

            raw = {
                "is_fixation": is_fix,
                "fixation_intervals": interval_merging(np.where(is_fix)[0]),
                "is_saccade": is_sac,
                "saccade_intervals": interval_merging(np.where(is_sac)[0]),
                "is_pursuit": is_pur,
                "pursuit_intervals": interval_merging(np.where(is_pur)[0]),
            }

            return filter_ternary_intervals_by_duration(
                raw,
                sampling_frequency=mt.config["sampling_frequency"],
                min_fix_duration=mt.config["min_fix_duration"],
                max_fix_duration=mt.config["max_fix_duration"],
                min_pursuit_duration=mt.config["min_pursuit_duration"],
                max_pursuit_duration=mt.config["max_pursuit_duration"],
            )

        raise ValueError("task must be 'binary' or 'ternary'.")
