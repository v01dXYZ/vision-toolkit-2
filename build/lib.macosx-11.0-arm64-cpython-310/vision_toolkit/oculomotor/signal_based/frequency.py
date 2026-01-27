# -*- coding: utf-8 -*-
import numpy as np

from vision_toolkit.oculomotor.signal_based.signal_based_base import SignalBased
from vision_toolkit.utils.spectral_factory import csd_, periodogram_, welch_
from vision_toolkit.utils.velocity_distance_factory import process_speed_components
from vision_toolkit.visualization.oculomotor.frequency import plot_periodogram


class FrequencyAnalysis(SignalBased):
    def __init__(self, input, **kwargs):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        verbose = kwargs.get("verbose", True)

        if verbose:
            print("Processing Frequency Analysis...")

        super().__init__(input, **kwargs)

        if verbose:
            print("...Frequency Analysis done")


    def _clean_signal(self, x, status=None, interpolate=True):
        
        x = np.asarray(x, dtype=float)
    
        if status is not None:
            status = np.asarray(status)
            # adapte la règle selon ton codage (0/1, 1=valid, etc.)
            valid = status.astype(bool)
            x = x.copy()
            x[~valid] = np.nan
    
        if interpolate and np.isnan(x).any():
            idx = np.arange(len(x))
            good = ~np.isnan(x)
            if good.sum() >= 2:
                x[~good] = np.interp(idx[~good], idx[good], x[good])
            else:
                # pas assez de points valides
                return None
    
        return x

    
    def periodogram(self, periodogram_data_type='velocity', silent=False):
        """
        
    
        Parameters
        ----------
        periodogram_data_type : TYPE
            DESCRIPTION.
        silent : TYPE, optional
            DESCRIPTION. The default is False.
    
        Raises
        ------
        ValueError
            DESCRIPTION.
    
        Returns
        -------
        TYPE
            DESCRIPTION.
    
        """
        s_f = self.config["sampling_frequency"]
        status = self.data_set.get("status", None)
    
        densities = dict()
    
        # prépare les signaux source
        if periodogram_data_type == "velocity":
            sp = process_speed_components(self.data_set, self.config)[0:2, :]
            data = {"x_array": sp[0], "y_array": sp[1]}
        elif periodogram_data_type == "position":
            data = {"x_array": self.data_set["x_array"], "y_array": self.data_set["y_array"]}
        else:
            raise ValueError('periodogram_data_type must be "position" or "velocity"')
    
        for _dir in ["x_array", "y_array"]:
            x = data[_dir]
    
            x = self._clean_signal(x, status=status, interpolate=True)
            if x is None:
                raise ValueError("Not enough valid samples for spectral analysis")
    
            nperseg = x.shape[-1]
            freqs, p_xx = periodogram_(x, fs=s_f, nperseg=nperseg)
            densities.update({_dir[0]: p_xx})
    
        if self.config["display_results"] and not silent:
            plot_periodogram(freqs, densities)
    
        return {"frequencies": freqs, "spectral_densities": densities}


    def welch_periodogram(self,
                          periodogram_data_type="velocity", 
                          Welch_samples_per_segment=256,
                          silent=False,):
        """
        

        Parameters
        ----------
        periodogram_data_type : TYPE
            DESCRIPTION.
        Welch_samples_per_segment : TYPE
            DESCRIPTION.
        silent : TYPE, optional
            DESCRIPTION. The default is False.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        s_f = self.config["sampling_frequency"]
        status = self.data_set.get("status", None)
    
        densities = dict()
    
        # prépare les signaux source
        if periodogram_data_type == "velocity":
            sp = process_speed_components(self.data_set, self.config)[0:2, :]
            data = {"x_array": sp[0], "y_array": sp[1]}
    
        elif periodogram_data_type == "position":
            data = {
                "x_array": self.data_set["x_array"],
                "y_array": self.data_set["y_array"],
            }
    
        else:
            raise ValueError('periodogram_data_type must be "position" or "velocity"')
    
        for _dir in ["x_array", "y_array"]:
            x = data[_dir]
    
            x = self._clean_signal(x, status=status, interpolate=True)
            if x is None:
                raise ValueError("Not enough valid samples for spectral analysis")
    
            freqs, p_xx = welch_(x, fs=s_f, nperseg=Welch_samples_per_segment)
            densities.update({_dir[0]: p_xx})
    
        if self.config["display_results"] and not silent:
            plot_periodogram(freqs, densities)
    
        return {"frequencies": freqs, "spectral_densities": densities}

 
 

class CrossFrequencyAnalysis:
    def __init__(self, input, **kwargs):
        """


        Parameters
        ----------
        input : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
         
        verbose = kwargs.get("verbose", True)
        if verbose:
            print("Processing Cross Frequency Analysis...")
        
        self.s_f = kwargs.get("sampling_frequency", None)
        assert self.s_f is not None, "Sampling frequency must be specified"

        #self.type_ = kwargs.get("csd_data_type", "velocity")

        self.fa_1 = FrequencyAnalysis(input[0], **kwargs)
        self.fa_2 = FrequencyAnalysis(input[1], **kwargs)

        # Configs are the same
        self.config = self.fa_1.config 

        if verbose:
            print("...Cross Frequency Analysis done")



    def verbose(self, add_=None):
        if self.config["verbose"]:
            print("\n --- Config used: ---\n")

            for it in self.config.keys():
                print(
                    "# {it}:{esp}{val}".format(
                        it=it, esp=" " * (50 - len(it)), val=self.config[it]
                    )
                )

            if add_ is not None:
                for it in add_.keys():
                    print(
                        "# {it}:{esp}{val}".format(
                            it=it, esp=" " * (50 - len(it)), val=add_[it]
                        )
                    )
            print("\n")

    def cross_spectral_density(self, cross_data_type='velocity'):
        """


        Parameters
        ----------
        type_ : TYPE
            DESCRIPTION.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        s_f = self.s_f

        if cross_data_type == "velocity":
            print(self.fa_1.config)
            sp_1 = process_speed_components(self.fa_1.data_set, self.fa_1.config)[
                0:2, :
            ]
            sp_2 = process_speed_components(self.fa_2.data_set, self.fa_2.config)[
                0:2, :
            ]

            data_1 = dict({"x_array": sp_1[0], "y_array": sp_1[1]})
            data_2 = dict({"x_array": sp_2[0], "y_array": sp_2[1]})
        
        elif cross_data_type == "position":
            pass
        
        else:
            raise ValueError('cross_data_type must be "position" or "velocity"')
            
        c_densities = dict()
        
        status1 = self.fa_1.data_set.get("status", None)
        status2 = self.fa_2.data_set.get("status", None)
        
        for _dir in ["x_array", "y_array"]:
            if cross_data_type == "velocity":
                x_1 = data_1[_dir]
                x_2 = data_2[_dir]

            elif cross_data_type == "position":
                x_1 = self.fa_1.data_set[_dir]
                x_2 = self.fa_2.data_set[_dir]
            
            x_1 = self.fa_1._clean_signal(x_1, status=status1, interpolate=True)
            x_2 = self.fa_2._clean_signal(x_2, status=status2, interpolate=True)
            
            if x_1 is None or x_2 is None:
                raise ValueError("Not enough valid samples for cross spectral analysis")

            freqs, p_xy = csd_(
                x_1, x_2, fs=s_f, window="hann", nperseg=max(len(x_1), len(x_2))
            )

            c_densities.update({_dir[0]: p_xy})

        if self.config["display_results"]:
            plot_periodogram(freqs, c_densities)

        results = dict({"frequencies": freqs, "cross_spectral_densities": c_densities})

        return results


    def welch_cross_spectral_density(self, cross_data_type='velocity', samples_per_segment=256, silent=False):
        """


        Parameters
        ----------
        type_ : TYPE
            DESCRIPTION.
        samples_per_segment : TYPE
            DESCRIPTION.
        silent : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        
        s_f = self.s_f

        if cross_data_type == "velocity":
            sp_1 = process_speed_components(self.fa_1.data_set, self.fa_1.config)[
                0:2, :
            ]
            sp_2 = process_speed_components(self.fa_2.data_set, self.fa_2.config)[
                0:2, :
            ]

            data_1 = dict({"x_array": sp_1[0], "y_array": sp_1[1]})
            data_2 = dict({"x_array": sp_2[0], "y_array": sp_2[1]})

        elif cross_data_type == "position":
            pass
        
        else:
            raise ValueError('cross_data_type must be "position" or "velocity"')
            
        c_densities = dict()

        status1 = self.fa_1.data_set.get("status", None)
        status2 = self.fa_2.data_set.get("status", None)
        
        for _dir in ["x_array", "y_array"]:
            if cross_data_type == "velocity":
                x_1 = data_1[_dir]
                x_2 = data_2[_dir]

            elif cross_data_type == "position":
                x_1 = self.fa_1.data_set[_dir]
                x_2 = self.fa_2.data_set[_dir]

            x_1 = self.fa_1._clean_signal(x_1, status=status1, interpolate=True)
            x_2 = self.fa_2._clean_signal(x_2, status=status2, interpolate=True)
            
            if x_1 is None or x_2 is None:
                raise ValueError("Not enough valid samples for cross spectral analysis")

            freqs, p_xy = csd_(x_1, x_2, fs=s_f, nperseg=samples_per_segment)
            c_densities.update({_dir[0]: p_xy})

        if self.config["display_results"]:
            if not silent:
                plot_periodogram(freqs, c_densities)

        results = dict({"frequencies": freqs, "cross_spectral_densities": c_densities})

        return results


    def signal_coherency(self, cross_data_type='velocity', samples_per_segment=256):
        """
        Returns
        -------
        results : dict
            Dictionary containing frequencies and signal coherencies.
        """
        pxy_s = self.welch_cross_spectral_density(
            cross_data_type, samples_per_segment, silent=True
        )
    
        pxx_s = self.fa_1.welch_periodogram(cross_data_type, samples_per_segment, silent=True)
        pyy_s = self.fa_2.welch_periodogram(cross_data_type, samples_per_segment, silent=True)
    
        freqs = pxy_s["frequencies"]
    
        coherencies = dict()
    
        for _dir in ["x", "y"]:
            p_xy = pxy_s["cross_spectral_densities"][_dir]
            p_xx = pxx_s["spectral_densities"][_dir]
            p_yy = pyy_s["spectral_densities"][_dir]
    
            den = p_xx * p_yy
            c_xy = (np.abs(p_xy) ** 2) / (den + 1e-30)
            coherencies.update({_dir: c_xy})
    
        if self.config["display_results"]:
            plot_periodogram(freqs, coherencies, cross=True)
    
        results = dict({"frequencies": freqs, "signal_coherencies": coherencies})
    
        return results


def periodogram(input, **kwargs):
    type_ = kwargs.get("periodogram_data_type", "velocity")

    if isinstance(input, FrequencyAnalysis):
        results = input.periodogram(type_)
        input.verbose(dict({"periodogram_data_type": type_}))

    else:
        frequency_analysis = FrequencyAnalysis(input, **kwargs)
        results = frequency_analysis.periodogram(type_)
        frequency_analysis.verbose(dict({"periodogram_data_type": type_}))

    return results


def welch_periodogram(input, **kwargs):
    type_ = kwargs.get("periodogram_data_type", "velocity")
    samples_per_segment = kwargs.get("Welch_samples_per_segment", 256)

    if isinstance(input, FrequencyAnalysis):
        results = input.welch_periodogram(type_, samples_per_segment)
        input.verbose(
            dict(
                {
                    "periodogram_data_type": type_,
                    "Welch_samples_per_segment": samples_per_segment,
                }
            )
        )

    else:
        frequency_analysis = FrequencyAnalysis(input, **kwargs)
        results = frequency_analysis.welch_periodogram(type_, samples_per_segment)
        frequency_analysis.verbose(
            dict(
                {
                    "periodogram_data_type": type_,
                    "Welch_samples_per_segment": samples_per_segment,
                }
            )
        )

    return results


def cross_spectral_density(input, **kwargs):
    cross_data_type = kwargs.get("cross_data_type", "velocity")

    if isinstance(input, CrossFrequencyAnalysis):
        results = input.cross_spectral_density(cross_data_type)
        input.verbose(dict({"cross_data_type": cross_data_type}))

    else:
        assert (
            len(input) == 2 and type(input) == list
        ), "Input must be a CrossFrequencyAnalysis instance or list of two csv"
        assert (
            type(input[0]) == str and type(input[1]) == str
        ), "Input must be a CrossFrequencyAnalysis instance or list of two csv"

        cross_analysis = CrossFrequencyAnalysis([input[0], input[1]], **kwargs)
        results = cross_analysis.cross_spectral_density(cross_data_type)
        cross_analysis.verbose(dict({"cross_data_type": cross_data_type}))

    return results


def welch_cross_spectral_density(input, **kwargs):
    cross_data_type = kwargs.get("cross_data_type", "velocity")
    samples_per_segment = kwargs.get("Welch_samples_per_segment", 256)

    if isinstance(input, CrossFrequencyAnalysis):
        results = input.welch_cross_spectral_density(cross_data_type, samples_per_segment)
        input.verbose(
            dict(
                {
                    "cross_data_type": cross_data_type,
                    "Welch_samples_per_segment": samples_per_segment,
                }
            )
        )

    else:
        assert (
            len(input) == 2 and type(input) == list
        ), "Input must be a CrossFrequencyAnalysis instance or list of two csv"
        assert (
            type(input[0]) == str and type(input[1]) == str
        ), "Input must be a CrossFrequencyAnalysis instance or list of two csv"

        cross_analysis = CrossFrequencyAnalysis([input[0], input[1]], **kwargs)
        results = cross_analysis.welch_cross_spectral_density(
            cross_data_type, samples_per_segment
        )
        cross_analysis.verbose(
            dict(
                {
                    "cross_data_type": cross_data_type,
                    "Welch_samples_per_segment": samples_per_segment,
                }
            )
        )

    return results


def signal_coherency(input, **kwargs):
    cross_data_type = kwargs.get("cross_data_type", "velocity")
    samples_per_segment = kwargs.get("Welch_samples_per_segment", 256)

    if isinstance(input, CrossFrequencyAnalysis):
        results = input.signal_coherency(cross_data_type, samples_per_segment)
        input.verbose(
            dict(
                {
                    "cross_data_type": cross_data_type,
                    "Welch_samples_per_segment": samples_per_segment,
                }
            )
        )
    else:
        assert (
            len(input) == 2 and type(input) == list
        ), "Input must be a CrossFrequencyAnalysis instance or list of two csv"
        assert (
            type(input[0]) == str and type(input[1]) == str
        ), "Input must be a CrossFrequencyAnalysis instance or list of two csv"

        cross_analysis = CrossFrequencyAnalysis([input[0], input[1]], **kwargs)
        results = cross_analysis.signal_coherency(cross_data_type, samples_per_segment)
        cross_analysis.verbose(
            dict(
                {
                    "cross_data_type": cross_data_type,
                    "Welch_samples_per_segment": samples_per_segment,
                }
            )
        )

    return results
