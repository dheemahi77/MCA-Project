import numpy as np
from PyEMD import EMD


class EMDProcessor:
    """
    Class for processing time series data using Empirical Mode Decomposition (EMD)
    """

    def __init__(self):
        """Initialize the EMD processor"""
        self.emd = EMD()
        self.imfs = None
        self.residual = None

    def decompose(self, data):
        """
        Decompose a time series into Intrinsic Mode Functions (IMFs)

        Parameters:
        -----------
        data : numpy.ndarray
            Time series data

        Returns:
        --------
        imfs : numpy.ndarray
            IMF components
        residual : numpy.ndarray
            Residual component (trend)
        """
        # Decompose the signal
        self.imfs = self.emd.emd(data)

        # Extract residual (last IMF is considered the residual/trend)
        if len(self.imfs) > 0:
            self.residual = self.imfs[-1]
            self.imfs = self.imfs[:-1]
        else:
            self.residual = np.zeros_like(data)

        return self.imfs, self.residual

    def combine_imfs(self, imf_indices):
        """
        Combine selected IMFs

        Parameters:
        -----------
        imf_indices : list
            List of IMF indices to combine

        Returns:
        --------
        numpy.ndarray
            Combined IMF components
        """
        if self.imfs is None:
            raise ValueError("Must run decompose() first")

        return np.sum(self.imfs[imf_indices], axis=0)

    def get_high_frequency(self, threshold=3):
        """
        Get high-frequency components (first 'threshold' IMFs)

        Parameters:
        -----------
        threshold : int
            Number of IMFs to treat as high-frequency (default: 3)

        Returns:
        --------
        numpy.ndarray
            Summed high-frequency IMF components
        """
        if self.imfs is None:
            raise ValueError("Must run decompose() first")

        if len(self.imfs) <= threshold:
            return self.combine_imfs(list(range(len(self.imfs))))
        else:
            return self.combine_imfs(list(range(threshold)))

    def get_low_frequency(self, threshold=3):
        """
        Get low-frequency components (IMFs after 'threshold')

        Parameters:
        -----------
        threshold : int
            Number of IMFs to skip from high-frequency (default: 3)

        Returns:
        --------
        numpy.ndarray
            Summed low-frequency IMF components
        """
        if self.imfs is None:
            raise ValueError("Must run decompose() first")

        if len(self.imfs) <= threshold:
            return np.zeros_like(self.residual)
        else:
            return self.combine_imfs(list(range(threshold, len(self.imfs))))

    def adaptive_split(self, data=None, threshold=3):
        """
        Adaptively split a decomposed signal into high-frequency,
        low-frequency, and trend components.

        Parameters:
        -----------
        data : numpy.ndarray, optional
            Time series data. If provided, decompose() is called automatically.
            If None, uses IMFs from a previously called decompose().
        threshold : int
            Number of leading IMFs treated as high-frequency (default: 3)

        Returns:
        --------
        high_freq : numpy.ndarray
            High-frequency components (first `threshold` IMFs summed)
        low_freq : numpy.ndarray
            Low-frequency components (remaining IMFs summed)
        trend : numpy.ndarray
            Residual / trend component (last IMF from raw decomposition)

        Example:
        --------
        # Option 1: pass data directly
        high_freq, low_freq, trend = emd_processor.adaptive_split(data=price_array)

        # Option 2: decompose first, then split
        emd_processor.decompose(price_array)
        high_freq, low_freq, trend = emd_processor.adaptive_split()
        """
        if data is not None:
            self.decompose(data)

        if self.imfs is None:
            raise ValueError(
                "No IMFs found. Either pass data to adaptive_split() "
                "or call decompose() before adaptive_split()."
            )

        high_freq = self.get_high_frequency(threshold)
        low_freq  = self.get_low_frequency(threshold)
        trend     = self.residual

        return high_freq, low_freq, trend

    def get_imf_count(self):
        """
        Return the number of IMFs (excluding the residual/trend)

        Returns:
        --------
        int
            Number of IMFs, or 0 if decompose() has not been called yet.
        """
        if self.imfs is None:
            return 0
        return len(self.imfs)

    def reconstruct(self):
        """
        Reconstruct the original signal from all IMFs and the residual.

        Returns:
        --------
        numpy.ndarray
            Reconstructed time series
        """
        if self.imfs is None or self.residual is None:
            raise ValueError("Must run decompose() first")

        return np.sum(self.imfs, axis=0) + self.residual