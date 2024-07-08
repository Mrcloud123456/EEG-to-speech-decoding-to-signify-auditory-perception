# -*- coding: utf-8 -*-

import numpy as np
from tools import generate_cfs, generate_bands, make_fir_filter

def spyral(input, fs, electrodes, n_carriers, spread, **kwargs):
    """Spyral: vocoder that utilizes multiple sinusoidal carriers to simulate current spread

    Parameters
    ----------
    input : array
        The input signal
    fs : scalar
        The sampling frequency
    electrodes : scalar or array
        If scalar, represents the number of electrodes linearly distributed on an ERB scale between analysis_lo and analysis_hi.
        If array, each element represents the corresponding best frequency of each electrode.
    n_carriers : scalar
        Number of tone carriers
    spread : scalar
        Current spread in dB/octave (negative!!)
    **kwargs : keyword arguments
        analysis_lo : scalar
            Lower bound of analysis filters in Hz [default = 120]
        analysis_hi : scalar
            Upper bound of analysis filters in Hz [default = 8658]
        analysis_cutoffs : array
            Array of cutoff frequencies to use. analysis_hi and lo are ignored. Must be one more than the number of electrodes.
        carrier_lo : scalar
            Lower bound of carriers in Hz [default = 20]
        carrier_hi : scalar
            Higher bound of carriers in Hz [default = 20,000]
        filt_env : scalar
            Envelope filter cutoff in Hz [default = 50]
        in_phase : bool
            If True, carriers are in phase [default = False]

    Returns
    -------
    out : array
        Vocoded input

    Example
    -------
    >>> out = spyral(signal, 44100, 20, 80, -8)
    """
    
    analysis_lo = kwargs.get('analysis_lo', 120)  # Lower bound of analysis filters, default = 120 Hz
    analysis_hi = kwargs.get('analysis_hi', 8658)  # Upper bound of analysis filters, default = 8658 Hz
    analysis_cutoffs = kwargs.get('analysis_cutoffs', None)  # User-specified cutoff frequencies for analysis bands
    carrier_lo = kwargs.get('carrier_lo', 20)  # Lower bound of carriers, default = 20 Hz
    carrier_hi = kwargs.get('carrier_hi', 20000)  # Higher bound of carriers, default = 20,000 Hz
    filt_env = kwargs.get('filt_env', 50)  # Envelope filter cutoff, default = 50 Hz
    in_phase = kwargs.get('in_phase', False)  # Flag for carriers in phase, default = False

    fs = np.float32(fs)  # Convert sampling frequency to float32

    rms_in = np.sqrt(np.mean(np.power(input, 2)))  # Root mean square of the input signal
    lp_filter = make_fir_filter(0, filt_env, fs)  # Generate a low-pass filter with cutoff at filt_env Hz

    if np.isscalar(electrodes):
        cfs = np.array(generate_cfs(analysis_lo, analysis_hi, electrodes))  # Generate center frequencies for electrodes
    else:
        cfs = np.array(electrodes)  # Use provided center frequencies for electrodes

    carrier_fs = generate_cfs(carrier_lo, carrier_hi, n_carriers)  # Generate frequencies for tone carriers
    t = np.arange(0, len(input) / fs, 1 / fs)  # Time vector based on input length and sampling frequency
    t_carrier = np.zeros((n_carriers, len(input)))  # Matrix to hold carrier signals

    if analysis_cutoffs is None:
        ip_bands = analysis_cutoffs  # Use user-specified cutoffs
    else:
        ip_bands = np.array(generate_bands(analysis_lo, analysis_hi, cfs.size))  # Generate analysis band limits

    ip_bank = np.zeros((cfs.size, 512))  # Matrix to hold analysis filters
    envelope = np.zeros((cfs.size, len(input)))  # Matrix to hold envelopes extracted per electrode
    mixed_envelope = np.zeros((n_carriers, len(input)))  # Matrix to hold mixed envelopes for modulating carriers

    # Envelope extraction
    for j in range(cfs.size):
        ip_bank[j, :] = make_fir_filter(ip_bands[j, 0], ip_bands[j, 1], fs)  # Generate analysis filterbank
        speechband = np.convolve(input, ip_bank[j, :], mode='same')  # Convolve input with analysis filter
        envelope[j, :] = np.convolve(np.maximum(speechband, 0), lp_filter, mode='same')  # Apply low-pass filter to envelope

    # Calculate weights for power envelopes
    for i in range(n_carriers):
        for j in range(cfs.size):
            mixed_envelope[i, :] += 10. ** (spread / 10. * np.abs(np.log2(cfs[j] / carrier_fs[i]))) * envelope[j, :] ** 2.

    # Take square root to convert back to amplitudes
    mixed_envelope = np.sqrt(mixed_envelope)
    out = np.zeros(len(input))

    # Generate carriers, modulate them with mixed envelopes
    for i in range(n_carriers):
        if in_phase:
            t_carrier[i, :] = np.sin(2 * np.pi * (carrier_fs[i] * t + np.random.rand()))  # Generate carriers in phase
        else:
            t_carrier[i, :] = np.sin((2. * np.pi * carrier_fs[i] * t) + np.random.rand())  # Generate carriers with random phase

        out += mixed_envelope[i, :] * t_carrier[i, :]  # Modulate carriers with mixed envelopes

    # Normalize the output signal based on input RMS
    return out * (np.sqrt(np.mean(np.square(input))) / np.sqrt(np.mean(np.square(out))))
