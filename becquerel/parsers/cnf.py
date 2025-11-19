"""Read in a Canberra CNF file.

The CNF data format is not readily available to build a custom parser so this
code is based on the CNF parsing code in the open source xylib project:
    https://github.com/wojdyr/xylib
"""

import datetime
import struct
import warnings
from pathlib import Path

import numpy as np

from ..core import calibration
from .parsers import BecquerelParserError

def _parse_vms_time(data_bytes):
    """Convert 64-bit VMS time (bytes) to datetime."""
    if len(data_bytes) < 8:
        return None
    val = int.from_bytes(data_bytes, byteorder='little')
    
    # Check for Empty (0) or Deleted (All Fs)
    if val == 0 or val == 0xFFFFFFFFFFFFFFFF:
        return None

    # VMS Time: 100ns ticks since Nov 17, 1858
    # Unix Offset: 3506716800 seconds
    try:
        t = (val / 10000000.0) - 3506716800
        if t < 0:
            return None 
        return datetime.datetime.utcfromtimestamp(t)
    except (ValueError, OverflowError):
        return None

def _parse_vms_duration(data_bytes):
    """Convert 64-bit VMS duration (ticks) to seconds."""
    if len(data_bytes) < 8:
        return 0.0
    val = int.from_bytes(data_bytes, byteorder='little')
    if val == 0:
        return 0.0
    
    # Bitwise inversion (One's Complement) on 64-bit unsigned
    val_inv = val ^ 0xFFFFFFFFFFFFFFFF
    return val_inv * 1.0e-7

def _from_pdp11(data, index):
    """Convert 32-bit PDP-11 float to double."""
    if index + 4 > len(data):
        return 0.0
    
    if (data[index + 1] & 0x80) == 0:
        sign = 1
    else:
        sign = -1

    exb = ((data[index + 1] & 0x7F) << 1) + ((data[index] & 0x80) >> 7)
    if exb == 0:
        return np.nan if sign == -1 else 0.0

    h = (
        data[index + 2] / 16777216.0
        + data[index + 3] / 65536.0
        + (128 + (data[index] & 0x7F)) / 256.0
    )
    return sign * h * pow(2.0, exb - 128.0)

def read(filename, verbose=False, cal_kwargs=None):
    """Parse the CNF file and return a dictionary of data.

    Parameters
    ----------
    filename : str | pathlib.Path
        The filename of the CNF file to read.
    verbose : bool (optional)
        Whether to print out debugging information. By default False.
    cal_kwargs : dict or None (optional)
        Kwargs to override the Calibration parameters read from file.

    Returns
    -------
    data : dict
        Dictionary of data that can be used to instantiate a Spectrum.
    cal : Calibration
        Energy calibration stored in the file.
    """
    filename = Path(filename)
    if verbose:
        print(f"Reading CNF file {filename}")

    with filename.open("rb") as f:
        file_bytes = f.read()
    
    file_len = len(file_bytes)

    # --- STEP 1: GLOBAL SIGNATURE SCAN ---
    # We scan the ENTIRE file for section headers.
    # Signature: [Type Byte] [20] [01] [00]
    # This bypasses potentially corrupted or incomplete header tables in new files.
    headers = []
    offset = 0
    while offset < file_len:
        # Find magic bytes \x20\x01
        loc = file_bytes.find(b'\x20\x01', offset)
        if loc == -1:
            break
        
        h_start = loc - 1
        if h_start < 0 or h_start + 14 > file_len: 
            offset = loc + 2
            continue

        d_off = int.from_bytes(file_bytes[h_start+10:h_start+14], 'little')
        block_type = file_bytes[h_start]
        
        if 0 < d_off < file_len:
            headers.append({'type': block_type, 'd_off': d_off})
        
        offset = loc + 2

    if not headers:
        raise BecquerelParserError("No valid CNF headers found.")

    # --- STEP 2: SELECT ACTIVE ACQUISITION BLOCK ---
    # Filter for Type 0 (Acq) blocks.
    # Logic: Iterate through all candidates. The "Active" block is the one
    # that contains valid Live Time (> 0). Genie 4 files often have empty
    # templates at the start which must be ignored.
    best_acq = None
    acq_candidates = [h for h in headers if h['type'] == 0]
    
    for h in acq_candidates:
        off = h['d_off']
        if off + 40 > file_len:
            continue
        
        ptr_date = int.from_bytes(file_bytes[off+36:off+38], 'little')
        if ptr_date > 0:
            dl = off + 48 + ptr_date + 1
            if dl + 24 <= file_len:
                lt = _parse_vms_duration(file_bytes[dl+16 : dl+24])
                if lt > 0.0:
                    best_acq = h
                    # Keep searching to find the *last* valid one (appended data)
    
    # Fallback: If no block has valid time (e.g. derived/geometry files), 
    # use the last available block.
    if not best_acq:
        if acq_candidates:
            best_acq = acq_candidates[-1]
            if verbose:
                print("Warning: No Acquisition block with valid time found. Using last candidate.")
        else:
            raise BecquerelParserError("No Acquisition Block found.")

    acq_start = best_acq['d_off']
    if verbose:
        print(f"Selected Acquisition Block at offset {acq_start}")

    # --- STEP 3: EXTRACT METADATA ---
    ptr_cal = int.from_bytes(file_bytes[acq_start+34:acq_start+36], 'little')
    ptr_date = int.from_bytes(file_bytes[acq_start+36:acq_start+38], 'little')
    
    data = {
        'start_time': None,
        'realtime': 1.0,
        'livetime': 1.0,
        'sample_name': '',
        'sample_description': ''
    }

    if ptr_date > 0:
        dl = acq_start + 48 + ptr_date + 1
        dt = _parse_vms_time(file_bytes[dl : dl+8])
        if dt:
            data['start_time'] = dt
        else:
            # Default for empty files
            data['start_time'] = datetime.datetime(1970, 1, 1)
        
        rt = _parse_vms_duration(file_bytes[dl+8 : dl+16])
        lt = _parse_vms_duration(file_bytes[dl+16 : dl+24])
        
        if rt > 0: data['realtime'] = rt
        if lt > 0: data['livetime'] = lt

    # Sample Info (Type 1)
    # Find the Type 1 block closest to the selected Acquisition block.
    # Usually, the last Sample block corresponds to the last Acquisition block.
    sam_candidates = [h for h in headers if h['type'] == 1]
    if sam_candidates:
        s_off = sam_candidates[-1]['d_off']
        if s_off + 128 <= file_len:
            # Name is at +48 (32 bytes), Description at +80 (48 bytes)
            data['sample_name'] = file_bytes[s_off+48:s_off+80].decode('latin-1', 'replace').strip('\x00')
            data['sample_description'] = file_bytes[s_off+80:s_off+128].decode('latin-1', 'replace').strip('\x00')

    # Fill required fields
    data['sample_id'] = ""
    data['sample_type'] = ""
    data['sample_unit'] = ""
    data['user_name'] = ""

    # --- STEP 4: READ SPECTRUM ---
    # 1. Get Channel Count from Header (Offset 186)
    count_offset = acq_start + 186
    num_channels = 0
    if count_offset + 2 <= file_len:
        chunks = int.from_bytes(file_bytes[count_offset:count_offset+2], 'little')
        num_channels = chunks * 256
    
    if num_channels == 0:
        if verbose:
            print("Warning: Header channel count is 0. Defaulting to 8192.")
        num_channels = 8192

    # 2. Find Spectrum Block (Type 5)
    spec_candidates = [h for h in headers if h['type'] == 5]
    if not spec_candidates:
        raise BecquerelParserError("No Spectrum Block found.")
    
    # Use the last one (matches Active Data logic)
    best_spec = spec_candidates[-1]
    spec_start = best_spec['d_off']
    
    data_start = spec_start + 512  # Skip 512 byte header
    data_len = num_channels * 4
    
    # Read Data
    if data_start + data_len > file_len:
        if verbose:
            print("Warning: File truncated, padding spectrum with zeros.")
        raw_counts = file_bytes[data_start : file_len]
        raw_counts += b'\x00' * (data_len - len(raw_counts))
    else:
        raw_counts = file_bytes[data_start : data_start + data_len]
        
    counts = np.frombuffer(raw_counts, dtype='<u4').astype(float)

    # Legacy Artifact Cleanup
    # If counts[0] or counts[1] match the time integers, zero them out.
    if len(counts) > 1:
        c0, c1 = int(counts[0]), int(counts[1])
        rt_int, lt_int = int(data['realtime']), int(data['livetime'])
        
        if c0 > 0 and (abs(c0 - rt_int) <= 1 or abs(c0 - lt_int) <= 1):
            counts[0] = 0.0
        if c1 > 0 and (abs(c1 - rt_int) <= 1 or abs(c1 - lt_int) <= 1):
            counts[1] = 0.0

    data['counts'] = counts

    # --- STEP 5: CALIBRATION ---
    offset_cal = acq_start + 48 + 32 + ptr_cal
    cal_coeff = None
    
    if ptr_cal > 0 and offset_cal < file_len:
        base_idx = offset_cal + 36
        if base_idx + 12 <= file_len:
            c = []
            for i in range(3):
                c.append(_from_pdp11(file_bytes, base_idx + (4 * i)))
            
            # If linear term is not zero, it's a valid calibration
            if c[1] != 0.0:
                cal_coeff = c + [0.0] # Add cubic term for compatibility

    if cal_coeff is None:
        cal_coeff = [0.0, 1.0, 0.0, 0.0]

    # Clean strings
    for k, v in data.items():
        if isinstance(v, str):
            data[k] = v.strip()

    # Create calibration object
    if cal_kwargs is None:
        cal_kwargs = {}
    cal = calibration.Calibration.from_polynomial(cal_coeff, **cal_kwargs)

    return data, cal