#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read_eprimetxt.py
=================
Generic Python reader for E-Prime text files (*.txt exported from E-DataAid).

Mirrors the behaviour of the MATLAB ``read_eprimetxt.m`` by K. N'Diaye
(https://github.com/kndiaye/matlab).

E-Prime text files are UTF-16LE encoded.  The file is structured as:

    *** Header Start ***
    VersionPersist: 1
    LevelName: Session
    LevelName: Block
    LevelName: Trial
    ...
    *** Header End ***
        Level: 2
        *** LogFrame Start ***
        Procedure: SomeProc
        Variable1: value1
        Variable2: 42
        *** LogFrame End ***
        ...

Public API
----------
read_eprimetxt(filename)
    Returns (df, header):
        df     – pandas DataFrame, one row per LogFrame, all variables as
                 columns. Numeric-looking values are cast to float.
                 A '_level' column indicates the nesting level (1 = session,
                 2 = block, 3 = trial, etc.).
        header – dict of session-level fields from the *** Header *** section
                 (LevelName entries are collected into a list).

Convenience helpers
-------------------
get_header(filename)        -> dict        (header only, fast)
get_variable(df, varname)   -> pd.Series   (extract one column, NaN-safe)
list_variables(df)          -> list of str  (all column names, sorted)
to_numeric_matrix(df)       -> (np.ndarray, list)
                               numeric columns only, as a 2-D array + names
to_text_matrix(df)          -> (list-of-lists, list)
                               text columns only

Example
-------
    from read_eprimetxt import read_eprimetxt

    df, header = read_eprimetxt('MyExperiment_SubjectN.txt')

    # Show all variable names
    print(df.columns.tolist())

    # Get onset times for a specific procedure
    task_rows = df[df['Procedure'] == 'TaskProc']
    print(task_rows[['Stimulus.OnsetTime', 'Response.RT']])

Command-line usage
------------------
    python read_eprimetxt.py path/to/file.txt [varname1 varname2 ...]

    Without variable names   – prints all column names (like MATLAB nargout=1).
    With variable names      – prints those columns only.

Author : Arun Sasidharan  (adapted from MATLAB original by Karim N'Diaye)
Date   : 2026-03-28
"""

import codecs
import os
import sys
import glob
import re
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _open_eprime(filename):
    """
    Open an E-Prime text file and return its lines, handling UTF-16LE (with or
    without BOM), UTF-16, and plain UTF-8 variants.
    """
    # E-Prime typically writes UTF-16LE with a BOM (bytes FF FE at start).
    # The Python 'utf-16' codec auto-detects the BOM; 'utf-16le' does not
    # expect one. We try both, plus UTF-8 fallbacks.
    for enc in ('utf-16', 'utf-16le', 'utf-8-sig', 'utf-8'):
        try:
            with codecs.open(filename, 'r', encoding=enc) as fh:
                lines = fh.readlines()
            # Sanity check: E-Prime files always start with the header marker
            first_nonempty = next(
                (l.strip() for l in lines if l.strip()), '')
            if ('*** Header Start ***' in first_nonempty or
                    'VersionPersist' in first_nonempty):
                return lines
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise IOError(
        f"Cannot decode file (not a valid E-Prime text export?): {filename}")


def _parse_kv(line):
    """
    Split a stripped 'Key: Value' line into (key, value).
    Returns (None, None) for blank lines and '*** ... ***' markers.
    """
    line = line.strip()
    if not line or line.startswith('***'):
        return None, None
    if ': ' in line:
        k, _, v = line.partition(': ')
        return k.strip(), v.strip()
    # Edge case: key with empty value (e.g. 'SomeField: ')
    if line.endswith(':'):
        return line[:-1].strip(), ''
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Core parser
# ─────────────────────────────────────────────────────────────────────────────

def read_eprimetxt(filename):
    """
    Parse an E-Prime text file into a DataFrame of LogFrame records.

    Mirrors ``[hnames, A, B, tvals, rowlevel] = read_eprimetxt(filename)``
    in the MATLAB original, but returns a tidy pandas DataFrame instead of
    separate numeric/text matrices (use :func:`to_numeric_matrix` and
    :func:`to_text_matrix` if you need the MATLAB-style split).

    Parameters
    ----------
    filename : str
        Path to the E-Prime text file.

    Returns
    -------
    df : pd.DataFrame
        One row per ``*** LogFrame ***`` block.  All field names become
        column headers; values are numeric (float) when possible, otherwise
        string.  Special columns:

        ``_level``   – nesting level of the frame (int; 1=session, 2=block,
                       3=trial, …).  Corresponds to ``rowlevel`` in MATLAB.
        ``_frame_n`` – sequential frame index (0-based).

    header : dict
        Session-level fields from the ``*** Header Start/End ***`` section.
        ``header['LevelName']`` is a list of level name strings
        (Session, Block, Trial, …).
    """
    lines = _open_eprime(filename)

    # ── 1. Split into header section and log section ─────────────────────────
    header = {}
    log_start_idx = 0

    in_header = False
    for idx, line in enumerate(lines):
        s = line.strip()
        if '*** Header Start ***' in s:
            in_header = True
            continue
        if '*** Header End ***' in s:
            in_header = False
            log_start_idx = idx + 1
            break
        if in_header:
            k, v = _parse_kv(line)
            if k is not None:
                if k == 'LevelName':
                    header.setdefault('LevelName', []).append(v)
                else:
                    header[k] = v

    log_lines = lines[log_start_idx:]

    # ── 2. Parse LogFrame blocks ──────────────────────────────────────────────
    frames = []
    current_frame = None
    current_level = 1   # default nesting level

    for line in log_lines:
        s = line.strip()
        if not s:
            continue

        # Detect nesting level (appears between frames, e.g. "        Level: 3")
        m = re.match(r'^Level:\s*(\d+)$', s)
        if m:
            current_level = int(m.group(1))
            continue

        if '*** LogFrame Start ***' in s:
            current_frame = {'_level': current_level}
            continue

        if '*** LogFrame End ***' in s:
            if current_frame is not None:
                frames.append(current_frame)
            current_frame = None
            continue

        if current_frame is not None:
            k, v = _parse_kv(line)
            if k is not None:
                # If the same key appears more than once in a frame, keep last
                current_frame[k] = v

    # ── 3. Build DataFrame ────────────────────────────────────────────────────
    if not frames:
        return pd.DataFrame(), header

    df = pd.DataFrame(frames)
    df.insert(0, '_frame_n', range(len(df)))

    # ── 4. Numeric conversion ─────────────────────────────────────────────────
    # Try to convert each column to numeric; leave as string on failure.
    for col in df.columns:
        if col.startswith('_'):
            continue
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            # Only replace if *all* non-NaN values successfully converted
            # (avoids silently converting mixed columns to all-NaN)
            orig_nonempty = df[col].notna() & (df[col] != '')
            conv_nonempty = converted.notna()
            if (orig_nonempty & ~conv_nonempty).any():
                pass   # keep as string – column has real text values
            else:
                df[col] = converted
        except (TypeError, ValueError):
            pass

    return df, header


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience helpers  (mirror MATLAB usage patterns)
# ─────────────────────────────────────────────────────────────────────────────

def get_header(filename):
    """
    Return only the header dict without parsing all LogFrames.
    Useful for quickly reading subject metadata.
    """
    lines = _open_eprime(filename)
    header = {}
    in_header = False
    for line in lines:
        s = line.strip()
        if '*** Header Start ***' in s:
            in_header = True
            continue
        if '*** Header End ***' in s:
            break
        if in_header:
            k, v = _parse_kv(line)
            if k is not None:
                if k == 'LevelName':
                    header.setdefault('LevelName', []).append(v)
                else:
                    header[k] = v
    return header


def list_variables(df):
    """
    Return a sorted list of all variable (column) names, excluding internal
    columns (_frame_n, _level).
    Equivalent to MATLAB ``hnames`` with nargout == 1.
    """
    return sorted(c for c in df.columns if not c.startswith('_'))


def get_variable(df, varname):
    """
    Extract a single variable column as a Series.

    Parameters
    ----------
    df      : pd.DataFrame  (from read_eprimetxt)
    varname : str

    Returns
    -------
    pd.Series
    """
    if varname not in df.columns:
        raise KeyError(f"Variable '{varname}' not found. "
                       f"Available: {list_variables(df)}")
    return df[varname]


def to_numeric_matrix(df):
    """
    Return the numeric-only columns as a 2-D numpy array and their names.
    Equivalent to the ``A`` output of the MATLAB function.

    Returns
    -------
    A      : np.ndarray, shape (n_frames, n_numeric_cols)
    hnames : list of str
    """
    num_cols = [c for c in df.columns
                if not c.startswith('_') and
                pd.api.types.is_numeric_dtype(df[c])]
    A = df[num_cols].to_numpy(dtype=float)
    return A, num_cols


def to_text_matrix(df):
    """
    Return the text-only columns as a list-of-lists and their names.
    Equivalent to the ``B`` output of the MATLAB function.

    Returns
    -------
    B      : list of lists  [n_frames][n_text_cols]
    hnames : list of str
    """
    txt_cols = [c for c in df.columns
                if not c.startswith('_') and
                not pd.api.types.is_numeric_dtype(df[c])]
    B = df[txt_cols].fillna('').values.tolist()
    return B, txt_cols


def filter_by_level(df, level):
    """
    Return only rows at a specific nesting level.

    Parameters
    ----------
    df    : pd.DataFrame
    level : int  (1 = session, 2 = block, 3 = trial, …)
    """
    if '_level' not in df.columns:
        raise KeyError("DataFrame has no '_level' column.")
    return df[df['_level'] == level].copy()


# ─────────────────────────────────────────────────────────────────────────────
#  Command-line entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    filepath = sys.argv[1]
    varnames = sys.argv[2:] if len(sys.argv) > 2 else None

    print(f"Reading: {filepath}")
    df, header = read_eprimetxt(filepath)

    # Print header
    print(f"\n{'─'*60}")
    print("HEADER")
    print(f"{'─'*60}")
    for k, v in header.items():
        print(f"  {k}: {v}")

    if varnames is None:
        # No variables specified → list all headers (MATLAB nargout==1 mode)
        print(f"\n{'─'*60}")
        print(f"VARIABLES  ({len(list_variables(df))} total, {len(df)} frames)")
        print(f"{'─'*60}")
        for v in list_variables(df):
            dtype = 'numeric' if pd.api.types.is_numeric_dtype(df[v]) else 'text'
            print(f"  {v}  [{dtype}]")
    else:
        # Print requested variables
        print(f"\n{'─'*60}")
        pd.set_option('display.max_rows', 50)
        pd.set_option('display.float_format', '{:.3f}'.format)
        cols = [c for c in varnames if c in df.columns]
        missing = [c for c in varnames if c not in df.columns]
        if missing:
            print(f"WARNING: variables not found: {missing}")
        if cols:
            print(df[cols].to_string(index=False))
