# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Miscellaneous utility classes and functions."""

import ctypes
import fnmatch
import importlib
import inspect
import numpy as np
import os
import shutil
import sys
import types
import io
import pickle
import re
import requests
import html
import hashlib
import glob
import tempfile
import urllib
import urllib.request
import uuid

from distutils.util import strtobool
from typing import Any, List, Tuple, Union
import os 
import click
import re
import json

# Util classes
# ------------------------------------------------------------------------------------------

import pandas as pd
import nltk 
def get_father_dir(splited_dir):
    from pathlib import Path
    path = Path(splited_dir)
    whole_dataset_dir=path.parent.absolute()
    return whole_dataset_dir

def setup_training_loop_kwargs(config_kwargs):
    args=EasyDict()
    for key,value in config_kwargs.items():
        setattr(args,key,value)
 
    run_desc=""
    
    return run_desc,args

def setup(config_kwargs,has_log=True):
     
 
    # Setup training options.
    run_desc, args = setup_training_loop_kwargs(config_kwargs)

    outdir=args.outdir
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    args.cur_run_id=cur_run_id
    assert not os.path.exists(args.run_dir)
    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    
    # Print options.
    print()
    print(f'Training options:  ')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    if has_log:
        # Launch processes.
        print('Launching processes...')
        logger= Logger(file_mode='a', should_flush=False)  
        logger.write(text=json.dumps(args, indent=2))
    else:
        logger=None
    return args,logger
 
import pandas as pd  

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
   

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        if len(text) == 0: 
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None
 