"""
GPUtil - GPU utilization

A Python module for programmically getting the GPU utilization from NVIDA
GPUs using nvidia-smi

Original Author: Anders Krogh Mortensen (anderskm)
Date:   16 January 2017
Web:    https://github.com/anderskm/gputil

LICENSE

MIT License

Copyright (c) 2017 anderskm

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from subprocess import Popen, PIPE
from distutils import spawn

import platform
import os

def safe_float_cast(str_number: str) -> float:
    try:
        number = float(str_number)
    except ValueError:
        number = float('nan')
    return number


class GPU:
    def __init__(self, ID: int) -> None:
        self.id = ID
        self.uuid = None
        self.load = 0.
        self.mem_total = 0.
        self.mem_used = 0.
        self.mem_util = 0.
        self.memoryFree = 0.
        self.driver = None
        self.name = None
        self.serial = None
        self.display_mode = None
        self.display_active = None
        self.temperature = 0.
        ret, ids = self.update()
        if not ret:
            print(f"RMSGD: GPU ID {ID} was not found")
            print(f"RMSGD: Valid GPU IDs are {ids}")
            raise ValueError

    def update(self) -> None:
        if platform.system() == "Windows":
            # If the platform is Windows and nvidia-smi
            # could not be found from the environment path,
            # try to find it from system drive with default installation path
            nvidia_smi = spawn.find_executable('nvidia-smi')
            if nvidia_smi is None:
                nvidia_smi = f"{os.environ['systemdrive']}\\Program Files\\NVIDIA " +\
                    "Corporation\\NVSMI\\nvidia-smi.exe"
        else:
            nvidia_smi = "nvidia-smi"

        # Get ID, processing and memory utilization for all GPUs
        try:
            p = Popen([nvidia_smi, "--query-gpu=index,uuid,utilization.gpu," +
                       "memory.total,memory.used,memory.free,driver_version," +
                       "name,gpu_serial,display_active,display_mode," +
                       "temperature.gpu",
                       "--format=csv,noheader,nounits"], stdout=PIPE)
            stdout, stderror = p.communicate()
        except Exception:
            return []
        output = stdout.decode('UTF-8')
        lines = output.split(os.linesep)
        num_devices = len(lines)-1
        ids = list()
        for g in range(num_devices):
            line = lines[g]
            vals = line.split(', ')
            for i in range(12):
                if (i == 0):
                    device_id = int(vals[i])
                    ids.append(device_id)
                elif (i == 1):
                    uuid = vals[i]
                elif (i == 2):
                    gpu_util = safe_float_cast(vals[i])/100
                elif (i == 3):
                    mem_total = safe_float_cast(vals[i])
                elif (i == 4):
                    mem_used = safe_float_cast(vals[i])
                elif (i == 5):
                    mem_free = safe_float_cast(vals[i])
                elif (i == 6):
                    driver = vals[i]
                elif (i == 7):
                    gpu_name = vals[i]
                elif (i == 8):
                    serial = vals[i]
                elif (i == 9):
                    display_active = vals[i]
                elif (i == 10):
                    display_mode = vals[i]
                elif (i == 11):
                    temp_gpu = safe_float_cast(vals[i])
            if device_id == self.id:
                self.uuid = uuid
                self.load = gpu_util
                self.mem_util = float(mem_used)/float(mem_total)
                self.mem_total = mem_total
                self.mem_used = mem_used
                self.mem_free = mem_free
                self.driver = driver
                self.name = gpu_name
                self.serial = serial
                self.display_mode = display_mode
                self.display_active = display_active
                self.temperature = temp_gpu
                return (True, None)
        return (False, ids)
