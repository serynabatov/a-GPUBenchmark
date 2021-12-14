import os
import subprocess
import logging

directory = '/home/nabatov/a-GPUBench/output/'
depth = 5
logging.basicConfig(level=logging.INFO)

for root, dirs, files in os.walk(directory):
    if root[len(directory):].count(os.sep) == depth:
        s_command = "python3.6 /home/nabatov/a-GPUBench/host_scripts/collect_data.py -a pytorch {}".format(root)
        logging.info(s_command)
        subprocess.call(s_command, shell=True, executable="/bin/sh")
        os.remove("/home/nabatov/a-GPUBench/pytorch.csv")
