#!/usr/bin/python3
"""
Copyright 2018 Marco Lattuada

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile

import provider

class Provider(provider.Provider):
    """
    Class to manage the execution of experiments on the host

    Attributes
    ----------
    _abs_root: str
        The absolute path containing the root of the library"

    _list_file_name: str
        The file containing the list of experiments to be executed

    Methods
    -------
    run_experiment()
        Run the experiments

    copy_list_to_target()
        Copy the file with the list of experiments to be executed
    """

    _abs_root = None

    _list_file_name = None

    def copy_list_to_target(self, list_file_name):
        """
        Copy the list_file_name in a temporary location

        Parameters
        ----------
        list_file_name: str
            The name of the file to be copied
        """
        shutil.copyfile(list_file_name, os.path.join(tempfile.gettempdir(), os.path.basename(list_file_name)))
        self._list_file_name = list_file_name

    def run_experiment(self):
        """
        Run the experiments
        """
        extra_options = ""
        if self._args.profile:
            extra_options = extra_options + " --profile " + self._args.profile

        if self._args.mail:
            extra_options = extra_options + " --mail "+ self._args.mail

        if self._args.debug:
            extra_options = extra_options + " --debug"

        utility = __import__("utility")
        root_project = utility.get_project_root()

        command = "screen -d -m " + os.path.join(root_project, "vm_scripts", "launch_local_experiment.py") + " -a " + self._args.application + " --parameters-list /tmp/" + os.path.basename(self._list_file_name) + extra_options + " --repetitions " + str(self._args.repetitions) + " --output " + self._args.output
        logging.info("command is %s", command)

        cmd = subprocess.Popen(command, shell=True, executable='/bin/bash')
        retcode = cmd.wait()
        if retcode == 0:
            logging.info("launched local experiment")
        else:
            logging.error("Error in launching local experiment")
            sys.exit(1)

def parse_args(parser):
    """
    Add to the command line parser the options related to the host (none so far)
    """
    return
