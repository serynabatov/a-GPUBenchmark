"""
Copyright 2018-2019 Marco Lattuada

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
import subprocess
import sys

import provider

class Provider(provider.Provider):
    """
    Class to manage the execution of experiments on a remote server accessed through ssh

    Attributes
    ----------
    _abs_root: str
        The absolute path containing the root of the library"

    Methods
    -------
    run_experiment()
        Run the experiments on the remote server

    copy_list_to_target()
        Copy the file with the list of experiments to be executed on the remote server
    """
    _abs_root = None

    def __init__(self, config, args):
        """
        Arguments
        ---------
        config: dict of str: dict of str: str
            The dictionary created from configuration file

        args
            The command line arguments parsed
        """
        super().__init__(config, args)
        if not config.has_section("inhouse"):
            logging.error("inhouse section missing in configuration file")
            sys.exit(1)
        if not config.has_option("inhouse", "username"):
            logging.error("inouse section has not username field in configuration file")
            sys.exit(1)
        self._remote_user = config["inhouse"]["username"]

        #The absolute path of the current file
        abs_script = os.path.realpath(__file__)

        #The root directory of the script
        self._abs_root = os.path.dirname(abs_script)

    def copy_list_to_target(self, list_file_name):
        """
        Copy the list_file_name to target

        Parameters
        ----------
        list_file_name: str
            The name of the file to be copied
        """
        self._list_file_name = list_file_name

        if not self._config.has_section("inhouse"):
            logging.error("inhouse section missing in configuration file")
            sys.exit(1)
        if not self._config.has_option("inhouse", "address"):
            logging.error("inouse section has not address field in configuration file")
            sys.exit(1)

        #The private ssh key
        private_key = os.path.join(os.path.abspath(os.path.join(self._abs_root, "..")), "keys", "id_rsa")
        os.chmod(private_key, 0o600)

        rsync_command = "rsync -a -e \"ssh -i " + private_key + " -o StrictHostKeyChecking=no\" " + list_file_name + " " + self._remote_user + "@" + self._config["inhouse"]["address"] + ":/tmp"
        logging.info("rsync command is %s", rsync_command)
        cmd = subprocess.Popen(rsync_command, shell=True)
        retcode = cmd.wait()
        if retcode == 0:
            logging.info("rsync completed")
        else:
            logging.error("Error in SSH")
            sys.exit(-1)

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

        remote_command = "screen -d -m /home/" + self._remote_user + "/a-GPUBench/vm_scripts/launch_local_experiment.py -a " + self._args.application + " --parameters-list /tmp/" + os.path.basename(self._list_file_name) + extra_options + " --repetitions " + str(self._args.repetitions) + " --output " + self._args.output
        logging.info("remote command is %s", remote_command)

        ssh_command = "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i " + os.path.join(self._abs_root, "..", "keys", "id_rsa") + " " + self._remote_user + "@" + self._config["inhouse"]["address"] + " " + remote_command
        logging.info("ssh command is %s", ssh_command)
        cmd = subprocess.Popen(ssh_command, shell=True)
        retcode = cmd.wait()
        if retcode == 0:
            logging.info("SSH completed")
        else:
            logging.error("Error in SSH")
            sys.exit(1)

def parse_args(parser):
    """
    Add to the command line parser the options related to the remote server (none so far)
    """
    return

