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
import sys

import provider

class Provider(provider.Provider):
    """
    Class to manage VMs created on Microsoft Azure

    Attributes
    ----------
    _azinterface: module
        The python3 module for using Microsoft Azure API

    _group_name: str
        The group name to be used in generation of VMs

    _location: str
        The Microsoft Azure location where VMs must be created

    _image: str
        The name of the image to be used for created VMs

    _size: str
        The size (i.e., the type) of VM to be created

    _subscription_name: str
        The name of the subscription to be used

    Methods
    -------
    run_experiment()
        Run the experiments on the Microsoft Azure VM

    copy_list_to_target()
        Copy the file with the list of experiments to be executed on the Microsoft Azure VM
    """
    _azinterface = None

    _group_name = None

    _location = None

    _image = None

    _size = None

    _subscription_name = None

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

        #The absolute path of the current file
        abs_script = os.path.realpath(__file__)

        #The root directory of the script
        abs_root = os.path.dirname(abs_script)

        sys.path.append(os.path.join(abs_root, "azure"))
        self._azinterface = __import__("azinterface")

        if self._args.group_name:
            group_name = self._args.group
        else:
            group_name = config["azure"]["default_group_name"]

        if self._args.location:
            location = self._args.location
        else:
            location = config["azure"]["default_location"]

        if self._args.image:
            image = self._args.image
        else:
            image = config["azure"]["default_image"]

        remote_user = config["azure"]["username"]

        #Check required parameters
        if not self._args.size:
            logging.error("--size is mandatory for azure provider")
            sys.exit(1)
        self._size = self._args.size
        if not self._args.subscription:
            logging.error("Subscription must be set for azure vm")
            sys.exit(1)

        #Check if key files exists
        if not os.path.exists(os.path.join(abs_root, "..", "keys", "id_rsa")) or not os.path.exists(os.path.join(abs_root, "..", "keys", "id_rsa.pub")):
            logging.error("Please add id_rsa and id_rsa.pub in keys")
            sys.exit(1)

        #Check if ssmtp configuration file exists
        if self._args.mail:
            if not os.path.exists(os.path.join(abs_root, "..", "vm_scripts", "revaliases")) or not os.path.exists(os.path.join(abs_root, "..", "vm_scripts", "ssmtp.conf")):
                logging.error("--mail option cannot be used without ssmtp configuration files")
                sys.exit(1)

        sys.path.append(os.path.join(abs_root, "azure"))
        import create_vm
        #Set the subscription
        self._azinterface.az_execute_command("account set --subscription " + self._args.subscription)
        if self._args.subscription == "058fad11-b4fb-4ac8-a3c2-9f8b76c4cdac":
            subscription_name = "temp1"
        elif self._args.subscription == "0a4671e3-d0d3-4ae9-89b1-1189163b65e7":
            subscription_name = "temp2"
        elif self._args.subscription == "294fff78-08a2-4e7e-b463-f2bb947a5abb":
            subscription_name = "temp3"
        elif self._args.subscription == "beadafc1-a1f6-4ff9-b8fe-e1db131dbf92":
            subscription_name = "temp4"
        else:
            subscription_name = self._azinterface.az_subscritpion_name(self._args.subscription).replace(" ", "_")

        #Create vm
        create_vm.create_vm(subscription_name, group_name, self._args.debug, location, self._args.size, image, True, self._args.reuse, remote_user, config["azure"]["prefix"])

    def copy_list_to_target(self, list_file_name):
        """
        Copy the list_file_name to target

        Parameters
        ----------
        list_file_name: str
            The name of the file to be copied
        """
        self._list_file_name = list_file_name
        self._azinterface.az_vm_rsync_to(self._list_file_name, self._subscription_name, self._location, self._size, "/tmp", self._config["azure"]["prefix"], self._remote_user)

    def run_experiment(self):
        """
        Run the experiments"
        """
        extra_options = ""
        if self._args.profile:
            extra_options = extra_options + " --profile " + self._args.profile

        if not self._args.not_shutdown:
            extra_options = extra_options + " --shutdown"

        if self._args.mail:
            extra_options = extra_options + " --mail "+ self._args.mail

        remote_command = "screen -d -m /home/" + self._remote_user + "/a-GPUBench/vm_scripts/launch_local_experiment.py -a " + self._args.application + " --parameters-list /tmp/" + os.path.basename(self._list_file_name) + extra_options + " --repetitions " + str(self._args.repetitions) + " --subscription " + self._subscription_name  + " --output " + self._args.output
        logging.info("remote command is %s", remote_command)
        self._azinterface.az_vm_ssh_command_invoke(self._subscription_name, self._location, self._size, remote_command, self._config["azure"]["prefix"], self._remote_user)

def parse_args(parser):
    """
    Add to the command line parser the options related to Microsoft Azure
    """
    parser.add_argument("--group-name", help="Azure: The name of the resource group")
    parser.add_argument("--location", help="Azure: The cluster location")
    parser.add_argument("--size", help="Azure: The size (aka the type) of the VM to be created")
    parser.add_argument("--image", help="Azure: The image to be used during creation of VM")
    parser.add_argument("--not-shutdown", help="Azure: Do not shutdown the vm", default=False, action="store_true")
    parser.add_argument("--subscription", help="Azure: The subscription to be used")
    parser.add_argument("--reuse", help="Azure: If true, an already running VM is reused", default=False, action="store_true")

