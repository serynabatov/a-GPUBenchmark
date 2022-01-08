#!/usr/bin/python3.6
"""
Copyright 2021 Sergei Nabatov

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

import xmltodict
import os
import logging
import sys
import subprocess


def main():
    #Elements used in grid search
    network_type = ['alexnet', 'mobilenet_v2', 'resnet', 'vgg16']
    #network_depth = [1,2,3] #for resnet
    optimizers = ['adam', 'sgd']
    momentum = [0.7, 0.8, 0.9]
    learning_rate = [0.001, 0.0001, 0.00001]
    batch_size = [16, 32, 64]
    seed = [1, 42, 100]
    #base xml used in order to fill it with information and run a training session
    config_name = 'base_grid2.xml'
    #path of config files
    base_path = '/opt/app/apps/pytorch/confs/'
    mom = 0.9
    for network in network_type:
    	for optimizer in optimizers:
    		for lr in learning_rate:
    			for bs in batch_size:
    				for mom in momentum:
    					for s in seed:
    						xml_file = os.path.join(base_path, config_name)
    						if not os.path.exists(xml_file):
    							logging.error("XML file %s not found", xml_file)
    							sys.exit(1)
    						# Load XML file
    						with open(xml_file) as fd:
    							doc = xmltodict.parse(fd.read(), force_list={'input_class'})
    							
    						#fill xml with information obtained from grid search
    						#try with one configuration
    						doc['pytorch_configuration']['network_type'] = network
    						doc['pytorch_configuration']['optimizer'] = optimizer
    						doc['pytorch_configuration']['momentum'] = mom #not always used, but setted anyway
    						doc['pytorch_configuration']['lr'] = lr
    						doc['pytorch_configuration']['batch_size'] = bs
    						doc['pytorch_configuration']['seed'] = s
    						#save configuration
    						with open(os.path.join(base_path, 'test_1.xml'), 'w') as result_file:
    							result_file.write(xmltodict.unparse(doc))
    						
    						#modify grid_conf file specify which configuration to use
    						f = open(os.path.join(base_path,'grid_conf'), "w")
    						f.write("configuration="+"test_1")
    						f.close()
    						
    						remote_command = "python3.6 /opt/app/vm_scripts/launch_local_experiment.py " + \
    						"-a pytorch --parameters-list /opt/app/apps/pytorch/confs/grid_conf --repetitions 1 --output /opt/app/output/{}/{}/{}/{}/{}/{} --profile 10".format(network, optimizer, lr, bs, mom, s)
    						logging.info("remote command is %s", remote_command)
    						
    						cmd = subprocess.Popen(remote_command, shell=True)
    						retcode = cmd.wait()
    						if retcode == 0:
    							logging.info("launched experiment")
    						else:
    							logging.error("Error in launching  experiment")
    							sys.exit(1)


if __name__ == "__main__":
    main()
