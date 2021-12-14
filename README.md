# a-GPUBench

Framework composed of a collection of python script to run, profile and collect data about applications exploiting GPUs. Application runs can be started by means of launch_experiment.py script.

The framework can be used with different machines and different applicaitons.
The target architecture already supported by this version are:
- inhouse server
- Microsoft Azure VMs

The application already supported by this version are:
- CNN training with pytorch
- CNN training with tensorflow

The framework can be configured via .ini configuration file.
An example of configuration file is available in configurations/default.ini.

Support to new providers can be provided by adding a python package under providers.
The package must provide the following functions:
- copy_list_to_target: to copy the list of experiments to be run from localhost to target
- initialize: to initialize the target architecture
- parse_args: to add command line arguments specific of the target architecture
- run_experiment: to run the experiment(s)

Support to new applications (not limited to python implementations) can be provided by adding a python package under apps which wrap them.
The package must provide the following functions:
- compute_configuration_name: to compute the name of the configuration of an experiment
- collect_data: to parse the output of an experiment and generate results
- main: to execute experiement(s)

The code in this repository is licensed under the terms of the
[Apache License version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

# How To Run It

Before running the container you should know the name of the user that will use the application and his/her id under the system. After you checked it you should change the lines **37-38** in the DockerFile 

```
RUN useradd -u {id} {username}
USER {username}
```

After you've done it you should build the image using the following command:

```
docker build -t {USER}/{NAME}:{VERSION}
```

where USER is the name of the user that will run it (it is not necessary but it is useful if you'll run the image not on your own PC or the server is the shared with other users), NAME is the name of image and VERSION is the version. All parameters are represented as for example: 

```
docker build -t nabatov/pytorchtest:v2
```

Then we need to run it and I use the configuration that is not default for this example. Actually I want to run the grid_search.py that is working with the configuration under the folder /apps/pytorch/confs/base_grid2.xml with the patience equals to 10

```
nvidia-docker run -ti --user $(id -u):$(id -g) {USER}/{NAME}:{VERSION} python3.6 /opt/app/grid_search.py
```

In this command you nee to specify only the image that you've built. But if you want to run so-called local experiment you should use the following command:

```
nvidia-docker run -ti --user $(id -u):$(id -g) {USER}/{NAME}:{VERSION} python3.6 /opt/app/launch_experiment.py --params
```

--params mean the parameters that you should pass to the script launch_experiment.py

If you want to download the results from the container you should do the following command:

```
docker cp -r ${container_name}:/opt/app/output/. /path/on/host
```
