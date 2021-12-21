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

# How To Run It (Localy)

Before running the container you should know the name of the user that will use the application and his/her id under the system. After you checked it you should change the lines **9, 30, 37** in the DockerFile 

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

If you want to download the raw results from the container you should do the following command:

```
docker cp -r ${container_name}:/opt/app/output/. /path/on/host
```

If you want to preprocess them, then you should run the collect.py script after the results are computed, but it's actualy not recommended since the results' folder name will contain the information of the experiments and it is not predetermined.

```
nvidia-docker run -ti --user $(id -u):$(id -g) {USER}/{NAME}:{VERSION} python3.6 /opt/app/collect.py
```

# How To Run It (client)

We can run the docker as the client and execute the server in the "screen mode". The server doesn't need to be a running docker container. 
To execute it you need to run the command 

```
docker build -t nabatov/pytorch:v3
```

This time we didn't use the nvidia as the base image but ubuntu since it uses less space and since on the client we don't perform any valueble computations, then we don't need this kind of thing to be there.

Then we run the docker as following

```
docker run -ti --user $(id -u):$(id -g) {USER}/{NAME}:{VERSION} python3.6 /opt/app/launch_experiment.py --params
```

# How To Run It (client + server)

Before starting you need to be sure that you've got the same RSA key.
In terms of test you can at first generate the shared RSA key the following way

```
ssh-keygen -t rsa -b 4096 -f ./id_rsa_shared
```

Then build server image

```
docker build -t nabatov/pytorchtest:v4
```

Then we create the network of containers (to assign ip address to the container)

```
docker network create --subnet=172.18.0.0/16 mynet123
```

Then we mount keys and create the user we need to "be" in this container. 
The username should be specified in ~/.ssh.

```
docker run --net mynet123 --ip 172.18.0.22 -v ~/.ssh:/home/user/.ssh:ro -ti --user $(id -u):$(id -g) {USER}/{NAME}:{VERSION}
```

After you can ping the docker, and also you could try to use ssh, it means that it works and you can start the containers locally
