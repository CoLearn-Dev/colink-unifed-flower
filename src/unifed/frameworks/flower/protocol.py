import os
import json
import sys
import subprocess
import tempfile
from typing import List

import colink as CL

from unifed.frameworks.flower.util import store_error, store_return, GetTempFileName, get_local_ip

pop = CL.ProtocolOperator(__name__)
UNIFED_TASK_DIR = "unifed:task"

def load_config_from_param_and_check(param: bytes):
    unifed_config = json.loads(param.decode())
    framework = unifed_config["framework"]
    assert framework == "flower"
    deployment = unifed_config["deployment"]
    if deployment["mode"] != "colink":
        raise ValueError("Deployment mode must be colink")
    return unifed_config

def run_external_process_and_collect_result(cl: CL.CoLink, participant_id,  role: str, server_ip: str):
    with GetTempFileName() as temp_log_filename, \
        GetTempFileName() as temp_output_filename:
        # note that here, you don't have to create temp files to receive output and log
        # you can also expect the target process to generate files and then read them

        # start training procedure
        print(f"{role}_{participant_id} starts popen...")
        process = subprocess.Popen(
            [
                "python",  
                # takes 4 args: mode(client/server), participant_id, output, and logging destination
                f"{role}.py",
                "config.json",
                str(participant_id),
                str(server_ip),
                # temp_output_filename,
                # temp_log_filename,
            ],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        # gather result
        print(f"{role}_{participant_id} starts gathering results...")
        stdout, stderr = process.communicate()
        print(f"{role}_{participant_id} done!")
        print(stdout.decode())
        print(stderr.decode())
        returncode = process.returncode
        with open(temp_output_filename, "rb") as f:
            output = f.read()
        cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:output", stdout.decode())
        with open(temp_log_filename, "rb") as f:
            log = f.read()
        cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:log", stderr.decode())
        return json.dumps({
            "server_ip": server_ip,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "returncode": returncode,
        })


@pop.handle("unifed.flower:server")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_server(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    unifed_config = load_config_from_param_and_check(param)
    flower_config = unifed_config
    flower_config["training_param"] = flower_config["training"]
    flower_config.pop("training")
    flower_config["training_param"]["epochs"] = flower_config["training_param"]["global_epochs"]
    flower_config["bench_param"] = flower_config["deployment"]
    with open("config.json", "w") as cf:
        json.dump(flower_config, cf)
    # for certain frameworks, clients need to learn the ip of the server
    # in that case, we get the ip of the current machine and send it to the clients
    server_ip = get_local_ip()
    cl.send_variable("server_ip", server_ip, [p for p in participants if p.role == "client"])
    # run external program
    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    return run_external_process_and_collect_result(cl, participant_id, "server", server_ip)


@pop.handle("unifed.flower:client")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_client(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    unifed_config = load_config_from_param_and_check(param)
    flower_config = unifed_config
    flower_config["training_param"] = flower_config["training"]
    flower_config.pop("training")
    flower_config["training_param"]["epochs"] = flower_config["training_param"]["global_epochs"]
    flower_config["bench_param"] = flower_config["deployment"]
    with open("config.json", "w") as cf:
        json.dump(flower_config, cf)
    # get the ip of the server
    server_in_list = [p for p in participants if p.role == "server"]
    assert len(server_in_list) == 1
    p_server = server_in_list[0]
    server_ip = cl.recv_variable("server_ip", p_server).decode()
    # run external program
    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    return run_external_process_and_collect_result(cl, participant_id, "client", server_ip)
