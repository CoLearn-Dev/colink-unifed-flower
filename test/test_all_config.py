import glob
import json
import pytest
import copy
import os

import flbenchmark.datasets
import flbenchmark.logging

import colink as CL


def simulate_with_config(config_file_path):
    from unifed.frameworks.flower.protocol import pop, UNIFED_TASK_DIR
    case_name = config_file_path.split("/")[-1].split(".")[0]
    with open(config_file_path, "r") as cf:
        config = json.load(cf)
    # convert config format
    flower_config = copy.deepcopy(config)
    flower_config["training_param"] = flower_config["training"]
    flower_config.pop("training")
    flower_config["bench_param"] = flower_config["deployment"]
    with open("config.json", "w") as cf:
        json.dump(flower_config, cf)
    # load dataset
    flbd = flbenchmark.datasets.FLBDatasets('../data')
    val_dataset = None
    if config['dataset'] == 'reddit':
        train_dataset, test_dataset, val_dataset = flbd.leafDatasets(config['dataset'])
    elif config['dataset'] == 'femnist':
        train_dataset, test_dataset = flbd.leafDatasets(config['dataset'])
    else:
        train_dataset, test_dataset = flbd.fateDatasets(config['dataset'])
    train_data_base = os.path.abspath('../csv_data/'+config['dataset']+'_train')
    test_data_base = os.path.abspath('../csv_data/'+config['dataset']+'_test')
    val_data_base = os.path.abspath('../csv_data/'+config['dataset']+'_val')
    flbenchmark.datasets.convert_to_csv(train_dataset, out_dir=train_data_base)
    if test_dataset is not None:
        flbenchmark.datasets.convert_to_csv(test_dataset, out_dir=test_data_base)
    if val_dataset is not None:
        flbenchmark.datasets.convert_to_csv(val_dataset, out_dir=val_data_base)

    # use instant server for simulation
    ir = CL.InstantRegistry()
    # TODO: confirm the format of `participants``
    config_participants = config["deployment"]["participants"]
    cls = []
    participants = []
    for _, role in config_participants:  # given user_ids are omitted and we generate new ones here
        cl = CL.InstantServer().get_colink().switch_to_generated_user()
        pop.run_attach(cl)
        participants.append(CL.Participant(user_id=cl.get_user_id(), role=role))
        cls.append(cl)
    task_id = cls[0].run_task("unifed.flower", json.dumps(config), participants, True)
    results = {}
    def G(key):
        r = cl.read_entry(f"{UNIFED_TASK_DIR}:{task_id}:{key}")
        if r is not None:
            if key == "log":
                return [json.loads(l) for l in r.decode().split("\n") if l != ""]
            return r.decode() if key != "return" else json.loads(r)
    for cl in cls:
        cl.wait_task(task_id)
        results[cl.get_user_id()] = {
            "output": G("output"),
            "log": G("log"),
            "return": G("return"),
            "error": G("error"),
        }
    return case_name, results


def test_load_config():
    # load all config files under the test folder
    config_file_paths = glob.glob("test/configs/*.json")
    assert len(config_file_paths) > 0


def convert_metric(config):
    AUC = ['breast_horizontal', 'default_credit_horizontal', 'give_credit_horizontal',
        'breast_vertical', 'default_credit_vertical', 'give_credit_vertical']
    ACC = ['vehicle_scale_horizontal', 'vehicle_scale_vertical', 'femnist', 'celeba', 'reddit']
    ERR = ['student_horizontal', 'motor_vertical', 'dvisits_vertical',  'student_vertical']
    if config['dataset'] in AUC:
        metrics = 'auc'
    elif config['dataset'] in ACC:
        metrics = 'accuracy'
    elif config['dataset'] in ERR:
        metrics = 'mse'
    else:
        raise NotImplementedError('Dataset {} is not supported.'.format(config['dataset']))
    with open('./log/0.log', 'r') as f:
        lines = f.readlines()
    out = []
    for line in lines:
        log = json.loads(line)
        if log.get('event') is not None and log['event'] == 'model_evaluation' and log['action'] == 'end':
            real_metrics = {metrics: log['metrics']['target_metric']}
            log['metrics'] = real_metrics
            out.append(json.dumps(log)+'\n')
        else:
            out.append(line)
    with open('./log/0.log', 'w') as f:
        f.writelines(out)


@pytest.mark.parametrize("config_file_path", glob.glob("test/configs/*.json"))
def test_with_config(config_file_path):
    if "skip" in config_file_path:
        pytest.skip("Skip this test case")
    results = simulate_with_config(config_file_path)
    for r in results[1].values():
        print(r["return"]["stderr"])
    assert all([r["error"] is None and r["return"]["returncode"] == 0 for r in results[1].values()])


if __name__ == "__main__":
    from pprint import pprint
    import time
    nw = time.time()
    target_case = "test/configs/case_0.json"
    # print(json.dumps(simulate_with_config(target_case), indent=2))
    results = simulate_with_config(target_case)
    for r in results[1].values():
        print(r["return"]["stdout"])
        print(r["return"]["stderr"])
    convert_metric(json.load(open("config.json", 'r')))
    flbenchmark.logging.get_report('./log')
    print("Time elapsed:", time.time() - nw)
