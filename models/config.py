from models.tinyProp import TinyPropParams

def get_tinyprop_config(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name in ["mnist", "fashionmnist"]:
        return {
            "tinyprop_params": TinyPropParams(S_min=0.05, S_max=0.5, zeta=0.25, number_of_layers=2),
            "skip_threshold": 0.005,
            "full_flops_per_batch": 1e6,
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9
            }
        }

    elif dataset_name in ["cifar10", "cifar100"]:
        return {
            "tinyprop_params": TinyPropParams(S_min=0.1, S_max=0.3, zeta=0.7, number_of_layers=3),
            "skip_threshold": 0.01,
            "full_flops_per_batch": 3e6,
            "phi_min": 0.5,
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 5e-4
            }
        }

    else:
        raise ValueError(f"No config defined for dataset: {dataset_name}")
