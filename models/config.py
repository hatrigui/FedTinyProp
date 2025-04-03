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

    elif dataset_name == "cifar10":
        return {
            "tinyprop_params": TinyPropParams(S_min=0.05, S_max=0.7, zeta=0.9, number_of_layers=5),
            "skip_threshold": 1e-5,
            "full_flops_per_batch": 2e6,
            "phi_min": 0.3,  
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9
            }
        }

    elif dataset_name == "cifar100":
        return {
            "tinyprop_params": TinyPropParams(S_min=0.1, S_max=0.8, zeta=0.95, number_of_layers=3),
            "skip_threshold": 1e-3,
            "full_flops_per_batch": 2.5e6,
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9
            }
        }

    else:
        raise ValueError(f"No config defined for dataset: {dataset_name}")
