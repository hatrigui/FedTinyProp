from models.tinyProp import TinyPropParams

def get_tinyprop_config(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name in ["mnist", "fashionmnist"]:
        return {
            "tinyprop_params": TinyPropParams(
                S_min=0.5,
                S_max=0.95,
                zeta=0.5,
                number_of_layers=4,
                random_skip=False
            ),
            "optimizer": {
                "type": "sgd",
                "lr": 0.0005,
                "momentum": 0.7,
                "weight_decay": 1e-4
            },
            "lr_scheduler": {
                "type": "cosine",
                "T_max": 100,
                "eta_min": 0.00001,
                "warmup_epochs": 5,
                "warmup_start_lr": 0.00001
            },
            "gradient_clip": 1.0,
            "batch_size": 32,
            "num_epochs": 1,
            "label_smoothing": 0.1,
            "quantization": {
                "bits": 8,
                "enabled": True
            }
        }

    elif dataset_name in ["cifar10", "cifar100"]:
        return {
            "tinyprop_params": TinyPropParams(
                S_min=0.5,
                S_max=0.95,
                zeta=0.5,
                number_of_layers=11,
                random_skip=False
            ),
            "optimizer": {
                "type": "sgd",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 5e-4
            },
            "lr_scheduler": {
                "type": "cosine",
                "T_max": 100,
                "eta_min": 0.0001,
                "warmup_epochs": 5,
                "warmup_start_lr": 0.001
            },
            "gradient_clip": 1.0,
            "batch_size": 32,
            "num_epochs": 1,
            "label_smoothing": 0.1,
            "data_augmentation": {
                "random_crop": True,
                "random_horizontal_flip": True,
                "random_rotation": 5,
                "color_jitter": {
                    "brightness": 0.2,
                    "contrast": 0.2,
                    "saturation": 0.2
                }
            },
            "quantization": {
                "bits": 8,
                "enabled": True
            }
        }

    elif dataset_name == "har":
        return {
            "tinyprop_params": TinyPropParams(
                S_min=0.5,
                S_max=0.95,
                zeta=0.5,
                number_of_layers=5,
                random_skip=False
            ),
            "optimizer": {
                "type": "sgd",
                "lr": 0.005,
                "momentum": 0.9
            },
            "lr_scheduler": {
                "type": "cosine",
                "T_max": 50,
                "eta_min": 1e-5
            },
            "gradient_clip": 1.0,
            "batch_size": 32,
            "num_epochs": 1,
            "quantization": {
                "bits": 8,
                "enabled": True
            }
        }
        
    elif dataset_name == "speechcommands":
        return {
            "tinyprop_params": TinyPropParams(
                S_min=0.5,            # Start sparse but conservative
                S_max=0.95,            # Allow aggressive pruning if possible
                zeta=0.5,            # Moderate adaptivity
                number_of_layers=6,   # CNN+GRU model has ~6 weight layers
                random_skip=False
            ),
            "optimizer": {
                "type": "sgd",       # Adam is usually better for speech/MFCC
                "lr": 0.001,
                "weight_decay": 1e-4
            },
            "lr_scheduler": {
                "type": "cosine",
                "T_max": 50,
                "eta_min": 1e-5,
                "warmup_epochs": 3,
                "warmup_start_lr": 1e-5
            },
            "gradient_clip": 1.0,
            "batch_size": 32,         # MFCC inputs are compact enough
            "num_epochs": 1,          # Overridable during experiments
            "label_smoothing": 0.1,
            "quantization": {
                "bits": 8,
                "enabled": True
            },
            "mfcc_config": {
                "sample_rate": 16000,
                "n_mfcc": 40,
                "n_fft": 1024,
                "hop_length": 512
            }
        }

    

    else:
        raise ValueError(f"No config defined for dataset: {dataset_name}")
