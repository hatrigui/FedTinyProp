import torch
from models.model import get_tinyprop_model

def avg_aggregate(models, model_name, tinyprop_params):

    global_model = get_tinyprop_model(model_name, tinyprop_params)
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [m.state_dict()[key].float() for m in models], dim=0
        ).mean(0)

    global_model.load_state_dict(global_dict)
    return global_model
