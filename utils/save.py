import torch
import os


def write_scalar(writer, scalar_name, scalar, step):
    writer.add_scalar(scalar_name, scalar, step)


def save_model(score, step, model, saves, log_path):

    score = round(score, 4)
    model_path = os.path.join(log_path, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    new_save = [score, step, os.path.join(model_path, f'model_{step}_{score}.pt')]

    if score > saves[-1][0]:
        saves.append(new_save)
        torch.save(model.state_dict(), new_save[2])
        saves.sort(key=lambda x: x[0], reverse=True)
        if os.path.exists(saves[-1][2]):
            os.remove(saves[-1][2])
        return saves[:-1]

    else:
        return saves

