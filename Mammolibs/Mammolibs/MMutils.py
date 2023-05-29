import torch
import os


def convert_seconds_to_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    time_string = ""
    if hours > 0:
        time_string += f"{int(hours)}h"
    if minutes > 0:
        time_string += f"{int(minutes)}m"
    if remaining_seconds > 0:
        time_string += f"{int(remaining_seconds)}s"

    return time_string


def print_iteration_stats(iter, train_loss, val_loss, n_iters, time_per_n_iters): # noqa
    print("Iter:{:5d}  |  Tr_loss: {:.4f}  |  Val_loss: {:.4f}  |  Time per {} iter: {}".format( # noqa
        iter, train_loss, val_loss, n_iters, convert_seconds_to_time(time_per_n_iters))) # noqa


def save_model(model, file_path, iter):
    save_path = os.path.join(file_path, 'model_iter_{}.pth'.format(iter)) # noqa
    torch.save(model.state_dict(), save_path)
    if os.path.exists(save_path):
        print(f'{save_path} saved successfully')
    else:
        print(f'Failed to save {save_path}')
