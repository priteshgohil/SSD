import logging
import os

import torch
import torch.utils.data
import torch.distributed as dist
from tqdm import tqdm
import time
import datetime
import logging

from ssd.data.build import make_data_loader
from ssd.data.datasets.evaluation import evaluate

from ssd.utils import dist_util, mkdir
from ssd.utils.dist_util import synchronize, is_main_process

#from ssd.engine.trainer import reduce_loss_dict

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = dist_util.all_gather(predictions_per_gpu)
    if not dist_util.is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("SSD.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def compute_on_dataset(model, data_loader, device, summary_writer, iteration):
    results_dict = {}
    all_loss_dict = {}
    total_loss=[]
    # cls_loss=[]
    # reg_loss=[]
    logger = logging.getLogger("SSD.inference")
    start_eval_time = time.time()  # Record evaluation time
    for i, batch in enumerate(tqdm(data_loader)):
        # per_iteration_time = time.time()
        images, targets, image_ids = batch
        print("engine/inference.py: img size from dataloader {}".format(images.shape))
        print("engine/inference.py: targets from dataloader {}".format(targets))
        cpu_device = torch.device("cpu")
        with torch.no_grad():
            outputs, loss_dict = model(images.to(device), targets=targets.to(device))
            # initialize once
            if all_loss_dict=={}:
                for key in loss_dict.keys():
                    all_loss_dict[key] = list()
            # calculate loss
            loss = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            l_dict_reduced = reduce_loss_dict(loss_dict)
            l_reduced = sum(loss for loss in l_dict_reduced.values())

            outputs = [o.to(cpu_device) for o in outputs]
        total_loss.append(l_reduced)
        for key, value in l_dict_reduced.items():
            all_loss_dict[key].append(value)
        # cls_loss.append(l_dict_reduced['cls_loss'])
        # reg_loss.append(l_dict_reduced['reg_loss'])
        # print("Debug: l_dict_reduced {}".format(l_dict_reduced))
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, outputs)}
        )
    # print("Debug: all_loss_dict {}".format(all_loss_dict))
    # print("Debug: sum of all_loss_dict {}/{}".format(sum(all_loss_dict['reg_loss']), len(all_loss_dict['reg_loss'])))
    total_eval_time = int(time.time() - start_eval_time)
    total_time_str = str(datetime.timedelta(seconds=total_eval_time))
    logger.info("Total inference time: {} (Avg. {:.4f} s / it)".format(total_time_str, total_eval_time / i+1))
    if summary_writer:
        global_step = iteration
        summary_writer.add_scalar('val_losses/total_loss', sum(total_loss) / len(total_loss), global_step=global_step)
        for loss_name, loss_item in all_loss_dict.items():
            summary_writer.add_scalar('val_losses/{}'.format(loss_name), sum(loss_item)/len(loss_item), global_step=global_step)
    # print("total_losses: {} \nSum: {} entries: {}".format(total_loss,sum(total_loss),len(total_loss)))
    # print("loss: {}".format(loss))
    # print("final loss_dict: {}".format(loss_dict))
    # # Log losses
    # if summary_writer:
    #     global_step = iteration
    #     summary_writer.add_scalar('val_losses/total_loss', sum(total_loss)/len(total_loss), global_step=global_step)
    #     summary_writer.add_scalar('val_losses/cls_loss', sum(cls_loss) / len(cls_loss), global_step=global_step)
    #     summary_writer.add_scalar('val_losses/reg_loss', sum(total_loss) / len(total_loss), global_step=global_step)
    return results_dict


def inference(model, data_loader, dataset_name, device, output_folder=None, use_cached=False, **kwargs):
    dataset = data_loader.dataset
    logger = logging.getLogger("SSD.inference")
    logger.info("Evaluating {} dataset({} images):".format(dataset_name, len(dataset)))
    predictions_path = os.path.join(output_folder, 'predictions.pth')
    summary_writer = kwargs['summary_writer']
    del kwargs['summary_writer']
    iteration = kwargs['iteration']
    if use_cached and os.path.exists(predictions_path):
        logger.info("Using saved model predictions.pth at {}:".format(output_folder))
        predictions = torch.load(predictions_path, map_location='cpu')
    else:
        predictions = compute_on_dataset(model, data_loader, device, summary_writer, iteration)
        synchronize()
        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return
    if output_folder:
        torch.save(predictions, predictions_path)
    return evaluate(dataset=dataset, predictions=predictions, output_dir=output_folder, **kwargs)

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = dist_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

@torch.no_grad()
def do_evaluation(cfg, model, distributed, **kwargs):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    data_loaders_val = make_data_loader(cfg, is_train=False, distributed=distributed)
    eval_results = []
    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        if not os.path.exists(output_folder):
            mkdir(output_folder)
        eval_result = inference(model, data_loader, dataset_name, device, output_folder, **kwargs)
        eval_results.append(eval_result)
    return eval_results
