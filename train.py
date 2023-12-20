import os

import torch
from config import getConfig
from customlog import CustomLogger
from data import JSONLEventData
from torch.utils.data import DataLoader
import wandb
from model import ImprovisedSasrec
from utils import saveModel, tryRestoreStateDict
def main():
    config=getConfig()
    wandb.init(project="sasrec",dir=os.path.join(config["train_dir"],"wandb"))
    model=ImprovisedSasrec(trainset.num_items, config["max_len"],config["hidden_size"],config["dropout_rate"],config["num_heads"],config["sampling_style"],device=config["device"])
    model.to(model.device)
    wandb.watch(model, log_freq=100)
    _,epoch_start_idx=tryRestoreStateDict(model,config["device"],config["train_dir"],config["state_dict_path"])
    logger=CustomLogger(log_file=os.path.join(config["train_dir"], 'log.txt'))
    trainset=JSONLEventData(path=config["dataset"],
                            stats_file=config["stats_file"],
                            sub_category="train",
                            max_seqlen=config["max_len"],
                            num_in_batch_negatives=config["num_batch_negatives"],
                            num_uniform_negatives=config["num_uniform_negatives"],
                            reject_uniform_session_items=config["reject_uniform_session_items"],
                            reject_in_batch_items=config["reject_in_batch_items"],
                            sampling_style=config["sampling_style"],
                            device=config["device"])
    trainloader=DataLoader(trainset,
                                drop_last=True,
                                batch_size=config["batch_size"],
                                shuffle=config["shuffle"],
                                pin_memory=True,
                                persistent_workers=True,
                                num_workers=os.cpu_count() or 2,
                                collate_fn=trainset.dynamic_collate)
    testset=JSONLEventData(path=config["dataset"],
                            stats_file=config["stats_file"],
                            sub_category="test",
                            max_seqlen=config["max_len"],
                        num_in_batch_negatives=config["num_batch_negatives"],
                        num_uniform_negatives=config["num_uniform_negatives"],
                        reject_uniform_session_items=config["reject_uniform_session_items"],
                        reject_in_batch_items=config["reject_in_batch_items"],
                        sampling_style=config["sampling_style"],
                        device=config["device"])
    testloader=DataLoader(testset,
                                drop_last=True,
                                batch_size=config["batch_size"],
                                shuffle=False,
                                pin_memory=True,
                                persistent_workers=True,
                                num_workers=os.cpu_count() or 2,
                                collate_fn=trainset.dynamic_collate)
    # if config["inference_only"]:
    #     model.eval()
    #     score = model.evaluate()
    #     logger.log("INFERENCE",score,True)
    #     exit()
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.98))
    logger=CustomLogger(os.path.join(config["train_dir"],"log.txt"))
    for epoch in range(epoch_start_idx, config["num_epochs"] + 1):
        logger.log("",f"Epoch {epoch}",True)
        for step in range(config["num_batch"]):
            batch=next(iter(trainloader))
            loss=model.train_step(batch,step,logger,optimizer,bce_criterion)
            wandb.log({"loss": loss})
        model.validate_step(next(iter(testloader)),epoch,logger,bce_criterion)
        saveModel(model,epoch,config["train_dir"])
if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()