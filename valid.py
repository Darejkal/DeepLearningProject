import os

import torch
from config import getConfig
from customlog import CustomLogger
from data import JSONLEventData
from torch.utils.data import DataLoader

from model import ImprovisedSasrec
from utils import saveModel, tryRestoreStateDict
def main():
    config=getConfig()
    logger=CustomLogger(log_file=os.path.join(config["train_dir"], 'log.txt'))
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
    model=ImprovisedSasrec(trainset.num_items, config["max_len"],config["hidden_size"],config["dropout_rate"],config["num_heads"],config["sampling_style"],device=config["device"])
    model.to(model.device)
    _,epoch_start_idx=tryRestoreStateDict(model,config["device"],config["train_dir"],config["state_dict_path"])
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.98))
    logger=CustomLogger(os.path.join(config["train_dir"],"log.txt"))
    results=(0,0)
    count=0
    for batch in next(iter(testloader)):
        recall,mrr,_=model.validate_step(batch,0,logger)
        results[0]+=recall
        results[1]+=mrr 
        count+=1
    logger.log("",f"Avg_recall:{recall.numpy()/count}, Avg_result:{mrr.numpy()/count}")
if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()