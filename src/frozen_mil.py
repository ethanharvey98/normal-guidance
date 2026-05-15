import argparse
import os
import time
import pandas as pd
import torch
# Importing our custom module(s)
import datasets
import losses
import models
import utils

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="oasis-3.py")
    parser.add_argument("--alpha", default=0.0, help="TODO (default: 0.0)", type=float)
    parser.add_argument("--batch_size", default=64, help="Batch size (default: 64)", type=int)
    parser.add_argument("--beta", default=1.0, help="TODO (default: 1.0)", type=float)
    parser.add_argument("--criterion", default="ERM", help="TODO (default: \"ERM\")", type=str)
    parser.add_argument("--dataset_dir", default="", help="Directory to dataset (default: \"\")", type=str)
    parser.add_argument("--embedding_level", action="store_true", default=False, help="Whether or not to use the embedding-level approach (default: False)")
    parser.add_argument("--epochs", default=1000, help="Number of epochs (default: 1000)", type=int)
    parser.add_argument("--experiments_dir", default="", help="Directory to save experiments (default: \"\")", type=str)
    parser.add_argument("--lr", default=0.01, help="Learning rate (default: 0.01)", type=float)
    parser.add_argument("--model_name", default="test", help="Model name (default: \"test\")", type=str)
    parser.add_argument("--pooling", default="max", help="Pooling operation (default: \"max\")", type=str)
    parser.add_argument("--save", action="store_true", default=False, help="Whether or not to save the model (default: False)")
    parser.add_argument("--seed", default=42, help="TODO (default: 42)", type=int)
    parser.add_argument("--weight_decay", default=0.0, help="Weight decay (default: 0.0)", type=float)
    parser.add_argument("--neighbors", default=1, help="Number of neighbors for SmAP pooling (default: 1)", type=int)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    os.makedirs(args.experiments_dir, exist_ok=True)
    
    train_data = torch.load(f"{args.dataset_dir}/train.pth", map_location=torch.device("cpu"), weights_only=False)
    val_data = torch.load(f"{args.dataset_dir}/val.pth", map_location=torch.device("cpu"), weights_only=False)
    test_data = torch.load(f"{args.dataset_dir}/test.pth", map_location=torch.device("cpu"), weights_only=False)
    
    train_dataset = datasets.MILTensorDataset(train_data["X"], train_data["lengths"], train_data["y"])
    #train_dataset = datasets.MILTensorDataset(train_data["X"], train_data["lengths"], train_data["y"], train_data["lengths_y"])
    val_dataset = datasets.MILTensorDataset(val_data["X"], val_data["lengths"], val_data["y"])
    #val_dataset = datasets.MILTensorDataset(val_data["X"], val_data["lengths"], val_data["y"], val_data["lengths_y"])
    test_dataset = datasets.MILTensorDataset(test_data["X"], test_data["lengths"], test_data["y"])
    #test_dataset = datasets.MILTensorDataset(test_data["X"], test_data["lengths"], test_data["y"], test_data["lengths_y"])
    
    shuffled_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=utils.collate_fn, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn)
        
    if args.embedding_level:
        model = models.PoolClf(in_features=train_data["X"].shape[1], out_features=1, pooling=args.pooling, neighbors=args.neighbors)
    else:
        model = models.ClfPool(in_features=train_data["X"].shape[1], out_features=1, pooling=args.pooling, neighbors=args.neighbors)
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    
    assert args.criterion in ["ERM", "L1", "L2", "GuidedL1"]
    if args.criterion == "ERM":
        criterion = losses.ERMLoss(criterion=torch.nn.BCEWithLogitsLoss())
    elif args.criterion == "L1":
        criterion = losses.L1Loss(alpha=args.alpha, criterion=torch.nn.BCEWithLogitsLoss())
    elif args.criterion == "L2":
        criterion = losses.L2Loss(alpha=args.alpha, criterion=torch.nn.BCEWithLogitsLoss())
    elif args.criterion == "GuidedL1":
        criterion = losses.GuidedAttentionL1Loss(alpha=args.alpha, beta=args.beta, criterion=torch.nn.BCEWithLogitsLoss())
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    
    columns = ["epoch", "test_auroc", "test_auprc", "test_bal_acc", "test_loss", "test_nll", "train_auroc", "train_auprc", "train_bal_acc", "train_loss", "train_nll", "train_sec/epoch", "val_auroc", "val_auprc", "val_bal_acc", "val_loss", "val_nll"]
    model_history_df = pd.DataFrame(columns=columns)

    for epoch in range(args.epochs):
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start_time = time.time()

        shuffled_train_metrics = utils.train_one_epoch(model, criterion, optimizer, shuffled_train_loader)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_end_time = time.time()        
        
        #train_metrics = utils.evaluate(model, criterion, train_loader)
        train_metrics = shuffled_train_metrics
        val_metrics = utils.evaluate(model, criterion, val_loader)
        test_metrics = utils.evaluate(model, criterion, test_loader)
        
        row = [epoch, test_metrics["auroc"], test_metrics["auprc"], test_metrics["bal_acc"], test_metrics["loss"], test_metrics["nll"], train_metrics["auroc"], train_metrics["auprc"], train_metrics["bal_acc"], train_metrics["loss"], train_metrics["nll"], epoch_end_time - epoch_start_time, val_metrics["auroc"], val_metrics["auprc"], val_metrics["bal_acc"], val_metrics["loss"], val_metrics["nll"]]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        model_history_df.to_csv(f"{args.experiments_dir}/{args.model_name}.csv")
    
        val_auroc_series = model_history_df[model_history_df.train_auroc > model_history_df.val_auroc].val_auroc
        if args.save and epoch == (val_auroc_series.idxmax() if not val_auroc_series.empty else None):
            torch.save(model.state_dict(), f"{args.experiments_dir}/{args.model_name}.pt")
