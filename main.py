import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from src.datasets import ThingsMEGDataset
from src.models import CLIPPretrainedModel
from src.utils import set_seed, train_one_epoch, validate

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(project="MEG-classification", config=args)

    # Dataset and DataLoader
    train_set = ThingsMEGDataset("train", args.data_dir)
    val_set = ThingsMEGDataset("val", args.data_dir)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = CLIPPretrainedModel(num_classes=40, seq_len=512, in_channels=306, dropout_rate=args.dropout_rate).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)

        if args.use_wandb:
            wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{logdir}/model_best.pt")

        print(f"Epoch {epoch + 1}/{args.epochs} | train loss: {train_loss:.3f} | train acc: {train_acc:.3f} | val loss: {val_loss:.3f} | val acc: {val_acc:.3f}")

    torch.save(model.state_dict(), f"{logdir}/model_last.pt")

if __name__ == "__main__":
    run()
