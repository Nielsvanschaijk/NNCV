import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from deeplabv3plus_resnet101 import DeepLabV3Plus

from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype
)

from unet import UNet

def remap_label_small(mask, ignore_index=255):
    """
    Keeps only 'small' classes at their original Cityscapes train IDs:
      5   = pole
      6   = traffic light
      7   = traffic sign
      11  = person
      12  = rider
      17  = motorcycle
      18  = bicycle
    Everything else, including original ignore (255), is set to `ignore_index`.
    """
    new_mask = torch.full_like(mask, fill_value=ignore_index)
    small_classes = [5, 6, 7, 11, 12, 17, 18]
    for c in small_classes:
        new_mask[mask == c] = c  # keep the same ID
    new_mask[mask == 255] = ignore_index
    return new_mask


def remap_label_medium(mask, ignore_index=255):
    """
    Keeps only 'medium' classes at their original Cityscapes train IDs:
      1  = sidewalk
      3  = wall
      4  = fence
      13 = car
    Everything else is set to `ignore_index`.
    """
    new_mask = torch.full_like(mask, fill_value=ignore_index)
    medium_classes = [1, 3, 4, 13]
    for c in medium_classes:
        new_mask[mask == c] = c
    new_mask[mask == 255] = ignore_index
    return new_mask


def remap_label_big(mask, ignore_index=255):
    """
    Keeps only 'big' classes at their original Cityscapes train IDs:
      0  = road
      2  = building
      8  = vegetation
      9  = terrain
      10 = sky
      14 = truck
      15 = bus
      16 = train
    Everything else is set to `ignore_index`.
    """
    new_mask = torch.full_like(mask, fill_value=ignore_index)
    big_classes = [0, 2, 8, 9, 10, 14, 15, 16]
    for c in big_classes:
        new_mask[mask == c] = c
    new_mask[mask == 255] = ignore_index
    return new_mask

# Convert raw Cityscapes "ID" labels to "train IDs" (0..18 or 255 for ignore).
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Dictionary to map train IDs -> colors for visualization
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # black for ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    """
    Convert a prediction mask (B x 1 x H x W) of train IDs into a color image (B x 3 x H x W).
    """
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image

def train_single_model(
    args,
    device,
    train_dataloader,
    valid_dataloader,
    model,
    label_remap_function,        # e.g. remap_label_small / remap_label_medium / remap_label_big
    model_label="small",         # just a string to differentiate (small/medium/big)
):

    """
    Trains a single U-Net model on either small, medium, or big classes, according to label_remap_function.
    Returns the best validation loss.
    """

    # Initialize W&B run specifically for this subset
    run_name = f"{args.experiment_id}_{model_label}"
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=run_name,
        config=vars(args),  # hyperparameters
        reinit=True,        # allow multiple wandb runs in one script
    )

    # Define loss/optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare output directory
    output_dir = os.path.join("checkpoints", run_name)
    os.makedirs(output_dir, exist_ok=True)

    best_valid_loss = float('inf')
    current_best_model_path = None

    # Training loop
    for epoch in range(args.epochs):
        print(f"[{model_label}] Epoch {epoch+1:03}/{args.epochs:03}")

        # ---- TRAIN
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            # Convert raw label IDs -> train IDs (0..18, 255)
            labels = convert_to_train_id(labels)
            # Then remap to keep only the subset (small/medium/big)
            labels = label_remap_function(labels)

            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)  # remove channel dim

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            step = epoch * len(train_dataloader) + i
            wandb.log({
                f"train_loss_{model_label}": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=step)

        # ---- VALIDATE
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                labels = label_remap_function(labels)

                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_losses.append(loss.item())

                # Log a few sample predictions
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1).unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    # Convert predictions & labels to color images
                    predictions_color = convert_train_id_to_color(predictions)
                    labels_color = convert_train_id_to_color(labels)

                    # Use torchvision.utils.make_grid if you want multiple images side by side
                    predictions_img = make_grid(predictions_color.cpu(), nrow=4)
                    labels_img = make_grid(labels_color.cpu(), nrow=4)

                    # Permute for wandb logging (H, W, C)
                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        f"predictions_{model_label}": [wandb.Image(predictions_img)],
                        f"labels_{model_label}": [wandb.Image(labels_img)],
                    }, step=epoch * len(train_dataloader) + i)

        valid_loss = sum(valid_losses) / len(valid_losses)
        wandb.log({f"valid_loss_{model_label}": valid_loss}, step=epoch * len(train_dataloader) + i)

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if current_best_model_path:
                os.remove(current_best_model_path)
            current_best_model_path = os.path.join(
                output_dir, 
                f"best_model_{model_label}-epoch={epoch:03}-val_loss={valid_loss:.4f}.pth"
            )
            torch.save(model.state_dict(), current_best_model_path)

    # Final save
    final_model_path = os.path.join(
        output_dir,
        f"final_model_{model_label}-epoch={epoch:03}-val_loss={valid_loss:.4f}.pth"
    )
    torch.save(model.state_dict(), final_model_path)
    print(f"[{model_label}] Training complete! Best valid loss = {best_valid_loss:.4f}")

    wandb.finish()
    return best_valid_loss

def compose_predictions(
    pred_small: torch.Tensor, 
    pred_medium: torch.Tensor,
    pred_big: torch.Tensor,
    ignore_index=255
):
    """
    Combine the three predictions so that:
      small > medium > big > background
    pred_small, pred_medium, pred_big have shape (B, H, W) each.
    Returns a combined prediction of shape (B, H, W).
    """
    # Start with everything at ignore_index (255).
    final = torch.full_like(pred_small, fill_value=ignore_index)

    # Where small != 255, take small.
    mask_small = (pred_small != ignore_index)
    final[mask_small] = pred_small[mask_small]

    # Where final is still 255 and medium != 255, take medium.
    mask_medium = (final == ignore_index) & (pred_medium != ignore_index)
    final[mask_medium] = pred_medium[mask_medium]

    # Where final is still 255 and big != 255, take big.
    mask_big = (final == ignore_index) & (pred_big != ignore_index)
    final[mask_big] = pred_big[mask_big]

    return final

def get_args_parser():
    parser = ArgumentParser("Training script for multiple models (small/medium/big).")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to Cityscapes data")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    return parser

def main(args):
    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize W&B once for the entire multi-model run
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args),
        reinit=True
    )

    # Basic transforms
    transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),  # scale to [0,1]
        Normalize((0.5,), (0.5,)),
    ])

    # Datasets
    train_dataset = Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=transform)
    valid_dataset = Cityscapes(args.data_dir, split="val",   mode="fine", target_type="semantic", transforms=transform)

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Models
    model_small = UNet(in_channels=3, n_classes=19).to(device)
    model_medium = UNet(in_channels=3, n_classes=19).to(device)
    model_big = DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights=None,  # <-- Do NOT load ImageNet pretrained weights
        in_channels=3,
        classes=19
    ).to(device)

    # Optimizers
    optimizer_small  = AdamW(model_small.parameters(),  lr=args.lr)
    optimizer_medium = AdamW(model_medium.parameters(), lr=args.lr)
    optimizer_big    = AdamW(model_big.parameters(),    lr=args.lr)

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Create an output directory
    os.makedirs("checkpoints", exist_ok=True)

    best_val_loss = float("inf")

    ##########################################################
    # TRAINING LOOP (all 3 models in the same loop)
    ##########################################################
    for epoch in range(args.epochs):
        print(f"Epoch [{epoch+1}/{args.epochs}]")

        # ---- TRAIN
        model_small.train()
        model_medium.train()
        model_big.train()

        train_losses_small = []
        train_losses_medium = []
        train_losses_big = []

        for images, labels in train_dataloader:
            images = images.to(device)
            # Convert raw label to train_id
            labels_trainid = convert_to_train_id(labels)

            # SMALL
            labels_small = remap_label_small(labels_trainid)
            labels_small = labels_small.long().squeeze(1).to(device)

            optimizer_small.zero_grad()
            out_small = model_small(images)
            loss_small = criterion(out_small, labels_small)
            loss_small.backward()
            optimizer_small.step()
            train_losses_small.append(loss_small.item())

            # MEDIUM
            labels_medium = remap_label_medium(labels_trainid)
            labels_medium = labels_medium.long().squeeze(1).to(device)

            optimizer_medium.zero_grad()
            out_medium = model_medium(images)
            loss_medium = criterion(out_medium, labels_medium)
            loss_medium.backward()
            optimizer_medium.step()
            train_losses_medium.append(loss_medium.item())

            # BIG
            labels_big = remap_label_big(labels_trainid)
            labels_big = labels_big.long().squeeze(1).to(device)

            optimizer_big.zero_grad()
            out_big = model_big(images)
            loss_big = criterion(out_big, labels_big)
            loss_big.backward()
            optimizer_big.step()
            train_losses_big.append(loss_big.item())

        avg_loss_small  = sum(train_losses_small)  / len(train_losses_small)
        avg_loss_medium = sum(train_losses_medium) / len(train_losses_medium)
        avg_loss_big    = sum(train_losses_big)    / len(train_losses_big)

        # Log training losses
        wandb.log({
            "train_loss_small":  avg_loss_small,
            "train_loss_medium": avg_loss_medium,
            "train_loss_big":    avg_loss_big,
            "epoch": epoch + 1
        })

        # ---- VALIDATION
        model_small.eval()
        model_medium.eval()
        model_big.eval()

        val_losses_small = []
        val_losses_medium = []
        val_losses_big = []

        with torch.no_grad():
            # We'll log just 1 composite image each epoch (for demonstration)
            logged_images = False

            for i, (images, labels) in enumerate(valid_dataloader):
                images = images.to(device)
                labels_trainid = convert_to_train_id(labels)

                # SMALL
                labels_small = remap_label_small(labels_trainid).long().squeeze(1).to(device)
                out_small = model_small(images)
                loss_small = criterion(out_small, labels_small)
                val_losses_small.append(loss_small.item())

                # MEDIUM
                labels_medium = remap_label_medium(labels_trainid).long().squeeze(1).to(device)
                out_medium = model_medium(images)
                loss_medium = criterion(out_medium, labels_medium)
                val_losses_medium.append(loss_medium.item())

                # BIG
                labels_big = remap_label_big(labels_trainid).long().squeeze(1).to(device)
                out_big = model_big(images)
                loss_big = criterion(out_big, labels_big)
                val_losses_big.append(loss_big.item())

                if not logged_images:
                    # Generate predictions
                    pred_small  = out_small.softmax(dim=1).argmax(dim=1)   # (B,H,W)
                    pred_medium = out_medium.softmax(dim=1).argmax(dim=1)  # (B,H,W)
                    pred_big    = out_big.softmax(dim=1).argmax(dim=1)     # (B,H,W)

                    # Compose final prediction
                    composed_pred = compose_predictions(
                        pred_small, pred_medium, pred_big, ignore_index=255
                    )
                    # Convert to color for logging
                    composed_pred_color = convert_train_id_to_color(composed_pred.unsqueeze(1))
                    composed_pred_img = make_grid(composed_pred_color.cpu(), nrow=4)
                    composed_pred_img = composed_pred_img.permute(1, 2, 0).numpy()

                    # Convert ground-truth to color as well
                    labels_color = convert_train_id_to_color(labels_trainid)
                    labels_color_grid = make_grid(labels_color.cpu(), nrow=4)
                    labels_color_grid = labels_color_grid.permute(1, 2, 0).numpy()

                    wandb.log({
                        "val_composed_prediction": [wandb.Image(composed_pred_img)],
                        "val_ground_truth": [wandb.Image(labels_color_grid)],
                    })
                    logged_images = True

        avg_val_small  = sum(val_losses_small)  / len(val_losses_small)
        avg_val_medium = sum(val_losses_medium) / len(val_losses_medium)
        avg_val_big    = sum(val_losses_big)    / len(val_losses_big)

        val_loss = (avg_val_small + avg_val_medium + avg_val_big) / 3.0  # or define your own weighting

        wandb.log({
            "val_loss_small":  avg_val_small,
            "val_loss_medium": avg_val_medium,
            "val_loss_big":    avg_val_big,
            "val_loss_combined": val_loss
        })

        # Save best combined model weights (optionally you can save each model separately too)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_small": model_small.state_dict(),
                "model_medium": model_medium.state_dict(),
                "model_big": model_big.state_dict(),
            }, os.path.join("checkpoints", f"best_models_epoch={epoch+1}_val={val_loss:.4f}.pth"))
            print(f"New best model saved at epoch {epoch+1} with val_loss {val_loss:.4f}")

    wandb.finish()
    print("Training complete.")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)