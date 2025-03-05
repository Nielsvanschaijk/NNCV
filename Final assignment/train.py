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

def get_args_parser():
    parser = ArgumentParser("Training script for multiple U-Net models (small/medium/big).")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to Cityscapes data")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    return parser

def main(args):
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Basic transforms
    transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ])

    # Load train/val splits from Cityscapes
    train_dataset = Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=transform)
    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=transform)

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_small = UNet(in_channels=3, n_classes=19).to(device)
    _ = train_single_model(
        args=args,
        device=device,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        model=model_small,
        label_remap_function=remap_label_small,
        model_label="small"
    )

    model_medium = UNet(in_channels=3, n_classes=19).to(device)
    _ = train_single_model(
        args=args,
        device=device,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        model=model_medium,
        label_remap_function=remap_label_medium,
        model_label="medium"
    )

    model_big = DeepLabV3Plus(
    encoder_name="resnet101",     # or "resnet50", "mobilenet_v2", "timm-resnest101", etc.
    encoder_weights="imagenet",   # use pretrained backbone weights
    in_channels=3,
    classes=19                    # Cityscapes has 19 valid train IDs
    ).to(device)
    
    _ = train_single_model(
        args=args,
        device=device,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        model=model_big,
        label_remap_function=remap_label_big,
        model_label="big"
    )

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
