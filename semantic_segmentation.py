import argparse

from torchgeo.tasks import SemanticSegmentation
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from perda import PerdaDataModule, Perda


ROOT = "./"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--pre-trained-weights",
        type=str,
        default=None,
        help="Pre-trained weights to initialize the model with. If set to None, its weights are randomly initialized."
    )
    # TODO: Benchmark other Sentinel-2 pre-trained weights available on TorchGeo: https://docs.torchgeo.org/en/stable/api/models.html#sentinel-2
    # and compare against randomly initialized weights
    # Also consider comparing with pre-trained ImageNet weights

    args = parser.parse_args()

    # Download, check and unpack the three splits into data/
    for split in ("train", "val", "test"):
        ds = Perda(root=f"{ROOT}/data", split=split, download=False)
        print(split, len(ds), "samples")

    dm = PerdaDataModule(
        root=f"{ROOT}/data",
        weights=args.pre_trained_weights,
        batch_size=8,
        num_workers=2,
    )

    task = SemanticSegmentation(
        model="unet",
        backbone="resnet18",
        weights=args.pre_trained_weights,
        in_channels=13,
        num_classes=len(Perda.classes),
        ignore_index=Perda.ignore_index,
        freeze_backbone=True,
    )

    filename = f"Unet_ResNet18_{args.pre_trained_weights}_weights" if args.pre_trained_weights else "Unet_ResNet18_random_weights"
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{ROOT}/models/semantic_segmentation",
        filename=filename,
        save_top_k=2,
        monitor="val_loss",
        mode="min",
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=False,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(task, datamodule=dm)
    trainer.test(task, datamodule=dm)


if __name__ == "__main__":
    main()