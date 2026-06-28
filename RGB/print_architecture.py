import segmentation_models_pytorch as smp


def main() -> None:
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=3,
    )
    print(model)


if __name__ == "__main__":
    main()
