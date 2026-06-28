# ResNet34-H95 Preprocessing Protocol

- Load the original image and convert it to grayscale uint8.
- Resize the grayscale uint8 image to 512x512 with bilinear interpolation for the full experiment.
- Compute the JPEG Q95 recompression residual from the resized uint8 grayscale image.
- Residual is `abs(resized_gray - jpeg_q95_resized_gray)` normalized by `percentile_99 + 1e-8` and clipped to `[0, 1]`.
- Resize masks to 512x512 with nearest-neighbor interpolation only.
- Final model input is `[2, 512, 512]`: channel 0 grayscale, channel 1 H95.
- Final mask is `[1, 512, 512]`.

Method name: grayscale + H95.
