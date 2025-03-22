## Datasets

You can download the nearest neighbor datasets used in the examples from Hugging Face:

- [nearest-neighbors-datasets](https://huggingface.co/datasets/habedi/nearest-neighbors-datasets)

Make sure that the downloaded datasets are stored in the `example/data/nearest-neighbors-datasets` directory.

### Using Hugging Face CLI Client

You can use [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli) to download the data:

```shell
huggingface-cli download habedi/nearest-neighbors-datasets --repo-type dataset --local-dir nearest-neighbors-datasets
```

The command must be run in `example/data` directory.
