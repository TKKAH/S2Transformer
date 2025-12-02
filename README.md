# S$^2$Transformer

The code for paper: [S2Transformer: Scalable Structured Transformers for Global Station Weather Forecasting](https://arxiv.org/html/2509.19648v2):

- We propose a novel Spatial Structured Attention Block for GSWF, which not only perceives spatial structure but also considers both spatial proximity and global correlation.
- Building on the proposed block, we develop a multiscale GSWF model S$^2$Transformer by gradually increasing the subgraph scales. The resulting model is scalable and can produce structured spatial correlation.
- Our proposed method is effective yet easy to implement. We evaluate its efficacy and efficiency on global station weather datasets from medium to large sizes. **It can achieve performance improvements up to 16.8% over time series forecasting baselines while maintaining low running costs.**

## Get Started

1. Install Pytorch and necessary dependencies.
```
pip install -r requirements.txt
```
2. Download the dataset from [[Code Ocean]](https://codeocean.com/capsule/0341365/tree/v1). And place them under the `./dataset` folder.

3. Train and evaluate the model with the following scripts.

```shell
bash ./scripts/Global_Temp/S2Transformer.sh 
bash ./scripts/Global_Wind/S2Transformer.sh 
```

Note: Since the raw data for Global Temp and Global Wind from the NCEI has been multiplied by ten times, the actual MSE and MAE for these two benchmarks should be divided by 100 and 10 respectively.

## Citation

If you find this repo useful, please cite our paper.

```
@article{chen2025s,
  title={S $\^{} 2$ Transformer: Scalable Structured Transformers for Global Station Weather Forecasting},
  author={Chen, Hongyi and Li, Xiucheng and Chen, Xinyang and Cheng, Yun and Li, Jing and Chen, Kehai and Nie, Liqiang},
  journal={arXiv preprint arXiv:2509.19648},
  year={2025}
}
```

