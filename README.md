# CrossCert
## About

This is the code for CrossCert.



## Requirement

We test the code on python==3.8.16, torch==2.0.1,timm==0.6.13



## How to use

First, users need to train a masking-basd recovery (PC) following train_model.py and a voting-based one following train_drs.py. Users can also download the corresponding checkpoint of PC from https://drive.google.com/drive/folders/1Ewks-NgJHDlpeAaGInz_jZ6iczcYNDlN (Thanks for PC's good open source.) and put it into ./checkpoints or set by users. Here is an example:

```python
python train_drs.py --model vit_base_patch16_224 --dataset cifar100 --ablation_size 19
```

After trainning, users need to calculate the prediction results of mutants, where pc_certification.py for masking-based and certification_drs.py for voting-based. Here is an example:

```python
python pc_certification.py --model vit_base_patch16_224_cutout2_128 --dataset cifar100 --num_mask 6 --patch_size 39
```

```python
python certification_drs.py --model vit_base_patch16_224_cifar100_drs_19 --dataset cifar100 --ablation_size 19
```

Finally, calculate the final result of CrossCert by CrossCert_sta.py. Both CrossCert and CrossCert-base would be calculated by the script. Here is an example:

```python
python -u CrossCert_sta.py --dataset cifar100 --patch_size 39 --ablation_size 19
```



## Acknowledgment

This implementation is partly based on https://github.com/inspire-group/PatchCleanser/tree/main
