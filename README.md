# MIND Training Scripts

## Environment

Recommended PyTorch environment:

```powershell
conda activate mind_torch_gpu
```

## CROWN on MINDlarge

```powershell
python .\train_crown.py
```

Useful overrides:

```powershell
python .\train_crown.py --batch_size 16 --epoch 8
python .\train_crown.py --mode test --test_model_path .\best_model\large\CROWN-CROWN\#1\CROWN-CROWN
```

## LIME on MINDlarge

`train_lime.py` defaults to `--lifetime_type fixed` because the standard MIND release does not include the lifetime-augmented behavior fields required by the original research preprocessing.

```powershell
python .\train_lime.py
```

Useful overrides:

```powershell
python .\train_lime.py --batch_size 16 --epoch 8
python .\train_lime.py --mode test --test_model_path .\best_model\large\LIME-CROWN-CROWN\#1\LIME-CROWN-CROWN
```

## Outputs

- Validation metrics are written into `results/large/...`
- Raw ranking outputs are written into `test/res/large/...`
- Codabench submission artifacts are written into `prediction/large/.../#run_index/`
  - `prediction.txt`
  - `prediction.zip`
