# run_torchbench

This repo contains a list of stand alone scripts for running torchbench models one by one.

## Dependencies

### 1. Torch XLA2:

Please install following the instructions in 
https://github.com/pytorch/xla/tree/master/experimental/torch_xla2

### 2. TorchBench

**Important:** Please git clone torchbench in this directory and install from source

```bash
pip install torchvision torchaudio
git clone https://github.com/pytorch/benchmark.git
cd benchmark
python3 install.py
```

**NOTE:** `python install.py` will install ALL models. If you are only 
interested in running one model, you can install it via `python install.py model_name`
this way it will be much faster.

**NOTE:*** if you met error like `ERROR: Cannot install -r requirements.txt (line 12) because these package versions have conflicting dependencies...`, please try to reinstall env with:
```bash
pip3 uninstall torch torchvision torchaudio -y
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

### 3. Run one torch bench model under torch xla2

```python
python models/name_of_the_model.py
```


For example,
```python
python models/BERT_pytorch.py
```

The scripts are identical per model except the model name.
The goal is for people to add loggings and edit the script as they see fit.
And able to focus just one model.


