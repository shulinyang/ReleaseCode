# pytorch-starter-kit
A demo project and quick starter kit for PyTorch.

## Structure

```
.
├── core
│   ├── __init__.py
│   ├── demo.py
│   ├── evaluate.py
│   └── train.py
├── datasets
│   └── __init__.py
├── external
├── models
│   └── __init__.py
├── config.py
├── init_path.py
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
└── utils.py
```

- `core`: core functions of project
  - `demo.py`: script for inference model on single image
  - `evaluate.py`: script for evaluating model on test set
  - `train.py`: script for training model on trainval set
- `datasets`: custom datasets and data loader
- `external`: external libraries
- `models`: custom network definition
- `config.py`: configurations of project
- `init_path.py`: script for inserting python path
- `LICENSE`: license file
- `main.py`: main script of project
- `README.md`: readme and documentation
- `requirements.txt`: python package requirements
- `utils.py`: utilities code
