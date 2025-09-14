# DenseNet CIFAR-100 Implementation

A PyTorch implementation of various DenseNet architectures trained on the CIFAR-100 dataset, with extensive customization options for model architecture, hyperparameters, and training strategies.

## Features

- **Multiple DenseNet Variants**: DenseNet for CIFAR, DenseNet-121, DenseNet-169, DenseNet-201, and DenseNet-264
- **Flexible Configuration**: Adjustable growth rate, dropout rate, optimizer, and learning rate schedules
- **Comprehensive Training**: Data augmentation, model checkpointing, and training visualization
- **Easy Experimentation**: Command-line interface for configuring all aspects of training

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- matplotlib
- numpy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AestheticVoyager/densenet-pytorch.git
cd densenet-pytorch
```

2. Install required packages:
```bash
pip install torch torchvision matplotlib numpy
```

## Usage

### Basic Training

Train the default DenseNet model for CIFAR-100:
```bash
python densenet_cifar100.py
```

### Advanced Training Options

Train a specific DenseNet variant with custom parameters:
```bash
python densenet_cifar100.py --model 121 --growth_rate 32 --dropout 0.3 --epochs 200 --lr 0.05 --scheduler cosine
```

### All Command-line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--model` | DenseNet variant | `cifar` | `cifar`, `121`, `169`, `201`, `264` |
| `--growth_rate` | Growth rate for DenseNet | `12` | Integer value |
| `--dropout` | Dropout rate | `0.2` | Float between 0.0-1.0 |
| `--batch_size` | Batch size for training | `64` | Integer value |
| `--epochs` | Number of training epochs | `300` | Integer value |
| `--lr` | Initial learning rate | `0.1` | Float value |
| `--scheduler` | Learning rate scheduler | `multistep` | `multistep`, `cosine`, `plateau` |
| `--optimizer` | Optimizer | `sgd` | `sgd`, `adam` |
| `--weight_decay` | Weight decay | `1e-4` | Float value |
| `--momentum` | Momentum for SGD | `0.9` | Float value |

### Example Usage Scenarios

1. **Train DenseNet-121 with Adam optimizer:**
```bash
python densenet_cifar100.py --model 121 --optimizer adam --lr 0.001 --weight_decay 1e-5
```

2. **Train DenseNet-201 with high dropout and cosine annealing:**
```bash
python densenet_cifar100.py --model 201 --dropout 0.5 --scheduler cosine --epochs 400
```

3. **Train the original CIFAR DenseNet with reduced growth rate:**
```bash
python densenet_cifar100.py --model cifar --growth_rate 8 --batch_size 128
```

## Model Variants

This implementation supports the following DenseNet architectures:

1. **DenseNet-CIFAR**: The original configuration from the DenseNet paper for CIFAR datasets
2. **DenseNet-121**: 121-layer model with bottleneck layers
3. **DenseNet-169**: 169-layer model with bottleneck layers  
4. **DenseNet-201**: 201-layer model with bottleneck layers
5. **DenseNet-264**: 264-layer model with bottleneck layers

## Output Files

After training, the following files are generated:

- `densenet{model}_cifar100_best.pth`: Best model weights
- `training_curves_densenet{model}.png`: Training loss and accuracy curves
- `results_densenet{model}.txt`: Summary of training results and parameters

## Results

Expected performance on CIFAR-100:

| Model | Parameters | Top-1 Accuracy |
|-------|------------|----------------|
| DenseNet-CIFAR | ~0.8M | ~75-77% |
| DenseNet-121 | ~8.0M | ~76-78% |
| DenseNet-169 | ~14.2M | ~77-79% |
| DenseNet-201 | ~20.0M | ~77-79% |
| DenseNet-264 | ~33.3M | ~78-80% |

Note: Actual results may vary based on hyperparameter settings and training duration.

### Personal Results

These are the results I got on my setup with a single NVIDIA GeForce RTX 3080 GPU:

| Epoch | Train Loss | Train Acc (%) | Test Loss | Test Acc (%) | Learning Rate |
|-------|------------|---------------|-----------|--------------|---------------|
| 1     | 4.019      | 8.10          | 3.703     | 13.48        | 0.100000      |
| 10    | 2.018      | 45.44         | 1.895     | 49.02        | 0.100000      |
| 20    | 1.590      | 55.65         | 1.715     | 53.87        | 0.100000      |
| 30    | 1.427      | 59.51         | 1.501     | 59.03        | 0.100000      |
| 40    | 1.353      | 61.06         | 1.491     | 58.92        | 0.100000      |
| 50    | 1.315      | 62.36         | 1.355     | 61.64        | 0.100000      |
| 60    | 1.268      | 63.48         | 1.336     | 63.32        | 0.100000      |
| 70    | 1.241      | 64.05         | 1.410     | 61.37        | 0.100000      |
| 80    | 1.229      | 64.46         | 1.295     | 64.14        | 0.100000      |
| 90    | 1.217      | 64.69         | 1.291     | 64.43        | 0.100000      |
| 100   | 1.211      | 64.95         | 1.309     | 63.72        | 0.100000      |
| 110   | 1.189      | 65.54         | 1.255     | 64.96        | 0.100000      |
| 120   | 1.185      | 65.70         | 1.347     | 63.03        | 0.100000      |
| 130   | 1.178      | 65.88         | 1.256     | 65.18        | 0.100000      |
| 140   | 1.179      | 65.81         | 1.195     | 66.67        | 0.100000      |

- **Architecture**: DenseNet-CIFAR
- **Growth Rate**: 12
- **Dropout Rate**: 0.2
- **Parameters**: 814,644
- **Best Validation Accuracy**: 66.67% (at epoch 140)


## Customization

### Adding New Architectures

To add a new DenseNet variant, extend the model dictionary in the `main()` function:

```python
model_dict = {
    'cifar': densenet_cifar,
    '121': densenet121,
    '169': densenet169,
    '201': densenet201,
    '264': densenet264,
    'custom': densenet_custom  # Add your custom variant
}
```

### Modifying Data Augmentation

Edit the transform compositions in the `main()` function to modify data augmentation strategies:

```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # Add rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add color jitter
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{densenet-cifar100,
  author = {Your Name},
  title = {DenseNet CIFAR-100 Implementation},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/densenet-cifar100}}
}
```

## References

- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) - Original DenseNet paper
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Blog Post](https://aestheticvoyager.github.io/aesvoy/posts/DenseNet/)

## Troubleshooting

### Common Issues

1. **Out of Memory Error**: Reduce batch size or model size
2. **Slow Training**: Use a smaller model or enable GPU acceleration
3. **Poor Accuracy**: Try increasing epochs, adjusting learning rate, or using data augmentation


### Getting Help

If you encounter any problems:
1. Check that all dependencies are installed correctly
2. Ensure you have sufficient GPU memory for your selected batch size
3. Open an issue on GitHub with details about your problem

