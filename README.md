# FedRDA: Hierarchical Noise Detection for Federated Finger Vein Recognition

This repository contains the implementation of **FedRDA**, a robust federated learning framework designed for finger vein recognition that effectively handles label noise in distributed environments.

## 🚀 Key Features

- **Hierarchical Noise Detection**: Two-stage framework identifying label noise at both client and sample levels
- **Dynamic Pseudo-Label Learning**: Adaptive label correction using improved Adam-RDA loss with uncertainty entropy guidance
- **Gradient-Aware Adaptive Aggregation**: Distance-aware and gradient consistency-based model aggregation strategy
- **Privacy Preservation**: Maintains user privacy while training on distributed finger vein data

## 📊 Performance

FedRDA achieves approximately **14% accuracy improvement** over existing methods under high noise rate conditions, while maintaining robust performance across various noise levels (5%-50%).

## 🛠️ Installation

```bash
git clone https://github.com/xieyuchen111/FedRDA.git
cd FedRDA
pip install -r requirements.txt
```

## 📖 Usage

### Basic Training
```bash
python main.py --dataset SDUMLA --noise_rate 0.2 --num_clients 20
```

### Configuration Options
- `--dataset`: Choose from SDUMLA, MMCBNU6000, FV-USM, or Joint
- `--noise_rate`: Label noise rate (0.05-0.5)
- `--num_clients`: Number of federated clients
- `--epochs`: Number of communication rounds

## 📁 Datasets

The framework supports the following finger vein datasets:
- **SDUMLA**: 636 classes, 3,816 samples
- **MMCBNU 6000**: 600 classes, 6,000 samples  
- **FV-USM**: 492 classes, 2,952 samples
- **Joint Datasets**: Combined dataset with 1,728 classes

## 🎯 Results

| Method | SDUMLA (20%) | MMCBNU (20%) | FV-USM (20%) | Joint (20%) |
|--------|--------------|--------------|--------------|-------------|
| FedAvg | 80.48% | 77.15% | 62.93% | 73.28% |
| FedProx | 81.24% | 78.93% | 64.52% | 75.84% |
| **FedRDA** | **87.65%** | **84.53%** | **72.13%** | **83.87%** |

## 📄 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{xie2023fedrda,
  title={FedRDA: Hierarchical Noise Detection for Federated Finger Vein Recognition},
  author={Xie, Yuchen and Ren, Hengyi and He, Hanyu and Fei, Shurui and Guo, Jian and Sun, Lijuan},
  journal={Journal of LaTeX Class Files},
  year={2023}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please contact:
- Yuchen Xie: xieyuchen@njfu.edu.cn

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
