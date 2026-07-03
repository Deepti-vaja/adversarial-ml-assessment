# Experiment Run: fraud_cnn_baseline (AML-CNN-BASELINE)

**Run ID**: da2caaa6e9d64582a0eadcfbd6c59ce6  
**Experiment**: AML-CNN-BASELINE  
**Status**: FINISHED  

## Run Parameters

- **data..augmentation..color_jitter..brightness**: 0.2
- **data..augmentation..color_jitter..contrast**: 0.2
- **data..augmentation..color_jitter..hue**: 0.1
- **data..augmentation..color_jitter..saturation**: 0.2
- **data..augmentation..random_crop..padding**: 4
- **data..augmentation..random_crop..size**: 32
- **data..augmentation..random_horizontal_flip..p**: 0.5
- **data..batch_size**: 64
- **data..dataset_dir**: ./data/cifar10
- **data..num_workers**: 0
- **data..pin_memory**: True
- **data..val_ratio**: 0.1
- **model..dropout_conv**: 0.3
- **model..dropout_fc**: 0.5
- **model..in_channels**: 3
- **model..num_classes**: 3
- **seed**: 42
- **training..batch_size**: 64
- **training..checkpoint_dir**: ./models/checkpoints
- **training..checkpoint_name**: fraud_cnn_baseline.pth
- **training..epochs**: 1
- **training..learning_rate**: 0.001
- **training..optimizer**: AdamW
- **training..weight_decay**: 0.0001

## Run Metrics

- **clean_test_accuracy**: 0.735100
- **clean_test_loss**: 0.616576
- **train_acc**: 0.670422
- **train_loss**: 0.740512
- **training_overhead_hours**: 0.080141
- **val_acc**: 0.732400
- **val_loss**: 0.627595

## Run Tags

- **mlflow.runName**: fraud_cnn_baseline
- **mlflow.source.name**: C:\Users\dell\Desktop\New project\adversarial-ml-assessment\training\train_baseline.py
- **mlflow.source.type**: LOCAL
- **mlflow.user**: dell