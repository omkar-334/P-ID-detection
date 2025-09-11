from trainer import Trainer
from utils import plot_training_curves


def train_task(
    task,
    wrappers,
    config,
    device,
    train_loaders,
    val_loaders,
    test_loaders,
):
    print(f"\n=== Training {task.upper()} model ({wrappers[task].model_type}) ===")
    trainer_task = Trainer(wrappers[task], device=device, lr=config["learning_rate"])
    trainer_task.train(
        train_loaders[task],
        val_loader=val_loaders[task],
        epochs=config["epochs"],
    )

    print(f"\nEvaluating {task.upper()} model on test set...")
    metrics_task = trainer_task.evaluate(
        test_loaders[task],
        iou_threshold=config["iou_threshold"],
        confidence_threshold=config["confidence_threshold"],
        class_names=wrappers[task].class_names,
    )

    print(f"\n--- {task.upper()} RESULTS ---")
    print(f"mAP: {metrics_task['mAP']:.4f}")
    print(f"Mean IoU: {metrics_task['mean_iou']:.4f}")
    print(f"Precision: {metrics_task['mean_precision']:.4f}")
    print(f"Recall: {metrics_task['mean_recall']:.4f}")
    print(f"F1-Score: {metrics_task['mean_f1']:.4f}")
    print(f"Inference Time: {metrics_task['avg_inference_time']:.4f} sec/image")

    plot_training_curves(trainer_task, save_path=f"models/training_curves_{task}.png")

    model_path_task = wrappers[task].save_model(
        config=config,
        metrics=metrics_task,
        name=f"{task}_final_model.pth",
    )
    print(f"{task.upper()} model saved as '{model_path_task}'")
    print(f"Training curves saved as 'models/training_curves_{task}.png'")

    return trainer_task, metrics_task
