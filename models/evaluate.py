"""
Evaluation module for Alzheimer's detection models.
"""

from monai.engines import SupervisedEvaluator
from monai.handlers import StatsHandler, CheckpointLoader
from monai.metrics import ROCAUCMetric
from monai.inferers import SimpleInferer


def evaluate_model(model, test_loader, device):
    """Evaluate the model using MONAI SupervisedEvaluator."""
    metric_name = "ROCAUC"
    metric = ROCAUCMetric()

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=model,
        inferer=SimpleInferer(),
        postprocessing=None,
        metrics=[metric],
        metric_name=metric_name,
    )

    stats_handler = StatsHandler(output_transform=lambda x: None)
    checkpoint_handler = CheckpointLoader(load_path="./checkpoints/best_model.pt", load_dict={"model": model})

    evaluator.add_event_handler(event_name=evaluator.EVENT_EPOCH, handler=stats_handler)
    evaluator.add_event_handler(event_name=evaluator.EVENT_EPOCH, handler=checkpoint_handler)

    evaluator.run()
    return evaluator.state.metrics
