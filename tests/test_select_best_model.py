# tests/test_select_best_model.py
from unittest.mock import MagicMock, patch
import pytest

def test_select_best_model_promotes():
    # -----------------------------
    # Arrange mocks
    # -----------------------------
    mock_exp = MagicMock()
    mock_exp.experiment_id = "1"

    mock_run_best = MagicMock()
    mock_run_best.info.run_id = "run_123"

    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = mock_exp
    mock_client.search_runs.return_value = [mock_run_best]

    mock_registered = MagicMock()
    mock_registered.version = 1

    # Patch exactly where they are used
    with patch("src.models.select_best_model.MlflowClient", return_value=mock_client), \
         patch("src.models.select_best_model.mlflow.register_model", return_value=mock_registered) as mock_register:

        # Import function under test
        from src.models.select_best_model import select_best_model

        # -----------------------------
        # Act
        # -----------------------------
        run_id, version = select_best_model(metric="roc_auc", experiment_name="heart-disease")

        # -----------------------------
        # Assert returned values
        # -----------------------------
        assert run_id == "run_123"
        assert version == 1

        # -----------------------------
        # Assert Mlflow calls
        # -----------------------------
        mock_client.get_experiment_by_name.assert_called_once_with("heart-disease")
        mock_client.search_runs.assert_called_once_with(
            experiment_ids=[mock_exp.experiment_id],
            order_by=["metrics.roc_auc DESC"],
            max_results=1
        )
        mock_register.assert_called_once_with(model_uri="runs:/run_123/model", name="HeartDiseaseClassifier")
        mock_client.transition_model_version_stage.assert_called_once_with(
            name="HeartDiseaseClassifier",
            version=mock_registered.version,
            stage="Production",
            archive_existing_versions=True
        )
