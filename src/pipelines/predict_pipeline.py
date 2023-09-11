from prefect import flow, task

from src.components.data_transformation import DataTransformation


@task
def data_transformation(result):
    """
    Task for data transformation.

    Args:
        result (dict): Dictionary containing online data.

    Returns:
        dict: Dictionary containing transformed train-test data.
    """
    online_data = result["online_data"]

    pred_data = DataTransformation().initiate_data_transformation(online_data)
    return {"pred_data": pred_data}


@task
def model_prediction(result):
    """
    Task for initiating model prediction using transformed data.

    Args:
        result (dict): Dictionary containing transformed prediction data.

    Returns:
        np.array: Array containing predictions of the model.
    """
    pred_data = result["pred_data"]

    return predictions


@flow(name="prediction_pipeline")
def prediction_pipeline():
    """
    Prefect flow that orchestrates the execution of data transformation,
    and model prediction tasks.
    """
    result_ingestion = data_ingestion()
    result_transformation = data_transformation(result=result_ingestion)
    model_training(result=result_transformation)


if __name__ == "__main__":
    retrain_pipeline()
