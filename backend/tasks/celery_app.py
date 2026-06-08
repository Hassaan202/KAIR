from celery import Celery

celery_app = Celery(
    "suparco",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
    include=["tasks.preprocess_task", "tasks.training_task", "tasks.inference_task"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_redirect_stdouts=False,
)
