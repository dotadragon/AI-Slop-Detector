from app.ml.image_detector import analyze_image
from app.tasks.celery_app import celery_app


@celery_app.task(name="slop.image.analyze")
def analyze_image_task(file_path: str) -> dict:
    return analyze_image(file_path)
