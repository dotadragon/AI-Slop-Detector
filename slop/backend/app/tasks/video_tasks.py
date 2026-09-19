from app.ml.video_detector import analyze_video
from app.tasks.celery_app import celery_app


@celery_app.task(name="slop.video.analyze")
def analyze_video_task(file_path: str, job_id: str) -> dict:
    return analyze_video(file_path, job_id)
