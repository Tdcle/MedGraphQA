from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from fastapi import APIRouter


router = APIRouter(tags=["metrics"])


@router.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

