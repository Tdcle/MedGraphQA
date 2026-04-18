from fastapi import APIRouter, Depends

from app.api.deps import get_container
from app.core.container import ServiceContainer

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health(container: ServiceContainer = Depends(get_container)):
    return {
        "status": "ok",
        "neo4j_connected": container.graph_service.ping(),
        "postgres_connected": container.entity_repository.ping(),
        "elasticsearch_connected": container.entity_search_index.ping(),
    }
