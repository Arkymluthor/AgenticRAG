import logging
import json
import azure.functions as func
import azure.durable_functions as df

from .orchestrator import main as orchestrator_main

logger = logging.getLogger(__name__)


async def main(event: func.EventGridEvent, starter: str) -> func.HttpResponse:
    """
    Azure Function triggered by EventGrid events.
    This function starts the document ingestion orchestrator when a new blob is created.
    
    Args:
        event: The EventGrid event that triggered the function
        starter: The client binding for the Durable Functions starter
        
    Returns:
        HTTP response indicating the orchestration status
    """
    logger.info(f"EventGrid trigger processed an event: {event.event_type}")
    
    try:
        # Parse the event data
        event_data = event.get_json()
        logger.info(f"Event data: {json.dumps(event_data)}")
        
        # Check if this is a blob created event
        if event.event_type != "Microsoft.Storage.BlobCreated":
            logger.info(f"Ignoring event type: {event.event_type}")
            return func.HttpResponse(
                f"Event type {event.event_type} is not processed by this function",
                status_code=200
            )
        
        # Extract blob URI from the event
        blob_uri = event_data.get("url")
        if not blob_uri:
            logger.error("No blob URI found in event data")
            return func.HttpResponse(
                "No blob URI found in event data",
                status_code=400
            )
        
        # Start the orchestrator
        client = df.DurableOrchestrationClient(starter)
        instance_id = await client.start_new("DocIngestOrchestrator", None, blob_uri)
        
        logger.info(f"Started orchestration with ID = '{instance_id}' for blob {blob_uri}")
        
        # Return HTTP response with orchestration status
        return client.create_check_status_response(req=None, instance_id=instance_id)
        
    except Exception as e:
        logger.exception(f"Error processing event: {str(e)}")
        return func.HttpResponse(
            f"Error processing event: {str(e)}",
            status_code=500
        )


# Register the orchestrator function
df.app.register_orchestration_function(orchestrator_main)

# Register activity functions
df.app.register_activity_function("DetectDocumentType", "activities.detect_type.run")
df.app.register_activity_function("ExtractText", "activities.extract_text.run")
df.app.register_activity_function("ChunkText", "activities.chunk_text.run")
df.app.register_activity_function("EmbedChunks", "activities.embed_chunks.run")
df.app.register_activity_function("PushIndex", "activities.push_index.run")
