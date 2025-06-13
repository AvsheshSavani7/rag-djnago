import requests
from django.conf import settings
import logging
from .models import ApiRequestLog

logger = logging.getLogger(__name__)


def call_node_api(endpoint, method='GET', data=None, params=None):
    """
    Utility function to call the Node.js API

    Args:
        endpoint (str): API endpoint (without base URL)
        method (str): HTTP method (GET, POST, etc.)
        data (dict): Data to be sent in the request body (for POST, PUT, etc.)
        params (dict): URL parameters (for GET requests)

    Returns:
        dict: Response from the Node.js API
    """

    url = f"{settings.NODE_API_BASE_URL}/{endpoint.lstrip('/')}"
    headers = {
        'Content-Type': 'application/json',
    }

    logger.info(f"Making {method} request to {url}")
    if data:
        logger.info(f"Request data: {data}")
    if params:
        logger.info(f"Request params: {params}")

    log_entry_created = False
    try:
        if method.upper() == 'GET':
            response = requests.get(url, params=params, headers=headers)
        elif method.upper() == 'POST':
            logger.info(f"Sending POST request with data: {data}")
            response = requests.post(url, json=data, headers=headers)
            logger.info(f"Response status code1: {response.status_code}")
            logger.info(f"Response headers: {response.headers}")
            try:
                logger.info(f"Response content: {response.json()}")
            except:
                logger.info(f"Raw response content: {response.content}")
        elif method.upper() == 'PUT':
            response = requests.put(url, json=data, headers=headers)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()  # This will raise an exception for 4XX/5XX status codes
        response_data = response.json()

        # ✅ Log the request using MongoEngine
        try:
            ApiRequestLog(
                endpoint=endpoint,
                method=method.upper(),
                status_code=response.status_code,
                request_data=data if data else params,
                response_data=response_data
            ).save()
            log_entry_created = True
        except Exception as log_error:
            logger.error(f"Error logging API request: {log_error}")

        return response_data

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Node.js API: {e}")
        logger.error(f"Request URL: {url}")
        logger.error(f"Request method: {method}")
        logger.error(f"Request data: {data}")
        logger.error(f"Request params: {params}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response content: {e.response.content}")

        # Log error only if no previous log was created
        if not log_entry_created:
            try:
                ApiRequestLog(
                    endpoint=endpoint,
                    method=method.upper(),
                    status_code=getattr(e.response, 'status_code', 500),
                    request_data=data if data else params,
                    response_data={'error': str(
                        e.response.content.decode('utf-8'))}
                ).save()
            except Exception as log_error:
                logger.error(f"Error logging failed API request: {log_error}")

        raise Exception(f"Error communicating with Node.js API: {str(e)}")
