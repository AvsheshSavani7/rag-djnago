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

    log_entry_created = False
    try:
        if method.upper() == 'GET':
            response = requests.get(url, params=params, headers=headers)
        elif method.upper() == 'POST':
            response = requests.post(url, json=data, headers=headers)
        elif method.upper() == 'PUT':
            response = requests.put(url, json=data, headers=headers)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

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

        response.raise_for_status()
        return response_data

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Node.js API: {e}")

        # Log error only if no previous log was created
        if not log_entry_created:
            try:
                ApiRequestLog(
                    endpoint=endpoint,
                    method=method.upper(),
                    status_code=getattr(e.response, 'status_code', 500),
                    request_data=data if data else params,
                    response_data={'error': str(e)}
                ).save()
            except Exception as log_error:
                logger.error(f"Error logging failed API request: {log_error}")

        raise Exception(f"Error communicating with Node.js API: {str(e)}")
