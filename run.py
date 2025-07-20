import logging

import uvicorn

from main import app

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    logger.info("Starting FAST API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
