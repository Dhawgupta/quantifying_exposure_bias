import logging

def setup_logger():
    """Sets up the logger at test.log

    Returns:
        logging.Logger: The logger object
    """    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("test.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    logger.info("<New run>")
    return logger

logger = setup_logger()