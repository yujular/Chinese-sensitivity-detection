import logging


def epoch_log(epoch, data, logger):
    """Log epoch information.
    Args:
        epoch: int, epoch number
        data: dict, including 'acc', 'f1'
        logger: logger object
    """
    # Logging
    logger.info(','.join([str(epoch)] + [str(s) for s in data]))


def step_log(step, loss, logger):
    """Log step information.
    Args:
        step: int, step number
        loss: float, loss value
        logger: logger object
    """

    # Logging
    logger.info(','.join([str(step), str(loss.item())]))


def get_csv_logger(log_file_name,
                   title='',
                   log_format='%(message)s',
                   log_level=logging.INFO):
    """Get csv logger.

    Args:
        log_file_name: file name
        title: first line in file
        log_format: default: '%(message)s'
        log_level: default: logging.INFO

    Returns:
        csv logger
    """
    logger = logging.getLogger(log_file_name)
    logger.setLevel(log_level)
    file_handler = logging.FileHandler(log_file_name, 'w')
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    if title:
        logger.info(title)
    return logger
