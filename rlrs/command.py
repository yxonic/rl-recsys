import fret


@fret.command
def train(ws, args):
    logger = ws.logger('train')
    trainer = ws.build_module()
    logger.info("[%s] %s", ws, trainer)
