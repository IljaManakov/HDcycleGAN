def backward(loss, **kwargs):
    loss.backward(**kwargs)


class OptimizerCollection(object):

    def __init__(self, *optimizers):

        for optimizer in optimizers:
            if not hasattr(optimizer, 'backward'):
                setattr(optimizer, 'backward', backward)
        self.optimizers = optimizers

    def backward(self, losses):
        for loss, optimizer in zip(losses, self.optimizers):
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
