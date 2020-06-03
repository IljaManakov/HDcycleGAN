from trainer import Trainer, events, Config
from trainer.handlers import EventSave
from optimizer import OptimizerCollection
import torch as pt
from functools import partial
from itertools import chain

pt.cuda.set_device(0)

print('initializing trainer...', end='')
trainer = Trainer.from_config_file('config.py', False)

config = Config.from_file('config.py')

gen_optimizer = config.OPTIMIZER(chain(trainer.model.generator['hn'].parameters(),
                                       trainer.model.generator['ln'].parameters())
                                 , **config.optimizer)
disc_optimizer = config.OPTIMIZER(trainer.model.discriminator.parameters(), **config.optimizer)
optimizers = []
for optimizer in (disc_optimizer, gen_optimizer):
    if hasattr(config, 'APEX'):
        optimizer = config.APEX(optimizer, **config.apex)
    optimizers.append(optimizer)
trainer.optimizer = OptimizerCollection(*optimizers)
trainer.backward_pass = trainer.optimizer.backward

sample = next(iter(trainer.dataloader))[:4]


def sample_inference(trainer, part, sample, ind, *args, **kwargs):
    return part(trainer._transform(sample)[0][ind])


trainer.register_event_handler(events.EACH_STEP, sample_inference, name='gen_hn', interval=100, sample=sample,
                               part=trainer.model.generator['hn'], ind=0)
trainer.register_event_handler(events.EACH_STEP, sample_inference, name='gen_ln', interval=100, sample=sample,
                               part=trainer.model.generator['ln'], ind=1)
#trainer.register_event_handler(events.EACH_STEP, trainer, name='sample', interval=250, sample=sample)
trainer.register_event_handler(events.EACH_EPOCH, EventSave(), monitor=False)
#trainer.monitor(name='criterion.ln_discriminator_loss')
#trainer.monitor(name='criterion.hn_discriminator_loss')
trainer.monitor(name='criterion.discriminator_loss')
trainer.monitor(name='criterion.ln_generator_loss')
trainer.monitor(name='criterion.hn_generator_loss')
trainer.monitor(name='criterion.cycle_loss')
print('done!')

print('\ncommencing training!')
trainer.train(n_epochs=100, resume=True)





