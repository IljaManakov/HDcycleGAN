from trainer import Trainer, events

print('initializing trainer...', end='')
trainer = Trainer.from_config_file('config.py')
sample = next(iter(trainer.dataloader))

trainer.register_event_handler(events.EACH_STEP, trainer, name='sample', interval=100, sample=sample)
trainer.monitor(name='criterion.discriminator_loss')
trainer.monitor(name='criterion.ln_generator_loss')
trainer.monitor(name='criterion.hn_generator_loss')
trainer.monitor(name='criterion.cycle_loss')
print('done!')

print('\ncommencing training!')
trainer.train(n_epochs=10)





