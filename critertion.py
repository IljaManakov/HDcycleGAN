from collections import namedtuple

import torch as pt
from torch.nn import L1Loss, CrossEntropyLoss


class ClassificationLoss(object):

    def __init__(self, cycle_loss=L1Loss, discriminator_loss=CrossEntropyLoss, cycle_factor=10, mode='classification'):

        self.cycle_factor = cycle_factor
        self.cyc_loss = cycle_loss()
        self.disc_loss = discriminator_loss()
        self.mode = mode

    def __call__(self, prediction, sample):

        # convert components to namedtuples for easier handling
        images = namedtuple('images', ('hn', 'ln'))(*sample[0])
        targets = namedtuple('targets', ('hn', 'ln'))(*sample[1])
        cycled = namedtuple('cycled', ('hn', 'ln'))(*prediction.cycled)

        # reals are flipped due to architecture
        scores = namedtuple('scores', ('real', 'fake', 'pool_fake'))
        hn_scores = scores(prediction.ln_scores.real, prediction.hn_scores.fake, prediction.hn_scores.pool_fake)
        ln_scores = scores(prediction.hn_scores.real, prediction.ln_scores.fake, prediction.ln_scores.pool_fake)

        f_disc_hn, f_disc_ln, f_gen_hn, f_gen_ln = self.generate_fake_targets(hn_scores, ln_scores)
        loss = self.discriminator_loss(hn_scores, ln_scores, targets, f_disc_hn, f_disc_ln) + \
               self.ln_generator_loss(ln_scores, f_gen_ln) + self.hn_generator_loss(hn_scores, f_gen_hn) + \
               self.cycle_loss(images, cycled)

        return loss

    def generate_fake_targets(self, hn_scores, ln_scores):
        
        disc_hn = pt.zeros_like(hn_scores.pool_fake)
        disc_ln = pt.zeros_like(ln_scores.pool_fake)
        gen_hn = pt.ones(hn_scores.fake)
        gen_ln = pt.ones(ln_scores.fake)

        if self.mode == 'classification':
            batch_size = hn_scores.pool_fake.shape[0]
            disc_hn = pt.zeros(batch_size).to(hn_scores.pool_fake).long()
            disc_ln = pt.zeros(batch_size).to(ln_scores.pool_fake).long()
            gen_hn = pt.ones(batch_size).to(hn_scores.fake).long()
            gen_ln = 2*pt.ones(batch_size).to(ln_scores.fake).long()

        return disc_hn, disc_ln, gen_hn, gen_ln

    def discriminator_loss(self, hn_scores, ln_scores, targets, f_disc_hn, f_disc_ln):
        return self.disc_loss(hn_scores.real, targets.hn) + self.disc_loss(ln_scores.real, targets.ln) + \
               self.disc_loss(hn_scores.pool_fake, f_disc_hn) +  self.disc_loss(ln_scores.pool_fake, f_disc_ln)

    def ln_generator_loss(self, ln_scores, f_gen_ln):
        return self.disc_loss(ln_scores.fake, f_gen_ln)

    def hn_generator_loss(self, hn_scores, f_gen_hn):
        return self.disc_loss(hn_scores.fake, f_gen_hn)

    def cycle_loss(self, images, cycled):
        return self.cycle_factor * (self.cyc_loss(cycled.hn, images.hn) + self.cyc_loss(cycled.ln, images.ln))

