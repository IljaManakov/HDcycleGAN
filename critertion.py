"""
Copyright 2019 Ilja Manakov

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


from collections import namedtuple

import torch as pt
from torch.nn import L1Loss, CrossEntropyLoss


class ClassificationLoss(object):

    def __init__(self, cycle_loss=L1Loss, discriminator_loss=CrossEntropyLoss, cycle_factor=10):

        self.cycle_factor = cycle_factor
        self.cyc_loss = cycle_loss()
        self.disc_loss = discriminator_loss()

    def __call__(self, prediction, sample):

        # convert components to namedtuples for easier handling
        images = namedtuple('images', ('hn', 'ln'))(*sample[0])
        cycled = namedtuple('cycled', ('hn', 'ln'))(*prediction.cycled)

        # reals are flipped due to architecture
        scores = namedtuple('scores', ('real', 'fake', 'pool_fake'))
        hn_scores = scores(prediction.ln_scores.real.float(),
                           prediction.hn_scores.fake.float(),
                           prediction.hn_scores.pool_fake.float())
        ln_scores = scores(prediction.hn_scores.real.float(),
                           prediction.ln_scores.fake.float(),
                           prediction.ln_scores.pool_fake.float())

        loss = self.hn_discriminator_loss(hn_scores) + self.ln_discriminator_loss(ln_scores) +\
               self.ln_generator_loss(ln_scores) + self.hn_generator_loss(hn_scores) + \
               self.cycle_loss(images, cycled)

        return loss

    def generate_fake_targets(self, hn_scores, ln_scores):

        batch_size = hn_scores.pool_fake.shape[0]
        disc_hn = pt.zeros(batch_size).to(hn_scores.pool_fake).long()
        disc_ln = pt.zeros(batch_size).to(ln_scores.pool_fake).long()
        gen_hn = pt.ones(batch_size).to(hn_scores.fake).long()
        gen_ln = 2*pt.ones(batch_size).to(ln_scores.fake).long()

        return disc_hn, disc_ln, gen_hn, gen_ln

    def hn_discriminator_loss(self, hn_scores):
        return self.disc_loss(hn_scores.real, pt.ones_like(hn_scores.real)) +\
               self.disc_loss(hn_scores.pool_fake, pt.zeros_like(hn_scores.pool_fake))

    def ln_discriminator_loss(self, ln_scores):
        return self.disc_loss(ln_scores.real, pt.ones_like(ln_scores.real)) +\
               self.disc_loss(ln_scores.pool_fake, pt.zeros_like(ln_scores.pool_fake))

    def ln_generator_loss(self, ln_scores):
        return self.disc_loss(ln_scores.fake, pt.ones_like(ln_scores.fake))

    def hn_generator_loss(self, hn_scores):
        return self.disc_loss(hn_scores.fake, pt.ones_like(hn_scores.fake))

    def cycle_loss(self, images, cycled):
        return self.cycle_factor * (self.cyc_loss(cycled.hn.float(), images.hn) + self.cyc_loss(cycled.ln.float(), images.ln))


class ClassificationLossHD(object):

    def __init__(self, cycle_loss=L1Loss, discriminator_loss=CrossEntropyLoss, cycle_factor=10):

        self.cycle_factor = cycle_factor
        self.cyc_loss = cycle_loss()
        self.disc_loss = discriminator_loss()

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

        batch_size = hn_scores.pool_fake.shape[0]
        disc_hn = pt.zeros(batch_size).to(hn_scores.pool_fake).long()
        disc_ln = pt.zeros(batch_size).to(ln_scores.pool_fake).long()
        gen_hn = pt.ones(batch_size).to(hn_scores.fake).long()
        gen_ln = 2*pt.ones(batch_size).to(ln_scores.fake).long()

        return disc_hn, disc_ln, gen_hn, gen_ln

    def discriminator_loss(self, hn_scores, ln_scores, targets, f_disc_hn, f_disc_ln):
        return self.disc_loss(hn_scores.real.float(), targets.hn.view(-1)) +\
               self.disc_loss(ln_scores.real.float(), targets.ln.view(-1)) + \
               self.disc_loss(hn_scores.pool_fake.float(), f_disc_hn) +\
               self.disc_loss(ln_scores.pool_fake.float(), f_disc_ln)

    def ln_generator_loss(self, ln_scores, f_gen_ln):
        return self.disc_loss(ln_scores.fake.float(), f_gen_ln)

    def hn_generator_loss(self, hn_scores, f_gen_hn):
        return self.disc_loss(hn_scores.fake.float(), f_gen_hn)

    def cycle_loss(self, images, cycled):
        return self.cycle_factor * (self.cyc_loss(cycled.hn.float(), images.hn.float()) +
                                    self.cyc_loss(cycled.ln.float(), images.ln.float()))

