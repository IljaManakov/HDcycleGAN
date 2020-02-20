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

        disc_hn_loss = self.hn_discriminator_loss(hn_scores)
        disc_ln_loss =  self.ln_discriminator_loss(ln_scores)
        gen_loss = self.ln_generator_loss(ln_scores) + self.hn_generator_loss(hn_scores) + self.cycle_loss(images, cycled)

        return disc_hn_loss, disc_ln_loss, gen_loss

    def hn_discriminator_loss(self, hn_scores):
        return self.disc_loss(hn_scores.real, self.create_target(hn_scores.real, 1)) +\
               self.disc_loss(hn_scores.pool_fake, self.create_target(hn_scores.pool_fake, 0))

    def ln_discriminator_loss(self, ln_scores):
        return self.disc_loss(ln_scores.real, self.create_target(ln_scores.real, 1)) +\
               self.disc_loss(ln_scores.pool_fake, self.create_target(ln_scores.pool_fake, 0))

    def ln_generator_loss(self, ln_scores):
        return self.disc_loss(ln_scores.fake, self.create_target(ln_scores.fake, 1))

    def hn_generator_loss(self, hn_scores):
        return self.disc_loss(hn_scores.fake, self.create_target(hn_scores.fake, 1))

    def cycle_loss(self, images, cycled):
        return self.cycle_factor * (self.cyc_loss(cycled.hn.float(), images.hn) +
                                    self.cyc_loss(cycled.ln.float(), images.ln))

    def create_target(self, tensor, value):
        if value:
            return value*pt.ones_like(tensor)
        else:
            return pt.zeros_like(tensor)


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

        # f_disc_hn, f_disc_ln, f_gen_hn, f_gen_ln = self.generate_fake_targets(hn_scores, ln_scores)
        fake_t, hn_t, ln_t = self.generate_fake_targets(hn_scores)
        disc_loss = self.discriminator_loss(hn_scores, ln_scores, fake_t, hn_t, ln_t)
        gen_loss = self.ln_generator_loss(ln_scores, ln_t) + self.hn_generator_loss(hn_scores, hn_t)
        cyc_loss = self.cycle_loss(images, cycled)

        return 0.5*disc_loss, gen_loss + cyc_loss

    def generate_fake_targets(self, hn_scores):

        template = hn_scores.pool_fake.float()
        batch_size = template.shape[0]
        # disc_hn = pt.zeros(batch_size).to(hn_scores.pool_fake).long()
        # disc_ln = 2*pt.zeros(batch_size).to(ln_scores.pool_fake).long()
        # gen_hn = pt.ones(batch_size).to(hn_scores.fake).long()
        # gen_ln = 2*pt.ones(batch_size).to(ln_scores.fake).long()
        fake_t = pt.tensor([[1, 0, 0]]*batch_size).to(template)
        hn_t = pt.tensor([[0, 1, 0]]*batch_size).to(template)
        ln_t = pt.tensor([[0, 0, 1]]*batch_size).to(template)

        return fake_t, hn_t, ln_t

    def discriminator_loss(self, hn_scores, ln_scores, fake_t, hn_t, ln_t):
        return self.disc_loss(hn_scores.real.float(), hn_t) +\
               self.disc_loss(ln_scores.real.float(), ln_t) + \
               self.disc_loss(hn_scores.pool_fake.float(), fake_t) +\
               self.disc_loss(ln_scores.pool_fake.float(), fake_t)

    def ln_generator_loss(self, ln_scores, f_gen_ln):
        return self.disc_loss(ln_scores.fake.float(), f_gen_ln)

    def hn_generator_loss(self, hn_scores, f_gen_hn):
        return self.disc_loss(hn_scores.fake.float(), f_gen_hn)

    def cycle_loss(self, images, cycled):
        return self.cycle_factor * (self.cyc_loss(cycled.hn.float(), images.hn.float()) +
                                    self.cyc_loss(cycled.ln.float(), images.ln.float()))

