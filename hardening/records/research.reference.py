"""Paper-level M4 tensor references, separate from generated code.
Inputs/parameters own their device; new allocations follow inputs. Executable fidelity gate:
validate-m4.ts + validate-m4.py (GPU required by default, explicit --device cpu only).
These formulas document the mathematical target; the gate compares full emitted graphs against
an independent TypeScript interpreter with aligned weights/inputs/noise, not training accuracy.
"""
import torch
import torch.nn.functional as F


def patchify(x, patch=2):
    b, c, h, w = x.shape
    assert h % patch == w % patch == 0
    # Explicit patch slicing is independent of the TENSA transpose sequence.
    return torch.stack([x[:, :, y:y+patch, z:z+patch].reshape(b, -1)
                        for y in range(0, h, patch) for z in range(0, w, patch)], dim=1)


def dense_block(x, weights):
    for w, b in weights:
        growth = F.relu(F.conv2d(x, w, b))
        x = torch.cat((x, growth), dim=1)
    return x


def vae_sample(mu, logvar, epsilon=None):
    epsilon = torch.randn_like(mu) if epsilon is None else epsilon
    return mu + (0.5 * logvar).exp() * epsilon


def elbo(reconstruction, x, mu, logvar, beta=0.01):
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
    return F.mse_loss(reconstruction, x) + beta * kl


def lora(x, base, down, up):
    return base(x) + 0.5 * (x @ down @ up)


def distill(student, teacher, label, temperature=2.0):
    soft = F.softmax(teacher.detach() / temperature, dim=-1)
    kd = -(soft * F.log_softmax(student / temperature, dim=-1)).sum(-1).mean()
    return 0.5 * temperature**2 * kd + 0.5 * F.cross_entropy(student, label)


def moco_logits_and_queue(query, key, bank, temperature=0.2):
    key = key.detach()
    logits = torch.cat(((query * key).sum(-1, keepdim=True), query @ bank.T), -1) / temperature
    next_bank = torch.cat((bank[key.shape[0]:], key), 0)
    return logits, next_bank


def gradient_penalty(critic, real, fake, mix):
    # Real input-gradient graph, deliberately retained for the outer critic derivative.
    xhat = (mix * real + (1 - mix) * fake.detach()).requires_grad_(True)
    score = critic(xhat)
    gx, = torch.autograd.grad(score.sum(), xhat, create_graph=True)
    return (gx.flatten(1).norm(dim=1) - 1).square().mean()


def diffuse(x, logit_alpha, epsilon=None):
    epsilon = torch.randn_like(x) if epsilon is None else epsilon
    alpha = logit_alpha.sigmoid()
    return alpha.sqrt() * x + (1-alpha).sqrt() * epsilon, epsilon


def actor_critic(policy, value, action, advantage, target):
    chosen = F.log_softmax(policy, -1).gather(-1, action[:, None]).squeeze(-1)
    return -(chosen * advantage.detach()).mean() + 0.5 * F.mse_loss(value, target)
