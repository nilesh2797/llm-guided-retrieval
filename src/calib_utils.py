import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def get_bimodal_gmm_intrsxn(theta, plot=False):
    gmm = GaussianMixture(n_components=2)
    theta_reshaped = theta.reshape(-1, 1)
    gmm.fit(theta_reshaped)

    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_

    x = np.linspace(min(theta), max(theta), 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    responsibilities = gmm.predict_proba(x)
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    if plot:
        plt.hist(theta, bins=30, density=True, alpha=0.5, color='gray')
        plt.plot(x, pdf, '-k', label='GMM Total')
        plt.plot(x, pdf_individual, '--', label='GMM Components')
        plt.title('GMM Fit to Theta Distribution')
        plt.xlabel('Theta')
        plt.ylabel('Density')

    # Highlight the intersection point of the two components in the GMM total i.e. the minimum point between the two peaks
    # find the xrange between the two peaks
    x = x.flatten()
    x_range = x[(x >= min(means)) & (x <= max(means))]
    y_range = pdf[(x >= min(means)) & (x <= max(means))]
    if y_range.size == 0:
        return 0.0
    min_idx = np.argmin(y_range)
    intersection_x = x_range[min_idx]
    if plot:
        intersection_y = y_range[min_idx]
        plt.plot(intersection_x, intersection_y, 'ro', label='Intersection Point')
        plt.axvline(intersection_x, color='r', linestyle='--')
        plt.text(intersection_x, intersection_y, f'  Intersection\n  ({intersection_x:.2f}, {intersection_y:.2f})', color='r')

    # get minimum bound of 80% confidence interval of GMM with highest mean
    gmm_idx = np.argmax(means)
    mean = means[gmm_idx][0]
    std_dev = np.sqrt(covariances[gmm_idx][0][0])
    conf_int_lower = mean - 1.28 * std_dev  # 80%
    if plot:
        plt.axvline(conf_int_lower, color='g', linestyle='--', label='80% CI Lower Bound')
        plt.text(conf_int_lower, 0, f'  80% CI Lower\n  Bound ({conf_int_lower:.2f})', color='g', rotation=90, verticalalignment='bottom')
        plt.legend()
        plt.show()
    return intersection_x # max(conf_int_lower, intersection_x)

class PLLinearPriorModel(nn.Module):
    def __init__(self, total_N, num_slates, tau=5.0, lambda_mse=0.5):
        super().__init__()
        self.total_N = total_N
        self.num_slates = num_slates
        self.tau = tau
        self.lambda_mse = lambda_mse

        # Parameters
        self.theta = nn.Parameter(torch.randn(total_N) * 0.01)
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b_s = nn.Parameter(torch.zeros(num_slates))

    def forward(self, slates, scores, lens):
        """
        slates: (S, maxK) long
        scores: (S, maxK) float
        lens: (S,) long
        """
        S, maxK = slates.shape
        # gather theta for all slates
        t = self.theta[slates]*self.tau  # (S, maxK)

        # mask out padding
        mask = torch.arange(maxK, device=slates.device)[None, :] < lens[:, None]  # (S, maxK)
        t_masked = t.masked_fill(~mask, float("-inf"))  # so exp(-inf)=0

        # compute cumulative logsumexp from right to left for each slate
        # trick: flip and do cumsum in exp-space
        exp_t = torch.exp(t_masked)
        # reverse along dim=1 for cumsum
        rev_exp = torch.flip(exp_t, dims=[1])
        rev_cumsum = torch.cumsum(rev_exp, dim=1)
        cumexp = torch.flip(rev_cumsum, dims=[1])  # (S, maxK)
        logcumexp = torch.log(cumexp + 1e-12)

        # loglikelihood per slate:
        # sum of t for actual K minus sum of logcumexp
        ll = (t * mask).sum(dim=1) - (logcumexp * mask).sum(dim=1)
        nll = -ll.mean()  # average across slates

        # MSE prior loss:
        # predicted scores = a * t + b_s[s] for each slate s
        b_per_point = self.b_s[:, None].expand_as(t)
        pred = self.a * t + b_per_point
        sc = scores * self.tau
        weights = torch.clamp(1/(1+torch.exp(-1*(scores-0.5))), min=0.1)  # weight more for higher scores
        mse_loss = (((pred - sc) ** 2) * weights)[mask].sum() / weights[mask].sum()  # average over all valid points

        return (1-self.lambda_mse)*nll + self.lambda_mse * mse_loss, nll, mse_loss

class CalibModel:
    def __init__(self,
                 total_N,
                 tau=5.0,
                 lr=1e-2,
                 weight_decay=1e-4,
                 maxiter=200,
                 clip_grad=1.0,
                 lambda_mse=1.0):
        self.total_N = total_N
        self.tau = tau
        self.lr = lr
        self.weight_decay = weight_decay
        self.maxiter = maxiter
        self.clip_grad = clip_grad
        self.lambda_mse = lambda_mse
        self.slates = []
        self.scores = []
        self.model = None
        self.trained = False
        # we keep full-space theta here:
        self.theta_full = np.full((total_N,), float(0.0))

    def add(self, relevance_scores):
        if isinstance(relevance_scores, dict):
            relevance_scores = sorted(list(relevance_scores.items()), key=lambda x: x[1], reverse=True)
        elif isinstance(relevance_scores, (list, tuple)) and len(relevance_scores) > 0 and isinstance(relevance_scores[0], (tuple, list)) and len(relevance_scores[0]) == 2:
            relevance_scores = sorted(relevance_scores, key=lambda x: x[1], reverse=True)
        else:
            raise ValueError('relevance_scores should be a dict or list of (idx, score) pairs')
        
        slate, scores = list(zip(*relevance_scores))
        if not all([score <= 1.001 and score >= -0.001 for score in scores]):
            print(f'Warning: Scores should be in [0, 1]: {scores}')
        self.slates.append(slate)
        self.scores.append(scores)
        self.trained = False

    def fit(self, verbose=False):
        # Build mapping from original indices to compact 0..M-1
        new_mapping = {}
        for slate in self.slates:
            for idx in slate:
                if idx not in new_mapping:
                    new_mapping[idx] = len(new_mapping)

        remap_slates = [[new_mapping[idx] for idx in slate] for slate in self.slates]
        M = len(new_mapping)
        if M == 0:
            return

        # Build tensors
        maxK = max(len(s) for s in remap_slates)
        S = len(remap_slates)
        slates_tensor = torch.zeros((S, maxK), dtype=torch.long)
        scores_tensor = torch.zeros((S, maxK), dtype=torch.float)
        lens_tensor = torch.zeros(S, dtype=torch.long)
        for i, (slate, sc) in enumerate(zip(remap_slates, self.scores)):
            lens_tensor[i] = len(slate)
            slates_tensor[i, :len(slate)] = torch.tensor(slate, dtype=torch.long)
            scores_tensor[i, :len(sc)] = torch.tensor(sc, dtype=torch.float)

        # Initialise model with compact size M
        self.model = PLLinearPriorModel(M, S, tau=self.tau, lambda_mse=self.lambda_mse)
        # self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        slates_tensor = slates_tensor.to(self.model.theta.device)
        scores_tensor = scores_tensor.to(self.model.theta.device)
        lens_tensor = lens_tensor.to(self.model.theta.device)
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=self.lr,
                                weight_decay=self.weight_decay)

        for it in range(self.maxiter):
            optimizer.zero_grad()
            loss, nll_loss, mse_loss = self.model(slates_tensor, scores_tensor, lens_tensor)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            optimizer.step()
            if verbose and (it % 50 == 0 or it == self.maxiter - 1):
                print(f"Iter {it}, loss={loss.item():.4f}, nll={nll_loss.item():.4f}, mse={mse_loss.item():.4f}")

        # write back into full theta space
        theta_compact = self.model.theta.detach().cpu()
        full_theta = torch.full((self.total_N,), float(0.0))
        for orig_idx, compact_idx in new_mapping.items():
            full_theta[orig_idx] = theta_compact[compact_idx]
        self.theta_full = full_theta.numpy() * self.a
        self.theta_full /= self.theta_full.max()  # normalize to max 1.0
        self.new_mapping = new_mapping  # keep for reference
        self.trained = True

    @property
    def theta(self):
        return self.theta_full

    @property
    def a(self):
        return self.model.a.detach().cpu().item() if self.model is not None else getattr(self, 'loaded_a', 1.0)

    @property
    def b_s(self):
        return self.model.b_s.detach().cpu().numpy() if self.model is not None else getattr(self, 'loaded_b_s', np.zeros(len(self.slates)))

    def to_dict(self):
        return {
            'total_N': self.total_N,
            'tau': self.tau,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'maxiter': self.maxiter,
            'clip_grad': self.clip_grad,
            'lambda_mse': self.lambda_mse,
            'slates': self.slates,
            'scores': self.scores,
        }

    def load_dict(self, d):
        self.__dict__.update({k: v for k, v in d.items() if k not in ['theta', 'b_s']})
        if 'slates' in d and 'scores' in d:
            self.fit()  # refit model to get theta, a, b_s
        return self