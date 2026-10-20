# SNS_recommend

Post recommendation for a social network service using **collaborative filtering on implicit feedback (likes)**, implemented in PyTorch on GPU. Personal project, 2021.

## Approach

- **Input:** like logs (`account`, `post`, timestamp) → binary user × post interaction matrix (`preprocess.ipynb`)
- **Model:** matrix factorization for implicit feedback — likes are treated as preference with confidence weighting, and user/post latent factors are optimized with L2 regularization (`CF.ipynb`; latent dim 200, λ = 40)
- **Output:** predicted preference score for every (user, post) pair → top-N unseen posts per user (`application.ipynb`)

Running the factorization on GPU lets the full interaction matrix be processed in mini-batches.

## Files

| File | Purpose |
|---|---|
| `preprocess.ipynb` | Build the interaction matrix from `likes.csv` → `like_p.csv` |
| `CF.ipynb` | Train the factorization model, export `predict.csv` |
| `application.ipynb` | Pick recommendations per user from predicted scores |
| `test.ipynb` | Sanity checks |
| `likes.csv` | Anonymized like log (opaque IDs only) |

## Notes / limitations
- Likes are a weak signal; richer implicit feedback (watch time, ratings) would improve accuracy.
- GPU is required as written (`torch.device("cuda")`).
