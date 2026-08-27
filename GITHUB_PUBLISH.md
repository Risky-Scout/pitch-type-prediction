# Publish to GitHub

No API or external data connection is required.

From the repository root:

```bash
git init
git add .
git commit -m "Pitch type probability model"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Before making the repository public, confirm with the recruiter whether the task dataset may be redistributed. This package intentionally excludes the raw CSV by default while including the fitted model, exact data SHA-256, split manifest, predictions, and all code needed to reproduce training once the supplied CSV is placed under `data/`.
