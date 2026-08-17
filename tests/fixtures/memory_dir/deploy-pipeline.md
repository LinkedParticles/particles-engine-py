# Deploy pipeline

2024-03-02

- The staging deployment runs on Jenkins; the `staging-deploy` Jenkins job builds and ships it.
- Production deploys require a signed release tag before the pipeline will run.
