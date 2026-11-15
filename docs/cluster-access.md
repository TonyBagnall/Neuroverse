# Cluster access rules

Confirmed by Tony on10November2026. Applies to Iridis6, IridisX and Hali.

- No external network calls originating from a cluster. This includes package
  managers (pip/conda), git network operations, curl/wget, dataset downloads,
  model-weight downloads, telemetry and automatic update checks. Tony performs
  any necessary external calls himself.
- Access only Tony's own file space and queues/jobs. Do not inspect other users'
  files, jobs or logs. Shared paths require explicit authorisation.
- Heavy computation must run on allocated compute nodes through Slurm, never
  on login nodes. This includes training, substantial preprocessing, large data
  audits and analysis. Login nodes are for lightweight navigation and scheduler
  control/inspection, subject to the cluster's rules.
- Southampton VPN is used for Iridis6/IridisX; Hali requires UEA VPN.

The locally inspected farm-management system is documented in
[tsml-eval farm management](tsml-eval-farm-management.md). Its existing defaults
do not override these rules, particularly automatic email or network activity.

The10November connection test authenticated to Iridis6 and IridisX using
existing SSH keys. Hali timed out on the Southampton VPN. No remote commands
or shells were requested, and the test connections were closed.

These rules do not themselves authorise job submission, cancellation, file
transfers or remote changes. Follow the scope of the current user request.
