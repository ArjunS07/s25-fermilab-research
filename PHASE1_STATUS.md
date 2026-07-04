# Phase 1 Grid — Status

**BLOCKED (2.5h+) — cluster-side storage provisioner outage, not fixable from our side.
RECOMMEND ESCALATING TO NRP SUPPORT — this will not self-resolve from anything we can do.**
All 6 jobs launched but none has started training. Root cause identified: the
`rook-system.cephfs.csi.ceph.com` provisioner (`csi-cephfsplugin-provisioner-5d568d65cc-9brbk`)
is timing out / stuck on every volume operation cluster-wide, not just contention on our
specific PVC. A brand-new PVC (`as-jet-train-pvc-2`) hit the identical failure signature,
which rules out "stale lock on the old volume" as the cause. Holding off on killing/recreating
jobs until the provisioner recovers — doing so now would just reproduce the same stuck state
with nothing running in the meantime.

## Timeline of diagnosis
1. All 6 phase1 pods (a-f) stuck in `ContainerCreating`/`Pending`, all failing to mount the
   shared PVC `as-jet-train-pvc` (PV `pvc-ca6e3f8b-...`, confirmed Bound, RWX rook-cephfs —
   not an RWO conflict).
2. Mount events on every pod: first `MountVolume.MountDevice failed ... DeadlineExceeded`,
   then repeated `... an operation with the given Volume ID ... already exists` (CSI driver
   stuck retrying against its own in-flight-operation lock).
3. Checked for zombie pods from prior runs holding the mount — none found; all pods on the
   PVC are these 6 jobs.
4. Created a new PVC (`as-jet-train-pvc-2`, same spec: 1Ti RWX rook-cephfs) and repointed all
   6 YAMLs' `claimName` at it, to sidestep a possible stale per-volume-ID lock.
5. New PVC never bound — `ProvisioningFailed: rpc error: code = DeadlineExceeded` followed by
   `code = Aborted: an operation with the given Volume ID pvc-8c277bd0-... already exists`,
   the *same* failure pattern on a brand-new volume ID. Confirms this is the provisioner
   itself, not anything specific to the old PVC.

## What's needed
This requires NRP/Nautilus platform admin intervention (restart of the stuck
`csi-cephfsplugin-provisioner` pod, or whatever is causing it to time out on every op).
We have no RBAC access to the `rook`/`rook-system` namespaces to fix this ourselves
(`kubectl auth can-i` confirmed no list/delete access there). Recommend filing a report with
NRP support referencing:
- provisioner pod: `csi-cephfsplugin-provisioner-5d568d65cc-9brbk`
- affected PVCs: `as-jet-train-pvc` (PV `pvc-ca6e3f8b-9181-4366-a747-020e4def5845`,
  underlying stuck volume ID `0001-0004-rook-0000000000000001-43f75f00-8e11-4f09-9823-6ea62d1ba849`)
  and `as-jet-train-pvc-2` (stuck volume ID `pvc-8c277bd0-6a36-42ee-b774-63fa753795c2`)
- namespace: `cms-ml`

## Other notes
- kubectl throws intermittent TLS cert warnings against the API server VIP
  (`x509: certificate is valid for 10.96.0.1, ... not 67.58.53.147`) — mostly benign/retriable,
  but did cause one hard failure on `kubectl create` (worked on retry).
- GIT_COMMIT pinned to `a8f5d55fe2600c63c65bdbc2b8812b11645206ea` for all 6 jobs — this part
  is not in question, unrelated to the storage issue.
- All 6 phase1 YAMLs currently point at `as-jet-train-pvc-2` (the new, also-stuck PVC). If we
  decide to revert to the original PVC while waiting on admins, that's a one-line edit back
  per file.

| run | pod | state | latest loss | issues | artifacts |
|---|---|---|---|---|---|
| a | as-jet-train-job-g30-phase1-a-jwpxw | ContainerCreating | — | PVC mount stuck (provisioner outage) | — |
| b | as-jet-train-job-g30-phase1-b-7m95v | ContainerCreating | — | PVC mount stuck (provisioner outage) | — |
| c | as-jet-train-job-g30-phase1-c-6nq4b | ContainerCreating | — | PVC mount stuck (provisioner outage) | — |
| d | as-jet-train-job-g30-phase1-d-m87fw | ContainerCreating | — | PVC mount stuck (provisioner outage) | — |
| e | as-jet-train-job-g30-phase1-e-tp8vt | ContainerCreating | — | PVC mount stuck (provisioner outage) | — |
| f | as-jet-train-job-g30-phase1-f-zpslp | Pending | — | scheduling + PVC mount stuck | — |

_Note: pods a-f above are still running against the OLD PVC (`as-jet-train-pvc`) — we have
NOT deleted/recreated them yet, since the new PVC also failed to bind. Nothing has been
killed._

## Check-in log
- ~04:26 (30-min check): no change. All 6 pods still non-Running (a/b/c/d/e in
  ContainerCreating, f Pending), same `DeadlineExceeded` mount events on the old PVC.
  New PVC `as-jet-train-pvc-2` still `Pending` after 21+ min — provisioner outage ongoing,
  unresolved on the platform side. No logs to tail (nothing Running), no terminal states,
  nothing relaunched. Continuing 30-min monitor.
- ~04:38 (30-min check): no real change. `kubectl get pods -l job-name=...-d` transiently
  returned empty (flaky API server VIP backend) — re-verified pod d (`...-d-m87fw`) still
  exists and is `ContainerCreating`, same stuck mount. All 6 still non-Running (a/b/c/d/e
  ContainerCreating, f Pending). New PVC `as-jet-train-pvc-2` still `Pending` after 53+ min.
  No logs to tail, no terminal states, nothing relaunched/killed. Continuing 30-min monitor.
- ~05:10 (30-min check): no change. All 6 still non-Running (a/b/c/d/e ContainerCreating,
  f Pending), ages now 100-105min. New PVC `as-jet-train-pvc-2` still `Pending` after 89+
  min. Provisioner outage still unresolved. No logs to tail, no terminal states, nothing
  relaunched/killed. Continuing 30-min monitor. Outage has now persisted 1.5+ hours —
  may be worth escalating to NRP support if it drags on much longer.
- ~05:45 (30-min check): no change. All 6 still non-Running (a/b/c/d/e ContainerCreating,
  f Pending), ages now 131-136min (2h11m-2h16m). New PVC `as-jet-train-pvc-2` still
  `Pending` after 120min (2h). **Outage has now crossed 2 hours with zero movement on
  either PVC** — recommending escalation to NRP support (see top of file). No logs to
  tail, no terminal states, nothing relaunched/killed. Continuing 30-min monitor pending
  user decision on escalation.
- ~06:17 (30-min check): pod f cleared node scheduling (was blocked on node taints/resource
  availability, unrelated to storage) and moved from `Pending` to `ContainerCreating` — but
  then immediately hit the identical stuck-mount failure (`Aborted: operation with the given
  Volume ID ... already exists`) as a/b/c/d/e. So: not real progress on the actual blocker.
  All 6 now non-Running via the same storage failure (a/b/c/d/e/f all ContainerCreating).
  Both PVCs still stuck (old Bound-but-unmountable, new still Pending, 153+ min). **Outage
  at 2.5+ hours, zero platform-side movement. Standing recommendation: escalate to NRP
  support — nothing left to try from our side.** No logs to tail, no terminal states,
  nothing relaunched/killed. Continuing 30-min monitor.
- ~06:50 (30-min check): no change. All 6 still non-Running (ContainerCreating), ages now
  3h16m-3h21m. New PVC `as-jet-train-pvc-2` still `Pending` at 3h5m. **Outage now at 3+
  hours, zero platform-side movement. Standing recommendation to escalate to NRP support
  still unconfirmed as acted on.** No logs to tail, no terminal states, nothing
  relaunched/killed. Continuing 30-min monitor.
- ~07:22 (30-min check): no change. All 6 still non-Running (ContainerCreating), ages now
  3h48m-3h53m. New PVC `as-jet-train-pvc-2` still `Pending` at 3h37m. **Outage now at
  ~3.9 hours, zero platform-side movement. Standing escalation recommendation to NRP
  support still unconfirmed as acted on.** No logs to tail, no terminal states, nothing
  relaunched/killed. Continuing 30-min monitor.
- ~07:54 (30-min check): no change. All 6 still non-Running (ContainerCreating), ages now
  4h20m-4h25m. New PVC `as-jet-train-pvc-2` still `Pending` at 4h9m. **Outage now at ~4.4
  hours, zero platform-side movement. Standing escalation recommendation to NRP support
  still unconfirmed as acted on.** No logs to tail, no terminal states, nothing
  relaunched/killed. Continuing 30-min monitor.
- ~08:25 (30-min check): no change. All 6 still non-Running (ContainerCreating), ages now
  4h51m-4h56m. New PVC `as-jet-train-pvc-2` still `Pending` at 4h40m. **Outage now at ~5
  hours, zero platform-side movement. Standing escalation recommendation to NRP support
  still unconfirmed as acted on.** No logs to tail, no terminal states, nothing
  relaunched/killed. Continuing 30-min monitor.
- ~08:56 (30-min check): no change. All 6 still non-Running (ContainerCreating), ages now
  5h22m-5h27m. New PVC `as-jet-train-pvc-2` still `Pending` at 5h11m. **Outage now at ~5.5
  hours, zero platform-side movement. Standing escalation recommendation to NRP support
  still unconfirmed as acted on.** No logs to tail, no terminal states, nothing
  relaunched/killed. Continuing 30-min monitor.
- ~09:28 (30-min check): no change. All 6 still non-Running (ContainerCreating), ages now
  5h54m-5h59m. New PVC `as-jet-train-pvc-2` still `Pending` at 5h43m. **Outage now at ~6
  hours, zero platform-side movement. Standing escalation recommendation to NRP support
  still unconfirmed as acted on.** No logs to tail, no terminal states, nothing
  relaunched/killed. Continuing 30-min monitor.
- ~09:59 (30-min check): no change. All 6 still non-Running (ContainerCreating), ages now
  6h25m-6h30m. New PVC `as-jet-train-pvc-2` still `Pending` at 6h14m. **Outage now at ~6.5
  hours, zero platform-side movement. Standing escalation recommendation to NRP support
  still unconfirmed as acted on.** No logs to tail, no terminal states, nothing
  relaunched/killed. Continuing 30-min monitor.
- ~10:31 (30-min check): no change. Pod a transiently missing from list query (flaky API
  VIP, same known issue) — re-verified via exact lookup + job status, still exists and
  `ContainerCreating` at 7h3m. All 6 still non-Running (ContainerCreating), ages now
  6h57m-7h3m. New PVC `as-jet-train-pvc-2` still `Pending` at 6h46m. **Outage now at ~7
  hours, zero platform-side movement. Standing escalation recommendation to NRP support
  still unconfirmed as acted on.** No logs to tail, no terminal states, nothing
  relaunched/killed. Continuing 30-min monitor.
- ~11:03 (30-min check): no change. All 6 still non-Running (ContainerCreating), ages now
  7h29m-7h34m. New PVC `as-jet-train-pvc-2` still `Pending` at 7h18m. **Outage now at ~7.5
  hours, zero platform-side movement. Standing escalation recommendation to NRP support
  still unconfirmed as acted on.** No logs to tail, no terminal states, nothing
  relaunched/killed. Continuing 30-min monitor.
- ~11:35 (30-min check): no change. All 6 still non-Running (ContainerCreating), ages now
  flat 8h. New PVC `as-jet-train-pvc-2` still `Pending` at 7h50m. **Outage now at 8 hours,
  zero platform-side movement. Standing escalation recommendation to NRP support still
  unconfirmed as acted on.** No logs to tail, no terminal states, nothing relaunched/killed.
  Continuing 30-min monitor.
- ~12:07 (30-min check): no change. All 6 still non-Running (ContainerCreating), ages now
  flat 8h. New PVC `as-jet-train-pvc-2` still `Pending`, also at 8h. **Outage now at 8.5
  hours, zero platform-side movement. Standing escalation recommendation to NRP support
  still unconfirmed as acted on.** No logs to tail, no terminal states, nothing
  relaunched/killed. Continuing 30-min monitor.
- ~12:39 (30-min check): no change. All 6 still non-Running (ContainerCreating), ages now
  flat 9h. New PVC `as-jet-train-pvc-2` still `Pending`, at 8h. **Outage now at 9 hours,
  zero platform-side movement. Standing escalation recommendation to NRP support still
  unconfirmed as acted on.** No logs to tail, no terminal states, nothing relaunched/killed.
  Continuing 30-min monitor.
- ~13:11 (30-min check): no change. All 6 still non-Running (ContainerCreating), ages now
  flat 9h. New PVC `as-jet-train-pvc-2` still `Pending`, also flat 9h. **Outage now at 9.5
  hours, zero platform-side movement. Standing escalation recommendation to NRP support
  still unconfirmed as acted on.** No logs to tail, no terminal states, nothing
  relaunched/killed. Continuing 30-min monitor.

_Last updated: 2026-07-04 (~13:11)_
