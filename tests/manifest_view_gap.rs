//! Reproduction: background compaction races a single **sequential** writer.
//!
//! Run with:
//! ```text
//! cargo test --features tokio --test manifest_view_gap -- --ignored --nocapture
//! ```
//!
//! Under seal-every-commit + minor compaction (threshold 1, tick 250 ms) a
//! lone writer — no concurrent callers, no tombstones, no mixed write shapes,
//! key-only predicates throughout — intermittently observes two defects, both
//! traceable to the compaction worker's manifest churn:
//!
//! 1. `Manifest(Backend(PreconditionFailed))` from a plain
//!    `Transaction::commit().await` under `CommitAckMode::Strict` +
//!    `WalSyncPolicy::Always`: the commit's manifest CAS races the compaction
//!    worker's concurrent manifest edit and the conflict surfaces to the
//!    *user* rather than being retried internally.
//!
//! 2. A strictly-acked row briefly **disappearing** from the committing
//!    handle's own scans — two `db.scan()` calls issued one statement apart
//!    disagree. The `remove_ssts` + `add_ssts` apply sequence is observable
//!    as an intermediate generation in which the merged segment is absent
//!    from the live view, so rows vanish for that generation. A freshly
//!    opened handle reads the same store correctly, so the on-disk state is
//!    sound and it is the live view transition that carries the gap.
//!
//! Both classes are orthogonal to the stale-version-predicate bug fixed on
//! this branch by `deb56d3` (see `tests/stale_version_filter.rs`): they are
//! visibility/CAS races, not MVCC filter races. A long-lived handle — which
//! is how an embedded engine has to hold the DB — observes both under any
//! compaction cadence; larger `minor_segment_threshold` / `periodic_tick`
//! shrink the window but do not close it.
use std::{sync::Arc, time::Duration};

use tonbo::db::wal_tuning::WalConfigExt;
use tonbo::db::{BatchesThreshold, CommitAckMode, CompactionOptions, DbBuilder, WalConfig, WalSyncPolicy};
use tonbo::prelude::*;

#[derive(Record, Debug, Clone)]
struct R {
    #[metadata(k = "tonbo.key", v = "true")]
    key: String,
    deleted_at: i64,
}

fn rows(rs: Vec<R>) -> tonbo::prelude::arrow_array::RecordBatch {
    let mut builders = <R>::new_builders(rs.len());
    builders.append_rows(rs);
    builders.finish().into_record_batch()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "repro: background compaction races sequential commits (see module docs)"]
async fn strict_commits_race_background_compaction() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let db = DbBuilder::from_schema(<R>::schema())
        .expect("schema")
        .on_disk(dir.path().join("db"))
        .expect("on disk")
        .with_commit_ack_mode(CommitAckMode::Strict)
        .wal_config(WalConfig::default().sync_policy(WalSyncPolicy::Always))
        .with_seal_policy(Arc::new(BatchesThreshold { batches: 1 }))
        .with_compaction_options(
            CompactionOptions::new().periodic_tick(Duration::from_millis(250)),
        )
        .with_minor_compaction(1, 0)
        .open()
        .await
        .expect("open");

    let rounds = 60;
    let mut commit_failures = 0usize;
    let mut visibility_misses = 0usize;

    for i in 0..rounds {
        let mut tx = db.begin_transaction().await.expect("tx");
        tx.upsert_batch(&rows(vec![R {
            key: "s".into(),
            deleted_at: 0,
        }]))
        .expect("upsert");
        if let Err(err) = tx.commit().await {
            commit_failures += 1;
            println!("COMMIT-FAIL i={i}: {err:?}");
            break;
        }

        let limited = db
            .scan()
            .filter(Expr::eq("key", ScalarValue::from("s".to_string())))
            .limit(1)
            .collect()
            .await
            .expect("scan limited");
        let plain = db
            .scan()
            .filter(Expr::eq("key", ScalarValue::from("s".to_string())))
            .collect()
            .await
            .expect("scan plain");
        let lc: usize = limited.iter().map(|b| b.num_rows()).sum();
        let pc: usize = plain.iter().map(|b| b.num_rows()).sum();
        if lc == 0 || pc == 0 {
            visibility_misses += 1;
            println!("VISIBILITY-MISS i={i} limited={lc} plain={pc}");
        }
    }

    println!(
        "SUMMARY rounds={rounds} commit_failures={commit_failures} \
         visibility_misses={visibility_misses}"
    );
    assert_eq!(
        commit_failures, 0,
        "a strict commit must not surface the compaction worker's manifest CAS conflict"
    );
    assert_eq!(
        visibility_misses, 0,
        "a strictly-acked row must be visible to the committing handle's next scan"
    );
}
