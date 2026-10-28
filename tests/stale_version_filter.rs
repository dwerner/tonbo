//! Regression: a value-predicate scan must never return a superseded MVCC
//! version, even after background compaction has rewritten segments.
//!
//! Bug (observed on a live handle): the read plan prunes/filters SST physical
//! rows with the *user* predicate before MVCC version resolution. When a key's
//! old version matches the predicate but the current version does not, the old
//! row survives pruning and is emitted — the current version is invisible
//! because it failed the pre-merge filter. A freshly-opened handle (replaying
//! the manifest without the live seal state) reads correctly, proving the
//! on-disk state is sound and the live read path is at fault.
use std::{sync::Arc, time::Duration};

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use fusio::disk::LocalFs;
use fusio::executor::tokio::TokioExecutor;
use tonbo::db::{BatchesThreshold, CompactionOptions, DB, DbBuilder, Expr, ScalarValue};
use typed_arrow_dyn::{DynCell, DynRow};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("v", DataType::Int32, false),
    ]))
}

fn row(id: impl Into<String>, v: i32) -> DynRow {
    DynRow(vec![
        Some(DynCell::Str(id.into())),
        Some(DynCell::I32(v)),
    ])
}

async fn make_db(dir: &std::path::Path) -> DB<LocalFs, TokioExecutor> {
    DbBuilder::from_schema_key_name(schema(), "id")
        .expect("config")
        .on_disk(dir)
        .expect("on disk")
        .with_commit_ack_mode(tonbo::db::CommitAckMode::Strict)
        .with_seal_policy(Arc::new(BatchesThreshold { batches: 1 }))
        .with_compaction_options(
            CompactionOptions::new().periodic_tick(Duration::from_millis(100)),
        )
        .with_minor_compaction(2, 0)
        .open()
        .await
        .expect("open")
}

fn extract(batches: &[RecordBatch]) -> Vec<(String, i32)> {
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("id col");
        let vals = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("v col");
        for (id, v) in ids.iter().zip(vals.iter()) {
            if let (Some(id), Some(v)) = (id, v) {
                rows.push((id.to_string(), v));
            }
        }
    }
    rows.sort();
    rows
}

async fn commit(db: &DB<LocalFs, TokioExecutor>, rows: Vec<DynRow>) {
    let mut tx = db.begin_transaction().await.expect("tx");
    for r in rows {
        tx.upsert(r).expect("upsert");
    }
    tx.commit().await.expect("commit");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn value_filter_never_resurrects_old_version_after_compaction() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let db = make_db(dir.path()).await;

    // Original versions.
    commit(&db, vec![row("a".to_string(), 1), row("b".to_string(), 5)]).await;
    // Fillers: force seals/flushes so the originals reach immutable segments.
    for i in 0..4 {
        commit(&db, vec![row(format!("f{i}"), 0)]).await;
    }
    // Update: a: 1->9, b: 5->7 (one txn — both must become the only visible version).
    commit(&db, vec![row("a".to_string(), 9), row("b".to_string(), 7)]).await;
    // More fillers so the compactor has post-update input too.
    for i in 0..4 {
        commit(&db, vec![row(format!("h{i}"), 0)]).await;
    }

    let expect_all = |rows: Vec<(String, i32)>, ctx: &str| {
        let mut want = vec![
            ("a".to_string(), 9),
            ("b".to_string(), 7),
            ("f0".to_string(), 0),
            ("f1".to_string(), 0),
            ("f2".to_string(), 0),
            ("f3".to_string(), 0),
            ("h0".to_string(), 0),
            ("h1".to_string(), 0),
            ("h2".to_string(), 0),
            ("h3".to_string(), 0),
        ];
        want.sort();
        assert_eq!(rows, want, "{ctx}: full scan mismatch");
    };

    // Sanity before the compaction worker runs.
    let batches = db.scan().collect().await.expect("scan");
    expect_all(extract(&batches), "pre-drain unfiltered");

    // Let the background compaction worker do its job.
    tokio::time::sleep(Duration::from_millis(2500)).await;

    // Unfiltered: correct on the live handle (versions dedup post-filter).
    let batches = db.scan().collect().await.expect("scan");
    expect_all(extract(&batches), "post-drain unfiltered");

    // Value predicates on OLD values must match nothing...
    let stale_a = db
        .scan()
        .filter(Expr::eq("v", ScalarValue::from(1i32)))
        .collect()
        .await
        .expect("scan stale a");
    let stale_b = db
        .scan()
        .filter(Expr::eq("v", ScalarValue::from(5i32)))
        .collect()
        .await
        .expect("scan stale b");
    assert_eq!(
        extract(&stale_a),
        Vec::<(String, i32)>::new(),
        "old version a(v=1) resurrected by value filter on live handle"
    );
    assert_eq!(
        extract(&stale_b),
        Vec::<(String, i32)>::new(),
        "old version b(v=5) resurrected by value filter on live handle"
    );

    // ...and predicates on CURRENT values must return the current row.
    let fresh = db
        .scan()
        .filter(Expr::eq("v", ScalarValue::from(7i32)))
        .collect()
        .await
        .expect("scan fresh b");
    assert_eq!(extract(&fresh), vec![("b".to_string(), 7)]);

    // Combined key+value predicate must agree with the key predicate alone
    // intersected with current values (the mark-filter shape users rely on).
    let a_live = db
        .scan()
        .filter(Expr::and(vec![
            Expr::eq("id", ScalarValue::from("a".to_string())),
            Expr::lt("v", ScalarValue::from(9i32)),
        ]))
        .collect()
        .await
        .expect("scan a live");
    assert_eq!(
        extract(&a_live),
        Vec::<(String, i32)>::new(),
        "key+value filter leaked a's old version"
    );
}
