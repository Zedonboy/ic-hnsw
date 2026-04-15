//! Public Candid-facing types — these are the shapes callers see.

use candid::{CandidType, Principal};
use serde::Deserialize;

pub type CollectionId = u32;
pub type NodeId = u64;

// ── Collection config ─────────────────────────────────────────────────────────

#[derive(CandidType, Deserialize, Clone, Debug)]
pub enum DistanceMetric {
    /// 1 − cosine_similarity. Best for text / embedding model output.
    Cosine,
    /// Euclidean (L2) distance. Good for image / dense feature vectors.
    Euclidean,
    /// Negative dot product. Use when vectors are already unit-normalised.
    DotProduct,
}

/// Parameters that control the HNSW graph structure.
///
/// Sensible defaults for ICP (instruction-budget conscious):
/// - m = 8, ef_construction = 100  → fast insert, good recall up to ~50k nodes
/// - m = 16, ef_construction = 200 → higher quality, slower insert
#[derive(CandidType, Deserialize, Clone, Debug)]
pub struct CollectionConfig {
    /// Human-readable name.
    pub name: String,
    /// Dimensionality of vectors in this collection. All inserted vectors must match.
    pub dim: u32,
    /// Max connections per node (except layer 0 which uses 2×m). Default 8.
    pub m: u8,
    /// Candidate pool size during index construction. Higher = better recall,
    /// more instructions per insert. Default 100.
    pub ef_construction: u16,
    pub distance: DistanceMetric,
}

// ── Access control ────────────────────────────────────────────────────────────

#[derive(CandidType, Deserialize, Clone, Debug, PartialEq)]
pub enum Role {
    /// Full control: configure, insert, delete, search, grant/revoke access.
    Owner,
    /// Can insert and delete vectors. Cannot manage access.
    Writer,
    /// Read-only: can search and inspect collection info.
    Reader,
}

#[derive(CandidType, Deserialize, Clone, Debug)]
pub struct AccessEntry {
    pub principal: Principal,
    pub role: Role,
}

// ── Insert / Search ───────────────────────────────────────────────────────────

#[derive(CandidType, Deserialize)]
pub struct InsertRequest {
    pub collection_id: CollectionId,
    pub vector: Vec<f32>,
    /// Arbitrary bytes the caller wants stored with this vector (JSON, Candid, etc.).
    pub metadata: Vec<u8>,
}

#[derive(CandidType, Deserialize)]
pub struct SearchRequest {
    pub collection_id: CollectionId,
    pub vector: Vec<f32>,
    pub top_k: u32,
    /// Candidate pool during search. Higher = better recall, more cycles.
    /// Must be ≥ top_k. Default 50.
    pub ef_search: u32,
}

#[derive(CandidType, Clone, Debug)]
pub struct SearchResult {
    pub id: NodeId,
    /// Distance to the query (lower = more similar).
    pub distance: f32,
    pub metadata: Vec<u8>,
}

// ── Collection info (read-only view) ─────────────────────────────────────────

#[derive(CandidType, Clone, Debug)]
pub struct CollectionInfo {
    pub id: CollectionId,
    pub name: String,
    pub dim: u32,
    pub m: u8,
    pub ef_construction: u16,
    pub distance: DistanceMetric,
    pub node_count: u64,
    pub max_layer: u8,
    pub owner: Principal,
}
