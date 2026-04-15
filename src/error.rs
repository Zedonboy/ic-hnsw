//! Canister error type.

use candid::CandidType;
use serde::Deserialize;

#[derive(CandidType, Deserialize, Clone, Debug)]
pub enum HnswError {
    /// Caller does not have the required role.
    Unauthorized,
    /// Referenced collection does not exist.
    CollectionNotFound(u32),
    /// The collection already exists with that name.
    CollectionAlreadyExists,
    /// Vector dimensionality does not match the collection.
    DimensionMismatch { expected: u32, got: u32 },
    /// Referenced node does not exist.
    NodeNotFound(u64),
    /// ef_search must be ≥ top_k.
    InvalidSearchParams,
    /// m must be in 2..=64; ef_construction must be ≥ m.
    InvalidConfig(String),
    /// Internal storage error (should never happen in practice).
    Internal(String),
}

impl core::fmt::Display for HnswError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub type Result<T> = core::result::Result<T, HnswError>;
