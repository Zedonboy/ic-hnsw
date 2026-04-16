//! `ic_hnsw` — HNSW vector search for ICP canisters.
//!
//! # Usage modes
//!
//! ## As a library embedded in your own canister
//!
//! Add the crate to your `Cargo.toml` and call the macro once at the root of
//! your canister's `lib.rs`:
//!
//! ```ignore
//! // Cargo.toml
//! [dependencies]
//! ic_hnsw   = { git = "..." }
//! ic-cdk    = "0.17"
//! candid    = "0.10"
//!
//! [lib]
//! crate-type = ["cdylib", "rlib"]
//! ```
//!
//! ```ignore
//! // src/lib.rs
//! ic_hnsw::export_hnsw!();   // inlines all HNSW Candid endpoints
//!
//! #[ic_cdk::update]
//! fn my_other_method() { ... }
//!
//! ic_cdk::export_candid!();
//! ```
//!
//! You can also bypass the macro and call the API directly from your own
//! update/query handlers:
//!
//! ```ignore
//! #[ic_cdk::update]
//! fn my_insert(req: ic_hnsw::types::InsertRequest)
//!     -> ic_hnsw::error::Result<ic_hnsw::types::NodeId>
//! {
//!     // custom pre-processing …
//!     ic_hnsw::api::insert(req)
//! }
//! ```
//!
//! ## As a standalone canister
//!
//! Deploy the pre-built `cdylib` directly.  All endpoints are already wired:
//!
//! ```shell
//! dfx deploy ic_hnsw
//! ```
//!
//! All state is stored in thread-local `StableBTreeMap`s and survives upgrades
//! without any extra work from the caller.

pub mod api;
pub mod error;
pub mod hnsw;
pub mod storage;
pub mod types;

// ── Macro: inline all endpoints into the calling canister ─────────────────────

/// Inline all HNSW Candid endpoints as `#[update]`/`#[query]` functions.
///
/// Invoke this **once** at the top level of your canister's `lib.rs`, then
/// follow it with `ic_cdk::export_candid!()` to emit the `.did` file.
///
/// ```ignore
/// ic_hnsw::export_hnsw!();
///
/// // your own endpoints …
///
/// ic_cdk::export_candid!();
/// ```
///
/// `ic-cdk` and `candid` must be in your `[dependencies]`
/// (they are already transitive deps of `ic_hnsw`).
#[macro_export]
macro_rules! export_hnsw {
    () => {
        #[::ic_cdk::update]
        fn create_collection(
            config: $crate::types::CollectionConfig,
        ) -> $crate::error::Result<$crate::types::CollectionId> {
            $crate::api::create_collection(config)
        }

        #[::ic_cdk::update]
        fn delete_collection(
            id: $crate::types::CollectionId,
        ) -> $crate::error::Result<()> {
            $crate::api::delete_collection(id)
        }

        #[::ic_cdk::query]
        fn collection_info(
            id: $crate::types::CollectionId,
        ) -> $crate::error::Result<$crate::types::CollectionInfo> {
            $crate::api::collection_info(id)
        }

        #[::ic_cdk::update]
        fn grant_access(
            collection_id: $crate::types::CollectionId,
            principal:     ::candid::Principal,
            role:          $crate::types::Role,
        ) -> $crate::error::Result<()> {
            $crate::api::grant_access(collection_id, principal, role)
        }

        #[::ic_cdk::update]
        fn revoke_access(
            collection_id: $crate::types::CollectionId,
            principal:     ::candid::Principal,
        ) -> $crate::error::Result<()> {
            $crate::api::revoke_access(collection_id, principal)
        }

        #[::ic_cdk::query]
        fn list_access(
            collection_id: $crate::types::CollectionId,
        ) -> $crate::error::Result<::std::vec::Vec<$crate::types::AccessEntry>> {
            $crate::api::list_access(collection_id)
        }

        #[::ic_cdk::update]
        fn transfer_ownership(
            collection_id: $crate::types::CollectionId,
            new_owner:     ::candid::Principal,
        ) -> $crate::error::Result<()> {
            $crate::api::transfer_ownership(collection_id, new_owner)
        }

        #[::ic_cdk::update]
        fn insert(
            req: $crate::types::InsertRequest,
        ) -> $crate::error::Result<$crate::types::NodeId> {
            $crate::api::insert(req)
        }

        #[::ic_cdk::update]
        fn delete_node(
            collection_id: $crate::types::CollectionId,
            node_id:       $crate::types::NodeId,
        ) -> $crate::error::Result<()> {
            $crate::api::delete_node(collection_id, node_id)
        }

        #[::ic_cdk::query]
        fn search(
            req: $crate::types::SearchRequest,
        ) -> $crate::error::Result<::std::vec::Vec<$crate::types::SearchResult>> {
            $crate::api::search(req)
        }

        #[::ic_cdk::update]
        fn set_controller_only(enabled: bool) -> $crate::error::Result<()> {
            $crate::api::set_controller_only(enabled)
        }

        #[::ic_cdk::query]
        fn get_controller_only() -> bool {
            $crate::api::get_controller_only_setting()
        }
    };
}

// ── Standalone canister wiring ────────────────────────────────────────────────
// When this crate is compiled as a cdylib the endpoints below turn it into a
// fully-functional standalone canister.  Library users never see these — they
// generate their own wrappers via export_hnsw!() in their own crate root.

#[ic_cdk::update]
fn create_collection(
    config: types::CollectionConfig,
) -> error::Result<types::CollectionId> {
    api::create_collection(config)
}

#[ic_cdk::update]
fn delete_collection(id: types::CollectionId) -> error::Result<()> {
    api::delete_collection(id)
}

#[ic_cdk::query]
fn collection_info(
    id: types::CollectionId,
) -> error::Result<types::CollectionInfo> {
    api::collection_info(id)
}

#[ic_cdk::update]
fn grant_access(
    collection_id: types::CollectionId,
    principal:     candid::Principal,
    role:          types::Role,
) -> error::Result<()> {
    api::grant_access(collection_id, principal, role)
}

#[ic_cdk::update]
fn revoke_access(
    collection_id: types::CollectionId,
    principal:     candid::Principal,
) -> error::Result<()> {
    api::revoke_access(collection_id, principal)
}

#[ic_cdk::query]
fn list_access(collection_id: types::CollectionId) -> error::Result<Vec<types::AccessEntry>> {
    api::list_access(collection_id)
}

#[ic_cdk::update]
fn transfer_ownership(
    collection_id: types::CollectionId,
    new_owner:     candid::Principal,
) -> error::Result<()> {
    api::transfer_ownership(collection_id, new_owner)
}

#[ic_cdk::update]
fn insert(req: types::InsertRequest) -> error::Result<types::NodeId> {
    api::insert(req)
}

#[ic_cdk::update]
fn delete_node(
    collection_id: types::CollectionId,
    node_id:       types::NodeId,
) -> error::Result<()> {
    api::delete_node(collection_id, node_id)
}

#[ic_cdk::query]
fn search(req: types::SearchRequest) -> error::Result<Vec<types::SearchResult>> {
    api::search(req)
}

#[ic_cdk::update]
fn set_controller_only(enabled: bool) -> error::Result<()> {
    api::set_controller_only(enabled)
}

#[ic_cdk::query]
fn get_controller_only() -> bool {
    api::get_controller_only_setting()
}

ic_cdk::export_candid!();
