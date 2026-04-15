//! Stable-memory layout and `Storable` implementations.
//!
//! Memory map (managed by `MemoryManager`):
//!
//! | Memory ID | Contents |
//! |-----------|----------|
//! | 0         | `StableBTreeMap<u32, StoredCollection>` — collection metadata |
//! | 1         | `StableBTreeMap<NodeKey, StoredNode>` — vectors + HNSW edges |

use candid::{CandidType, Principal, decode_one, encode_one};
use ic_stable_structures::{
    DefaultMemoryImpl, StableBTreeMap, Storable,
    memory_manager::{MemoryId, MemoryManager, VirtualMemory},
    storable::Bound,
};
use serde::Deserialize;
use std::{borrow::Cow, cell::RefCell};

use crate::types::{CollectionId, DistanceMetric, NodeId, Role};

type Mem = VirtualMemory<DefaultMemoryImpl>;

// ── Stable collections ────────────────────────────────────────────────────────

/// Everything persisted for one collection.
#[derive(CandidType, Deserialize, Clone)]
pub struct StoredCollection {
    pub name: String,
    pub dim: u32,
    /// Connections per node above layer 0.
    pub m: u8,
    /// Connections at layer 0 (= 2 × m).
    pub m_max0: u8,
    pub ef_construction: u16,
    /// 0 = Cosine, 1 = Euclidean, 2 = DotProduct
    pub distance: u8,
    /// Entry-point node at the highest layer (None when empty).
    pub entry_point: Option<NodeId>,
    pub max_layer: u8,
    pub next_node_id: NodeId,
    pub node_count: u64,
    /// LCG state for random layer assignment.
    pub rng_state: u64,
    pub owner: Principal,
    pub writers: Vec<Principal>,
    pub readers: Vec<Principal>,
}

impl StoredCollection {
    pub fn distance_metric(&self) -> DistanceMetric {
        match self.distance {
            1 => DistanceMetric::Euclidean,
            2 => DistanceMetric::DotProduct,
            _ => DistanceMetric::Cosine,
        }
    }

    pub fn role_of(&self, caller: &Principal) -> Option<Role> {
        if &self.owner == caller {
            return Some(Role::Owner);
        }
        if self.writers.contains(caller) {
            return Some(Role::Writer);
        }
        if self.readers.contains(caller) {
            return Some(Role::Reader);
        }
        None
    }
}

impl Storable for StoredCollection {
    fn to_bytes(&self) -> Cow<[u8]> {
        Cow::Owned(encode_one(self).expect("encode StoredCollection"))
    }
    fn from_bytes(bytes: Cow<[u8]>) -> Self {
        decode_one(&bytes).expect("decode StoredCollection")
    }
    const BOUND: Bound = Bound::Unbounded;
}

// ── Stable nodes ──────────────────────────────────────────────────────────────

/// One vector and its HNSW edge lists.
///
/// `layers[l]` = neighbour `NodeId`s at layer `l`.
/// Length is always `max_layer + 1`.
#[derive(CandidType, Deserialize, Clone)]
pub struct StoredNode {
    pub vector: Vec<f32>,
    pub max_layer: u8,
    pub layers: Vec<Vec<NodeId>>,
    pub metadata: Vec<u8>,
}

impl Storable for StoredNode {
    fn to_bytes(&self) -> Cow<[u8]> {
        Cow::Owned(encode_one(self).expect("encode StoredNode"))
    }
    fn from_bytes(bytes: Cow<[u8]>) -> Self {
        decode_one(&bytes).expect("decode StoredNode")
    }
    const BOUND: Bound = Bound::Unbounded;
}

// ── NodeKey: (collection_id u32, node_id u64) = 12 bytes, fixed ──────────────

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct NodeKey {
    // Stored big-endian so BTreeMap order is (coll 0, all nodes), (coll 1, ...) etc.
    pub collection_id: CollectionId,
    pub node_id: NodeId,
}

impl NodeKey {
    pub fn new(collection_id: CollectionId, node_id: NodeId) -> Self {
        Self { collection_id, node_id }
    }
}

impl Storable for NodeKey {
    fn to_bytes(&self) -> Cow<[u8]> {
        let mut b = [0u8; 12];
        b[0..4].copy_from_slice(&self.collection_id.to_be_bytes());
        b[4..12].copy_from_slice(&self.node_id.to_be_bytes());
        Cow::Owned(b.to_vec())
    }
    fn from_bytes(bytes: Cow<[u8]>) -> Self {
        let collection_id = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
        let node_id       = u64::from_be_bytes(bytes[4..12].try_into().unwrap());
        Self { collection_id, node_id }
    }
    const BOUND: Bound = Bound::Bounded { max_size: 12, is_fixed_size: true };
}

// ── Thread-local stable state ─────────────────────────────────────────────────

thread_local! {
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));

    pub static COLLECTIONS: RefCell<StableBTreeMap<u32, StoredCollection, Mem>> =
        RefCell::new(StableBTreeMap::init(
            MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(0)))
        ));

    pub static NODES: RefCell<StableBTreeMap<NodeKey, StoredNode, Mem>> =
        RefCell::new(StableBTreeMap::init(
            MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(1)))
        ));
}

// ── Convenience wrappers ──────────────────────────────────────────────────────

pub fn get_collection(id: CollectionId) -> Option<StoredCollection> {
    COLLECTIONS.with(|c| c.borrow().get(&id))
}

pub fn put_collection(id: CollectionId, coll: StoredCollection) {
    COLLECTIONS.with(|c| c.borrow_mut().insert(id, coll));
}

pub fn remove_collection(id: CollectionId) {
    COLLECTIONS.with(|c| c.borrow_mut().remove(&id));
}

pub fn next_collection_id() -> CollectionId {
    COLLECTIONS.with(|c| {
        c.borrow()
            .last_key_value()
            .map(|(k, _)| k + 1)
            .unwrap_or(0)
    })
}

pub fn collection_exists_by_name(name: &str) -> bool {
    COLLECTIONS.with(|c| c.borrow().iter().any(|(_, v)| v.name == name))
}

pub fn get_node(collection_id: CollectionId, node_id: NodeId) -> Option<StoredNode> {
    NODES.with(|n| n.borrow().get(&NodeKey::new(collection_id, node_id)))
}

pub fn put_node(collection_id: CollectionId, node_id: NodeId, node: StoredNode) {
    NODES.with(|n| n.borrow_mut().insert(NodeKey::new(collection_id, node_id), node));
}

pub fn remove_node(collection_id: CollectionId, node_id: NodeId) {
    NODES.with(|n| n.borrow_mut().remove(&NodeKey::new(collection_id, node_id)));
}

/// Delete all nodes belonging to a collection.
pub fn remove_all_nodes(collection_id: CollectionId) {
    let prefix_start = NodeKey::new(collection_id, 0);
    let prefix_end   = NodeKey::new(collection_id, u64::MAX);

    let keys: Vec<NodeKey> = NODES.with(|n| {
        n.borrow()
            .range(prefix_start..=prefix_end)
            .map(|(k, _)| k)
            .collect()
    });

    NODES.with(|n| {
        let mut store = n.borrow_mut();
        for k in keys {
            store.remove(&k);
        }
    });
}
