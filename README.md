# ic_hnsw

Approximate nearest-neighbour vector search for the [Internet Computer](https://internetcomputer.org/).

Built on the [HNSW](https://arxiv.org/abs/1603.09320) (Hierarchical Navigable Small World) algorithm, backed by `ic-stable-structures` so all data survives canister upgrades automatically.

---

## Contents

- [What is HNSW?](#what-is-hnsw)
- [Architecture overview](#architecture-overview)
- [Usage modes](#usage-modes)
  - [Mode A — standalone canister](#mode-a--standalone-canister)
  - [Mode B — embedded library](#mode-b--embedded-library)
- [Concepts](#concepts)
  - [Collections](#collections)
  - [Distance metrics](#distance-metrics)
  - [HNSW parameters](#hnsw-parameters)
  - [Access control](#access-control)
  - [Controller-only gate](#controller-only-gate)
- [Tutorial](#tutorial)
  - [1. Create a collection](#1-create-a-collection)
  - [2. Insert vectors](#2-insert-vectors)
  - [3. Search](#3-search)
  - [4. Manage access](#4-manage-access)
  - [5. Lock the canister to controllers only](#5-lock-the-canister-to-controllers-only)
  - [6. Delete a node](#6-delete-a-node)
  - [7. Delete a collection](#7-delete-a-collection)
- [Embedding + search end-to-end](#embedding--search-end-to-end)
- [Calling from another canister](#calling-from-another-canister)
- [Instruction budget](#instruction-budget)
- [Tuning guide](#tuning-guide)
- [Candid interface reference](#candid-interface-reference)

---

## What is HNSW?

When you have thousands (or millions) of high-dimensional vectors and want to find the most similar ones to a query, comparing every vector one-by-one is too slow. HNSW builds a multi-layer graph where each node is connected to its closest neighbours. A search starts at the top (sparse) layer and greedily descends to the bottom (dense) layer, following edges toward the query vector. This finds approximate nearest neighbours in **O(log N)** time rather than O(N).

The tradeoff is construction cost and memory: each inserted vector triggers a bounded number of graph rewires, and each node stores a small edge list. For typical settings (M=8, ef=100) the index is compact enough to fit millions of vectors in ICP stable memory.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────┐
│  ic_hnsw  (this crate)                              │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌─────────────────┐ │
│  │ api.rs   │   │ hnsw.rs  │   │  storage.rs     │ │
│  │          │   │          │   │                 │ │
│  │ Public   │──▶│ Insert / │──▶│ StableBTreeMap  │ │
│  │ fn API   │   │ Search   │   │ (survives       │ │
│  │          │   │ algorithm│   │  upgrades)      │ │
│  └──────────┘   └──────────┘   └─────────────────┘ │
│       │                                             │
│  ┌────▼──────────────────┐                         │
│  │ export_hnsw!() macro  │  ← wire into your       │
│  │ #[update]/#[query]    │    canister with one     │
│  │ Candid endpoints      │    macro call            │
│  └───────────────────────┘                         │
└─────────────────────────────────────────────────────┘
```

Memory layout (managed by `MemoryManager`):

| Memory ID | Contents |
|-----------|----------|
| 0 | `StableBTreeMap<u32, StoredCollection>` — collection metadata |
| 1 | `StableBTreeMap<NodeKey, StoredNode>` — vectors + HNSW edge lists |
| 2 | `StableCell<bool>` — controller-only gate flag |

---

## Usage modes

### Mode A — standalone canister

Deploy `ic_hnsw` as its own canister. All endpoints are already wired in the compiled WASM.

**Using pre-built assets (Recommended)**

You can use the release asset links directly in your `dfx.json` to deploy without compiling:

```json
{
  "canisters": {
    "ic_hnsw": {
      "type": "custom",
      "candid": "https://github.com/Zedonboy/ic-hnsw/releases/latest/download/ic_hnsw.did",
      "wasm": "https://github.com/Zedonboy/ic-hnsw/releases/latest/download/ic_hnsw.wasm"
    }
  }
}
```

**Building from source**

```json
{
  "canisters": {
    "ic_hnsw": {
      "type": "rust",
      "package": "ic_hnsw",
      "candid": "ic_hnsw.did"
    }
  }
}
```

```shell
dfx deploy ic_hnsw
```

Your frontend, other canisters, or agents call it over the Candid interface.

---

### Mode B — embedded library

Embed the vector index directly inside your own canister. Your canister gains all HNSW endpoints alongside your own methods. Both share the same upgrade-safe stable memory.

**Step 1 — add the dependency**

```toml
# Cargo.toml
[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
ic_hnsw  = { git = "https://github.com/your-org/ic_hnsw" }
ic-cdk   = "0.17"
candid   = "0.10"
```

**Step 2 — expand the macro in your canister root**

```rust
// src/lib.rs

// Your own methods come first …
#[ic_cdk::update]
fn my_custom_method() { /* … */ }

// Inline all ten HNSW Candid endpoints:
ic_hnsw::export_hnsw!();

// Emit the combined .did file:
ic_cdk::export_candid!();
```

That is all. The macro expands to twelve `#[update]`/`#[query]` functions that delegate to `ic_hnsw::api::*`. The library owns its stable-memory segments and never interferes with yours.

**Calling the API programmatically** (without going through Candid) from within the same canister:

```rust
use ic_hnsw::{api, types::{CollectionConfig, DistanceMetric, InsertRequest, SearchRequest}};

// Inside an #[update] handler:
let coll_id = api::create_collection(CollectionConfig {
    name: "embeddings".into(),
    dim: 1536,
    m: 8,
    ef_construction: 100,
    distance: DistanceMetric::Cosine,
})?;

let node_id = api::insert(InsertRequest {
    collection_id: coll_id,
    vector: my_embedding,
    metadata: serde_json::to_vec(&my_doc_id).unwrap(),
})?;
```

---

## Concepts

### Collections

A collection is a named group of vectors that all share the same dimensionality and distance metric. Think of it like a table in a database: you create one per embedding model or use-case (e.g. one for 1536-dim OpenAI embeddings, another for 768-dim sentence embeddings).

The principal that calls `create_collection` becomes its **owner** and can grant other principals read or write access.

### Distance metrics

| Variant | Formula | When to use |
|---------|---------|-------------|
| `Cosine` | `1 − (a·b) / (‖a‖ ‖b‖)` | Text, document, and most embedding-model outputs. Direction matters, magnitude does not. |
| `Euclidean` | `√Σ(aᵢ−bᵢ)²` | Dense feature vectors where absolute magnitude is meaningful (e.g. image features). |
| `DotProduct` | `−(a·b)` | Unit-normalised vectors where you want raw similarity. Fastest to compute. |

When in doubt, use **Cosine** — it is the safe default for embedding model output.

Each collection has a default metric set at creation time. You can **override it per search** by supplying the optional `distance` field in `SearchRequest`. This lets you experiment with different metrics against the same stored vectors without rebuilding the index.

### HNSW parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `m` | 8 | Max edges per node above layer 0. Higher = better recall, more memory, slower insert. Range: 2–64. |
| `ef_construction` | 100 | Candidate pool size during index build. Higher = better graph quality, more cycles per insert. Must be ≥ `m`. |
| `ef_search` | per query | Candidate pool size during search. Higher = better recall, more cycles. Must be ≥ `top_k`. |

Layer-0 connections are always `2 × m` (HNSW paper convention).

### Access control

Each collection has one **Owner** and optional lists of **Writers** and **Readers**.

| Role | Permissions |
|------|-------------|
| `Owner` | Everything: configure, insert, delete, search, grant/revoke access, transfer ownership |
| `Writer` | Insert vectors, delete nodes, search |
| `Reader` | Search, view collection info |

Callers with no role receive `Unauthorized`.

### Controller-only gate

On top of collection-level roles there is a canister-wide **controller-only gate**. When enabled, every caller that is not a [canister controller](https://internetcomputer.org/docs/current/concepts/governance/#canister-controllers) receives `Unauthorized` before any collection check even runs.

The full decision tree:

```
Is the caller a canister controller?
  YES → always allowed, implicit Owner on every collection
  NO  → is the controller-only gate enabled?
          YES → Unauthorized (blocked entirely)
          NO  → normal collection role check (Owner / Writer / Reader)
```

Key properties:
- **Controllers always win.** Even when the gate is off, a controller can do anything on any collection without being listed as owner, writer, or reader.
- **Survives upgrades.** The flag is stored in a `StableCell` in stable memory (memory ID 2) so it persists across canister upgrades with no `pre_upgrade`/`post_upgrade` hooks needed.
- **Self-protecting.** Only a canister controller can call `set_controller_only`, so the gate cannot be disabled by a rogue writer or reader.

Use the gate for private deployments where the canister should only respond to the deploying identity or a small group of admins added as controllers via `dfx canister update-settings`.

---

## Tutorial

The examples below use the Candid textual format as you would see it in `dfx canister call` or the Candid UI. Equivalent calls work from any agent SDK (Rust, JavaScript, Motoko) or from within another canister.

### 1. Create a collection

```bash
dfx canister call ic_hnsw create_collection '(
  record {
    name = "articles";
    dim  = 1536 : nat32;
    m    = 8    : nat8;
    ef_construction = 100 : nat16;
    distance = variant { Cosine }
  }
)'
```

Returns `(variant { Ok = 0 : nat32 })` — the new collection ID.

A collection name must be unique. If it already exists you get `CollectionAlreadyExists`.

---

### 2. Insert vectors

Each insert takes a `collection_id`, a `vector` (list of `float32`), and a `metadata` blob. Metadata is opaque bytes you store alongside the vector — typically a serialised document ID, URL, or JSON object.

```bash
dfx canister call ic_hnsw insert '(
  record {
    collection_id = 0 : nat32;
    vector   = vec { 0.1; 0.4; -0.2; /* … 1536 values … */ };
    metadata = blob "\00\01\02\03"
  }
)'
```

Returns the assigned `NodeId` (a `nat64` counter starting at 0).

**Batch inserts** — there is no batch endpoint; call `insert` once per vector. For large initial loads, consider spreading inserts across multiple update calls to stay within the per-call instruction limit (see [Instruction budget](#instruction-budget)).

---

### 3. Search

```bash
dfx canister call ic_hnsw search '(
  record {
    collection_id = 0 : nat32;
    vector   = vec { 0.1; 0.4; -0.2; /* … same dimensionality … */ };
    top_k    = 5  : nat32;
    ef_search = 50 : nat32;
    distance  = null
  }
)'
```

`distance` is optional. Pass `null` (or omit the field) to use the collection's configured metric. To override it for this search:

```bash
dfx canister call ic_hnsw search '(
  record {
    collection_id = 0 : nat32;
    vector   = vec { 0.1; 0.4; -0.2; /* … */ };
    top_k    = 5  : nat32;
    ef_search = 50 : nat32;
    distance  = opt variant { Euclidean }
  }
)'
```

Returns a list of `SearchResult` records, sorted by distance (closest first):

```
(
  variant {
    Ok = vec {
      record { id = 42 : nat64; distance = 0.04 : float32; metadata = blob "…" };
      record { id = 7  : nat64; distance = 0.11 : float32; metadata = blob "…" };
      …
    }
  }
)
```

`ef_search` must be ≥ `top_k`. A value of `2 × top_k` to `4 × top_k` gives a good recall/speed tradeoff.

---

### 4. Manage access

**Grant write access to another principal:**

```bash
dfx canister call ic_hnsw grant_access '(
  0 : nat32,
  principal "aaaaa-aa",
  variant { Writer }
)'
```

**Grant read-only access:**

```bash
dfx canister call ic_hnsw grant_access '(
  0 : nat32,
  principal "bbbbb-bb",
  variant { Reader }
)'
```

**Revoke access:**

```bash
dfx canister call ic_hnsw revoke_access '(0 : nat32, principal "aaaaa-aa")'
```

**List all principals with access (owner only):**

```bash
dfx canister call ic_hnsw list_access '(0 : nat32)'
```

**Transfer ownership** (the previous owner becomes a Writer):

```bash
dfx canister call ic_hnsw transfer_ownership '(0 : nat32, principal "ccccc-cc")'
```

---

### 5. Lock the canister to controllers only

**Enable the gate** — from this point on, only canister controllers can call anything:

```bash
dfx canister call ic_hnsw set_controller_only '(true)'
```

**Check the current state:**

```bash
dfx canister call ic_hnsw get_controller_only '()'
# → (true)
```

**Add another principal as a canister controller** so they can also call in:

```bash
dfx canister update-settings ic_hnsw --add-controller <principal>
```

**Disable the gate** — reopens the canister to normal collection-level roles:

```bash
dfx canister call ic_hnsw set_controller_only '(false)'
```

> `set_controller_only` itself is controller-gated and cannot be called by writers or readers.

---

### 6. Delete a node

```bash
dfx canister call ic_hnsw delete_node '(0 : nat32, 42 : nat64)'
```

> **Note:** HNSW does not support true deletion with graph repair. The node's data and edges are removed and dangling references are silently skipped during search. For workloads with frequent deletions, periodically recreate the collection to compact the graph.

---

### 7. Delete a collection

Removes the collection and **all its vectors**. Owner only. This is irreversible.

```bash
dfx canister call ic_hnsw delete_collection '(0 : nat32)'
```

---

## Embedding + search end-to-end

A realistic pipeline: embed documents with an LLM, store the vectors in `ic_hnsw`, query by embedding the user's question.

```rust
// Pseudocode using the `irig` library (https://github.com/your-org/irig)
// inside a canister that embeds ic_hnsw via export_hnsw!().

use irig::providers::openai::{Client, TEXT_EMBEDDING_3_SMALL};
use ic_hnsw::{api, types::*};

// ── Index a document ──────────────────────────────────────────────────────────

async fn index_document(
    http:       &impl irig::http::HttpClient,
    openai_key: &str,
    coll_id:    CollectionId,
    doc_id:     u64,
    text:       &str,
) -> Result<NodeId, String> {
    let client = Client::new(http, openai_key);
    let model  = client.embedding_model(TEXT_EMBEDDING_3_SMALL);

    // Get the embedding (1536 dims for text-embedding-3-small)
    let embedding = model.embed_text(text).await.map_err(|e| e.to_string())?;
    let vector: Vec<f32> = embedding.vec.iter().map(|&x| x as f32).collect();

    // Store the vector; pack the document ID into metadata
    let node_id = api::insert(InsertRequest {
        collection_id: coll_id,
        vector,
        metadata: doc_id.to_be_bytes().to_vec(),
    }).map_err(|e| format!("{:?}", e))?;

    Ok(node_id)
}

// ── Query ─────────────────────────────────────────────────────────────────────

async fn semantic_search(
    http:       &impl irig::http::HttpClient,
    openai_key: &str,
    coll_id:    CollectionId,
    query:      &str,
    top_k:      u32,
) -> Result<Vec<(u64, f32)>, String> {
    let client = Client::new(http, openai_key);
    let model  = client.embedding_model(TEXT_EMBEDDING_3_SMALL);

    let embedding = model.embed_text(query).await.map_err(|e| e.to_string())?;
    let vector: Vec<f32> = embedding.vec.iter().map(|&x| x as f32).collect();

    let results = api::search(SearchRequest {
        collection_id: coll_id,
        vector,
        top_k,
        ef_search: top_k * 4,
        distance: None, // use the collection's configured metric
    }).map_err(|e| format!("{:?}", e))?;

    // Decode the doc_id we packed into metadata during insert
    let hits = results.into_iter().filter_map(|r| {
        let bytes: [u8; 8] = r.metadata.try_into().ok()?;
        Some((u64::from_be_bytes(bytes), r.distance))
    }).collect();

    Ok(hits)
}
```

---

## Calling from another canister

Use `ic_cdk::call` with the generated Candid types, or generate a Rust client from the `.did` file.

```rust
// In some other canister that wants to query ic_hnsw
use candid::Principal;
use ic_cdk::call;

async fn search_remote(
    hnsw_canister: Principal,
    collection_id: u32,
    vector:        Vec<f32>,
) -> Vec<(u64, f32)> {
    type SearchRequest = (u32, Vec<f32>, u32, u32); // (collection_id, vector, top_k, ef_search)

    let (result,): (Result<Vec<(u64, f32, Vec<u8>)>, String>,) = call(
        hnsw_canister,
        "search",
        (ic_hnsw::types::SearchRequest {
            collection_id,
            vector,
            top_k: 10,
            ef_search: 40,
            distance: None,
        },),
    )
    .await
    .expect("inter-canister call failed");

    result.unwrap_or_default()
        .into_iter()
        .map(|r| (r.id, r.distance))
        .collect()
}
```

---

## Instruction budget

Every `get_node` / `put_node` call is a `StableBTreeMap` lookup costing ~500 instructions. Rough worst-case estimates:

| Operation | Instruction estimate |
|-----------|---------------------|
| `insert` (M=8, ef=100, N=50k) | ~1.2 B |
| `insert` (M=16, ef=200, N=50k) | ~4.8 B |
| `search` (ef=50, N=50k) | ~200 M |
| `search` (ef=200, N=50k) | ~800 M |

ICP update call limit: **20 B instructions**. Query call limit: **5 B instructions** (but queries cannot write).

`search` is a query call — keep `ef_search` modest (50–200) for large collections.

For bulk loading, spread inserts across multiple update calls rather than doing them all in one message.

---

## Tuning guide

**Small collection (< 10k vectors), highest recall:**
```
m = 16, ef_construction = 200, ef_search = 200
```

**Medium collection (10k–100k), balanced:**
```
m = 8, ef_construction = 100, ef_search = 50
```

**Large collection (> 100k) or tight instruction budget:**
```
m = 4, ef_construction = 50, ef_search = 30
```

**General rules:**
- `ef_search` ≥ `top_k`. Start at `4 × top_k`, lower until recall degrades.
- Raising `m` helps recall more than raising `ef_construction` alone.
- Cosine distance on unit-normalised vectors and DotProduct give identical rankings; DotProduct is marginally cheaper to compute.
- After many deletions, rebuild the collection (create new, re-insert surviving nodes) to restore graph connectivity.

---

## Candid interface reference

```candid
type CollectionId    = nat32;
type NodeId          = nat64;

type DistanceMetric  = variant { Cosine; Euclidean; DotProduct };
type Role            = variant { Owner; Writer; Reader };

type CollectionConfig = record {
  name            : text;
  dim             : nat32;
  m               : nat8;
  ef_construction : nat16;
  distance        : DistanceMetric;
};

type CollectionInfo = record {
  id              : CollectionId;
  name            : text;
  dim             : nat32;
  m               : nat8;
  ef_construction : nat16;
  distance        : DistanceMetric;
  node_count      : nat64;
  max_layer       : nat8;
  owner           : principal;
};

type InsertRequest = record {
  collection_id : CollectionId;
  vector        : vec float32;
  metadata      : blob;
};

type SearchRequest = record {
  collection_id : CollectionId;
  vector        : vec float32;
  top_k         : nat32;
  ef_search     : nat32;
  distance      : opt DistanceMetric;
};

type SearchResult = record {
  id       : NodeId;
  distance : float32;
  metadata : blob;
};

type AccessEntry = record { principal : principal; role : Role };

type HnswError = variant {
  Unauthorized;
  CollectionNotFound    : nat32;
  CollectionAlreadyExists;
  DimensionMismatch     : record { expected : nat32; got : nat32 };
  NodeNotFound          : nat64;
  InvalidSearchParams;
  InvalidConfig         : text;
  Internal              : text;
};

service : {
  // Collection management (owner only except collection_info)
  create_collection  : (CollectionConfig)             -> (variant { Ok : CollectionId; Err : HnswError });
  delete_collection  : (CollectionId)                 -> (variant { Ok; Err : HnswError });
  collection_info    : (CollectionId)                 -> (variant { Ok : CollectionInfo; Err : HnswError }) query;

  // Access control (owner only)
  grant_access       : (CollectionId, principal, Role) -> (variant { Ok; Err : HnswError });
  revoke_access      : (CollectionId, principal)       -> (variant { Ok; Err : HnswError });
  list_access        : (CollectionId)                  -> (variant { Ok : vec AccessEntry; Err : HnswError }) query;
  transfer_ownership : (CollectionId, principal)       -> (variant { Ok; Err : HnswError });

  // Vector operations
  insert             : (InsertRequest)                 -> (variant { Ok : NodeId; Err : HnswError });
  delete_node        : (CollectionId, NodeId)          -> (variant { Ok; Err : HnswError });
  search             : (SearchRequest)                 -> (variant { Ok : vec SearchResult; Err : HnswError }) query;

  // Canister-level access gate (controller only)
  set_controller_only : (bool)                         -> (variant { Ok; Err : HnswError });
  get_controller_only : ()                             -> (bool) query;
};
```
