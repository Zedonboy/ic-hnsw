//! HNSW insert and search — pure algorithm, reads/writes via storage helpers.
//!
//! # Instruction budget notes (ICP)
//!
//! Every `get_node` call is a `StableBTreeMap` lookup: O(log N) instructions.
//! Typical worst-case per insert with M=8, ef_construction=100, N=50k:
//!   layers ≈ 3, ef×M per layer ≈ 800 node reads → ~2400 stable reads × ~500 instr each
//!   ≈ 1.2B instructions — safely within the 20B update limit.
//!
//! For larger collections reduce ef_construction or M, or batch inserts across
//! multiple messages using a queue pattern.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use crate::{
    storage::{get_node, put_node, StoredNode},
    types::{CollectionId, DistanceMetric, NodeId},
};

// ── Distance functions ────────────────────────────────────────────────────────

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32  = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32   = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32   = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { return 1.0; }
    1.0 - dot / (na * nb)
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    // Higher dot product = more similar, so negate for "smaller = closer" convention.
    -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

pub fn dist_fn(metric: &DistanceMetric) -> fn(&[f32], &[f32]) -> f32 {
    match metric {
        DistanceMetric::Cosine      => cosine_distance,
        DistanceMetric::Euclidean   => euclidean_distance,
        DistanceMetric::DotProduct  => dot_product_distance,
    }
}

// ── Candidate: used in both search heaps ─────────────────────────────────────

#[derive(Clone, PartialEq)]
pub struct Candidate {
    pub dist: f32,
    pub id: NodeId,
}

impl Eq for Candidate {}

/// Default `Ord` = max-heap by distance (furthest first).
/// Use `Reverse<Candidate>` for a min-heap (closest first).
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ── Random layer assignment ───────────────────────────────────────────────────

/// LCG step. Returns a value in `[0, 1)` and advances `state`.
fn lcg_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // Use high 32 bits for better distribution.
    ((*state >> 32) as f64) / (u32::MAX as f64 + 1.0)
}

/// Draw a random insertion layer using the standard HNSW exponential distribution.
/// `m_l = 1 / ln(m)` is the normalisation factor from the original paper.
pub fn random_layer(m: u8, rng: &mut u64) -> u8 {
    let ml = 1.0 / (m as f64).ln();
    let r  = lcg_f64(rng);
    (-r.ln() * ml).floor() as u8
}

// ── Core search primitive ─────────────────────────────────────────────────────

/// Greedy best-first search within one layer.
///
/// Returns up to `ef` candidates sorted by distance (closest first).
///
/// `entry_points` is a slice of already-known close nodes to seed the search.
/// At upper layers this is typically a single node; at layer 0 it's the result
/// of the layer-1 search.
pub fn search_layer(
    coll_id:      CollectionId,
    query:        &[f32],
    entry_points: &[NodeId],
    ef:           usize,
    layer:        u8,
    distance:     fn(&[f32], &[f32]) -> f32,
) -> Vec<Candidate> {
    let mut visited: HashSet<NodeId>              = HashSet::new();
    let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new(); // min-heap
    let mut results:    BinaryHeap<Candidate>          = BinaryHeap::new(); // max-heap

    for &ep in entry_points {
        if let Some(node) = get_node(coll_id, ep) {
            let d = distance(query, &node.vector);
            candidates.push(Reverse(Candidate { dist: d, id: ep }));
            results.push(Candidate { dist: d, id: ep });
            visited.insert(ep);
        }
    }

    while let Some(Reverse(closest)) = candidates.pop() {
        // If the closest unexplored node is further than the worst result we've
        // accumulated, no future exploration can improve the result set.
        let worst = results.peek().map(|c| c.dist).unwrap_or(f32::MAX);
        if closest.dist > worst {
            break;
        }

        let node = match get_node(coll_id, closest.id) {
            Some(n) => n,
            None    => continue,
        };

        // Visit this node's neighbours at the requested layer.
        let neighbours = node
            .layers
            .get(layer as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        for &n_id in neighbours {
            if visited.contains(&n_id) {
                continue;
            }
            visited.insert(n_id);

            let n_node = match get_node(coll_id, n_id) {
                Some(n) => n,
                None    => continue,
            };
            let d = distance(query, &n_node.vector);
            let worst_now = results.peek().map(|c| c.dist).unwrap_or(f32::MAX);

            if d < worst_now || results.len() < ef {
                candidates.push(Reverse(Candidate { dist: d, id: n_id }));
                results.push(Candidate { dist: d, id: n_id });
                if results.len() > ef {
                    results.pop(); // evict the furthest
                }
            }
        }
    }

    // Return sorted closest-first.
    let mut out: Vec<Candidate> = results.into_vec();
    out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
    out
}

// ── Neighbour selection heuristic ─────────────────────────────────────────────

/// Select the best `m` neighbours from `candidates` (sorted closest-first).
///
/// This is the heuristic from Algorithm 4 of the HNSW paper. Instead of
/// naively keeping the M closest, it selects neighbours that extend coverage
/// in new *directions*, preventing the degenerate case where all M edges point
/// the same way and leave one side of the node unreachable.
///
/// Rule: include candidate `c` only if it is closer to the inserting node than
/// it is to any already-selected neighbour. Fill remaining slots from discarded
/// candidates if fewer than `m` pass the filter.
fn select_neighbours_heuristic(
    coll_id:    CollectionId,
    candidates: &[Candidate],  // sorted closest-first
    m:          usize,
    distance:   fn(&[f32], &[f32]) -> f32,
) -> Vec<NodeId> {
    let mut selected:  Vec<Candidate> = Vec::with_capacity(m);
    let mut discarded: Vec<Candidate> = Vec::new();

    'outer: for c in candidates {
        if selected.len() >= m {
            break;
        }
        let c_node = match get_node(coll_id, c.id) {
            Some(n) => n,
            None    => continue,
        };
        for sel in &selected {
            let sel_node = match get_node(coll_id, sel.id) {
                Some(n) => n,
                None    => continue,
            };
            // If c is closer to an already-selected neighbour than to the query,
            // it doesn't add directional coverage — skip it.
            if distance(&c_node.vector, &sel_node.vector) < c.dist {
                discarded.push(c.clone());
                continue 'outer;
            }
        }
        selected.push(c.clone());
    }

    // Fill remaining slots from discarded (preserves recall under sparse graphs).
    for c in discarded {
        if selected.len() >= m {
            break;
        }
        selected.push(c);
    }

    selected.into_iter().map(|c| c.id).collect()
}

// ── Insert ────────────────────────────────────────────────────────────────────

/// Insert a vector into the HNSW graph for `coll_id`.
///
/// Returns the assigned `NodeId`. The caller is responsible for persisting
/// the updated `StoredCollection` (entry_point, max_layer, counters) after
/// this call returns.
pub fn insert(
    coll_id:         CollectionId,
    node_id:         NodeId,
    vector:          Vec<f32>,
    metadata:        Vec<u8>,
    entry_point:     Option<NodeId>,
    current_max_layer: u8,
    m:               u8,
    m_max0:          u8,
    ef_construction: u16,
    distance:        fn(&[f32], &[f32]) -> f32,
    rng:             &mut u64,
) -> InsertResult {
    let node_layer = random_layer(m, rng);

    // Persist the new node immediately (empty edge lists — we fill them below).
    let new_node = StoredNode {
        vector:    vector.clone(),
        max_layer: node_layer,
        layers:    vec![vec![]; node_layer as usize + 1],
        metadata,
    };
    put_node(coll_id, node_id, new_node);

    // If the graph was empty, this node becomes the entry point.
    let ep = match entry_point {
        None => {
            return InsertResult {
                node_id,
                new_entry_point:   Some(node_id),
                new_max_layer:     node_layer,
            };
        }
        Some(ep) => ep,
    };

    let m       = m as usize;
    let m_max0  = m_max0 as usize;
    let ef      = ef_construction as usize;

    // ── Phase 1: navigate from top layer down to node_layer + 1 ──────────────
    // We just want to arrive in the right neighbourhood — single greedy step per layer.
    let mut ep_set = vec![ep];
    for layer in (node_layer + 1..=current_max_layer).rev() {
        let found = search_layer(coll_id, &vector, &ep_set, 1, layer, distance);
        if !found.is_empty() {
            ep_set = found.iter().map(|c| c.id).collect();
        }
    }

    // ── Phase 2: insert at each layer node_layer..=0 ─────────────────────────
    for layer in (0..=node_layer.min(current_max_layer)).rev() {
        let candidates = search_layer(
            coll_id, &vector, &ep_set, ef, layer, distance,
        );

        let m_at_layer = if layer == 0 { m_max0 } else { m };
        let neighbours = select_neighbours_heuristic(
            coll_id, &candidates, m_at_layer, distance,
        );

        // Wire new node → neighbours.
        {
            let mut node = get_node(coll_id, node_id)
                .expect("node just inserted");
            node.layers[layer as usize] = neighbours.clone();
            put_node(coll_id, node_id, node);
        }

        // Wire neighbours → new node (bidirectional), then prune if needed.
        for &n_id in &neighbours {
            let mut n_node = match get_node(coll_id, n_id) {
                Some(n) => n,
                None    => continue,
            };

            // Ensure the neighbour has an edge list at this layer.
            // (Older nodes might not have been initialised for this layer
            //  if the graph grew layers after they were inserted.)
            while n_node.layers.len() <= layer as usize {
                n_node.layers.push(vec![]);
            }

            n_node.layers[layer as usize].push(node_id);

            // Prune back to m_at_layer if over-connected.
            if n_node.layers[layer as usize].len() > m_at_layer {
                let n_vec = n_node.vector.clone();
                let mut current: Vec<Candidate> = n_node.layers[layer as usize]
                    .iter()
                    .filter_map(|&id| {
                        get_node(coll_id, id).map(|n| Candidate {
                            dist: distance(&n_vec, &n.vector),
                            id,
                        })
                    })
                    .collect();
                current.sort_by(|a, b| {
                    a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal)
                });
                n_node.layers[layer as usize] =
                    select_neighbours_heuristic(coll_id, &current, m_at_layer, distance);
            }

            put_node(coll_id, n_id, n_node);
        }

        // The best candidates at this layer seed the search at the next lower layer.
        if !candidates.is_empty() {
            ep_set = candidates.iter().map(|c| c.id).collect();
        }
    }

    let new_entry_point = if node_layer > current_max_layer {
        Some(node_id)
    } else {
        None
    };

    InsertResult {
        node_id,
        new_entry_point,
        new_max_layer: node_layer.max(current_max_layer),
    }
}

/// What the caller must persist into `StoredCollection` after an insert.
pub struct InsertResult {
    pub node_id: NodeId,
    /// `Some(id)` if this node became the new entry point (its layer exceeded the old max).
    pub new_entry_point: Option<NodeId>,
    pub new_max_layer: u8,
}

// ── KNN search ────────────────────────────────────────────────────────────────

/// Find the `top_k` approximate nearest neighbours of `query`.
pub fn knn_search(
    coll_id:      CollectionId,
    query:        &[f32],
    entry_point:  NodeId,
    max_layer:    u8,
    top_k:        usize,
    ef_search:    usize,
    distance:     fn(&[f32], &[f32]) -> f32,
) -> Vec<Candidate> {
    // Navigate down the hierarchy to layer 1 with greedy single-result steps.
    let mut ep_set = vec![entry_point];

    for layer in (1..=max_layer).rev() {
        let found = search_layer(coll_id, query, &ep_set, 1, layer, distance);
        if !found.is_empty() {
            ep_set = found.iter().map(|c| c.id).collect();
        }
    }

    // Full ef_search at layer 0 for the final candidate pool.
    let mut results = search_layer(
        coll_id, query, &ep_set, ef_search.max(top_k), 0, distance,
    );

    results.truncate(top_k);
    results
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(v: &[f32]) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn cosine_identical() {
        let a = unit_vec(&[1.0, 0.0, 0.0]);
        assert!((cosine_distance(&a, &a)).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = unit_vec(&[1.0, 0.0]);
        let b = unit_vec(&[0.0, 1.0]);
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn layer_distribution() {
        // Roughly geometric: layer 0 >> layer 1 >> layer 2 ...
        let mut rng = 0xdeadbeef_u64;
        let mut counts = [0u32; 8];
        for _ in 0..10_000 {
            let l = random_layer(8, &mut rng) as usize;
            counts[l.min(7)] += 1;
        }
        // Layer 0 must be most frequent.
        assert!(counts[0] > counts[1]);
        assert!(counts[1] > counts[2]);
    }
}
