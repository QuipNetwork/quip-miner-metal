//! Graph validation, coloring, and dense tile layout.

use std::collections::HashSet;

use quip_solver_core::IsingGraph;

use crate::AneError;

pub(crate) const MAX_NODES: usize = 16_384;
pub(crate) const MAX_EDGES: usize = 163_840;
pub(crate) const MAX_DEGREE: usize = 20;
pub(crate) const MAX_LANES: usize = 128;
pub(crate) const TILE_CHANNELS: usize = 4_096;
pub(crate) const MAX_TILES: usize = 24;

/// Physical qubit labels absent from the Advantage2 processor.
///
/// Duplicated from `src/topology.rs` because the root crate depends on this
/// one, so the dependency cannot run the other way. The class counts that
/// `advantage2_colors` derives from this table are pinned by a test, which
/// fails if either copy drifts.
const ADVANTAGE2_MISSING_LABELS: [usize; 223] = [
    21, 31, 76, 91, 93, 169, 181, 215, 234, 239, 249, 285, 316, 327, 328, 340, 348, 351, 354, 363,
    364, 370, 373, 376, 380, 381, 400, 406, 441, 451, 465, 484, 495, 496, 510, 518, 530, 534, 544,
    555, 556, 568, 570, 577, 585, 623, 630, 738, 753, 762, 769, 780, 790, 801, 817, 863, 880, 918,
    972, 1008, 1018, 1020, 1031, 1139, 1173, 1218, 1268, 1276, 1281, 1283, 1308, 1395, 1413, 1436,
    1537, 1549, 1590, 1598, 1674, 1702, 1842, 1844, 1845, 1866, 1871, 1895, 1913, 1979, 1993, 2002,
    2004, 2049, 2072, 2111, 2123, 2149, 2171, 2230, 2355, 2377, 2386, 2403, 2491, 2516, 2550, 2587,
    2612, 2641, 2642, 2659, 2680, 2682, 2756, 2758, 2759, 2782, 2795, 2838, 2911, 2912, 2926, 2927,
    2940, 3060, 3108, 3110, 3112, 3122, 3198, 3205, 3206, 3212, 3218, 3225, 3240, 3252, 3253, 3264,
    3265, 3266, 3267, 3268, 3277, 3278, 3280, 3281, 3289, 3290, 3298, 3309, 3312, 3325, 3382, 3434,
    3437, 3458, 3467, 3534, 3545, 3546, 3551, 3651, 3698, 3705, 3722, 3792, 3796, 3827, 3888, 3953,
    3961, 3985, 3997, 4058, 4071, 4082, 4083, 4119, 4120, 4121, 4132, 4133, 4134, 4155, 4172, 4177,
    4186, 4189, 4199, 4202, 4220, 4226, 4237, 4259, 4270, 4279, 4334, 4350, 4374, 4386, 4388, 4393,
    4412, 4424, 4447, 4450, 4476, 4508, 4575, 4579, 4604, 4608, 4641, 4650, 4684, 4686, 4720, 4723,
    4725, 4761, 4768, 4778, 4780,
];

/// Number of logical variables on the Advantage2 processor.
const ADVANTAGE2_NODES: usize = 4_577;
/// Physical label space the logical variables are drawn from.
const ADVANTAGE2_PHYSICAL: usize = 4_800;

/// Derives the Advantage2 four-colouring from Zephyr coordinates.
///
/// Returns `None` for any graph this does not recognise, including one that
/// has the right variable count but an edge joining two nodes of one colour.
/// The check is against every edge, not only those with a nonzero coupling,
/// because the topology is fixed across jobs and an edge that carries zero
/// now may carry a coupling in the next job.
///
/// Four classes beat the greedy eight on the engine, because each sweep
/// walks its classes in strict sequence. See
/// `docs/perf/2026-09-18-ane-four-coloring.md`.
fn advantage2_colors(graph: &IsingGraph) -> Option<Vec<usize>> {
    if graph.h.len() != ADVANTAGE2_NODES {
        return None;
    }
    let mut missing = ADVANTAGE2_MISSING_LABELS.iter().peekable();
    let mut colors = Vec::with_capacity(ADVANTAGE2_NODES);
    for physical in 0..ADVANTAGE2_PHYSICAL {
        if missing.peek() == Some(&&physical) {
            missing.next();
            continue;
        }
        let z = physical % 12;
        let j = physical / 12 % 2;
        let w = physical / (12 * 2 * 4) % 25;
        let u = physical / (12 * 2 * 4 * 25);
        colors.push(j + ((w + 2 * (z + u) + j) & 2));
    }
    if colors.len() != ADVANTAGE2_NODES {
        return None;
    }
    if graph
        .edges
        .iter()
        .any(|&(u, v)| u >= colors.len() || v >= colors.len() || colors[u] == colors[v])
    {
        return None;
    }
    Some(colors)
}

/// Colours greedily by descending degree, then ascending index.
///
/// The fallback for every graph `advantage2_colors` does not recognise.
fn greedy_colors(neighbors: &[Vec<(usize, i8)>]) -> Result<Vec<usize>, AneError> {
    let n = neighbors.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&node| (std::cmp::Reverse(neighbors[node].len()), node));
    let mut colors = vec![usize::MAX; n];
    for node in order {
        let mut used = [false; MAX_DEGREE + 1];
        for &(neighbor, _) in &neighbors[node] {
            if colors[neighbor] != usize::MAX {
                used[colors[neighbor]] = true;
            }
        }
        colors[node] = used
            .iter()
            .position(|&taken| !taken)
            .ok_or_else(|| AneError::Runtime("color bound violated".into()))?;
    }
    Ok(colors)
}

const FP16_BYTES: usize = 2;
const FP16_PAYLOAD_LIMIT: usize = 128 * 1024 * 1024;
const TOTAL_FP16_PAYLOAD_LIMIT: usize = 544 * 1024 * 1024;
const SURFACE_ALIGNMENT: usize = 65_536;

#[derive(Debug)]
pub(crate) struct ColorTile {
    #[cfg(test)]
    pub(crate) color: usize,
    pub(crate) nodes: Vec<usize>,
    pub(crate) output_channels: usize,
}

#[derive(Debug)]
pub(crate) struct PreparedGraph {
    pub(crate) node_count: usize,
    pub(crate) input_channels: usize,
    pub(crate) fields: Vec<i8>,
    pub(crate) neighbors: Vec<Vec<(usize, i8)>>,
    pub(crate) tiles: Vec<ColorTile>,
    #[cfg(test)]
    pub(crate) color_count: usize,
}

fn unit_coefficient(value: f64) -> Result<i8, AneError> {
    match value {
        -1.0 => Ok(-1),
        0.0 => Ok(0),
        1.0 => Ok(1),
        _ => Err(AneError::Capacity("coefficient must be -1, 0, or 1".into())),
    }
}

fn checked_mul(left: usize, right: usize, what: &str) -> Result<usize, AneError> {
    left.checked_mul(right)
        .ok_or_else(|| AneError::Capacity(format!("{what} overflow")))
}

fn checked_add(left: usize, right: usize, what: &str) -> Result<usize, AneError> {
    left.checked_add(right)
        .ok_or_else(|| AneError::Capacity(format!("{what} overflow")))
}

fn fp16_payload_bytes(output_channels: usize, input_channels: usize) -> Result<usize, AneError> {
    let cells = checked_mul(output_channels, input_channels, "weight payload cells")?;
    checked_mul(cells, FP16_BYTES, "weight payload bytes")
}

fn aligned_surface_bytes(channels: usize) -> Result<usize, AneError> {
    let cells = checked_mul(channels, MAX_LANES, "surface cells")?;
    let bytes = checked_mul(cells, FP16_BYTES, "surface bytes")?;
    let blocks = bytes.max(1).div_ceil(SURFACE_ALIGNMENT);
    checked_mul(blocks, SURFACE_ALIGNMENT, "aligned surface bytes")
}

fn check_program_bytes(tiles: &[ColorTile], input_channels: usize) -> Result<(), AneError> {
    let mut total_payload = 0usize;
    for tile in tiles {
        let payload = fp16_payload_bytes(tile.output_channels, input_channels)?;
        if payload > FP16_PAYLOAD_LIMIT {
            return Err(AneError::Capacity("FP16 payload exceeds 128 MiB".into()));
        }
        total_payload = checked_add(total_payload, payload, "total FP16 payload")?;
        let neighbor_surface = aligned_surface_bytes(input_channels)?;
        let output_surface = aligned_surface_bytes(tile.output_channels)?;
        let with_spins = checked_add(neighbor_surface, output_surface, "surface bytes")?;
        let with_thresholds = checked_add(with_spins, output_surface, "surface bytes")?;
        let _output_surface_total = checked_add(with_thresholds, output_surface, "surface bytes")?;
    }
    if total_payload >= TOTAL_FP16_PAYLOAD_LIMIT {
        return Err(AneError::Capacity(
            "total FP16 payload exceeds 544 MiB".into(),
        ));
    }
    Ok(())
}

pub(crate) fn prepare(graph: &IsingGraph) -> Result<PreparedGraph, AneError> {
    // Keep the measurement control fixed for the worker process lifetime.
    static FOUR_COLOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let four_color = *FOUR_COLOR.get_or_init(|| {
        !matches!(
            std::env::var("QUIP_ANE_MSA_FOUR_COLOR").as_deref(),
            Ok("0") | Ok("false")
        )
    });
    prepare_with_coloring(graph, four_color)
}

fn prepare_with_coloring(graph: &IsingGraph, four_color: bool) -> Result<PreparedGraph, AneError> {
    let node_count = graph.h.len();
    if node_count > MAX_NODES {
        return Err(AneError::Capacity(format!(
            "graph has {node_count} variables; maximum is {MAX_NODES}"
        )));
    }
    if graph.edges.len() > MAX_EDGES {
        return Err(AneError::Capacity(format!(
            "graph has {} edges; maximum is {MAX_EDGES}",
            graph.edges.len()
        )));
    }
    if graph.j.len() != graph.edges.len() {
        return Err(AneError::Capacity("edge and coupling counts differ".into()));
    }

    let mut fields = Vec::with_capacity(node_count);
    for &value in &graph.h {
        fields.push(unit_coefficient(value)?);
    }

    let mut seen = HashSet::with_capacity(graph.edges.len());
    let mut neighbors = vec![Vec::new(); node_count];
    for (&(u, v), &coupling) in graph.edges.iter().zip(&graph.j) {
        if u >= node_count || v >= node_count {
            return Err(AneError::Capacity("edge endpoint out of range".into()));
        }
        if u == v {
            return Err(AneError::Capacity("self-loop is not allowed".into()));
        }
        let canonical = (u.min(v), u.max(v));
        if !seen.insert(canonical) {
            return Err(AneError::Capacity("duplicate undirected edge".into()));
        }
        let coeff = unit_coefficient(coupling)?;
        if coeff != 0 {
            neighbors[u].push((v, coeff));
            neighbors[v].push((u, coeff));
        }
    }

    for nbrs in &mut neighbors {
        if nbrs.len() > MAX_DEGREE {
            return Err(AneError::Capacity(format!(
                "variable degree {} exceeds {MAX_DEGREE}",
                nbrs.len()
            )));
        }
        nbrs.sort_by_key(|&(index, _)| index);
    }

    let n = neighbors.len();
    // Prefer the Advantage2 four-colouring. It halves the dependent chain a
    // sweep walks and costs slightly less padded work, which measured 1.178
    // times faster per sweep with a 37.6% cheaper compile. Greedy remains
    // the fallback for every other graph.
    let colors = match four_color.then(|| advantage2_colors(graph)).flatten() {
        Some(colors) => colors,
        None => greedy_colors(&neighbors)?,
    };
    let color_count = colors.iter().copied().max().map_or(0, |last| last + 1);
    let input_channels = n.max(1).div_ceil(32) * 32;
    let mut tiles = Vec::new();
    for color in 0..color_count {
        let nodes: Vec<usize> = (0..n).filter(|&node| colors[node] == color).collect();
        for chunk in nodes.chunks(TILE_CHANNELS) {
            tiles.push(ColorTile {
                #[cfg(test)]
                color,
                nodes: chunk.to_vec(),
                output_channels: chunk.len().div_ceil(32) * 32,
            });
        }
    }

    if tiles.len() > MAX_TILES {
        return Err(AneError::Capacity(format!(
            "tile count {} exceeds {MAX_TILES}",
            tiles.len()
        )));
    }
    check_program_bytes(&tiles, input_channels)?;

    Ok(PreparedGraph {
        node_count,
        input_channels,
        fields,
        neighbors,
        tiles,
        #[cfg(test)]
        color_count,
    })
}

impl PreparedGraph {
    pub(crate) fn storage_order(&self) -> Vec<usize> {
        self.tiles
            .iter()
            .flat_map(|tile| tile.nodes.iter().copied())
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn tile_weights(&self, tile: &ColorTile) -> Result<Vec<i8>, AneError> {
        let len = tile
            .output_channels
            .checked_mul(self.input_channels)
            .ok_or_else(|| AneError::Runtime("weight matrix overflow".into()))?;
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(len)
            .map_err(|_| AneError::Runtime("weight matrix allocation failed".into()))?;
        weights.resize(len, 0);
        for (row, &node) in tile.nodes.iter().enumerate() {
            for &(neighbor, coupling) in &self.neighbors[node] {
                weights[row * self.input_channels + neighbor] = coupling;
            }
        }
        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    /// Loads the Advantage2 topology the production path runs on.
    fn advantage2_graph() -> IsingGraph {
        let fixture = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/advantage2-system1.edges"
        ));
        let edges: Vec<(usize, usize)> = fixture
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| {
                let mut ids = line
                    .split_whitespace()
                    .map(|id| id.parse().expect("node id"));
                (ids.next().expect("u"), ids.next().expect("v"))
            })
            .collect();
        assert_eq!(edges.len(), 41_515);
        IsingGraph::new(vec![0.0; 4_577], vec![1.0; edges.len()], edges)
    }

    /// Pins the four-colouring this crate derives from its own copy of the
    /// missing-label table. `src/topology.rs` holds the other copy, and the
    /// root crate depends on this one so the table cannot be shared. If
    /// either copy drifts, these counts move and this test fails.
    #[test]
    fn advantage2_colors_are_four_balanced_classes() {
        let graph = advantage2_graph();
        let colors = advantage2_colors(&graph).expect("Advantage2 topology is recognised");
        assert_eq!(colors.len(), 4_577);
        let mut counts = [0usize; 4];
        for &color in &colors {
            assert!(color < 4, "color {color} is outside the four classes");
            counts[color] += 1;
        }
        assert_eq!(counts, [1148, 1145, 1145, 1139]);
    }

    /// Every edge must join two different colours, including edges whose
    /// coupling is zero. The topology is fixed across jobs, so an edge that
    /// carries zero now may carry a coupling later.
    #[test]
    fn advantage2_coloring_is_proper_over_every_edge() {
        let graph = advantage2_graph();
        let colors = advantage2_colors(&graph).expect("Advantage2 topology is recognised");
        for &(u, v) in &graph.edges {
            assert_ne!(colors[u], colors[v], "edge ({u}, {v}) joins one colour");
        }
    }

    /// The prepared graph must carry four tiles rather than the greedy eight,
    /// because the tile count is what a sweep walks in sequence.
    #[test]
    fn advantage2_prepares_four_tiles_not_eight() {
        let prepared = prepare(&advantage2_graph()).expect("prepare");
        assert_eq!(prepared.tiles.len(), 4);
        let mut lengths: Vec<usize> = prepared.tiles.iter().map(|tile| tile.nodes.len()).collect();
        lengths.sort_unstable();
        assert_eq!(lengths, vec![1139, 1145, 1145, 1148]);
    }

    #[test]
    fn greedy_control_preserves_graph_and_has_eight_colors() {
        let graph = advantage2_graph();
        let four = super::prepare_with_coloring(&graph, true).unwrap();
        let greedy = super::prepare_with_coloring(&graph, false).unwrap();
        assert_eq!(four.color_count, 4);
        assert_eq!(greedy.color_count, 8);
        assert_eq!(four.neighbors, greedy.neighbors);
        assert_eq!(four.fields, greedy.fields);
        assert_prepared_invariants(&greedy);
    }

    /// A graph the recogniser does not know must still colour, by the greedy
    /// fallback, and must still produce a proper colouring.
    #[test]
    fn unknown_topology_falls_back_to_greedy() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let graph = IsingGraph::new(vec![0.0; 3], vec![1.0; 3], edges);
        assert!(advantage2_colors(&graph).is_none());
        let prepared = prepare(&graph).expect("prepare");
        // A triangle needs three colours, so three single-node tiles.
        assert_eq!(prepared.tiles.len(), 3);
    }

    use quip_solver_core::IsingGraph;

    use super::{advantage2_colors, prepare, PreparedGraph};
    use crate::AneError;

    const FP16_PAYLOAD_LIMIT: usize = 128 * 1024 * 1024;
    const TOTAL_FP16_PAYLOAD_LIMIT: usize = 544 * 1024 * 1024;

    fn isolated(nodes: usize) -> IsingGraph {
        IsingGraph::new(vec![0.0; nodes], Vec::new(), Vec::new())
    }

    fn assert_prepared_invariants(prepared: &PreparedGraph) {
        let mut seen = vec![false; prepared.node_count];
        let mut color_of = vec![None; prepared.node_count];
        let mut total_payload = 0usize;
        for tile in &prepared.tiles {
            assert_eq!(
                tile.output_channels,
                tile.nodes.len().div_ceil(32) * 32,
                "output channels must pad to a multiple of 32"
            );
            let payload = tile
                .output_channels
                .checked_mul(prepared.input_channels)
                .and_then(|cells| cells.checked_mul(2))
                .expect("payload byte count must not overflow");
            assert!(
                payload <= FP16_PAYLOAD_LIMIT,
                "payload {payload} exceeds 128 MiB"
            );
            total_payload = total_payload
                .checked_add(payload)
                .expect("total payload must not overflow");
            for &node in &tile.nodes {
                assert!(node < prepared.node_count);
                assert!(!seen[node], "variable {node} appears in more than one tile");
                seen[node] = true;
                color_of[node] = Some(tile.color);
            }
        }
        assert!(
            total_payload < TOTAL_FP16_PAYLOAD_LIMIT,
            "total payload {total_payload} is not below 544 MiB"
        );
        assert!(seen.iter().all(|&present| present));
        assert_eq!(prepared.fields.len(), prepared.node_count);
        assert_eq!(prepared.neighbors.len(), prepared.node_count);
        assert_eq!(
            prepared.input_channels,
            prepared.node_count.max(1).div_ceil(32) * 32
        );
        for (node, neighbors) in prepared.neighbors.iter().enumerate() {
            let Some(color) = color_of[node] else {
                panic!("variable {node} is missing a color");
            };
            let mut last = None;
            for &(neighbor, coupling) in neighbors {
                assert_ne!(coupling, 0, "zero couplings must be omitted");
                assert!(last.is_none_or(|prev| prev < neighbor));
                last = Some(neighbor);
                assert_ne!(
                    color_of[neighbor],
                    Some(color),
                    "nonzero edge {node}-{neighbor} stays inside color {color}"
                );
            }
        }
    }

    fn assert_capacity(graph: IsingGraph) {
        assert!(matches!(prepare(&graph), Err(AneError::Capacity(_))));
    }

    #[test]
    fn error_variants_display_the_contained_string() {
        assert_eq!(AneError::Capacity("too big".into()).to_string(), "too big");
        assert_eq!(AneError::Runtime("device".into()).to_string(), "device");
    }

    #[test]
    fn complete_twenty_one_uses_singleton_colors() {
        let edges: Vec<_> = (0..21)
            .flat_map(|u| ((u + 1)..21).map(move |v| (u, v)))
            .collect();
        let graph = IsingGraph::new(vec![0.0; 21], vec![1.0; edges.len()], edges);
        let prepared = prepare(&graph).unwrap();
        assert_eq!(prepared.tiles.len(), 21);
        assert!(prepared.tiles.iter().all(|tile| tile.nodes.len() == 1));
        assert!(prepared.tiles.iter().all(|tile| tile.output_channels == 32));
        assert_prepared_invariants(&prepared);
    }

    #[test]
    fn degree_twenty_one_is_rejected() {
        let edges: Vec<_> = (1..22).map(|v| (0, v)).collect();
        let graph = IsingGraph::new(vec![0.0; 22], vec![1.0; 21], edges);
        assert!(matches!(prepare(&graph), Err(AneError::Capacity(_))));
    }

    #[test]
    fn node_count_bounds_and_padding() {
        struct Case {
            nodes: usize,
            ok: bool,
            tiles: usize,
            input_channels: usize,
            output_channels: &'static [usize],
        }
        let cases = [
            Case {
                nodes: 0,
                ok: true,
                tiles: 0,
                input_channels: 32,
                output_channels: &[],
            },
            Case {
                nodes: 1,
                ok: true,
                tiles: 1,
                input_channels: 32,
                output_channels: &[32],
            },
            Case {
                nodes: 31,
                ok: true,
                tiles: 1,
                input_channels: 32,
                output_channels: &[32],
            },
            Case {
                nodes: 32,
                ok: true,
                tiles: 1,
                input_channels: 32,
                output_channels: &[32],
            },
            Case {
                nodes: 33,
                ok: true,
                tiles: 1,
                input_channels: 64,
                output_channels: &[64],
            },
            Case {
                nodes: 4097,
                ok: true,
                tiles: 2,
                input_channels: 4128,
                output_channels: &[4096, 32],
            },
            Case {
                nodes: 16_384,
                ok: true,
                tiles: 4,
                input_channels: 16_384,
                output_channels: &[4096, 4096, 4096, 4096],
            },
            Case {
                nodes: 16_385,
                ok: false,
                tiles: 0,
                input_channels: 0,
                output_channels: &[],
            },
        ];
        for case in cases {
            let result = prepare(&isolated(case.nodes));
            if !case.ok {
                assert!(
                    matches!(result, Err(AneError::Capacity(_))),
                    "nodes {} must be rejected",
                    case.nodes
                );
                continue;
            }
            let prepared =
                result.unwrap_or_else(|_| panic!("nodes {} must be accepted", case.nodes));
            assert_eq!(prepared.node_count, case.nodes);
            assert_eq!(prepared.input_channels, case.input_channels);
            assert_eq!(prepared.tiles.len(), case.tiles);
            let outputs: Vec<usize> = prepared
                .tiles
                .iter()
                .map(|tile| tile.output_channels)
                .collect();
            assert_eq!(outputs, case.output_channels);
            if case.nodes == 16_384 {
                assert!(prepared.tiles.iter().all(|tile| tile.nodes.len() == 4096));
            }
            assert_prepared_invariants(&prepared);
        }
    }

    #[test]
    fn malformed_edges_and_coefficients_are_rejected() {
        assert_capacity(IsingGraph::new(
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![(0, 1), (1, 0)],
        ));
        assert_capacity(IsingGraph::new(vec![0.0], vec![1.0], vec![(0, 0)]));
        assert_capacity(IsingGraph::new(vec![0.0], vec![0.0], vec![(0, 0)]));
        assert_capacity(IsingGraph::new(vec![0.0], vec![1.0], vec![(0, 1)]));
        assert_capacity(IsingGraph::new(vec![0.0, 0.0], vec![1.0], vec![]));
        assert_capacity(IsingGraph::new(vec![0.0, 0.0], vec![], vec![(0, 1)]));
        assert_capacity(IsingGraph::new(
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![(0, 1), (0, 1)],
        ));
        assert_capacity(IsingGraph::new(vec![f64::NAN], vec![], vec![]));
        assert_capacity(IsingGraph::new(
            vec![0.0, 0.0],
            vec![f64::NAN],
            vec![(0, 1)],
        ));
        assert_capacity(IsingGraph::new(vec![f64::INFINITY], vec![], vec![]));
        assert_capacity(IsingGraph::new(
            vec![0.0, 0.0],
            vec![f64::NEG_INFINITY],
            vec![(0, 1)],
        ));
        assert_capacity(IsingGraph::new(vec![0.5], vec![], vec![]));
        assert_capacity(IsingGraph::new(vec![0.0, 0.0], vec![0.5], vec![(0, 1)]));
        assert_capacity(IsingGraph::new(
            vec![0.0, 0.0],
            vec![1.0; 163_841],
            vec![(0, 1); 163_841],
        ));
    }

    #[test]
    fn zero_couplings_are_omitted_from_neighbors() {
        let graph = IsingGraph::new(vec![1.0, -1.0], vec![0.0], vec![(0, 1)]);
        let prepared = prepare(&graph).unwrap();
        assert_eq!(prepared.fields, vec![1, -1]);
        assert!(prepared.neighbors.iter().all(|list| list.is_empty()));
        assert_eq!(prepared.color_count, 1);
        assert_eq!(prepared.tiles.len(), 1);
        assert_eq!(prepared.tiles[0].nodes, vec![0, 1]);
        assert_prepared_invariants(&prepared);
        let weights = prepared.tile_weights(&prepared.tiles[0]).unwrap();
        assert_eq!(weights, vec![0; 32 * 32]);
    }

    #[test]
    fn three_node_path_colors_center_first_by_descending_degree() {
        // Nodes 0-1-2 with node 1 the degree-2 center (nodes 0 and 2 each
        // have degree 1). Descending-degree coloring must visit the center
        // first, so the color groups are [1] then [0, 2].
        let graph = IsingGraph::new(vec![0.0, 0.0, 0.0], vec![1.0, 1.0], vec![(0, 1), (1, 2)]);
        let prepared = prepare(&graph).unwrap();
        assert_eq!(prepared.color_count, 2);
        assert_eq!(prepared.tiles.len(), 2);
        assert_eq!(prepared.tiles[0].nodes, vec![1]);
        assert_eq!(prepared.tiles[1].nodes, vec![0, 2]);
        assert_prepared_invariants(&prepared);
    }

    #[test]
    fn three_node_tile_matches_padded_weight_array() {
        let graph = IsingGraph::new(
            vec![1.0, -1.0, 0.0],
            vec![1.0, -1.0, 1.0],
            vec![(0, 1), (1, 2), (2, 0)],
        );
        let prepared = prepare(&graph).unwrap();
        assert_eq!(prepared.fields, vec![1, -1, 0]);
        assert_eq!(prepared.input_channels, 32);
        assert_eq!(prepared.color_count, 3);
        assert_eq!(prepared.tiles.len(), 3);
        assert_prepared_invariants(&prepared);

        let expected = [
            {
                let mut weights = vec![0i8; 32 * 32];
                weights[1] = 1;
                weights[2] = 1;
                weights
            },
            {
                let mut weights = vec![0i8; 32 * 32];
                weights[0] = 1;
                weights[2] = -1;
                weights
            },
            {
                let mut weights = vec![0i8; 32 * 32];
                weights[0] = 1;
                weights[1] = -1;
                weights
            },
        ];
        for (tile, expected_weights) in prepared.tiles.iter().zip(expected) {
            assert_eq!(tile.nodes.len(), 1);
            assert_eq!(tile.output_channels, 32);
            assert_eq!(prepared.tile_weights(tile).unwrap(), expected_weights);
        }
    }
}
