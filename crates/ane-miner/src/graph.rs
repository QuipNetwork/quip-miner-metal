//! Graph validation, greedy coloring, and dense tile layout.

use std::collections::HashSet;

use quip_solver_core::IsingGraph;

use crate::AneError;

pub(crate) const MAX_NODES: usize = 16_384;
pub(crate) const MAX_EDGES: usize = 163_840;
pub(crate) const MAX_DEGREE: usize = 20;
pub(crate) const LANES: usize = 128;
pub(crate) const TILE_CHANNELS: usize = 4_096;
pub(crate) const MAX_PROGRAMS: usize = 24;

const FP16_BYTES: usize = 2;
const FP16_PAYLOAD_LIMIT: usize = 128 * 1024 * 1024;
const TOTAL_FP16_PAYLOAD_LIMIT: usize = 544 * 1024 * 1024;
const SURFACE_ALIGNMENT: usize = 65_536;

#[derive(Debug)]
pub(crate) struct ColorTile {
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
    let cells = checked_mul(channels, LANES, "surface cells")?;
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
    let color_count = colors.iter().copied().max().map_or(0, |last| last + 1);
    let input_channels = n.max(1).div_ceil(32) * 32;
    let mut tiles = Vec::new();
    for color in 0..color_count {
        let nodes: Vec<usize> = (0..n).filter(|&node| colors[node] == color).collect();
        for chunk in nodes.chunks(TILE_CHANNELS) {
            tiles.push(ColorTile {
                color,
                nodes: chunk.to_vec(),
                output_channels: chunk.len().div_ceil(32) * 32,
            });
        }
    }

    if tiles.len() > MAX_PROGRAMS {
        return Err(AneError::Capacity(format!(
            "program count {} exceeds {MAX_PROGRAMS}",
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
        color_count,
    })
}

impl PreparedGraph {
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
    use quip_solver_core::IsingGraph;

    use super::{prepare, PreparedGraph};
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
