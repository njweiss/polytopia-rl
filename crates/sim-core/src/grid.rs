//! Minimal grid helpers shared by map generation and (later) the state model.

use serde::{Deserialize, Serialize};

/// A tile position. `row` grows downward, `col` grows rightward; row-major order is
/// `(row, col)` lexicographic. The map is at most 255 wide, so `u8` is enough.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Pos {
    pub row: u8,
    pub col: u8,
}

impl Pos {
    pub const fn new(row: u8, col: u8) -> Pos {
        Pos { row, col }
    }
}

/// Chebyshev (king-move) distance.
pub fn chebyshev(a: Pos, b: Pos) -> u32 {
    let dr = (a.row as i32 - b.row as i32).unsigned_abs();
    let dc = (a.col as i32 - b.col as i32).unsigned_abs();
    dr.max(dc)
}

/// Every position of an `n`-wide board, row-major.
pub(crate) fn all_tiles(n: u8) -> impl Iterator<Item = Pos> {
    (0..n).flat_map(move |r| (0..n).map(move |c| Pos::new(r, c)))
}

/// The in-bounds 8-neighbors of `p`, row-major.
pub(crate) fn neighbors(size: u8, p: Pos) -> impl Iterator<Item = Pos> {
    within(size, p, 1).filter(move |&q| q != p)
}

/// The in-bounds tiles within Chebyshev `radius` of `p` (including `p`), row-major.
pub(crate) fn within(size: u8, p: Pos, radius: u32) -> impl Iterator<Item = Pos> {
    let span = |x: u8| {
        let x = u32::from(x);
        x.saturating_sub(radius) as u8..=x.saturating_add(radius).min(u32::from(size) - 1) as u8
    };
    let cols = span(p.col);
    span(p.row).flat_map(move |r| cols.clone().map(move |c| Pos::new(r, c)))
}
