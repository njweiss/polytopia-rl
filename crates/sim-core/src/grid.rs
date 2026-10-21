//! Minimal grid helpers shared by map generation and (later) the state model.

/// A tile position. `row` grows downward, `col` grows rightward; row-major order is
/// `(row, col)` lexicographic. The map is at most 255 wide, so `u8` is enough.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
