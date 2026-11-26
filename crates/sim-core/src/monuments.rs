//! v2 lighthouses and monuments (docs/MILESTONE_18).

use crate::grid::Pos;

/// The four corner positions of a `size x size` map in row-major order.
pub fn corners(_size: u8) -> [Pos; 4] {
    unimplemented!()
}

/// Index `0..=3` if `pos` is one of `corners(size)`, else `None`.
pub fn corner_index(_size: u8, _pos: Pos) -> Option<usize> {
    unimplemented!()
}
