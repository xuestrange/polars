// h: size of half window
// k: size of window
// alpha: slice of length k

use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use std::ops::{Add, Mul, Sub};
use num_traits::NumCast;

use polars_utils::float::IsFloat;
use polars_utils::sort::arg_sort_ascending;
use polars_utils::IdxSize;

use crate::legacy::kernels::rolling::Idx;
use crate::types::NativeType;

struct Block<'a, T: NativeType + IsFloat> {
    k: usize,
    tail: usize,
    n_element: usize,
    alpha: &'a [T],
    pi: &'a mut [u32],
    prev: &'a mut Vec<u32>,
    next: &'a mut Vec<u32>,
    // permutation index in alpha
    m: usize,
    // index in the list
    current_index: usize,
}

impl<T: NativeType + IsFloat> Debug for Block<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.n_element == 0 {
            return writeln!(f, "empty block");
        }
        writeln!(f, "elements in list: {}", self.n_element)?;
        writeln!(f, "m: {}", self.m)?;
        writeln!(f, "m_index: {}", self.current_index)?;
        writeln!(f, "α[m]: {:?}", self.alpha[self.m])?;

        let mut p = self.m as u32;

        // Find start.
        loop {
            p = self.prev[p as usize];
            if p as usize == self.tail {
                p = self.next[p as usize];
                break;
            }
        }

        // Find all elements from start.
        let mut current = Vec::with_capacity(self.n_element);
        for _ in 0..self.n_element {
            current.push(self.alpha[p as usize]);
            p = self.next[p as usize];
        }

        write!(f, "current buffer sorted: [")?;
        for (i, v) in current.iter().enumerate() {
            if i == self.current_index {
                write!(f, "[{v:?}], ")?;
            } else {
                let chars = if i == self.n_element - 1 { "" } else { ", " };
                write!(f, "{v:?}{chars}")?;
            }
        }
        write!(f, "]")
    }
}

impl<'a, T: IsFloat + PartialOrd + NativeType> Block<'a, T> {
    fn new(
        alpha: &'a [T],
        scratch: &'a mut Vec<u8>,
        prev: &'a mut Vec<u32>,
        next: &'a mut Vec<u32>,
    ) -> Self {
        let k = alpha.len();
        let pi = arg_sort_ascending(alpha, scratch);

        let m_index = k / 2;
        let m = pi[m_index] as usize;

        prev.resize(k + 1, 0 as u32);
        next.resize(k + 1, 0 as u32);
        let mut b = Self {
            k,
            pi,
            prev,
            next,
            m,
            current_index: m_index,
            n_element: k,
            tail: k,
            alpha,
        };
        b.init_links();
        b
    }

    fn capacity(&self) -> usize {
        self.alpha.len()
    }

    fn init_links(&mut self) {
        let mut p = self.tail;

        for &q in self.pi.iter() {
            self.next[p as usize] = q;
            self.prev[q as usize] = p as u32;

            p = q as usize;
        }
        self.next[p as usize] = self.tail as u32;
        self.prev[self.tail] = p as u32;
    }

    fn delete_link(&mut self, i: usize) {
        self.next[self.prev[i] as usize] = self.next[i];
        self.prev[self.next[i] as usize] = self.prev[i];
    }

    fn undelete_link(&mut self, i: usize) {
        self.next[self.prev[i] as usize] = i as u32;
        self.prev[self.next[i] as usize] = i as u32;
    }

    fn unwind(&mut self) {
        for i in (0..self.k).rev() {
            self.delete_link(i)
        }
        self.m = self.tail;
        self.n_element = 0;
    }

    fn set_median(&mut self) {
        // median index position
        let new_index = self.n_element / 2;
        self.traverse_to_index(new_index)
    }

    fn traverse_to_index(&mut self, i: usize) {
        match i as i64 - self.current_index as i64 {
            0 => {
                // pass
            },
            -1 => {
                self.current_index -= 1;
                self.m = self.prev[self.m as usize] as usize;
            },
            1 => {
                self.advance()
            },
            i64::MIN..=0 => {
                self.current_index -= i;
                for _ in i..0 {
                    self.m = self.prev[self.m as usize] as usize;
                }
            },
            _ => {
                self.current_index += i;
                for _ in 0..i {
                    self.m = self.next[self.m as usize] as usize;
                }
            },
        }
    }

    fn advance(&mut self) {
        self.current_index += 1;
        self.m = self.next[self.m] as usize;
    }

    fn reset(&mut self) {
        self.current_index = 0;
        self.m = self.next[self.tail] as usize;
    }

    fn delete(&mut self, i: usize) {
        let delete = self.get_pair(i);

        let current = self.get_pair(self.m);

        // delete from links
        self.delete_link(i);

        self.n_element -= 1;

        match delete.partial_cmp(&current).unwrap() {
            Ordering::Less => {
                // 1, 2, [3], 4, 5
                //    2, [3], 4, 5
                // the del changes index
                self.current_index -= 1
            },
            Ordering::Greater => {
                // 1, 2, [3], 4, 5
                // 1, 2, [3], 4
                // index position remains unaffected
            },
            Ordering::Equal => {
                // 1, 2, [3], 4, 5
                // 1, 2, [4], 5
                // go to next position because the link was deleted
                if self.n_element > self.current_index {
                    self.m = self.next[self.m as usize] as usize;
                } else {
                    // move to previous position because the link was deleted
                    // 1, [2],
                    // [1]
                    self.m = self.prev[self.m as usize] as usize
                }
            },
        };
    }

    fn undelete(&mut self, i: usize) {
        // undelete from links
        self.undelete_link(i);

        if self.at_end() {
            self.m = self.prev[self.m] as usize;
            self.n_element = 1;
            self.current_index = 0;
            return;
        }
        let added = self.get_pair(i);
        let current = self.get_pair(self.m);

        self.n_element += 1;

        match added.partial_cmp(&current).unwrap() {
            Ordering::Less => {
                //    2, [3], 4, 5
                // 1, 2, [3], 4, 5
                // the addition changes index
                self.current_index += 1
            },
            Ordering::Greater => {
                // 1, 2, [3], 4
                // 1, 2, [3], 4, 5
                // index position remains unaffected
            },
            Ordering::Equal => {
                // 1, 2,      4, 5
                // 1, 2, [3], 4, 5
                // go to prev position because hte link was added
                // self.m = self.prev[self.m as usize] as usize;
            },
        };
    }

    fn delete_set_median(&mut self, i: usize) {
        self.delete(i);
        self.set_median()
    }

    fn undelete_set_median(&mut self, i: usize) {
        self.undelete(i);
        self.set_median()
    }

    fn at_end(&self) -> bool {
        self.m == self.tail
    }

    fn peek(&self) -> Option<T> {
        if self.at_end() {
            None
        } else {
            Some(self.alpha[self.m as usize])
        }
    }

    fn get_pair(&self, i: usize) -> (T, u32) {
        (self.alpha[i], i as u32)
    }

    fn is_small(&self, i: usize) -> bool {
        self.at_end() || self.get_pair(i) < self.get_pair(self.m as usize)
    }
}

trait LenGet {
    type Item: NativeType;
    fn len(&self) -> usize;

    fn get(&mut self, i: usize) -> Self::Item;
}

impl<T: IsFloat + PartialOrd + NativeType> LenGet for &mut Block<'_, T> {
    type Item = T;

    fn len(&self) -> usize {
        self.n_element
    }

    fn get(&mut self, i: usize) -> Self::Item {
        self.traverse_to_index(i);
        self.peek().unwrap()
    }
}

struct BlockUnion<'a, T: IsFloat + PartialOrd + NativeType> {
    block_left: &'a mut Block<'a, T>,
    block_right: &'a mut Block<'a, T>,
    // Current index position of sorted merge.
    s: usize
}

impl<'a, T: IsFloat + PartialOrd + NativeType> BlockUnion<'a, T> {
    fn new(block_left: &'a mut Block<'a, T>, block_right: &'a mut Block<'a, T>, k: usize ) -> Self {
        let out = Self {
            block_left,
            block_right,
            s: 0
        };
        debug_assert_eq!(out.len(), k);

        out
    }
}

impl<T: IsFloat + PartialOrd + NativeType> LenGet for BlockUnion<'_, T> {
    type Item = T;

    fn len(&self) -> usize {
        self.block_left.n_element + self.block_right.n_element
    }

    fn get(&mut self, i: usize) -> Self::Item {
        // Simple case, all elements are left.
        if self.block_right.n_element == 0 {
            self.block_left.traverse_to_index(i);
            return self.block_left.peek().unwrap()
        }
        let steps = i + 1 - self.s;
        dbg!(steps);


        let left = self.block_left.peek().unwrap();
        let right = self.block_right.peek().unwrap();
        dbg!(left, right);
        match left.partial_cmp(&right).unwrap() {
            // On equality, take the left as that one was first.
            Ordering::Equal | Ordering::Less => {
                self.block_left.advance();
                left
            },
            Ordering::Greater => {
                self.block_right.advance();
                right
            }
        }


        // let i= if i < self.block_left.n_element {
        //     self.block_left.traverse_to_index(i);
        //     return self.block_left.peek().unwrap()
        // } else if self.block_left.n_element == 0 {
        //     i
        // } else{
        //     i % self.block_left.n_element
        // };
        // self.block_right.traverse_to_index(i);
        // self.block_right.peek().unwrap()
    }
}

struct MedianUpdate<M: LenGet> {
    inner: M
}

impl<M> MedianUpdate<M>
where M: LenGet,
<M as LenGet>::Item:
Sub<Output=<M as LenGet>::Item>
+ Mul<Output=<M as LenGet>::Item>
+ Add<Output=<M as LenGet>::Item>
+ NumCast
{
    fn new(inner: M) -> Self {
        Self {
            inner
        }
    }

    fn median(&mut self) -> M::Item {
        let lenght = self.inner.len();
        let length_f =  lenght as f64;
        let idx = ((length_f - 1.0) * 0.5).floor() as usize;

        let float_idx_top = (length_f - 1.0) * 0.5;
        let top_idx = float_idx_top.ceil() as usize;

        return if idx == top_idx {
            self.inner.get(idx)
        } else {
            let proportion : M::Item = NumCast::from(float_idx_top - idx as f64).unwrap();
            let vi = self.inner.get(idx);
            let vj = self.inner.get(top_idx);

            proportion * (vj - vi) + vi
        };
    }

}

fn rolling_median<T>(k: usize, slice: &[T])
where
    T: IsFloat + NativeType + PartialOrd + Sub + NumCast
+ Sub<Output=T>
+ Mul<Output=T>
+ Add<Output=T>
,
{
    let mut scratch_left = vec![];
    let mut prev_left = vec![];
    let mut next_left = vec![];

    let mut scratch_right = vec![];
    let mut prev_right = vec![];
    let mut next_right = vec![];

    let k = std::cmp::min(k, slice.len());
    let alpha = &slice[..k];

    let mut out = Vec::with_capacity(slice.len());

    let scratch_right_ptr = &mut scratch_right as *mut Vec<u8>;
    let scratch_left_ptr = &mut scratch_left as *mut Vec<u8>;
    let prev_right_ptr = &mut prev_right as *mut Vec<_>;
    let prev_left_ptr = &mut prev_left as *mut Vec<_>;
    let next_right_ptr = &mut next_right as *mut Vec<_>;
    let next_left_ptr = &mut next_left as *mut Vec<_>;

    let n_blocks = slice.len() / k;

    let mut block_left = unsafe { Block::new(alpha, &mut *scratch_left_ptr, &mut *prev_left_ptr, &mut *next_left_ptr) };
    let mut block_right = unsafe { Block::new(&alpha[..1], &mut *scratch_right_ptr, &mut *prev_right_ptr, &mut *next_right_ptr) };

    let ptr_left = &mut block_left as *mut Block<'_, _>;
    let ptr_right = &mut block_right as *mut Block<'_, _>;

    block_left.unwind();

    for i in 0..block_left.capacity() {
        block_left.undelete(i);

        let mut mu = MedianUpdate::new(&mut block_left);
        out.push(mu.median());
    }
    for i in 1..n_blocks {
        // Block left is now completely full as it is completely filled coming from the boundary effects.
        debug_assert!(block_left.n_element == k);

        // Windows state at this point.
        //
        //  - BLOCK_LEFT -- BLOCK_RIGHT -
        // |-------------||-------------|
        //   - WINDOW -
        // |--------------|
        let alpha = &slice[i * k..(i + 1) *k];
        block_right = unsafe { Block::new(alpha, &mut *scratch_right_ptr, &mut *prev_right_ptr, &mut *next_right_ptr) };

        // Time reverse the rhs so we can undelete in sorted order.
        block_right.unwind();

        // Here the window will move from BLOCK_LEFT into BLOCK_RIGHT
        for j in 0..block_right.capacity() {
            block_left.delete(j);
            block_right.undelete(j);

            dbg!(&block_left, &block_right);
            unsafe {
                let union = BlockUnion::new(&mut *ptr_left, &mut *ptr_right, k);
                out.push(dbg!(MedianUpdate::new(union).median()))

            }
        }
        unsafe {
            std::ptr::swap_nonoverlapping(ptr_left, ptr_right, 1);
            std::ptr::swap_nonoverlapping(scratch_left_ptr, scratch_right_ptr, 1);
            std::ptr::swap_nonoverlapping(prev_left_ptr, prev_right_ptr, 1);
            std::ptr::swap_nonoverlapping(next_left_ptr, next_right_ptr, 1);
        }
    }
    dbg!(&out);
    // mu.

    // let mut block_b = Block::new(h, alpha, &mut scratch_b, &mut prev_b, &mut next_b);
    // out.push(block_b.peek());
    //
    // for j in 1..b {
    //     block_a = block_b;
    //
    //     let alpha = &slice[j * k..(j + 1) *k];
    //     block_b = if j % 2 == 0 {
    //         Block::new(h, alpha, &mut scratch_b, &mut prev_b, &mut next_b)
    //     } else {
    //         Block::new(h, alpha, &mut scratch_a, &mut prev_a, &mut next_a)
    //     };
    //
    //     block_b.unwind();
    //     debug_assert_eq!(block_a.counter, h);
    //     debug_assert_eq!(block_b.counter, h);
    //
    //     for i in 0..k {
    //         block_a.delete(i);
    //         block_b.undelete(i);
    //         debug_assert!(block_a.counter + block_b.counter <= h);
    //
    //         if block_a.counter + block_b.counter <= h {
    //             if block_a.peek() <= block_b.peek() {
    //                 block_a.advance()
    //             } else {
    //                 block_b.advance()
    //             }
    //         }
    //         debug_assert_eq!(block_a.counter + block_b.counter, h)
    //         out.push(std::cmp::min(block_a.peek(), block_b.peek()));
    //     }
    //     debug_assert_eq!(block_a.counter, 0);
    //     debug_assert_eq!(block_b.counter, h);
    // }
    // dbg!(out);
}

mod test {
    use super::*;

    #[test]
    fn test_block() {
        //                    0, 1, 2, 3, 4, 5, 6, 7
        let values = [2, 8, 5, 9, 1, 3, 4, 10];
        let mut scratch = vec![];
        let mut prev = vec![];
        let mut next = vec![];
        let mut b = Block::new(&values, &mut scratch, &mut prev, &mut next);

        // Unwind to get temporal window
        b.unwind();

        // Insert window in the right order
        b.undelete_set_median(0);
        // [[2]]
        assert_eq!(b.peek(), Some(2));
        b.undelete_set_median(1);
        // [2, [8]]
        assert_eq!(b.peek(), Some(8));
        b.undelete_set_median(2);
        // [2, [5], 8]
        assert_eq!(b.peek(), Some(5));
        b.undelete_set_median(3);
        // [2, 5, [8], 9]
        assert_eq!(b.peek(), Some(8));
        b.undelete_set_median(4);
        // [1, 2, [5], 8, 9]
        assert_eq!(b.peek(), Some(5));
        b.undelete_set_median(5);
        // [1, 2, 3, [5], 8, 9]
        assert_eq!(b.peek(), Some(5));
        b.undelete_set_median(6);
        // [1, 2, 3, [4], 5, 8, 9]
        assert_eq!(b.peek(), Some(4));
        b.undelete_set_median(7);
        // [1, 2, 3, 4, [5], 8, 9, 10]
        assert_eq!(b.peek(), Some(5));

        // Now we will delete as the block` will leave the window.
        b.delete_set_median(0);
        // [1, 3, 4, [5], 8, 9, 10]
        assert_eq!(b.peek(), Some(5));
        b.delete_set_median(1);
        // [1, 3, 4, [5], 9, 10]
        assert_eq!(b.peek(), Some(5));
        b.delete_set_median(2);
        // [1, 3, [4],  9, 10]
        assert_eq!(b.peek(), Some(4));
        b.delete_set_median(3);
        // [1, 3, [4], 10]
        assert_eq!(b.peek(), Some(4));
        b.delete_set_median(4);
        // [3, [4], 10]
        assert_eq!(b.peek(), Some(4));
        b.delete_set_median(5);
        // [4, [10]]
        assert_eq!(b.peek(), Some(10));
        b.delete_set_median(6);
        // [[10]]
        assert_eq!(b.peek(), Some(10));
    }

    #[test]
    fn test_block_2() {
        let values = [9, 1, 2];
        let mut scratch = vec![];
        let mut prev = vec![];
        let mut next = vec![];
        let mut b = Block::new(&values, &mut scratch, &mut prev, &mut next);

        b.unwind();
        b.undelete_set_median(0);
        assert_eq!(b.peek(), Some(9));
        b.undelete_set_median(1);
        assert_eq!(b.peek(), Some(9));
        b.undelete_set_median(2);
        assert_eq!(b.peek(), Some(2));
    }

    #[test]
    fn test_block_union() {
        let alpha_a = [10, 4, 2];
        let alpha_b = [3, 4, 1];

        let mut scratch = vec![];
        let mut prev = vec![];
        let mut next = vec![];
        let mut a = Block::new(&alpha_a, &mut scratch, &mut prev, &mut next);

        let mut scratch = vec![];
        let mut prev = vec![];
        let mut next = vec![];
        let mut b = Block::new(&alpha_b, &mut scratch, &mut prev, &mut next);

        b.unwind();
        let mut aub = BlockUnion::new(&mut a, &mut b, alpha_a.len());
        assert_eq!(aub.len(), 3);
        // STEP 0
        // block 1:
        // i:  10, 4, 2
        // s:  2, 4, 10
        // block 2: empty
        assert_eq!(aub.get(0), 2);
        assert_eq!(aub.get(1), 4);
        assert_eq!(aub.get(2), 10);

        // STEP 1
        aub.block_left.reset();
        aub.block_left.delete(0);
        aub.block_right.undelete(0);
        assert_eq!(aub.len(), 3);
        // block 1:
        // i:  4, 2
        // s:  2, 4
        // block 2:
        // i:  3
        // s:  3
        // union s: [2, 3, 4]
        assert_eq!(aub.get(0), 2);
        assert_eq!(aub.get(1), 3);
        dbg!(&aub.block_left);
        dbg!(&aub.block_right);
        dbg!(aub.get(0));



        dbg!(aub.len());
    }

    #[test]
    fn test_median() {
        let values = [2.0, 8.0, 5.0, 9.0, 1.0, 2.0, 4.0, 2.0];
        let k = 3;

        //    block 1
        // i:  2, 8, 5
        // s:  2, 5, 8
        //              block 2
        // i:           9, 1, 2
        // s:           1, 2, 9

        rolling_median(k, &values);
    }
}
