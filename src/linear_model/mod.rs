pub mod glm;

pub use glm::LinearModel;

extern crate nalgebra as na;

use na::{DMatrix, DVector};
use float_cmp::approx_eq;

fn eq_with_nan_eq(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || approx_eq!(f64, a, b, epsilon = 0.00000003)
}

fn vec_compare(a: &[f64], b: &[f64]) -> bool {
    (a.len() == b.len()) &&
     a.iter()
      .zip(b)
      .all(|(x, y)| eq_with_nan_eq(*x, *y))
}

#[cfg(test)]
mod tests {
    use std::ops::Deref;

    use float_cmp::ApproxEq;

    use super::*;

    #[test]
    fn test_lm_train() {
        let x_val = DMatrix::from_row_slice(3, 4, &[2104., 5., 1., 45.,
                                                    1416., 3., 2., 40.,
                                                    852., 2., 1., 35.,]);
        let y_val = DVector::from_row_slice(&[460., 232., 178.]);
        let mut model = LinearModel::new(true);

        // initial weights, initial biased
        let mut w_init = DVector::from_row_slice(&vec![0.39133535, 18.75376741, -53.36032453, -26.42131618]);
        let mut b_init: f64 = 785.1811367994083;

        model.train(&x_val, &y_val, &w_init, b_init, 100, 0.005);
        let w_final = model
                        .coef()
                        .as_ref()
                        .unwrap()
                        .data
                        .as_vec();
        
        let margin = vec![-0.0027262357719640327f64, -6.271972627776752e-6_f64, -2.217455782253334e-6_f64, -6.92403390682254e-5_f64];

        println!("Weights: {:?}", w_final);
        assert_eq!(vec_compare(w_final, &margin), true);
    }
}