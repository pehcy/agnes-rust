extern crate nalgebra as na;

use std::usize;

use na::{U2, U3, ArrayStorage, VecStorage, Matrix, SMatrix, DMatrix, DVector, Scalar, Dim, OMatrix, DimMin, DefaultAllocator, allocator::Allocator, Dyn, Vector, VectorN};

pub struct GeneralLinearModel {
    thetas: Option<Vec<f64>>
}

/// Rust nalgebra:
/// Convert static matrix to dynamic function
fn static_to_dynamic_matrix<R, C>(mat: OMatrix<f64, R, C>) -> OMatrix<f64, na::Dyn, na::Dyn>
    where
    R: Dim + DimMin<C>,
    C: Dim,
    DefaultAllocator: Allocator<f64, R, C>
{
    let mut dyn_matrix = OMatrix::<f64, na::Dyn, na::Dyn>::zeros(mat.nrows(), mat.ncols());
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            dyn_matrix[(i, j)] = mat[(i, j)];
        }
    }
    dyn_matrix
}

impl GeneralLinearModel {
    pub fn new(&self) -> GeneralLinearModel {
        GeneralLinearModel { 
            thetas: None 
        }
    }

    // gradient descent
    pub fn train(
        &self, 
        inputs: &DMatrix<f64>,
        targets: &SMatrix<f64, 6, 1>,
        epochs: usize,
        alpha: f64,
    ) {
        // get number of row in input matrix
        let m: usize = inputs.ncols();
        let n_rows: usize = inputs.nrows();
        let mut thetas = DMatrix::from_vec(1, m, vec![0f64; n_rows]);

        for _ in 0..epochs {
            for i in 0..n_rows {
                let h_theta = thetas.clone() * &inputs.transpose();
                let errs = h_theta - targets;
                
                


                // let h_theta = thetas.dot(&inputs.row(i));
            }
        }
    }
}