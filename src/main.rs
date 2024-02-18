mod linear_model;
mod optim;

use linear_model::glm::LinearModel;

extern crate nalgebra as na;

use na::{DMatrix, DVector};

fn main() {
    /*
    let x_val = DMatrix::from_row_slice(3, 4, 
                                                    &[2104., 5., 1., 45.,
                                                           1416., 3., 2., 40.,
                                                            852., 2., 1., 35.,
                                                        ]);
    let y_val = DVector::from_row_slice(&[460., 232., 178.]);
    let mut model = LinearModel::new(true);
    let mut w_init = DVector::from_row_slice(&vec![0.39133535, 18.75376741, -53.36032453, -26.42131618]);
    let mut b_init: f64 = 785.1811367994083;

    model.train(&x_val, &y_val, &w_init, b_init, 100, 0.005);
    
    // model.fit(&x_val, &y_val);

    println!("Weights: {:?}", model.coef().as_ref().unwrap().data.as_vec());
    // [-0.0027262357719640327, -6.271972627776752e-6, -2.217455782253334e-6, -6.92403390682254e-5]
    */
    let y = DVector::from_row_slice(&[0.]);
    optim::convex::hessian_mtx(y, &example_fn, 0.0005_f32);
    // optim::convex::approx_fprime(DVector::<f32>::zeros(3), &example_fn, 0.05);
}

fn example_fn(value: f32) -> f32 {
    (3.0_f32 * value).exp() + (value - 0.1).exp() + (-value - 0.1).exp()
}
