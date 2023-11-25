mod linear_model;

use linear_model::glm::LinearModel;

extern crate nalgebra as na;

use na::{DMatrix, DVector};

fn main() {
    let x_val = DMatrix::from_row_slice(3, 4, 
                                                    &[2104., 5., 1., 45.,
                                                           1416., 3., 2., 40.,
                                                            852., 2., 1., 35.,
                                                        ]);
    println!("x val: {:?}", x_val);
    let y_val = DVector::from_row_slice(&[460., 232., 178.]);
    let mut model = LinearModel::new(true);
    model.train(&x_val, &y_val, 100, 0.005);
    
    // model.fit(&x_val, &y_val);

    println!("Weights: {:?}", model.coef());
}
