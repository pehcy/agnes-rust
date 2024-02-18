extern crate nalgebra as na;

use nalgebra::{DVector, DMatrix};

/// Obtain Hessian Matrix from a continuous function.
pub fn hessian_mtx(x0: DVector<f32>, func: &dyn Fn(f32) -> f32, eps: f32) -> DMatrix<f32> {
    // allocate space for Hessian
    let sizes = x0.shape();
    let n: usize = sizes.0;
    let mut hess = DMatrix::<f32>::zeros(n, n);
    let _f1 = approx_fprime(&x0, func, eps);

    let mut xx = DVector::<f32>::zeros(n);

    for i in 0..n {
        // temporary store the old value from initial matrix
        let temp = xx[(0,i)];
        xx[(0,i)] += eps;
        let _f2 = approx_fprime(&xx, func, eps);
        let delta = (_f2 - &_f1) / eps;
        hess.set_column(i,&delta);

        // restore to initial value
        xx[(0,i)] = temp;
    }
    println!("{:?}", hess);
    hess
}

/// Compute differentiation using finite difference method.
/// 
///          f(xk[i] + epsilon[i]) - f(xk[i])
/// f'[i] = ---------------------------------
///                     epsilon[i]
/// 
pub fn approx_fprime(x: &DVector<f32>, func: &dyn Fn(f32) -> f32, eps: f32) -> DVector<f32> {
    let xx = x.map(|entry| { func(entry) });
    let xx_prime = x.add_scalar(eps).map(|entry| { func(entry) });

    let diff: DVector<_> = (xx_prime - xx) / eps;
    diff
}
