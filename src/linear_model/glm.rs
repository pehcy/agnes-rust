extern crate nalgebra as na;

use na::{DMatrix, DVector};
use std::ops::Mul;

/// Linear Regression model
/// 
/// This model fits a linear model of the form 
/// of y = w_0 + w_1 * x_0 + w_2 * x_1 + ...
/// which aimed to minimize the residual sum of squared
/// between the observed value and target.
#[derive(Debug)]
pub struct LinearModel {
    w: Option<DVector<f64>>,
    fit_intercept: bool,
}

pub struct AnovaTest {
    sres: f64,
    tts: f64,
}

impl LinearModel {
    pub fn new(fit_intercept: bool) -> LinearModel {
        LinearModel {
            w: None,
            fit_intercept,
        }
    }

    /// Fit the model using formula (X.transpose() * X)^(-1) * X.transpose() * y
    /// 
    /// # Arguments
    /// 
    /// * `x_val` - parameters of shape 
    pub fn fit(&mut self, x_val: &DMatrix<f64>, y_val: &DVector<f64>) {
        if self.fit_intercept {
            let x_val = x_val.clone().insert_column(0, 1.0);
            self._fit(&x_val, y_val);
        }
        else {
            self._fit(x_val, y_val);
        }
    }

    fn _fit(&mut self, x_val: &DMatrix<f64>, y_val: &DVector<f64>) {
        self.w = Some(
            x_val
                .tr_mul(&x_val)
                .try_inverse()
                .unwrap()
                .mul(x_val.transpose())
                .mul(y_val)
        );
    }

    pub fn coef(&self) -> &Option<DVector<f64>> {
        &self.w
    }

    pub fn intercept(&self) -> Result<f64, String> {
        if !self.w.is_some() {
            return Err("Data input is unable to be fitted.".to_string());
        }
        if self.fit_intercept {
            return Ok(self.w.as_ref().unwrap()[0]);
        }
        
        return Err("Model was not fitted with intercept.".to_string());
    }

    /// Train linear model with gradient descent. Unlike `fit` function,
    /// this method required to loop through a given number of iterations 
    /// to obtain the optimal weights.
    /// 
    /// # Arguments
    /// 
    /// * `x_val` - parameters of shape 
    /// 
    /// * `lr` - learning rate, view as the step size of gradient descent.
    pub fn train(
        &mut self, 
        x_val: &DMatrix<f64>, 
        y_val: &DVector<f64>, 
        epochs: usize,
        lr: f64,
    ) {
        let mut thetas = DVector::from_row_slice(&vec![0f64; x_val.ncols()]);

        for i in 0..epochs {
            let costs_mtx = x_val.mul(&thetas) - y_val;
            let cost_total = costs_mtx.sum();

            thetas.add_scalar_mut(-lr / (x_val.nrows() as f64) * cost_total);
            println!("Iteration {}", i);
        }

        self.w = Some(thetas);
    }

}