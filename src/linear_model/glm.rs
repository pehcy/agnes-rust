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
}

pub struct AnovaTest {
    sres: f64,
    tts: f64,
}

impl LinearModel {
    pub fn new() -> LinearModel {
        LinearModel {
            w: None
        }
    }

    /// Fit the model using formula (X.transpose() * X)^(-1) * X.transpose() * y
    /// 
    /// # Arguments
    /// 
    /// * `x_val` - parameters of shape 
    pub fn fit(&mut self, x_val: &DMatrix<f64>, y_val: &DVector<f64>) {
        let x_val = x_val.clone().insert_column(0, 1.0);
        self._fit(&x_val, y_val);
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

        Ok(self.w.as_ref().unwrap()[0])
    }

    /// Train linear model with gradient descent. Unlike `fit` function,
    /// this method required iteration epochs to obtain 
    /// optimal weights.
    pub fn train(&mut self, x_val: &DMatrix<f64>, y_val: &DVector<f64>) {

    }

}