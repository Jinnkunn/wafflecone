use nalgebra::{DMatrix, RowDVector, SVD};

pub struct PCA {
    mean: Vec<f64>,
    components: DMatrix<f64>,
    n_components: usize,
}

impl PCA {
    pub fn new(n_components: usize) -> Self {
        PCA {
            mean: Vec::new(),
            components: DMatrix::zeros(0, 0),
            n_components
        }
    }

    pub fn fit(mut self, x: DMatrix<f64>) -> Self {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Subtract the mean
        self.mean = x.row_mean().transpose().as_slice().to_vec();

        assert_eq!(self.mean.len(), n_features, "Mean vector size does not match the number of features in the input data");

        let mean_vector = RowDVector::from_vec(self.mean.clone());

        // Subtract the mean from each row
        let x_centered: Vec<RowDVector<f64>> = x.row_iter()
            .map(|row| row - &mean_vector)
            .collect();

        // Convert the Vec<RowDVector<f64>> to DMatrix<f64>
        let x_centered = DMatrix::from_rows(&x_centered);

        // Compute covariance matrix
        let covariance_matrix = x_centered.transpose() * x_centered / (n_samples as f64 - 1.0);

        // Compute SVD
        let svd = SVD::new(covariance_matrix, true, true);
        let u = svd.u.unwrap();

        // Select the top n_components principal components and convert to DMatrix
        self.components = u.columns(0, self.n_components).clone_owned();
        self
    }


    pub fn transform(&self, x: DMatrix<f64>) -> DMatrix<f64> {
        let mean_vector = RowDVector::from_vec(self.mean.clone());
        let rows: Vec<_> = x.row_iter()
            .map(|row| row - &mean_vector)
            .collect();
        let x_centered = DMatrix::from_rows(&rows);
        x_centered * &self.components
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::DMatrix;

    #[test]
    fn test_pca() {
        let x = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let pca = PCA::new(2).fit(x.clone());
        let x_transformed = pca.transform(x);
        let x_transformed_expected = DMatrix::from_row_slice(
            3,
            2,
            &[-2.82842712, 0.0, 0.0, 0.0, 2.82842712, 0.0],
        );
        assert_abs_diff_eq!(x_transformed, x_transformed_expected, epsilon = 1e-6);
    }

    #[test]
    fn test_pca2() {
        let x = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let pca = PCA::new(1).fit(x.clone());
        let x_transformed = pca.transform(x);
        let x_transformed_expected = DMatrix::from_row_slice(
            3,
            1,
            &[-2.82842712, 0.0, 2.82842712],
        );
        assert_abs_diff_eq!(x_transformed, x_transformed_expected, epsilon = 1e-6);
    }
}