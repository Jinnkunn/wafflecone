pub fn z_score_noramlization(values: Vec<f64>) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::new();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let std_dev = (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64)
        .sqrt();
    for value in values {
        result.push((value - mean) / std_dev);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_z_score_noramlization() {
        // Test case 1
        let values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let expected1 = vec![
            -1.2649110640673518,
            -0.6324555320336759,
            0.0,
            0.6324555320336759,
            1.2649110640673518,
        ];
        assert_eq!(z_score_noramlization(values1), expected1);

        // Test case 2
        //     let values2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        //     let expected2 = vec![
        //         -1.2649110640673518,
        //         -0.6324555320336759,
        //         0.0,
        //         0.6324555320336759,
        //         1.2649110640673518,
        //     ];
        //     assert_eq!(z_score_noramlization(values2), expected2);

        //     // Test case 3
        //     let values3 = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        //     let expected3 = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        //     assert_eq!(z_score_noramlization(values3), expected3);
    }
}
