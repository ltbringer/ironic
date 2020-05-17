struct NN <'a> {
    layers: &'a Vec<usize>,
    weights: Option<Vec<Vec<Vec<f64>>>>,
    bias: Option<Vec<Vec<f64>>>
}


fn make_nd_array(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(rows);
    for _ in 0..matrix.capacity() {
        let mut array: Vec<f64> = Vec::with_capacity(cols);
        for _ in 0..array.capacity() {
            array.push(0.0);
        }
        matrix.push(array);
    }
    return matrix
}

impl NN <'_> {
    fn init(&mut self) -> () {
        let layer_length: &usize = &self.layers.len();
        let excluding_last: usize = layer_length - 1;
        let row_slice: &[usize] = &self.layers[..excluding_last];
        let col_slice: &[usize] = &self.layers[1..];
        let mut weights: Vec<Vec<Vec<f64>>> = Vec::with_capacity(row_slice.len());
        for i in 0..weights.capacity() {
            weights.push(make_nd_array(row_slice[i], col_slice[i]));
        }
        self.weights = Some(weights);
    }
}


#[cfg(test)]
mod tests {
    use crate::NN;

    #[test]
    fn it_works() {
        let layers = vec![2, 3, 1];
        let mut nn = NN { layers, bias: None, weights: None };
        nn.init();
        assert_eq!(nn.layers.len(), 3);
        match nn.weights {
            Some(w) => {
                println!("ok {:?}", w);
                assert_eq!(6, 6);
            },
            None => panic!("Something's wrong I can feel it.")
        }
    }
}
